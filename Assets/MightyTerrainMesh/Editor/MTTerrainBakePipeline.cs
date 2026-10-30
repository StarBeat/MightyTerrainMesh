namespace MightyTerrainMesh.EditorTools
{
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using Unity.Burst;
    using Unity.Collections;
    using Unity.Collections.LowLevel.Unsafe;
    using Unity.Jobs;
    using Unity.Mathematics;
    using UnityEditor;
    using UnityEditor.SceneManagement;
    using UnityEngine;

    /// <summary>
    /// 地形一键烘焙管线（静态 API，供 MTDataEditor / MTClipmapBaker / MTOneClickBakeWindow 复用）：
    /// 1. BakeMT      —— Terrain → MT 几何数据（mesh pack / 四叉树 / 高度图 / 细节 / 材质 / MTData 头）；
    /// 2. BakeClipmap —— Terrain → ClipmapVT 纹理源（L7/L8 全地形 + 区域高精块 + manifest，§7.1）；
    /// 3. PlaceInScene—— 场景中创建/更新 MTLoader + VTCreator +（可选）MTClipmapController。
    /// 重复烘焙覆盖同名资产且保留 MTData 的 GUID（场景引用不断）。
    /// </summary>
    public static class MTTerrainBakePipeline
    {
        // ============================================================
        // 1. MT 几何数据（逻辑提取自原 MTDataEditor，输出目录参数化）
        // ============================================================

        /// <summary>
        /// 烘焙 MT 几何数据。
        /// </summary>
        /// <param name="terrain">源地形</param>
        /// <param name="quadTreeDepth">四叉树深度（切片数 = 2^depth）</param>
        /// <param name="lodSettings">各 LOD 采样/坡度误差设置</param>
        /// <param name="datapack">每个 mesh 数据包文件容纳的 mesh 数</param>
        /// <param name="genUV2">是否生成 UV2</param>
        /// <param name="dataFolder">MTData 输出目录（Assets 下，如 Assets/ArtRes/BakedData/{name}）</param>
        /// <param name="meshDataFolder">mesh pack 输出目录（必须位于某 Resources 下且名为 MeshData，运行时装载机按 Resources.Load("MeshData/...") 读取）</param>
        /// <returns>创建/更新的 MTData 头资产</returns>
        public static MTData BakeMT(Terrain terrain, int quadTreeDepth, MeshLODCreate[] lodSettings,
            int datapack, bool genUV2, string dataFolder, string meshDataFolder)
        {
            if (terrain == null) { MTLog.LogError("no target terrain"); return null; }
            if (lodSettings == null || lodSettings.Length == 0) { MTLog.LogError("no lod setting"); return null; }
            datapack = Mathf.Max(1, datapack);

            // calculate min tri size
            int sliceCount = 1 << quadTreeDepth;
            int max_sub = lodSettings[0].Subdivision;
            float max_sub_grids = sliceCount * (1 << max_sub);
            float min_edge_len = Mathf.Max(terrain.terrainData.bounds.size.x, terrain.terrainData.bounds.size.z) / max_sub_grids;
            float minArea = min_edge_len * min_edge_len / 8f;
            //
            var tBnd = new Bounds(terrain.transform.TransformPoint(terrain.terrainData.bounds.center),
                terrain.terrainData.bounds.size);
            try
            {
                var dataCreateJob = new CreateDataJob(terrain, tBnd, quadTreeDepth, lodSettings, 0.5f * min_edge_len);
                for (int i = 0; i < int.MaxValue; ++i)
                {
                    dataCreateJob.Update();
                    EditorUtility.DisplayProgressBar("creating data", "scaning volumn", dataCreateJob.progress);
                    if (dataCreateJob.IsDone)
                        break;
                }
                dataCreateJob.EndProcess();
                //
                var tessellationJob = new TessellationDataJob(dataCreateJob.LODs, minArea);
                for (int i = 0; i < int.MaxValue; ++i)
                {
                    tessellationJob.Update();
                    EditorUtility.DisplayProgressBar("creating data", "tessellation", tessellationJob.progress);
                    if (tessellationJob.IsDone)
                        break;
                }
                //build quad tree data, this is the first level quadtree
                MTQuadTreeBuildNode treeRoot = new MTQuadTreeBuildNode(quadTreeDepth, tBnd.min, tBnd.max, Vector2.zero, Vector2.one);
                //output parameters
                EnsureAssetFolder(dataFolder);
                EnsureAssetFolder(meshDataFolder);
                var topFulllPath = ToFullPath(dataFolder);
                var meshFulllPath = ToFullPath(meshDataFolder);
                //
                MTData dataHeader = ScriptableObject.CreateInstance<MTData>();
                dataHeader.MeshDataPack = datapack;
                dataHeader.MeshPrefix = terrain.name;
                {
                    int packed = 0;
                    int start_meshId = -1;
                    MemoryStream stream = new MemoryStream();
                    for (int i = 0; i < tessellationJob.mesh.Length; ++i)
                    {
                        MTMeshData data = tessellationJob.mesh[i];
                        if (!treeRoot.AddMesh(data))
                        {
                            Debug.LogError("mesh can't insert into tree : " + data.meshId);
                        }
                        if (start_meshId < 0)
                            start_meshId = data.meshId;
                        EditorUtility.DisplayProgressBar("saving mesh data", "processing", (float)i / tessellationJob.mesh.Length);
                        if (packed % datapack == 0)
                        {
                            if (stream.Length > 0)
                            {
                                File.WriteAllBytes(string.Format("{0}/{1}_{2}.bytes", meshFulllPath, terrain.name, start_meshId), stream.ToArray());
                                stream.Close();
                                start_meshId = data.meshId;
                                stream = new MemoryStream();
                            }
                            packed = 0;
                            //预写入offset
                            for (int o = 0; o < datapack; ++o)
                            {
                                MTFileUtils.WriteInt(stream, 0);
                            }
                        }
                        //offset
                        int reserve = (int)stream.Position;
                        stream.Position = packed * sizeof(int);
                        MTFileUtils.WriteInt(stream, reserve);
                        stream.Position = reserve;
                        MTMeshUtils.Serialize(stream, data.lods[0]);
                        ++packed;
                    }
                    //last one
                    if (stream.Length > 0 && start_meshId >= 0)
                    {
                        File.WriteAllBytes(string.Format("{0}/{1}_{2}.bytes", meshFulllPath, terrain.name, start_meshId), stream.ToArray());
                        stream.Close();
                    }
                    AssetDatabase.Refresh();
                }
                {
                    List<MTQuadTreeNode> nodes = new List<MTQuadTreeNode>();
                    MTQuadTreeNode rootNode = new MTQuadTreeNode(0);
                    nodes.Add(rootNode);
                    ExportTree(treeRoot, rootNode, nodes);
                    EditorUtility.DisplayProgressBar("saving tree data", "processing", 0);
                    MemoryStream stream = new MemoryStream();
                    SerializeTrees(stream, nodes);
                    File.WriteAllBytes(string.Format("{0}/treeData.bytes", topFulllPath), stream.ToArray());
                    stream.Close();
                    AssetDatabase.Refresh();
                    dataHeader.TreeData = AssetDatabase.LoadAssetAtPath(string.Format("{0}/treeData.bytes", dataFolder), typeof(TextAsset)) as TextAsset;
                }
                List<string> detailMats = new List<string>();
                List<string> bakeAlbetoMats = new List<string>();
                List<string> bakeBumpMats = new List<string>();
                //materials for baking resource
                MTMatUtils.SaveMixMaterials(dataFolder, terrain.name, terrain, detailMats);
                //materials for baking texture
                MTMatUtils.SaveVTMaterials(dataFolder, terrain.name, terrain, bakeAlbetoMats, bakeBumpMats);
                dataHeader.DetailMats = new Material[detailMats.Count];
                for (int p = 0; p < detailMats.Count; ++p)
                {
                    dataHeader.DetailMats[p] = AssetDatabase.LoadAssetAtPath(detailMats[p], typeof(Material)) as Material;
                }
                dataHeader.BakeDiffuseMats = new Material[bakeAlbetoMats.Count];
                for (int p = 0; p < bakeAlbetoMats.Count; ++p)
                {
                    dataHeader.BakeDiffuseMats[p] = AssetDatabase.LoadAssetAtPath(bakeAlbetoMats[p], typeof(Material)) as Material;
                }
                dataHeader.BakeNormalMats = new Material[bakeBumpMats.Count];
                for (int p = 0; p < bakeBumpMats.Count; ++p)
                {
                    dataHeader.BakeNormalMats[p] = AssetDatabase.LoadAssetAtPath(bakeBumpMats[p], typeof(Material)) as Material;
                }
                //materials for baked texture
                Material bakedMat = new Material(Shader.Find("MT/TerrainVTLit"));
                bakedMat.EnableKeyword("_NORMALMAP");
                var bakedMatPath = string.Format("{0}/BakedMat.mat", dataFolder);
                var oldBakedMat = AssetDatabase.LoadAssetAtPath<Material>(bakedMatPath);
                if (oldBakedMat != null)
                    AssetDatabase.DeleteAsset(bakedMatPath);
                AssetDatabase.CreateAsset(bakedMat, bakedMatPath);
                dataHeader.BakedMat = AssetDatabase.LoadAssetAtPath(bakedMatPath, typeof(Material)) as Material;
                //export height map
                ExportHeightMap(dataHeader, terrain, topFulllPath, dataFolder);
                ExportDetails(dataHeader, terrain, topFulllPath, dataFolder);
                //
                string headerPath = string.Format("{0}/{1}.asset", dataFolder, terrain.name);
                var existing = AssetDatabase.LoadAssetAtPath<MTData>(headerPath);
                if (existing != null)
                {
                    // 重复烘焙：保留 GUID，场景中的 MTLoader.header 引用不断
                    EditorUtility.CopySerialized(dataHeader, existing);
                    Object.DestroyImmediate(dataHeader);
                    dataHeader = existing;
                    EditorUtility.SetDirty(dataHeader);
                }
                else
                {
                    AssetDatabase.CreateAsset(dataHeader, headerPath);
                }
                AssetDatabase.SaveAssets();
                return dataHeader;
            }
            finally
            {
                EditorUtility.ClearProgressBar();
                AssetDatabase.Refresh();
            }
        }

        // ============================================================
        // 2. ClipmapVT 纹理源（逻辑提取自原 MTClipmapBaker）
        // ============================================================

        private const float ClipmapBlockWorldSize = 32f; // 32m×32m 块（§7.1）

        /// <summary>烘焙压缩格式（§10）：ASTC 6x6/5x5（移动端）/ BC7（DX/桌面，D12）/ RGBA32（未压缩）。</summary>
        public enum ClipmapBakeFormat { RGBA32 = 0, ASTC = 1, BC7 = 2 }

        /// <summary>格式 → （albedo, normalAttr）目标 TextureFormat（通道打包不变，§10 Shader 免分支）。</summary>
        public static void BakeFormatsOf(ClipmapBakeFormat f, out TextureFormat fmtA, out TextureFormat fmtN)
        {
            switch (f)
            {
                case ClipmapBakeFormat.ASTC: fmtA = TextureFormat.ASTC_6x6; fmtN = TextureFormat.ASTC_5x5; break;
                case ClipmapBakeFormat.BC7:  fmtA = TextureFormat.BC7;      fmtN = TextureFormat.BC7;      break;
                default:                     fmtA = TextureFormat.RGBA32;   fmtN = TextureFormat.RGBA32;   break;
            }
        }

        /// <summary>
        /// 烘焙 Clipmap 纹理源到 outputRoot（须位于 Resources 下，如 Assets/Resources/ClipmapBaked）。
        /// 静态层 L_K（覆盖全地形，S/2040 texel）含离线生成的完整 Mip 链（§4.1）；
        /// 瓦片金字塔（§7.1）：L0 = 32m@2040²（1.57cm），区域模式仅烘焙 regions 内高精包（D8），
        /// ≥ coarseFullFromLevel 的粗级始终全量（片数少，保证顶层环面与近静态层可供流式直拷）。
        /// </summary>
        public static void BakeClipmap(Terrain terrain, int l7Res, int l8Res, ClipmapBakeFormat format,
            string outputRoot, List<Rect> regions, bool bakePyramid,
            bool pyramidFullCoverage = true, int pyramidCoarseFullFromLevel = 3,
            TextureCompressionQuality pyramidFineQuality = TextureCompressionQuality.Fast)
        {
            if (terrain == null) { MTLog.LogError("no target terrain"); return; }
            Directory.CreateDirectory(outputRoot);
            var src = ClipmapSourceRuntime.FromTerrain(terrain);
            var origin = terrain.transform.position;
            float size = terrain.terrainData.size.x;
            BakeFormatsOf(format, out var fmtA, out var fmtN);

            var manifest = new ClipmapManifest
            {
                textureDataVersion = ClipmapManifest.CurrentTextureDataVersion,
                originX = origin.x,
                originZ = origin.z,
                terrainSize = size,
                l7Res = l7Res,
                l8Res = l8Res,
                blockRes = 0,
                blockRects = new Rect[0], // 区域高精块已由瓦片金字塔取代（保留字段兼容旧数据）
                texFormat = (int)fmtA,
                texFormatN = (int)fmtN,
                materialChannels = 1,
            };

            var swTotal = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                // L7 仅供无金字塔旧数据的 Job 回退采样；直拷模式（烘焙金字塔存在时）运行时从不加载，跳过。
                // 静态层 L_K：全地形 S/2040 texel，完整 Mip 链（§4.1 离线生成）。
                if (!bakePyramid)
                    BakeClipmapLayer(src, "l7", l7Res, new float2(origin.x, origin.z), size / l7Res, false, fmtA, fmtN, outputRoot);
                else
                {
                    // 清掉旧烘焙残留的 L7（直拷模式不加载，避免 Viewer 显示过期数据）
                    foreach (var n in new[] { "l7_albedo", "l7_normal" })
                    {
                        string stale = Path.Combine(outputRoot, n + ".bytes");
                        if (File.Exists(stale)) File.Delete(stale);
                    }
                }
                BakeClipmapLayer(src, "l8", l8Res, new float2(origin.x, origin.z), size / l8Res, true, fmtA, fmtN, outputRoot);

                // 瓦片金字塔（§7.1 流式契约）：L0 与 clipmap L0 层同密度（1.57cm@2040²），逐级盒滤波合并
                if (bakePyramid)
                {
                    int n = BakeTilePyramid(src, terrain, 2040, ClipmapBlockWorldSize, fmtA, fmtN, outputRoot, manifest,
                        pyramidFullCoverage, regions, pyramidCoarseFullFromLevel, pyramidFineQuality);
                    Debug.Log($"[ClipmapBaker] 金字塔：{manifest.tileLevels} 级，L0 {n} 片，StreamingAssets 归档");
                }
                else { manifest.tileLevels = 0; }

                File.WriteAllText(Path.Combine(outputRoot, "manifest.json"), JsonUtility.ToJson(manifest, true));
                AssetDatabase.Refresh();
                Debug.Log($"[ClipmapBaker] 完成：L7 {l7Res}² / L8 {l8Res}² / 格式 {format} → {outputRoot}（总耗时 {swTotal.Elapsed.TotalSeconds:F1}s）");
            }
            finally
            {
                src.Dispose();
                EditorUtility.ClearProgressBar();
            }
        }

        private static void BakeClipmapLayer(ClipmapSourceRuntime src, string name, int res, float2 worldMin, float texel,
            bool mipChain, TextureFormat fmtA, TextureFormat fmtN, string outputRoot)
        {
            EditorUtility.DisplayProgressBar("Clipmap Baker", name, 0f);
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var pxA = new NativeArray<Color32>(res * res, Allocator.TempJob);
            var pxN = new NativeArray<Color32>(res * res, Allocator.TempJob);
            try
            {
                var jobA = src.CreateJob(0, true, worldMin, texel, res, pxA, res, 0);
                var jobN = src.CreateJob(0, false, worldMin, texel, res, pxN, res, 0);
                var hA = jobA.Schedule(res * res, 64);
                var hN = jobN.Schedule(res * res, 64);
                JobHandle.CompleteAll(ref hA, ref hN);

                // 诊断：albedo 均值与近黑像素占比（抽样 1/16）。
                // 均值≈0 且近黑≈100% → alphamap 权重全 0（地形无层/层数据异常）；
                // 均匀 ≈(128,128,128) → 层纹理不可读回退灰（Console 会有对应警告）。
                long sr = 0, sg = 0, sb = 0; int dark = 0, ns = 0;
                for (int i = 0; i < pxA.Length; i += 16)
                {
                    var c = pxA[i];
                    sr += c.r; sg += c.g; sb += c.b; ++ns;
                    if (c.r < 16 && c.g < 16 && c.b < 16) ++dark;
                }
                Debug.Log($"[ClipmapBaker] {name}_albedo 均值=({sr / ns},{sg / ns},{sb / ns}) 近黑像素={dark * 100 / ns}%");

                string pathA = Path.Combine(outputRoot, name + "_albedo.bytes");
                string pathN = Path.Combine(outputRoot, name + "_normal.bytes");
                SaveClipmapTex(pxA, res, pathA, true, fmtA, mipChain,
                    TextureCompressionQuality.Normal, out var rawA);
                SaveClipmapTex(pxN, res, pathN, false, fmtN, mipChain,
                    TextureCompressionQuality.Normal, out var rawN);
                VerifyBakedTex(pathA, rawA, res, mipChain, true, fmtA, name + "_albedo");
                VerifyBakedTex(pathN, rawN, res, mipChain, false, fmtN, name + "_normal");
                Debug.Log($"[ClipmapBaker] {name} 层完成：{res}²（耗时 {sw.Elapsed.TotalSeconds:F1}s）");
            }
            finally
            {
                pxA.Dispose();
                pxN.Dispose();
            }
        }

        /// <summary>估算金字塔级数与总片数（UI 显示用）。区域模式下 j&lt;coarseFullFrom 按区域面积近似。</summary>
        public static void EstimatePyramid(float terrainSize, float tileWorld0,
            bool fullCoverage, float regionArea, int coarseFullFrom, out int levels, out long tiles)
        {
            levels = Mathf.Max(1, Mathf.CeilToInt(Mathf.Log(terrainSize / tileWorld0, 2f)));
            tiles = 0;
            for (int i = 0; i < levels; ++i)
            {
                float tw = tileWorld0 * (1 << i);
                long g = Mathf.Max(1, Mathf.CeilToInt(terrainSize / tw));
                long total = g * g;
                if (fullCoverage || i >= coarseFullFrom) tiles += total;
                else tiles += System.Math.Min(total, (long)Mathf.CeilToInt(regionArea / (tw * tw)) + 2 * g); // +边缘
            }
        }

        // ---------------- 瓦片金字塔（§7.1 流式契约 / D8 区域高精包） ----------------

        /// <summary>金字塔向上合并时的 2×2 子块累积缓冲（(tileRes*2)² 组合区）。</summary>
        private class PyramidAccum
        {
            public NativeArray<Color32> a, n;
            public int arrived;
        }

        /// <summary>
        /// 烘焙瓦片金字塔：Level 0 = tileWorld0 米/tileRes²（与 clipmap L0 层同密度，1.57cm@2040²），
        /// 由地形直接采样。两种覆盖模式（D8）：
        /// - fullCoverage：全地形 L0 + 逐级盒滤波合并（盒滤波抗走样质量最优，粗级也从子块传播）；
        /// - 区域模式：L0 仅烘与 regions 相交的瓦片（高精包），区域父级（j&lt;coarseFullFrom）需 4 子块齐聚才生成；
        ///   j ≥ coarseFullFrom 的粗级全网格直接从地形按该级密度采样（片数少，保证顶层环面可供流式直拷）。
        /// 级数 = ceil(log2(S/tileWorld0))（顶级覆盖 S/2，与运行时 K=log2(S)−5 的顶层环面 footprint 对齐）。
        /// 输出 StreamingAssets 归档 tiles.bin + tiles.idx（L0 质量 fineQuality，≥L1 恒为 Fast），并回填 manifest。
        /// 进度条可取消（取消后需重新完整烘焙）。返回 L0 实际烘焙片数。
        /// </summary>
        public static int BakeTilePyramid(ClipmapSourceRuntime src, Terrain terrain, int tileRes, float tileWorld0,
            TextureFormat fmtA, TextureFormat fmtN, string outputRoot, ClipmapManifest manifest,
            bool fullCoverage, List<Rect> regions, int coarseFullFromLevel,
            TextureCompressionQuality fineQuality = TextureCompressionQuality.Fast)
        {
            float size = terrain.terrainData.size.x;
            var origin = terrain.transform.position;
            int levels = Mathf.Max(1, Mathf.CeilToInt(Mathf.Log(size / tileWorld0, 2f)));
            manifest.tileRes = tileRes;
            manifest.tileWorld0 = tileWorld0;
            manifest.tileLevels = levels;

            // 瓦片归档（StreamingAssets）：单文件 tiles.bin + 索引 tiles.idx，取代逐瓦片 .bytes
            // （8192m 全量 13 万文件 → 2 个文件；构建后 Resources 不可按路径读，归档走 File IO）。
            // 旧版逐文件 tiles/ 目录删除防残留。
            using var arc = new TileArchiveWriter(ToFullPath(StreamingRootOf(outputRoot)));
            // ASTC 双格式且有 astcenc → 外部进程池并行压缩；否则 Unity 内置（单线程）兜底
            using var comp = CreateTileCompressor(arc, fmtA, fmtN, tileRes);
            string legacyTileDir = Path.Combine(outputRoot, "tiles");
            if (Directory.Exists(legacyTileDir)) Directory.Delete(legacyTileDir, true);
            manifest.tileArchive = 1;

            // 区域模式：卷积传播止于 coarseFullFrom-1 级父块；粗级全量直接烘焙
            int directFrom = fullCoverage ? levels : Mathf.Clamp(coarseFullFromLevel, 1, levels - 1);

            // ---- L0 目标集合（区域模式严格按 regions；空列表 = 无高精包，与 EstimatePyramid 一致）----
            if (!fullCoverage && (regions == null || regions.Count == 0))
                Debug.LogWarning("[ClipmapBaker] 区域模式但未配置任何区域：L0 高精包为空（仅烘焙粗级全量层）");
            int gx0 = Mathf.Max(1, Mathf.CeilToInt(size / tileWorld0));
            var bake0 = new bool[gx0, gx0];
            int total0 = 0;
            for (int tz = 0; tz < gx0; ++tz)
            for (int tx = 0; tx < gx0; ++tx)
            {
                bool hit = fullCoverage;
                if (!hit && regions != null)
                {
                    var tileRect = new Rect(origin.x + tx * tileWorld0, origin.z + tz * tileWorld0, tileWorld0, tileWorld0);
                    foreach (var r in regions)
                        if (tileRect.Overlaps(r)) { hit = true; break; }
                }
                bake0[tx, tz] = hit;
                if (hit) ++total0;
            }

            var pending = new Dictionary<long, PyramidAccum>[levels];
            for (int i = 0; i < levels; ++i) pending[i] = new Dictionary<long, PyramidAccum>();
            try
            {
                // 父块优先扫描：同一父块的 2×2 子块连续烘焙，父块即时收尾释放——
                // 同时挂起的累积器 ≈ 层数个（行优先扫描会挂起 ~gx 个，2040² 下单个 33MB 不可接受）
                int gxP = (gx0 + 1) / 2;
                var seq = new List<int>(total0); // tx | tz<<16，保持父块优先顺序
                for (int pz = 0; pz < gxP; ++pz)
                for (int px = 0; px < gxP; ++px)
                for (int dz = 0; dz < 2; ++dz)
                for (int dx = 0; dx < 2; ++dx)
                {
                    int tx = px * 2 + dx, tz = pz * 2 + dz;
                    if (tx >= gx0 || tz >= gx0 || !bake0[tx, tz]) continue;
                    seq.Add(tx | (tz << 16));
                }

                // 双缓冲流水：压缩/存盘瓦片 i（主线程）与采样瓦片 i+1（worker 线程）重叠。
                // 此前 Schedule 后立即 CompleteAll，采样时主线程空等、压缩时 worker 全闲。
                int baked = 0;
                var swL0 = System.Diagnostics.Stopwatch.StartNew();
                var bufA = new NativeArray<Color32>[2];
                var bufN = new NativeArray<Color32>[2];
                var hA = new JobHandle[2];
                var hN = new JobHandle[2];
                void ScheduleL0(int slot, int seqIdx)
                {
                    int tx = seq[seqIdx] & 0xFFFF, tz = seq[seqIdx] >> 16;
                    bufA[slot] = new NativeArray<Color32>(tileRes * tileRes, Allocator.TempJob);
                    bufN[slot] = new NativeArray<Color32>(tileRes * tileRes, Allocator.TempJob);
                    float2 wmin = new float2(origin.x + tx * tileWorld0, origin.z + tz * tileWorld0);
                    float texel = tileWorld0 / tileRes;
                    var jA = src.CreateJob(0, true, wmin, texel, tileRes, bufA[slot], tileRes, 0);
                    var jN = src.CreateJob(0, false, wmin, texel, tileRes, bufN[slot], tileRes, 0);
                    hA[slot] = jA.Schedule(tileRes * tileRes, 256);
                    hN[slot] = jN.Schedule(tileRes * tileRes, 256);
                }
                void CancelL0(int slot) // 清理唯一挂起的槽位（取消路径）
                {
                    JobHandle.CompleteAll(ref hA[slot], ref hN[slot]);
                    if (bufA[slot].IsCreated) bufA[slot].Dispose();
                    if (bufN[slot].IsCreated) bufN[slot].Dispose();
                }
                if (seq.Count > 0) ScheduleL0(0, 0);
                for (int i = 0; i < seq.Count; ++i)
                {
                    int slot = i & 1;
                    int tx = seq[i] & 0xFFFF, tz = seq[i] >> 16;
                    if (EditorUtility.DisplayCancelableProgressBar("瓦片金字塔",
                            $"L0 {baked + 1}/{total0}  ({tx},{tz})", (float)baked / Mathf.Max(1, total0)))
                    {
                        CancelL0(slot);
                        Debug.LogWarning("[ClipmapBaker] 金字塔烘焙已取消（tiles/ 不完整，请重新完整烘焙）");
                        return baked;
                    }
                    if (i + 1 < seq.Count) ScheduleL0(slot ^ 1, i + 1); // 先发射下一片，再收尾本片
                    JobHandle.CompleteAll(ref hA[slot], ref hN[slot]);
                    comp.Compress(0, tx, tz, bufA[slot], bufN[slot], fineQuality);
                    PropagatePyramid(pending, levels, size, tileRes, tileWorld0, 0, tx, tz, bufA[slot], bufN[slot], comp, directFrom);
                    bufA[slot].Dispose(); bufN[slot].Dispose();
                    ++baked;
                }
                // 全量模式下所有父块应收尾完毕；区域模式下区域边缘之外的父块不生成属预期
                if (fullCoverage)
                    for (int i = 0; i < levels; ++i)
                        if (pending[i].Count != 0)
                            Debug.LogWarning($"[ClipmapBaker] 金字塔第 {i} 级有 {pending[i].Count} 个父块未收尾（网格非 2 幂？）");
                Debug.Log($"[ClipmapBaker] 金字塔 L0+逐级传播完成：{baked}/{total0} 片（耗时 {swL0.Elapsed.TotalSeconds:F1}s）");

                // ---- 粗级全量直接烘焙（区域模式；j = directFrom..levels-1）----
                for (int lvl = Mathf.Max(1, directFrom); lvl < levels; ++lvl)
                {
                    var swLvl = System.Diagnostics.Stopwatch.StartNew();
                    float tw = tileWorld0 * (1 << lvl);
                    int gc = Mathf.Max(1, Mathf.CeilToInt(size / tw));
                    int total = gc * gc, done = 0;
                    for (int tz = 0; tz < gc; ++tz)
                    for (int tx = 0; tx < gc; ++tx)
                    {
                        if (EditorUtility.DisplayCancelableProgressBar("瓦片金字塔（粗级全量）",
                                $"L{lvl} {done + 1}/{total}  ({tx},{tz})", (float)done / total))
                        {
                            Debug.LogWarning("[ClipmapBaker] 金字塔烘焙已取消（tiles/ 不完整，请重新完整烘焙）");
                            return baked;
                        }
                        BakeTileDirect(src, origin, tileRes, tw, lvl, tx, tz, comp);
                        ++done;
                    }
                    Debug.Log($"[ClipmapBaker] 金字塔 L{lvl} 粗级全量完成：{done} 片（耗时 {swLvl.Elapsed.TotalSeconds:F1}s）");
                }
                comp.Flush(); // 等待 astcenc 队列全部收尾写入归档（Unity 压缩路径为空操作）
                Debug.Log($"[ClipmapBaker] 瓦片归档：{arc.Count} 片 → {StreamingRootOf(outputRoot)}/tiles.bin");
                return baked;
            }
            finally
            {
                DisposePending(pending);
            }
        }

        private static void DisposePending(Dictionary<long, PyramidAccum>[] pending)
        {
            foreach (var map in pending)
            foreach (var kv in map)
            {
                kv.Value.a.Dispose();
                kv.Value.n.Dispose();
            }
        }

        /// <summary>按 (lvl, tx, tz) 的世界矩形从地形直接采样一片瓦片，压缩入队（粗级，质量 Fast）。</summary>
        private static void BakeTileDirect(ClipmapSourceRuntime src, Vector3 origin, int tileRes, float tileWorld,
            int lvl, int tx, int tz, ITileCompressor comp)
        {
            var pxA = new NativeArray<Color32>(tileRes * tileRes, Allocator.TempJob);
            var pxN = new NativeArray<Color32>(tileRes * tileRes, Allocator.TempJob);
            float2 wmin = new float2(origin.x + tx * tileWorld, origin.z + tz * tileWorld);
            float texel = tileWorld / tileRes;
            var jobA = src.CreateJob(0, true, wmin, texel, tileRes, pxA, tileRes, 0);
            var jobN = src.CreateJob(0, false, wmin, texel, tileRes, pxN, tileRes, 0);
            var hA = jobA.Schedule(tileRes * tileRes, 256);
            var hN = jobN.Schedule(tileRes * tileRes, 256);
            JobHandle.CompleteAll(ref hA, ref hN);
            comp.Compress(lvl, tx, tz, pxA, pxN, TextureCompressionQuality.Fast);
            pxA.Dispose(); pxN.Dispose();
        }

        /// <summary>子块到位后向上卷积：子块即时盒滤波降采样写入父象限（累积器仅 tileRes²，省 4× 内存）；
        /// 集齐父块的全部现存子块 → 边缘复制补齐 → 存盘 → 递归向上。
        /// 卷积止于 stopBeforeLevel 级（该级起由 BakeTileDirect 全量直接烘焙，或已到顶级）。</summary>
        private static void PropagatePyramid(Dictionary<long, PyramidAccum>[] pending, int levels, float terrainSize,
            int tileRes, float tileWorld0, int level, int tx, int tz,
            NativeArray<Color32> cA, NativeArray<Color32> cN,
            ITileCompressor comp, int stopBeforeLevel)
        {
            if (level + 1 >= levels || level + 1 >= stopBeforeLevel) return;
            int px = tx >> 1, pz = tz >> 1;
            int qx = tx & 1, qz = tz & 1;
            int q = tileRes / 2; // 父象限边长
            var map = pending[level];
            long key = ((long)px << 20) | (uint)pz;
            if (!map.TryGetValue(key, out var acc))
            {
                acc = new PyramidAccum
                {
                    a = new NativeArray<Color32>(tileRes * tileRes, Allocator.Persistent),
                    n = new NativeArray<Color32>(tileRes * tileRes, Allocator.Persistent)
                };
                map.Add(key, acc);
            }
            Downsample2x(cA, cN, tileRes, out var downA, out var downN);
            CopyQuadrantSmall(downA, acc.a, q, qx, qz);
            CopyQuadrantSmall(downN, acc.n, q, qx, qz);
            downA.Dispose();
            downN.Dispose();
            acc.arrived++;
            // 该父块在 level 网格内的实际子块数（地形边缘不足 2×2）
            int gc = Mathf.Max(1, Mathf.CeilToInt(terrainSize / (tileWorld0 * (1 << level))));
            int expect = Mathf.Clamp(gc - px * 2, 1, 2) * Mathf.Clamp(gc - pz * 2, 1, 2);
            if (acc.arrived < expect) return;
            // 缺失象限用最近现存象限复制（边缘复制，保证内容完整）
            int maxQx = Mathf.Clamp(gc - px * 2 - 1, 0, 1);
            int maxQz = Mathf.Clamp(gc - pz * 2 - 1, 0, 1);
            for (int z = 0; z <= 1; ++z)
            for (int x = 0; x <= 1; ++x)
            {
                if (x <= maxQx && z <= maxQz) continue;
                CopyQuadrantFrom(acc.a, q, Mathf.Min(x, maxQx), Mathf.Min(z, maxQz), x, z);
                CopyQuadrantFrom(acc.n, q, Mathf.Min(x, maxQx), Mathf.Min(z, maxQz), x, z);
            }
            map.Remove(key);
            try
            {
                // 粗级（≥L1）用 Fast：盒滤波已抹平细节，像素足迹大，Normal 的视觉收益不可见、耗时数倍
                comp.Compress(level + 1, px, pz, acc.a, acc.n, TextureCompressionQuality.Fast);
                PropagatePyramid(pending, levels, terrainSize, tileRes, tileWorld0, level + 1, px, pz, acc.a, acc.n, comp, stopBeforeLevel);
            }
            finally
            {
                acc.a.Dispose();
                acc.n.Dispose();
            }
        }

        /// <summary>
        /// 压缩一片瓦片并追加到归档（albedo 在前、normal 在后）。
        /// 两张缓存纹理分别对应 albedo（sRGB）/ normal（linear），GetRawTextureData 别名写出后立即可复用。
        /// </summary>
        private static void AppendTileCompressed(TileArchiveWriter arc, int level, int tx, int tz,
            NativeArray<Color32> pxA, NativeArray<Color32> pxN, int tileRes,
            TextureFormat fmtA, TextureFormat fmtN, TextureCompressionQuality quality)
        {
            long offset = arc.Position;
            var texA = GetBakeTex(tileRes, false, false);
            texA.SetPixelData(pxA, 0);
            texA.Apply(false);
            if (fmtA != TextureFormat.RGBA32)
                EditorUtility.CompressTexture(texA, fmtA, quality);
            var rawA = texA.GetRawTextureData<byte>();
            int lenA = rawA.Length;
            arc.Write(rawA);

            var texN = GetBakeTex(tileRes, false, true);
            texN.SetPixelData(pxN, 0);
            texN.Apply(false);
            if (fmtN != TextureFormat.RGBA32)
                EditorUtility.CompressTexture(texN, fmtN, quality);
            var rawN = texN.GetRawTextureData<byte>();
            int lenN = rawN.Length;
            arc.Write(rawN);

            arc.AddRecord(level, tx, tz, offset, lenA, lenN);
        }

        /// <summary>
        /// 瓦片归档写出器：tiles.bin（顺序追加的压缩字节流，albedo+normal 交替）+
        /// tiles.idx（二进制索引，Dispose 时落盘：magic 'MTTI' / 版本 1 / N × {level,tx,tz,offset,lenA,lenN}）。
        /// </summary>
        private sealed class TileArchiveWriter : System.IDisposable
        {
            private struct Rec { public int level, tx, tz; public long offset; public int lenA, lenN; }

            private readonly FileStream _fs;
            private readonly string _idxPath;
            private readonly List<Rec> _recs = new List<Rec>();

            public int Count => _recs.Count;
            public long Position => _fs.Position;

            public TileArchiveWriter(string fullDir)
            {
                Directory.CreateDirectory(fullDir);
                _fs = new FileStream(Path.Combine(fullDir, "tiles.bin"), FileMode.Create, FileAccess.Write,
                    FileShare.None, 1 << 20, FileOptions.SequentialScan);
                _idxPath = Path.Combine(fullDir, "tiles.idx");
            }

            public unsafe void Write(NativeArray<byte> raw)
            {
                var span = new System.ReadOnlySpan<byte>(
                    (byte*)NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(raw), raw.Length);
                _fs.Write(span);
            }

            public void Write(byte[] data, int offset, int length) => _fs.Write(data, offset, length);

            public void AddRecord(int level, int tx, int tz, long offset, int lenA, int lenN)
                => _recs.Add(new Rec { level = level, tx = tx, tz = tz, offset = offset, lenA = lenA, lenN = lenN });

            public void Dispose()
            {
                _fs.Dispose();
                using (var bw = new BinaryWriter(new FileStream(_idxPath, FileMode.Create, FileAccess.Write, FileShare.None)))
                {
                    bw.Write(0x4954544D); // 'MTTI'
                    bw.Write(1);          // 索引格式版本
                    bw.Write(_recs.Count);
                    foreach (var r in _recs)
                    {
                        bw.Write(r.level); bw.Write(r.tx); bw.Write(r.tz);
                        bw.Write(r.offset); bw.Write(r.lenA); bw.Write(r.lenN);
                    }
                }
            }
        }

        // ---------------- 瓦片压缩器（Unity 内置回退 / 外部编码器进程池：astcenc / Compressonator） ----------------

        private interface ITileCompressor : System.IDisposable
        {
            void Compress(int level, int tx, int tz,
                NativeArray<Color32> pxA, NativeArray<Color32> pxN, TextureCompressionQuality quality);
            void Flush();       // 等待全部压缩完成并写入归档（正常结束路径调用）
            int PendingCount { get; }
        }

        /// <summary>ASTC → astcenc、BC7 → Compressonator；双通道编码器齐备才用进程池，否则 Unity 内置主线程压缩。</summary>
        private static ITileCompressor CreateTileCompressor(TileArchiveWriter arc,
            TextureFormat fmtA, TextureFormat fmtN, int tileRes)
        {
            var encA = CreateTileEncoder(fmtA);
            var encN = CreateTileEncoder(fmtN);
            if (encA != null && encN != null)
                return new ProcessTileCompressor(arc, encA, encN, tileRes);
            if (encA != null || encN != null || IsAstc(fmtA) || IsAstc(fmtN) || fmtA == TextureFormat.BC7 || fmtN == TextureFormat.BC7)
                Debug.LogWarning("[ClipmapBaker] 外部编码器不齐备（ASTC→astcenc / BC7→Compressonator），" +
                                 "全程使用 Unity 内置压缩（较慢）；菜单 MightyTerrainMesh/ 下有对应下载项");
            return new UnityTileCompressor(arc, fmtA, fmtN, tileRes);
        }

        private static TileEncoder CreateTileEncoder(TextureFormat f)
        {
            if (IsAstc(f)) return MTAstcEncoder.Locate() != null ? new AstcTileEncoder(f) : null;
            if (f == TextureFormat.BC7) return MTBc7Encoder.Locate() != null ? new Bc7TileEncoder() : null;
            return null;
        }

        private static bool IsAstc(TextureFormat f)
            => f == TextureFormat.ASTC_6x6 || f == TextureFormat.ASTC_5x5;

        /// <summary>
        /// 单通道外部编码器适配：中间文件读写 + 运行参数 + 输出头剥离。
        /// 方向约定：ASTC 用 TGA 左下原点（与 Unity raw 行序一致）；BC 工具链用 DDS（top-down 严格约定，
        /// 写入时翻转让"图顶 = 纹理底"，输出块行序即与 Unity raw 一致）。
        /// </summary>
        private abstract class TileEncoder
        {
            protected readonly byte[] _buf = new byte[1 << 18]; // 转换暂存（仅主线程写输入用）
            public abstract TextureFormat Format { get; } // 目标压缩格式（Unity 压缩回退用）
            public abstract string InExt { get; }
            public abstract string OutExt { get; }
            public abstract void WriteInput(string path, NativeArray<Color32> px, int res);
            public abstract bool Run(string inPath, string outPath, bool srgb,
                TextureCompressionQuality quality, int jobs, out string err);
            public abstract int DataOffset(byte[] outFile); // 压缩数据起始偏移（-1 = 文件非法）
            public abstract NativeArray<Color32> ReadPixels(string path, int res); // Unity 压缩回退用
        }

        /// <summary>astcenc：TGA（32bpp BGRA，descriptor 0x08 左下原点——行序与 Unity raw 一致）。</summary>
        private sealed class AstcTileEncoder : TileEncoder
        {
            private readonly string _block;
            private readonly TextureFormat _format;
            public AstcTileEncoder(TextureFormat f)
            { _block = f == TextureFormat.ASTC_6x6 ? "6x6" : "5x5"; _format = f; }
            public override TextureFormat Format => _format;
            public override string InExt => ".tga";
            public override string OutExt => ".astc";
            public override bool Run(string inPath, string outPath, bool srgb,
                TextureCompressionQuality quality, int jobs, out string err)
                => MTAstcEncoder.Run(inPath, outPath, _block, srgb, quality, jobs, out err);
            public override int DataOffset(byte[] outFile) => 16; // .astc 文件头 16 字节

            public override unsafe void WriteInput(string path, NativeArray<Color32> px, int res)
            {
                using (var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None,
                           1 << 20, FileOptions.SequentialScan))
                {
                    var header = new byte[18];
                    header[2] = 2; // 非压缩真彩
                    header[12] = (byte)(res & 0xFF); header[13] = (byte)(res >> 8);
                    header[14] = (byte)(res & 0xFF); header[15] = (byte)(res >> 8);
                    header[16] = 32;
                    header[17] = 0x08; // 8 alpha 位 + 左下原点
                    fs.Write(header, 0, header.Length);
                    var src = (Color32*)NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(px);
                    int chunk = _buf.Length / 4;
                    for (int off = 0; off < px.Length; off += chunk)
                    {
                        int n = Mathf.Min(chunk, px.Length - off);
                        for (int i = 0; i < n; ++i)
                        {
                            var c = src[off + i];
                            _buf[i * 4] = c.b; _buf[i * 4 + 1] = c.g;
                            _buf[i * 4 + 2] = c.r; _buf[i * 4 + 3] = c.a;
                        }
                        fs.Write(_buf, 0, n * 4);
                    }
                }
            }

            public override unsafe NativeArray<Color32> ReadPixels(string path, int res)
            {
                var bytes = File.ReadAllBytes(path); // 头 18 字节、BGRA、左下原点（自己写出的格式）
                var px = new NativeArray<Color32>(res * res, Allocator.TempJob);
                fixed (byte* src = bytes)
                {
                    var dst = (Color32*)NativeArrayUnsafeUtility.GetUnsafePtr(px);
                    for (int i = 0; i < px.Length; ++i)
                    {
                        byte* s = src + 18 + i * 4;
                        dst[i] = new Color32(s[2], s[1], s[0], s[3]);
                    }
                }
                return px;
            }
        }

        /// <summary>Compressonator（BC7）：DDS 容器（128B 头 A8R8G8B8 未压缩，top-down——写入翻转行使图顶 = 纹理底）。</summary>
        private sealed class Bc7TileEncoder : TileEncoder
        {
            public override TextureFormat Format => TextureFormat.BC7;
            public override string InExt => ".dds";
            public override string OutExt => ".dds";
            public override bool Run(string inPath, string outPath, bool srgb,
                TextureCompressionQuality quality, int jobs, out string err)
                => MTBc7Encoder.Run(inPath, outPath, jobs, out err);

            /// <summary>DDS 数据偏移：DX10 扩展头（BC7 必有，'DX10' FourCC）→ 148，否则 128；-1 = 非法文件。</summary>
            public override int DataOffset(byte[] outFile)
            {
                if (outFile.Length < 128 || outFile[0] != 'D' || outFile[1] != 'D'
                    || outFile[2] != 'S' || outFile[3] != ' ') return -1;
                bool dx10 = outFile.Length >= 148 && outFile[84] == 'D' && outFile[85] == 'X'
                    && outFile[86] == '1' && outFile[87] == '0';
                return dx10 ? 148 : 128;
            }

            public override unsafe void WriteInput(string path, NativeArray<Color32> px, int res)
            {
                using (var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None,
                           1 << 20, FileOptions.SequentialScan))
                {
                    var bw = new BinaryWriter(fs, System.Text.Encoding.ASCII, true);
                    bw.Write(0x20534444);       // "DDS "
                    bw.Write(124);              // 头尺寸
                    bw.Write(0x0000100F);       // CAPS|HEIGHT|WIDTH|PITCH|PIXELFORMAT
                    bw.Write(res);              // height
                    bw.Write(res);              // width
                    bw.Write(res * 4);          // pitch
                    bw.Write(0);                // depth
                    bw.Write(0);                // mipmaps
                    for (int i = 0; i < 11; ++i) bw.Write(0); // reserved
                    bw.Write(32);               // ddspf.size
                    bw.Write(0x41);             // DDPF_RGB | DDPF_ALPHAPIXELS
                    bw.Write(0);                // fourCC（未压缩）
                    bw.Write(32);               // bitCount
                    bw.Write(0x00FF0000);       // R mask
                    bw.Write(0x0000FF00);       // G mask
                    bw.Write(0x000000FF);       // B mask
                    bw.Write((uint)0xFF000000); // A mask → 内存字节序 B,G,R,A
                    bw.Write(0x1000);           // DDSCAPS_TEXTURE
                    bw.Write(0); bw.Write(0); bw.Write(0); bw.Write(0);
                    bw.Flush();
                    // DDS top-down：先写纹理底行（px 行 0）→ 图顶 = 纹理底，输出块序与 Unity raw 一致
                    var src = (Color32*)NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(px);
                    int chunk = _buf.Length / 4;
                    for (int off = 0; off < px.Length; off += chunk)
                    {
                        int n = Mathf.Min(chunk, px.Length - off);
                        for (int i = 0; i < n; ++i)
                        {
                            var c = src[off + i];
                            _buf[i * 4] = c.b; _buf[i * 4 + 1] = c.g;
                            _buf[i * 4 + 2] = c.r; _buf[i * 4 + 3] = c.a;
                        }
                        fs.Write(_buf, 0, n * 4);
                    }
                    bw.Dispose();
                }
            }

            public override unsafe NativeArray<Color32> ReadPixels(string path, int res)
            {
                var bytes = File.ReadAllBytes(path);
                int off = DataOffset(bytes);
                if (off < 0) throw new System.Exception("DDS 输入非法: " + path);
                var px = new NativeArray<Color32>(res * res, Allocator.TempJob);
                fixed (byte* src = bytes)
                {
                    var dst = (Color32*)NativeArrayUnsafeUtility.GetUnsafePtr(px);
                    // 与写入对称：文件行 0（图顶）= 纹理底 = px 行 0，无需翻转
                    for (int i = 0; i < px.Length; ++i)
                    {
                        byte* s = src + off + (long)i * 4;
                        dst[i] = new Color32(s[2], s[1], s[0], s[3]);
                    }
                }
                return px;
            }
        }

        private sealed class UnityTileCompressor : ITileCompressor
        {
            private readonly TileArchiveWriter _arc;
            private readonly TextureFormat _fmtA, _fmtN;
            private readonly int _tileRes;
            public int PendingCount => 0;
            public UnityTileCompressor(TileArchiveWriter arc, TextureFormat fmtA, TextureFormat fmtN, int tileRes)
            { _arc = arc; _fmtA = fmtA; _fmtN = fmtN; _tileRes = tileRes; }
            public void Compress(int level, int tx, int tz,
                NativeArray<Color32> pxA, NativeArray<Color32> pxN, TextureCompressionQuality quality)
                => AppendTileCompressed(_arc, level, tx, tz, pxA, pxN, _tileRes, _fmtA, _fmtN, quality);
            public void Flush() { }
            public void Dispose() { }
        }

        /// <summary>
        /// 外部编码器进程池（astcenc / Compressonator，通道级适配）：采样线程只写中间文件入队，
        /// K 个 worker 各跑子进程压缩，完成的瓦片由主线程在采样间隙/收尾时按完成序追加归档
        /// （索引使文件内顺序无关）。单片失败回退 Unity 内置压缩（读回中间文件），烘焙不中断；
        /// 取消时杀掉全部子进程。
        /// </summary>
        private sealed class ProcessTileCompressor : ITileCompressor
        {
            private sealed class Item
            {
                public int level, tx, tz;
                public TextureCompressionQuality quality;
                public string inA, inN, outA, outN;
                public byte[] dataA, dataN;
                public int offA, offN; // 输出头剥离偏移（-1 = 非法）
                public bool ok;
                public string error;
            }

            private readonly TileArchiveWriter _arc;
            private readonly TileEncoder _encA, _encN;
            private readonly int _tileRes;
            private readonly string _tmpDir;
            private readonly System.Collections.Concurrent.BlockingCollection<Item> _queue
                = new System.Collections.Concurrent.BlockingCollection<Item>();
            private readonly System.Collections.Concurrent.ConcurrentQueue<Item> _completed
                = new System.Collections.Concurrent.ConcurrentQueue<Item>();
            private readonly System.Threading.Tasks.Task[] _workers;
            private int _queued, _drained;
            private volatile bool _abandoned;

            public int PendingCount => _queued - _drained;

            public ProcessTileCompressor(TileArchiveWriter arc, TileEncoder encA, TileEncoder encN, int tileRes)
            {
                _arc = arc; _encA = encA; _encN = encN; _tileRes = tileRes;
                _tmpDir = Path.Combine("Library", "MTTexEncTmp",
                    System.Guid.NewGuid().ToString("N").Substring(0, 8));
                Directory.CreateDirectory(_tmpDir);
                int cores = Mathf.Max(2, System.Environment.ProcessorCount);
                int workers = Mathf.Clamp(cores / 4, 1, 6);
                int jobs = Mathf.Clamp(cores / (workers * 2), 1, 4); // 采样 Burst 也在跑，避免过订阅
                Debug.Log($"[ClipmapBaker] 外部编码器并行压缩：{workers} 进程 × {jobs} 线程/进程" +
                          $"（albedo={encA.GetType().Name}，normal={encN.GetType().Name}）");
                _workers = new System.Threading.Tasks.Task[workers];
                for (int i = 0; i < workers; ++i)
                    _workers[i] = System.Threading.Tasks.Task.Run(() => WorkerLoop(jobs));
            }

            public void Compress(int level, int tx, int tz,
                NativeArray<Color32> pxA, NativeArray<Color32> pxN, TextureCompressionQuality quality)
            {
                string stem = $"l{level}_{tx}_{tz}";
                var item = new Item
                {
                    level = level, tx = tx, tz = tz, quality = quality,
                    inA = Path.Combine(_tmpDir, stem + "_a" + _encA.InExt),
                    inN = Path.Combine(_tmpDir, stem + "_n" + _encN.InExt),
                    outA = Path.Combine(_tmpDir, stem + "_a" + _encA.OutExt),
                    outN = Path.Combine(_tmpDir, stem + "_n" + _encN.OutExt),
                };
                _encA.WriteInput(item.inA, pxA, _tileRes);
                _encN.WriteInput(item.inN, pxN, _tileRes);
                _queue.Add(item);
                ++_queued;
                DrainCompleted();
            }

            private void WorkerLoop(int jobs)
            {
                foreach (var item in _queue.GetConsumingEnumerable())
                {
                    if (_abandoned) { CleanupFiles(item); continue; }
                    string errN = null;
                    bool okA = _encA.Run(item.inA, item.outA, true, item.quality, jobs, out string errA);
                    bool okN = okA && _encN.Run(item.inN, item.outN, false, item.quality, jobs, out errN);
                    item.error = errA ?? errN;
                    if (okA && okN)
                    {
                        try
                        {
                            item.dataA = File.ReadAllBytes(item.outA);
                            item.dataN = File.ReadAllBytes(item.outN);
                            item.offA = _encA.DataOffset(item.dataA);
                            item.offN = _encN.DataOffset(item.dataN);
                            item.ok = item.offA >= 0 && item.offN >= 0
                                && item.dataA.Length > item.offA && item.dataN.Length > item.offN;
                            if (!item.ok) item.error = "编码输出非法";
                        }
                        catch (System.Exception ex) { item.error = ex.Message; }
                    }
                    _completed.Enqueue(item);
                }
            }

            /// <summary>主线程：回收完成的瓦片写入归档；失败片读回中间文件用 Unity 压缩兜底。</summary>
            private void DrainCompleted()
            {
                while (_completed.TryDequeue(out var item))
                {
                    ++_drained;
                    if (item.ok)
                    {
                        long offset = _arc.Position;
                        int lenA = item.dataA.Length - item.offA, lenN = item.dataN.Length - item.offN;
                        _arc.Write(item.dataA, item.offA, lenA);
                        _arc.Write(item.dataN, item.offN, lenN);
                        _arc.AddRecord(item.level, item.tx, item.tz, offset, lenA, lenN);
                    }
                    else
                    {
                        Debug.LogWarning($"[ClipmapBaker] 外部编码失败 l{item.level}_{item.tx}_{item.tz}" +
                                         $"（{item.error}），回退 Unity 压缩");
                        var pxA = _encA.ReadPixels(item.inA, _tileRes);
                        var pxN = _encN.ReadPixels(item.inN, _tileRes);
                        AppendTileCompressed(_arc, item.level, item.tx, item.tz,
                            pxA, pxN, _tileRes, _encA.Format, _encN.Format, item.quality);
                        pxA.Dispose(); pxN.Dispose();
                    }
                    CleanupFiles(item);
                }
            }

            public void Flush()
            {
                _queue.CompleteAdding();
                var sw = System.Diagnostics.Stopwatch.StartNew();
                while (!System.Threading.Tasks.Task.WaitAll(_workers, 100))
                {
                    DrainCompleted();
                    EditorUtility.DisplayCancelableProgressBar("瓦片金字塔",
                        $"外部编码收尾（剩余 {PendingCount}）", 0.99f);
                }
                DrainCompleted();
                if (_queued > 0)
                    Debug.Log($"[ClipmapBaker] 外部编码收尾：{_queued} 任务（耗时 {sw.Elapsed.TotalSeconds:F1}s）");
            }

            public void Dispose()
            {
                _abandoned = true;
                if (!_queue.IsAddingCompleted) _queue.CompleteAdding();
                MTAstcEncoder.KillAllProcesses();
                try { System.Threading.Tasks.Task.WaitAll(_workers, 5000); } catch (System.Exception) { }
                DrainCompleted(); // 已完成结果仍写入归档（取消路径也保持 bin/idx 一致）
                _queue.Dispose();
                try { if (Directory.Exists(_tmpDir)) Directory.Delete(_tmpDir, true); }
                catch (System.Exception) { /* 杀毒软件占用等，下轮启动无关 */ }
            }

            private static void TryDelete(string path)
            {
                try { if (path != null && File.Exists(path)) File.Delete(path); }
                catch (System.Exception) { }
            }

            private static void CleanupFiles(Item item)
            {
                TryDelete(item.inA); TryDelete(item.inN);
                TryDelete(item.outA); TryDelete(item.outN);
            }
        }

        /// <summary>(q×q) 小块写入父缓冲（(2q)×(2q)）的 (qx,qz) 象限。</summary>
        private static void CopyQuadrantSmall(NativeArray<Color32> src, NativeArray<Color32> dst, int q, int qx, int qz)
        {
            int big = q * 2;
            for (int y = 0; y < q; ++y)
                NativeArray<Color32>.Copy(src, y * q, dst, (qz * q + y) * big + qx * q, q);
        }

        /// <summary>父缓冲（(2q)×(2q)）内象限间复制（边缘补齐用）。</summary>
        private static void CopyQuadrantFrom(NativeArray<Color32> buf, int q, int sx, int sz, int dx, int dz)
        {
            int big = q * 2;
            for (int y = 0; y < q; ++y)
                NativeArray<Color32>.Copy(buf, (sz * q + y) * big + sx * q,
                    buf, (dz * q + y) * big + dx * q, q);
        }

        [BurstCompile]
        private struct Downsample2xJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<Color32> src;
            [WriteOnly] public NativeArray<Color32> dst;
            public int srcRes;
            public int dstRes;

            public void Execute(int index)
            {
                int x = index % dstRes;
                int y = index / dstRes;
                var c00 = src[(2 * y) * srcRes + 2 * x];
                var c10 = src[(2 * y) * srcRes + 2 * x + 1];
                var c01 = src[(2 * y + 1) * srcRes + 2 * x];
                var c11 = src[(2 * y + 1) * srcRes + 2 * x + 1];
                dst[index] = new Color32(
                    (byte)((c00.r + c10.r + c01.r + c11.r + 2) / 4),
                    (byte)((c00.g + c10.g + c01.g + c11.g + 2) / 4),
                    (byte)((c00.b + c10.b + c01.b + c11.b + 2) / 4),
                    Vote(c00.a, c10.a, c01.a, c11.a));
            }
        }

        /// <summary>albedo/normal 并行做 2×2 盒滤波，RGB 带舍入平均，A 通道多数表决（并列取 c00）。</summary>
        private static void Downsample2x(NativeArray<Color32> srcA, NativeArray<Color32> srcN, int srcRes,
            out NativeArray<Color32> dstA, out NativeArray<Color32> dstN)
        {
            int dstRes = srcRes / 2;
            int count = dstRes * dstRes;
            dstA = new NativeArray<Color32>(count, Allocator.TempJob);
            dstN = new NativeArray<Color32>(count, Allocator.TempJob);
            var hA = new Downsample2xJob { src = srcA, dst = dstA, srcRes = srcRes, dstRes = dstRes }.Schedule(count, 64);
            var hN = new Downsample2xJob { src = srcN, dst = dstN, srcRes = srcRes, dstRes = dstRes }.Schedule(count, 64);
            JobHandle.CompleteAll(ref hA, ref hN);
        }

        /// <summary>4 候选多数表决：任何出现 ≥2 次的值胜出，全不同取 a。</summary>
        private static byte Vote(byte a, byte b, byte c, byte d)
        {
            if (a == b || a == c || a == d) return a;
            if (b == c || b == d) return b;
            if (c == d) return c;
            return a;
        }

        /// <summary>保存为纯 raw .bytes；尺寸和格式写入 manifest。targetFmt != RGBA32 时走 CompressTexture。</summary>
        private static void SaveClipmapTex(NativeArray<Color32> px, int res, string path, bool sRGB,
            TextureFormat targetFmt, bool mipChain,
            TextureCompressionQuality quality = TextureCompressionQuality.Normal)
        {
            SaveClipmapTex(px, res, path, sRGB, targetFmt, mipChain, quality, false, out _);
        }

        private static void SaveClipmapTex(NativeArray<Color32> px, int res, string path, bool sRGB,
            TextureFormat targetFmt, bool mipChain, TextureCompressionQuality quality, out byte[] raw)
        {
            SaveClipmapTex(px, res, path, sRGB, targetFmt, mipChain, quality, true, out raw);
        }

        // 压缩用 Texture2D 缓存：每瓦片 new/DestroyImmediate 一张 2040² RGBA32（16MB）churn 太大，
        // 按 (res, mipChain, linear) 复用（域重载自动清理）。压缩格式逐次调用指定，与缓存键无关。
        private static readonly Dictionary<long, Texture2D> _bakeTexCache = new Dictionary<long, Texture2D>();

        private static Texture2D GetBakeTex(int res, bool mipChain, bool linear)
        {
            long key = ((long)res << 2) | (mipChain ? 2 : 0) | (linear ? 1 : 0);
            if (!_bakeTexCache.TryGetValue(key, out var tex) || tex == null)
            {
                tex = new Texture2D(res, res, TextureFormat.RGBA32, mipChain, linear);
                _bakeTexCache[key] = tex;
            }
            // 上次调用可能已把纹理压成 ASTC/BC7，复用前重置回 RGBA32（SetPixelData 要求格式与数据匹配）
            else tex.Reinitialize(res, res, TextureFormat.RGBA32, mipChain);
            return tex;
        }

        private static void SaveClipmapTex(NativeArray<Color32> px, int res, string path, bool sRGB,
            TextureFormat targetFmt, bool mipChain, TextureCompressionQuality quality,
            bool captureRaw, out byte[] raw)
        {
            var tex = GetBakeTex(res, mipChain, !sRGB);
            tex.SetPixelData(px, 0);
            tex.Apply(mipChain); // mipChain=true 时离线生成完整 Mip 链（§4.1）
            if (targetFmt != TextureFormat.RGBA32)
                EditorUtility.CompressTexture(tex, targetFmt, quality);
            var nativeRaw = tex.GetRawTextureData<byte>();
            raw = captureRaw ? nativeRaw.ToArray() : null;
            WriteRaw(path, nativeRaw);
        }

        private static unsafe void WriteRaw(string path, NativeArray<byte> raw)
        {
            using (var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None,
                       64 * 1024, FileOptions.SequentialScan))
            {
                var data = new System.ReadOnlySpan<byte>(
                    (byte*)NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(raw), raw.Length);
                fs.Write(data);
            }
        }

        /// <summary>
        /// 烘焙自检：按配置重建并上传纹理，再对完整原始数据（含 mip 链）做零容差字节比对。
        /// 失败说明 保存→加载 链路（格式/尺寸/mip 布局）不一致，错误会直接打在 Console。
        /// </summary>
        private static void VerifyBakedTex(string path, byte[] expectedRaw, int res, bool mipChain, bool sRGB,
            TextureFormat fmt, string name)
        {
            try
            {
                var bytes = File.ReadAllBytes(path);
                var tex = new Texture2D(res, res, fmt, mipChain, !sRGB);
                tex.LoadRawTextureData(bytes);
                tex.Apply(false);
                var rb = tex.GetRawTextureData();
                if (rb.Length != expectedRaw.Length)
                {
                    Debug.LogError($"[ClipmapBaker] 自检失败 {name}：回读字节数 {rb.Length} != 基准 {expectedRaw.Length}");
                    Object.DestroyImmediate(tex);
                    return;
                }
                int diff = -1;
                for (int i = 0; i < rb.Length; ++i)
                {
                    if (rb[i] == expectedRaw[i]) continue;
                    diff = i;
                    break;
                }
                if (diff < 0)
                    Debug.Log($"[ClipmapBaker] 自检通过 {name}（{rb.Length} 字节回读一致）");
                else
                    Debug.LogError($"[ClipmapBaker] 自检失败 {name}：首处差异位于字节 {diff}");
                Object.DestroyImmediate(tex);
            }
            catch (System.Exception e)
            {
                Debug.LogError($"[ClipmapBaker] 自检异常 {name}：{e.Message}");
            }
        }

        // ============================================================
        // 3. 场景布置
        // ============================================================

        /// <summary>
        /// 在场景中创建/更新 MT 地形节点：
        /// - 确保 VTCreator 节点存在（MTLoader 运行时有依赖；Clipmap 激活时其组件会被自动禁用）；
        /// - 创建/更新 "{terrain.name}_MT" 节点（位于地形原点），挂 MTLoader 并连线；
        /// - 可选：创建/更新 MTClipmapController 并连线；
        /// - 可选：关闭源地形高度图渲染（保留碰撞与数据，便于重复烘焙）。
        /// 重复调用幂等：已存在的节点只做字段更新。
        /// </summary>
        public static GameObject PlaceInScene(Terrain terrain, MTData header, MTLODPolicy lodPolicy, Camera cam,
            bool addClipmapController, string clipmapResourcesRoot, bool disableTerrainRendering)
        {
            if (cam == null) cam = Camera.main;
            if (cam == null) Debug.LogWarning("[MTBake] 场景中未找到主相机，cullCamera/viewCamera 未连线");

            // VTCreator 节点
            var vtGo = GameObject.Find("VTCreator");
            if (vtGo == null)
            {
                vtGo = new GameObject("VTCreator");
                Undo.RegisterCreatedObjectUndo(vtGo, "Create VTCreator");
            }
            if (vtGo.GetComponent<MTVTCreator>() == null)
                Undo.AddComponent<MTVTCreator>(vtGo);

            // MTLoader 节点
            string goName = terrain.name + "_MT";
            var go = GameObject.Find(goName);
            if (go == null)
            {
                go = new GameObject(goName);
                Undo.RegisterCreatedObjectUndo(go, "Create MT Terrain");
            }
            Undo.RecordObject(go, "Setup MT Terrain");
            // 必须放在世界原点：烘焙数据（四叉树 bounds / mesh 顶点）本身就是世界空间，
            // 运行时 MTQuadTreeUtil 与 MTPooledRenderMesh 会再叠加 loader GO 的 position。
            // 若 GO 放在地形位置而非原点，所有 bounds 会被二次偏移，视锥剔除全部失败
            // （activeCmd 为空、地形不渲染）。项目既有演示场景中 loader GO 也均在原点。
            go.transform.position = Vector3.zero;
            var loader = go.GetComponent<MTLoader>();
            if (loader == null) loader = Undo.AddComponent<MTLoader>(go);
            loader.header = header;
            loader.lodPolicy = lodPolicy;
            loader.cullCamera = cam;
            loader.VTCreatorGo = vtGo;

            // Clipmap 控制器（全局唯一）
            if (addClipmapController)
            {
                var ccGo = GameObject.Find("MTClipmapController");
                if (ccGo == null)
                {
                    ccGo = new GameObject("MTClipmapController");
                    Undo.RegisterCreatedObjectUndo(ccGo, "Create ClipmapController");
                }
                var cc = ccGo.GetComponent<MTClipmapController>();
                if (cc == null) cc = Undo.AddComponent<MTClipmapController>(ccGo);
                cc.loader = loader;
                cc.viewCamera = cam;
                cc.sourceTerrain = terrain; // 兜底：无烘焙数据时 Terrain 实时采样
                cc.terrainOriginXZ = new Vector2(terrain.transform.position.x, terrain.transform.position.z);
                cc.terrainSize = terrain.terrainData.size.x;
                cc.bakedResourcesRoot = clipmapResourcesRoot;
            }

            if (disableTerrainRendering)
            {
                Undo.RecordObject(terrain, "Disable Terrain Rendering");
                terrain.drawHeightmap = false; // 保留碰撞体与 TerrainData（重复烘焙仍可读）
            }

            EditorSceneManager.MarkSceneDirty(go.scene);
            Selection.activeGameObject = go;
            return go;
        }

        // ============================================================
        // 分辨率推荐（§5.1/§7.1 密度契约）
        // ============================================================

        /// <summary>
        /// 按地形尺寸推荐烘焙分辨率（§3.1.1）：
        /// - 静态层 L_K：覆盖全地形，texel = S/2040（与环面同分辨率 2040²，离线完整 Mip 链）；
        /// - L7 粗源：仅 Job 回退路径用，texel ≈ S/512 即可。
        /// </summary>
        public static void RecommendClipmapResolutions(float terrainSize, out int l7Res, out int l8Res)
        {
            l7Res = ClampPow2(Mathf.RoundToInt(terrainSize / 2f), 256, 1024);
            l8Res = 2040;
        }

        static int ClampPow2(int v, int min, int max)
            => Mathf.Clamp(Mathf.NextPowerOfTwo(Mathf.Max(2, v)), min, max);

        // ============================================================
        // 目录工具
        // ============================================================

        /// <summary>确保 Assets 下的（可嵌套）目录存在，逐级创建；已存在则跳过。</summary>
        public static void EnsureAssetFolder(string assetPath)
        {
            if (AssetDatabase.IsValidFolder(assetPath)) return;
            var parts = assetPath.Split('/');
            string cur = parts[0]; // "Assets"
            for (int i = 1; i < parts.Length; ++i)
            {
                var next = cur + "/" + parts[i];
                if (!AssetDatabase.IsValidFolder(next))
                    AssetDatabase.CreateFolder(cur, parts[i]);
                cur = next;
            }
        }

        public static string ToFullPath(string assetPath)
        {
            if (!assetPath.StartsWith("Assets"))
            {
                Debug.LogError("[MTBake] 输出目录必须在 Assets 下: " + assetPath);
                return assetPath;
            }
            return Application.dataPath + assetPath.Substring("Assets".Length);
        }

        /// <summary>
        /// Resources 输出根 → 对应 StreamingAssets 目录（瓦片归档 tiles.bin/tiles.idx 的位置）。
        /// 归档必须经 File IO 读取：构建后 Resources 被打包不可按路径访问，StreamingAssets 保留为文件。
        /// </summary>
        public static string StreamingRootOf(string outputRoot)
        {
            string p = outputRoot.Replace('\\', '/');
            const string seg = "/Resources/";
            int i = p.IndexOf(seg, System.StringComparison.Ordinal);
            string rel = i >= 0 ? p.Substring(i + seg.Length) : new DirectoryInfo(outputRoot).Name;
            return "Assets/StreamingAssets/" + rel;
        }

        // ============================================================
        // MT 数据导出私有实现（与原 MTDataEditor 一致）
        // ============================================================

        private static void ExportTree(MTQuadTreeBuildNode root, MTQuadTreeNode node, List<MTQuadTreeNode> nodes)
        {
            if (root != null)
            {
                node.bnd = root.Bound;
                node.meshIdx = root.MeshID;
                node.lodLv = (byte)root.LODLv;
                if (root.SubNode != null)
                {
                    node.children = new int[root.SubNode.Length];
                    for (int i = 0; i < root.SubNode.Length; ++i)
                    {
                        var child = new MTQuadTreeNode(nodes.Count);
                        nodes.Add(child);
                        node.children[i] = child.cellIdx;
                    }
                    for (int i = 0; i < root.SubNode.Length; ++i)
                    {
                        var childIdx = node.children[i];
                        ExportTree(root.SubNode[i], nodes[childIdx], nodes);
                    }
                }
            }
        }

        private static void SerializeTrees(MemoryStream stream, List<MTQuadTreeNode> nodes)
        {
            MTFileUtils.WriteInt(stream, nodes.Count);
            foreach (var node in nodes)
            {
                node.Serialize(stream);
            }
        }

        private static void ExportHeightMap(MTData dataHeader, Terrain curentTarget, string topFulllPath, string folder0)
        {
            EditorUtility.DisplayProgressBar("saving height map", "processing", 0);
            dataHeader.HeightmapResolution = curentTarget.terrainData.heightmapResolution;
            dataHeader.HeightmapScale = curentTarget.terrainData.heightmapScale;
            float[,] heightData = curentTarget.terrainData.GetHeights(0, 0, dataHeader.HeightmapResolution, dataHeader.HeightmapResolution);
            byte[] heightBytes = new byte[dataHeader.HeightmapResolution * dataHeader.HeightmapResolution * 2];
            for (int hy = 0; hy < dataHeader.HeightmapResolution; ++hy)
            {
                for (int hx = 0; hx < dataHeader.HeightmapResolution; ++hx)
                {
                    float val = heightData[hy, hx] * 255f;
                    byte h = (byte)Mathf.FloorToInt(val);
                    byte l = (byte)Mathf.FloorToInt((val - h) * 255f);
                    heightBytes[hy * dataHeader.HeightmapResolution * 2 + hx * 2] = h;
                    heightBytes[hy * dataHeader.HeightmapResolution * 2 + hx * 2 + 1] = l;
                }
            }
            File.WriteAllBytes(string.Format("{0}/heightMap.bytes", topFulllPath), heightBytes);
            AssetDatabase.Refresh();
            dataHeader.HeightMap = AssetDatabase.LoadAssetAtPath(string.Format("{0}/heightMap.bytes", folder0), typeof(TextAsset)) as TextAsset;
        }

        private static void ExportDetails(MTData dataHeader, Terrain curentTarget, string topFulllPath, string folder0)
        {
            EditorUtility.DisplayProgressBar("saving details", "processing", 0);
            var original = curentTarget.terrainData.detailPrototypes;
            //combine layers of the same mesh
            List<MTDetailLayerData> lLayers = new List<MTDetailLayerData>();
            int[][,] layerDatas = new int[original.Length][,];
            for (int l = 0; l < original.Length; ++l)
            {
                var originalLayer = original[l];
                if (originalLayer.prototype == null)
                    continue;
                layerDatas[l] = curentTarget.terrainData.GetDetailLayer(0, 0, curentTarget.terrainData.detailWidth, curentTarget.terrainData.detailHeight, l);
                var layer = new MTDetailLayerData();
                layer.minWidth = originalLayer.minWidth;
                layer.maxWidth = originalLayer.maxWidth;
                layer.minHeight = originalLayer.minHeight;
                layer.maxHeight = originalLayer.maxHeight;
                layer.noiseSpread = originalLayer.noiseSpread;
                layer.dryColor = originalLayer.dryColor;
                layer.healthyColor = originalLayer.healthyColor;
                layer.prototype = originalLayer.prototype;
                lLayers.Add(layer);
            }
            dataHeader.DetailPrototypes = new MTDetailLayerData[lLayers.Count];
            dataHeader.DetailWidth = curentTarget.terrainData.detailWidth;
            dataHeader.DetailHeight = curentTarget.terrainData.detailHeight;
            dataHeader.DetailResolutionPerPatch = curentTarget.terrainData.detailResolutionPerPatch;
            if (dataHeader.DetailResolutionPerPatch <= 0 ||
                dataHeader.DetailHeight / dataHeader.DetailResolutionPerPatch > byte.MaxValue)
            {
                Debug.LogError("导出Detail失败，DetailResolutionPerPatch 数值太小");
                return;
            }
            //split data to patches, drop empty patches
            //layer header : [patch data offsets] 4 bytes each, -1 means no density
            //patch data : [patch block] 1 byte each
            //
            MemoryStream detailStream = new MemoryStream();
            int patch_x = Mathf.CeilToInt((float)dataHeader.DetailWidth / dataHeader.DetailResolutionPerPatch);
            int patch_y = Mathf.CeilToInt((float)dataHeader.DetailHeight / dataHeader.DetailResolutionPerPatch);
            byte[] patch_block = new byte[dataHeader.DetailResolutionPerPatch * dataHeader.DetailResolutionPerPatch];
            int[] patchDataOffsets = new int[patch_x * patch_y * lLayers.Count];
            //header占位
            foreach (var d in patchDataOffsets)
                MTFileUtils.WriteInt(detailStream, -1);
            for (int l = 0; l < lLayers.Count; ++l)
            {
                var layer = lLayers[l];
                dataHeader.DetailPrototypes[l] = layer;
                layer.maxDensity = 0;
                for (int py = 0; py < patch_y; ++py)
                {
                    for (int px = 0; px < patch_x; ++px)
                    {
                        int max_density = 0;
                        for (int sub_py = 0; sub_py < dataHeader.DetailResolutionPerPatch; ++sub_py)
                        {
                            for (int sub_px = 0; sub_px < dataHeader.DetailResolutionPerPatch; ++sub_px)
                            {
                                int hy = py * dataHeader.DetailResolutionPerPatch + sub_py;
                                int hx = px * dataHeader.DetailResolutionPerPatch + sub_px;
                                hy = Mathf.Min(hy, dataHeader.DetailHeight);
                                hx = Mathf.Min(hx, dataHeader.DetailWidth);
                                int density = layerDatas[l][hy, hx];
                                if (density > max_density)
                                    max_density = density;
                                patch_block[sub_py * dataHeader.DetailResolutionPerPatch + sub_px] = (byte)density;
                            }
                        }
                        var offsetDataIdx = l * patch_x * patch_y + py * patch_x + px;
                        if (max_density > 0)
                        {
                            patchDataOffsets[offsetDataIdx] = (int)detailStream.Position;
                            detailStream.Write(patch_block, 0, patch_block.Length);
                        }
                        else
                        {
                            patchDataOffsets[offsetDataIdx] = -1;
                        }
                        if (max_density > layer.maxDensity)
                            layer.maxDensity = max_density;
                    }
                }
            }
            detailStream.Position = 0;
            foreach (var d in patchDataOffsets)
                MTFileUtils.WriteInt(detailStream, d);
            File.WriteAllBytes(string.Format("{0}/details.bytes", topFulllPath), detailStream.ToArray());
            detailStream.Close();
            AssetDatabase.Refresh();
            dataHeader.DetailLayers = AssetDatabase.LoadAssetAtPath(string.Format("{0}/details.bytes", folder0), typeof(TextAsset)) as TextAsset;
        }
    }
}
