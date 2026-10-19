namespace MightyTerrainMesh.EditorTools
{
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using Unity.Collections;
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
            bool pyramidFullCoverage = true, int pyramidCoarseFullFromLevel = 3)
        {
            if (terrain == null) { MTLog.LogError("no target terrain"); return; }
            Directory.CreateDirectory(outputRoot);
            var src = ClipmapSourceRuntime.FromTerrain(terrain);
            var origin = terrain.transform.position;
            float size = terrain.terrainData.size.x;
            BakeFormatsOf(format, out var fmtA, out var fmtN);

            var manifest = new ClipmapManifest
            {
                originX = origin.x,
                originZ = origin.z,
                terrainSize = size,
                l7Res = l7Res,
                l8Res = l8Res,
                blockRes = 0,
                blockRects = new Rect[0], // 区域高精块已由瓦片金字塔取代（保留字段兼容旧数据）
                texFormat = (int)fmtA,
            };

            try
            {
                // L7 粗源（Job 回退路径用）/ 静态层 L_K（全地形 S/2040 texel，完整 Mip 链：§4.1 离线生成）
                BakeClipmapLayer(src, "l7", l7Res, new float2(origin.x, origin.z), size / l7Res, false, fmtA, fmtN, outputRoot);
                BakeClipmapLayer(src, "l8", l8Res, new float2(origin.x, origin.z), size / l8Res, true, fmtA, fmtN, outputRoot);

                // 瓦片金字塔（§7.1 流式契约）：L0 与 clipmap L0 层同密度（1.57cm@2040²），逐级盒滤波合并
                if (bakePyramid)
                {
                    int n = BakeTilePyramid(src, terrain, 2040, ClipmapBlockWorldSize, fmtA, fmtN, outputRoot, manifest,
                        pyramidFullCoverage, regions, pyramidCoarseFullFromLevel);
                    Debug.Log($"[ClipmapBaker] 金字塔：{manifest.tileLevels} 级，L0 {n} 片，tiles/ 目录");
                }
                else { manifest.tileLevels = 0; }

                File.WriteAllText(Path.Combine(outputRoot, "manifest.json"), JsonUtility.ToJson(manifest, true));
                AssetDatabase.Refresh();
                Debug.Log($"[ClipmapBaker] 完成：L7 {l7Res}² / L8 {l8Res}² / 格式 {format} → {outputRoot}");
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
                SaveClipmapTex(pxA, res, pathA, true, fmtA, mipChain);
                SaveClipmapTex(pxN, res, pathN, false, fmtN, mipChain);
                // 自检：回读 .bytes 与源像素比对，验证 保存→加载 往返一致（ASTC 有损，允许容差）
                VerifyBakedTex(pathA, pxA, res, mipChain, true, name + "_albedo");
                VerifyBakedTex(pathN, pxN, res, mipChain, false, name + "_normal");
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
            public Color32[] a, n;
            public int arrived;
        }

        /// <summary>
        /// 烘焙瓦片金字塔：Level 0 = tileWorld0 米/tileRes²（与 clipmap L0 层同密度，1.57cm@2040²），
        /// 由地形直接采样。两种覆盖模式（D8）：
        /// - fullCoverage：全地形 L0 + 逐级盒滤波合并（盒滤波抗走样质量最优，粗级也从子块传播）；
        /// - 区域模式：L0 仅烘与 regions 相交的瓦片（高精包），区域父级（j&lt;coarseFullFrom）需 4 子块齐聚才生成；
        ///   j ≥ coarseFullFrom 的粗级全网格直接从地形按该级密度采样（片数少，保证顶层环面可供流式直拷）。
        /// 级数 = ceil(log2(S/tileWorld0))（顶级覆盖 S/2，与运行时 K=log2(S)−5 的顶层环面 footprint 对齐）。
        /// 输出 tiles/l{级}_{x}_{z}_{albedo|normal}.bytes，并回填 manifest。进度条可取消（取消后需重新完整烘焙）。
        /// 返回 L0 实际烘焙片数。
        /// </summary>
        public static int BakeTilePyramid(ClipmapSourceRuntime src, Terrain terrain, int tileRes, float tileWorld0,
            TextureFormat fmtA, TextureFormat fmtN, string outputRoot, ClipmapManifest manifest,
            bool fullCoverage, List<Rect> regions, int coarseFullFromLevel)
        {
            float size = terrain.terrainData.size.x;
            var origin = terrain.transform.position;
            int levels = Mathf.Max(1, Mathf.CeilToInt(Mathf.Log(size / tileWorld0, 2f)));
            manifest.tileRes = tileRes;
            manifest.tileWorld0 = tileWorld0;
            manifest.tileLevels = levels;

            string tileDir = Path.Combine(outputRoot, "tiles");
            Directory.CreateDirectory(tileDir);
            foreach (var old in Directory.GetFiles(tileDir, "*.bytes")) File.Delete(old); // 清旧防残留

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
            // 父块优先扫描：同一父块的 2×2 子块连续烘焙，父块即时收尾释放——
            // 同时挂起的累积器 ≈ 层数个（行优先扫描会挂起 ~gx 个，2040² 下单个 33MB 不可接受）
            int gxP = (gx0 + 1) / 2;
            int baked = 0;
            for (int pz = 0; pz < gxP; ++pz)
            for (int px = 0; px < gxP; ++px)
            for (int dz = 0; dz < 2; ++dz)
            for (int dx = 0; dx < 2; ++dx)
            {
                int tx = px * 2 + dx, tz = pz * 2 + dz;
                if (tx >= gx0 || tz >= gx0 || !bake0[tx, tz]) continue;
                if (EditorUtility.DisplayCancelableProgressBar("瓦片金字塔",
                        $"L0 {baked + 1}/{total0}  ({tx},{tz})", (float)baked / Mathf.Max(1, total0)))
                {
                    Debug.LogWarning("[ClipmapBaker] 金字塔烘焙已取消（tiles/ 不完整，请重新完整烘焙）");
                    return baked;
                }
                BakeTileDirect(src, origin, tileRes, tileWorld0, 0, tx, tz, fmtA, fmtN, tileDir, out var pxA, out var pxN);
                PropagatePyramid(pending, levels, size, tileRes, tileWorld0, 0, tx, tz, pxA, pxN, fmtA, fmtN, tileDir, directFrom);
                pxA.Dispose(); pxN.Dispose();
                ++baked;
            }
            // 全量模式下所有父块应收尾完毕；区域模式下区域边缘之外的父块不生成属预期
            if (fullCoverage)
                for (int i = 0; i < levels; ++i)
                    if (pending[i].Count != 0)
                        Debug.LogWarning($"[ClipmapBaker] 金字塔第 {i} 级有 {pending[i].Count} 个父块未收尾（网格非 2 幂？）");

            // ---- 粗级全量直接烘焙（区域模式；j = directFrom..levels-1）----
            for (int lvl = Mathf.Max(1, directFrom); lvl < levels; ++lvl)
            {
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
                    BakeTileDirect(src, origin, tileRes, tw, lvl, tx, tz, fmtA, fmtN, tileDir, out var dA, out var dN);
                    dA.Dispose(); dN.Dispose();
                    ++done;
                }
            }
            return baked;
        }

        /// <summary>按 (lvl, tx, tz) 的世界矩形从地形直接采样一片瓦片并保存。</summary>
        private static void BakeTileDirect(ClipmapSourceRuntime src, Vector3 origin, int tileRes, float tileWorld,
            int lvl, int tx, int tz, TextureFormat fmtA, TextureFormat fmtN, string tileDir,
            out NativeArray<Color32> pxA, out NativeArray<Color32> pxN)
        {
            pxA = new NativeArray<Color32>(tileRes * tileRes, Allocator.TempJob);
            pxN = new NativeArray<Color32>(tileRes * tileRes, Allocator.TempJob);
            float2 wmin = new float2(origin.x + tx * tileWorld, origin.z + tz * tileWorld);
            float texel = tileWorld / tileRes;
            var jobA = src.CreateJob(0, true, wmin, texel, tileRes, pxA, tileRes, 0);
            var jobN = src.CreateJob(0, false, wmin, texel, tileRes, pxN, tileRes, 0);
            var hA = jobA.Schedule(tileRes * tileRes, 64);
            var hN = jobN.Schedule(tileRes * tileRes, 64);
            JobHandle.CompleteAll(ref hA, ref hN);
            SaveClipmapTex(pxA, tileRes, TilePath(tileDir, lvl, tx, tz, true), true, fmtA, false, TextureCompressionQuality.Normal);
            SaveClipmapTex(pxN, tileRes, TilePath(tileDir, lvl, tx, tz, false), false, fmtN, false, TextureCompressionQuality.Normal);
        }

        /// <summary>子块到位后向上卷积：子块即时盒滤波降采样写入父象限（累积器仅 tileRes²，省 4× 内存）；
        /// 集齐父块的全部现存子块 → 边缘复制补齐 → 存盘 → 递归向上。
        /// 卷积止于 stopBeforeLevel 级（该级起由 BakeTileDirect 全量直接烘焙，或已到顶级）。</summary>
        private static void PropagatePyramid(Dictionary<long, PyramidAccum>[] pending, int levels, float terrainSize,
            int tileRes, float tileWorld0, int level, int tx, int tz,
            NativeArray<Color32> cA, NativeArray<Color32> cN, TextureFormat fmtA, TextureFormat fmtN,
            string tileDir, int stopBeforeLevel)
        {
            if (level + 1 >= levels || level + 1 >= stopBeforeLevel) return;
            int px = tx >> 1, pz = tz >> 1;
            int qx = tx & 1, qz = tz & 1;
            int q = tileRes / 2; // 父象限边长
            var map = pending[level];
            long key = ((long)px << 20) | (uint)pz;
            if (!map.TryGetValue(key, out var acc))
            {
                acc = new PyramidAccum { a = new Color32[tileRes * tileRes], n = new Color32[tileRes * tileRes] };
                map.Add(key, acc);
            }
            CopyQuadrantSmall(Downsample2x(cA, tileRes), acc.a, q, qx, qz);
            CopyQuadrantSmall(Downsample2x(cN, tileRes), acc.n, q, qx, qz);
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
            var nA = new NativeArray<Color32>(acc.a, Allocator.TempJob);
            var nN = new NativeArray<Color32>(acc.n, Allocator.TempJob);
            SaveClipmapTex(nA, tileRes, TilePath(tileDir, level + 1, px, pz, true), true, fmtA, false, TextureCompressionQuality.Normal);
            SaveClipmapTex(nN, tileRes, TilePath(tileDir, level + 1, px, pz, false), false, fmtN, false, TextureCompressionQuality.Normal);
            PropagatePyramid(pending, levels, terrainSize, tileRes, tileWorld0, level + 1, px, pz, nA, nN, fmtA, fmtN, tileDir, stopBeforeLevel);
            nA.Dispose(); nN.Dispose();
        }

        private static string TilePath(string tileDir, int level, int tx, int tz, bool albedo)
            => Path.Combine(tileDir, $"l{level}_{tx}_{tz}_{(albedo ? "albedo" : "normal")}.bytes");

        /// <summary>(q×q) 小块写入父缓冲（(2q)×(2q)）的 (qx,qz) 象限。</summary>
        private static void CopyQuadrantSmall(Color32[] src, Color32[] dst, int q, int qx, int qz)
        {
            int big = q * 2;
            for (int y = 0; y < q; ++y)
                System.Array.Copy(src, y * q, dst, (qz * q + y) * big + qx * q, q);
        }

        /// <summary>父缓冲（(2q)×(2q)）内象限间复制（边缘补齐用）。</summary>
        private static void CopyQuadrantFrom(Color32[] buf, int q, int sx, int sz, int dx, int dz)
        {
            int big = q * 2;
            for (int y = 0; y < q; ++y)
                System.Array.Copy(buf, (sz * q + y) * big + sx * q,
                    buf, (dz * q + y) * big + dx * q, q);
        }

        /// <summary>
        /// 2×2 盒滤波降采样（srcRes → srcRes/2），RGB 带舍入平均；
        /// A 通道是细节层索引（语义不可平均），用多数表决（并列取 c00）。
        /// </summary>
        private static Color32[] Downsample2x(NativeArray<Color32> src, int srcRes)
        {
            int dst = srcRes / 2;
            var o = new Color32[dst * dst];
            for (int y = 0; y < dst; ++y)
            for (int x = 0; x < dst; ++x)
            {
                var c00 = src[(2 * y) * srcRes + 2 * x];
                var c10 = src[(2 * y) * srcRes + 2 * x + 1];
                var c01 = src[(2 * y + 1) * srcRes + 2 * x];
                var c11 = src[(2 * y + 1) * srcRes + 2 * x + 1];
                o[y * dst + x] = new Color32(
                    (byte)((c00.r + c10.r + c01.r + c11.r + 2) / 4),
                    (byte)((c00.g + c10.g + c01.g + c11.g + 2) / 4),
                    (byte)((c00.b + c10.b + c01.b + c11.b + 2) / 4),
                    Vote(c00.a, c10.a, c01.a, c11.a));
            }
            return o;
        }

        /// <summary>4 候选多数表决：任何出现 ≥2 次的值胜出，全不同取 a。</summary>
        private static byte Vote(byte a, byte b, byte c, byte d)
        {
            if (a == b || a == c || a == d) return a;
            if (b == c || b == d) return b;
            if (c == d) return c;
            return a;
        }

        /// <summary>保存为 .bytes（12B 头 w/h/fmt + 原始纹理数据）。targetFmt != RGBA32 时走 CompressTexture。</summary>
        private static void SaveClipmapTex(NativeArray<Color32> px, int res, string path, bool sRGB,
            TextureFormat targetFmt, bool mipChain,
            TextureCompressionQuality quality = TextureCompressionQuality.Best)
        {
            var tex = new Texture2D(res, res, TextureFormat.RGBA32, mipChain, !sRGB);
            tex.SetPixelData(px, 0);
            tex.Apply(mipChain); // mipChain=true 时离线生成完整 Mip 链（§4.1）
            TextureFormat fmt = TextureFormat.RGBA32;
            if (targetFmt != TextureFormat.RGBA32)
            {
                EditorUtility.CompressTexture(tex, targetFmt, quality);
                fmt = targetFmt;
            }
            // 含全部 mip 级，运行时 LoadRawTextureData 整体读回。
            // 注意：2018.3+ 返回 NativeArray<byte>（实例 ToArray），更早版本返回 byte[]
            // （走 System.Linq 扩展 ToArray）——两种情况下本行均可编译。
            byte[] raw = tex.GetRawTextureData().ToArray();
            using (var fs = new FileStream(path, FileMode.Create))
            using (var bw = new BinaryWriter(fs))
            {
                bw.Write(res);
                bw.Write(res);
                bw.Write((int)fmt);
                bw.Write(raw);
            }
            Object.DestroyImmediate(tex);
        }

        /// <summary>
        /// 烘焙自检：按 .bytes 头重建纹理并回读 mip0，与源像素在角点/中心采样比对。
        /// 失败说明 保存→加载 链路（格式/头/mip 布局）不一致，错误会直接打在 Console。
        /// </summary>
        private static void VerifyBakedTex(string path, NativeArray<Color32> src, int res, bool mipChain, bool sRGB, string name)
        {
            try
            {
                var bytes = File.ReadAllBytes(path);
                int w = System.BitConverter.ToInt32(bytes, 0);
                int h = System.BitConverter.ToInt32(bytes, 4);
                var fmt = (TextureFormat)System.BitConverter.ToInt32(bytes, 8);
                var tex = new Texture2D(w, h, fmt, mipChain, !sRGB);
                var payload = new byte[bytes.Length - 12];
                System.Buffer.BlockCopy(bytes, 12, payload, 0, payload.Length);
                tex.LoadRawTextureData(payload);
                tex.Apply(false);
                var rb = tex.GetPixels32(); // mip0
                Object.DestroyImmediate(tex);
                if (rb.Length != src.Length)
                {
                    Debug.LogError($"[ClipmapBaker] 自检失败 {name}：回读像素数 {rb.Length} != 源 {src.Length}");
                    return;
                }
                int maxErr = 0;
                int[] samples = { 0, res / 2 + (res / 2) * res, res * res - 1 };
                foreach (var i in samples)
                {
                    Color32 a = src[i], b = rb[i];
                    maxErr = Mathf.Max(maxErr, Mathf.Abs(a.r - b.r), Mathf.Abs(a.g - b.g),
                        Mathf.Abs(a.b - b.b), Mathf.Abs(a.a - b.a));
                }
                // ASTC 有损：编解码损耗可达 ~20/255（法线等高频内容），容差放宽；
                // 真正的链路错误（格式/布局/sRGB 错乱）偏差在 60+ 量级，依然能抓住
                int tolerance = fmt == TextureFormat.RGBA32 ? 0 : 24;
                if (maxErr <= tolerance)
                    Debug.Log($"[ClipmapBaker] 自检通过 {name}（采样最大误差 {maxErr}/255，容差 {tolerance}）");
                else
                    Debug.LogError($"[ClipmapBaker] 自检失败 {name}：回读与源数据偏差 {maxErr}/255（容差 {tolerance}）");
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
