namespace MightyTerrainMesh
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using Unity.Collections;
    using Unity.Mathematics;
    using UnityEngine;

    /// <summary>
    /// Clipmap 烘焙清单（对应 ClipmapVT.md §7.1 数据契约）。
    /// 文件布局（Resources/&lt;root&gt;/ 下）：
    ///   manifest.json / l7_albedo.bytes / l7_normal.bytes / l8_albedo.bytes / l8_normal.bytes
    ///   block_{i}_albedo.bytes / block_{i}_normal.bytes
    /// 纹理 .bytes 头：int32 width, height, textureFormat, 后接 GetRawTextureData 字节流。
    /// </summary>
    [Serializable]
    public class ClipmapManifest
    {
        public float originX;
        public float originZ;
        public float terrainSize;
        public int l7Res;
        public int l8Res;
        public int blockRes;
        public Rect[] blockRects = new Rect[0]; // 世界坐标（米）
        // 瓦片金字塔（§7.1 流式契约）：Level j 瓦片边长 = tileWorld0<<j 米，分辨率 tileRes，
        // 与 clipmap 层 j 密度 1:1；文件 tiles/l{j}_{x}_{z}_{albedo|normal}.bytes。
        // tileLevels = 0 表示无金字塔（旧版烘焙数据）。
        public int tileRes;
        public float tileWorld0;
        public int tileLevels;
        // 烘焙 albedo 目标格式（TextureFormat 枚举值：ASTC_6x6 / BC7 / RGBA32，normal 依 §10 规则配对）。
        // 0 = 旧数据（按 ASTC_6x6 假设，不匹配时加载报错提示重烘）。
        public int texFormat;
        // 细节层通道（已废弃，D14：splat 完全烘焙，运行时无 detail overlay；NormalAttr B/A 依 §4.1 为粗糙度/AO）。
        // 字段保留仅为反序列化兼容旧 manifest；运行时不再读取。
        public int detailLayers;
    }

    /// <summary>
    /// Clipmap 采样源运行时持有者：拥有全部 NativeArray（Persistent），负责构建与释放。
    /// 三种模式：Baked（烘焙数据）/ Terrain（TerrainData 实时采样）/ Test（测试图案）。
    /// </summary>
    public class ClipmapSourceRuntime : IDisposable
    {
        /// <summary>原型高精块加载上限（§7.1；正式版应改为按需流式，无硬上限）。</summary>
        public const int MaxBlocks = 64;

        public int Mode { get; private set; }

        // Terrain 模式数据
        private NativeArray<float> _alphamaps;
        private NativeArray<Color32> _layerPix0, _layerPix1, _layerPix2, _layerPix3;
        private NativeArray<Color32> _layerPix4, _layerPix5, _layerPix6, _layerPix7;
        private NativeArray<float> _heights;
        private int _alphaRes, _alphaLayers, _heightRes;
        private int4 _layerW, _layerH, _layerWB, _layerHB;
        private float4 _tileX, _tileY, _tileXB, _tileYB;
        private float4 _tileOffX, _tileOffY, _tileOffXB, _tileOffYB;
        private Vector3 _terrainPos, _terrainSize3;

        // Baked 模式数据
        private NativeArray<Color32> _l7A, _l7N, _l8A, _l8N;
        private NativeArray<float4> _blockRects;
        private NativeArray<Color32> _blockA, _blockN;
        private int _l7Res, _l8Res, _blockRes;
        private Vector2 _bakedOrigin;
        private float _bakedSize;

        // 瓦片金字塔流式缓存（Baked 模式可选）——瓦片以 GPU 纹理驻留，
        // 加载不解码（ASTC 平台保持压缩），更新 = Graphics.CopyTexture 矩形直拷，CPU 只做簿记。
        private bool _hasPyramid;
        private string _bakedRoot;
        private int _tileRes;
        private float _tileWorld0;
        private int _tileLevels;
        private int _tileBudget = 64;   // 驻留上限（对，albedo+normal 各 tileRes²）
        private int _residentTiles;
        private readonly Dictionary<long, TileEntry> _tiles = new Dictionary<long, TileEntry>();
        private readonly List<TileEntry> _loading = new List<TileEntry>();

        private class TileEntry
        {
            public int level, tx, tz;
            public Texture2D texA, texN;            // 非 null = 已驻留（格式 = 环面目标格式）
            public ResourceRequest reqA, reqN;
            public int lastUsedFrame;
            public bool failed;
            public int failCount;       // 连续失败次数（退避重试用）
            public int nextRetryFrame;  // 允许重新发起加载的最早帧
        }

        /// <summary>ASTC 硬件能力（决定环面与瓦片目标格式：ASTC 直拷 / RGBA32 解码回退）。</summary>
        public static bool AstcSupported { get; } = DetectAstc();

        /// <summary>BC7 硬件能力（DX/桌面，D12）。</summary>
        public static bool Bc7Supported { get; } = DetectBc7();

        /// <summary>瓦片期望目标格式（由 Controller 按环面格式设置；不一致的烘焙数据将判失败）。
        /// RGBA32 = 解码回退（瓦片任意格式加载时 CPU 解码）。</summary>
        public TextureFormat TileFormatA { get; set; } = TextureFormat.RGBA32;
        public TextureFormat TileFormatN { get; set; } = TextureFormat.RGBA32;

        /// <summary>烘焙清单声明的 albedo 格式（未声明的旧数据按 ASTC_6x6 假设）。</summary>
        public TextureFormat BakedTexFormatA { get; private set; } = TextureFormat.ASTC_6x6;

        /// <summary>瓦片驻留预算（对）。超出后 LRU 淘汰最久未用瓦片。</summary>
        public int TileBudget { get => _tileBudget; set => _tileBudget = Mathf.Max(8, value); }

        public bool HasPyramid => _hasPyramid;
        public int TileRes => _tileRes;
        public float TileWorld0 => _tileWorld0;
        public int Levels => _tileLevels;

        private static bool DetectAstc()
        {
            return SystemInfo.IsFormatSupported(
                       UnityEngine.Experimental.Rendering.GraphicsFormat.RGBA_ASTC6X6_SRGB,
                       UnityEngine.Experimental.Rendering.FormatUsage.Sample)
                && SystemInfo.IsFormatSupported(
                       UnityEngine.Experimental.Rendering.GraphicsFormat.RGBA_ASTC5X5_UNorm,
                       UnityEngine.Experimental.Rendering.FormatUsage.Sample);
        }

        private static bool DetectBc7()
        {
            return SystemInfo.IsFormatSupported(
                UnityEngine.Experimental.Rendering.GraphicsFormat.RGBA_BC7_SRGB,
                UnityEngine.Experimental.Rendering.FormatUsage.Sample);
        }

        public Vector2 BakedOrigin => _bakedOrigin;
        public float BakedSize => _bakedSize;

        /// <summary>测试模式：所有数组初始化为 0 长度，保证 Job 调度安全。</summary>
        public ClipmapSourceRuntime()
        {
            Mode = ClipmapStripJob.MODE_TEST;
            _alphamaps = new NativeArray<float>(0, Allocator.Persistent);
            _heights = new NativeArray<float>(0, Allocator.Persistent);
            _layerPix0 = new NativeArray<Color32>(0, Allocator.Persistent);
            _layerPix1 = new NativeArray<Color32>(0, Allocator.Persistent);
            _layerPix2 = new NativeArray<Color32>(0, Allocator.Persistent);
            _layerPix3 = new NativeArray<Color32>(0, Allocator.Persistent);
            _layerPix4 = new NativeArray<Color32>(0, Allocator.Persistent);
            _layerPix5 = new NativeArray<Color32>(0, Allocator.Persistent);
            _layerPix6 = new NativeArray<Color32>(0, Allocator.Persistent);
            _layerPix7 = new NativeArray<Color32>(0, Allocator.Persistent);
            _l7A = new NativeArray<Color32>(0, Allocator.Persistent);
            _l7N = new NativeArray<Color32>(0, Allocator.Persistent);
            _l8A = new NativeArray<Color32>(0, Allocator.Persistent);
            _l8N = new NativeArray<Color32>(0, Allocator.Persistent);
            _blockRects = new NativeArray<float4>(0, Allocator.Persistent);
            _blockA = new NativeArray<Color32>(0, Allocator.Persistent);
            _blockN = new NativeArray<Color32>(0, Allocator.Persistent);
        }

        // ---------------- 构建：Terrain ----------------

        public static ClipmapSourceRuntime FromTerrain(Terrain t)
        {
            var s = new ClipmapSourceRuntime { Mode = ClipmapStripJob.MODE_TERRAIN };
            var td = t.terrainData;
            s._terrainPos = t.transform.position;
            s._terrainSize3 = td.size;

            var am = td.GetAlphamaps(0, 0, td.alphamapWidth, td.alphamapHeight);
            s._alphaRes = td.alphamapWidth;
            s._alphaLayers = td.alphamapLayers;
            s._alphamaps = new NativeArray<float>(am.Length, Allocator.Persistent);
            int idx = 0;
            for (int y = 0; y < td.alphamapHeight; ++y)
                for (int x = 0; x < td.alphamapWidth; ++x)
                    for (int l = 0; l < s._alphaLayers; ++l)
                        s._alphamaps[idx++] = am[y, x, l];

            var heights = td.GetHeights(0, 0, td.heightmapResolution, td.heightmapResolution);
            s._heightRes = td.heightmapResolution;
            s._heights = new NativeArray<float>(s._heightRes * s._heightRes, Allocator.Persistent);
            for (int y = 0; y < s._heightRes; ++y)
                for (int x = 0; x < s._heightRes; ++x)
                    s._heights[y * s._heightRes + x] = heights[y, x];

            for (int l = 0; l < 8; ++l)
            {
                NativeArray<Color32> px = new NativeArray<Color32>(0, Allocator.Persistent);
                int w = 0, h = 0;
                float tx = 1, ty = 1, ox = 0, oy = 0;
                if (l < td.terrainLayers.Length)
                {
                    var layer = td.terrainLayers[l];
                    tx = layer.tileSize.x; ty = layer.tileSize.y;
                    ox = layer.tileOffset.x; oy = layer.tileOffset.y;
                    if (layer.diffuseTexture == null)
                    {
                        Debug.LogWarning($"[ClipmapSource] 地形层 {l} ({layer.name}) 无 diffuse 纹理，烘焙回退为灰色");
                    }
                    else
                    {
                        // 非可读纹理经 RenderTexture 中转读回（GPU Blit，无需修改导入设置）
                        var src = GetPixels32Any(layer.diffuseTexture);
                        px = new NativeArray<Color32>(src, Allocator.Persistent);
                        w = layer.diffuseTexture.width; h = layer.diffuseTexture.height;
                    }
                }
                switch (l)
                {
                    case 0: s._layerPix0 = px; break;
                    case 1: s._layerPix1 = px; break;
                    case 2: s._layerPix2 = px; break;
                    case 3: s._layerPix3 = px; break;
                    case 4: s._layerPix4 = px; break;
                    case 5: s._layerPix5 = px; break;
                    case 6: s._layerPix6 = px; break;
                    default: s._layerPix7 = px; break;
                }
                if (l < 4)
                {
                    s._layerW[l] = w; s._layerH[l] = h;
                    s._tileX[l] = tx; s._tileY[l] = ty;
                    s._tileOffX[l] = ox; s._tileOffY[l] = oy;
                }
                else
                {
                    int l4 = l - 4;
                    s._layerWB[l4] = w; s._layerHB[l4] = h;
                    s._tileXB[l4] = tx; s._tileYB[l4] = ty;
                    s._tileOffXB[l4] = ox; s._tileOffYB[l4] = oy;
                }
            }
            if (td.alphamapLayers > 8)
                Debug.LogWarning($"[ClipmapSource] 地形共 {td.alphamapLayers} 层，原型仅烘焙前 8 层，其余层将丢失");
            return s;
        }

        /// <summary>
        /// 读取纹理像素。纹理未开 Read/Write 时经 RenderTexture 中转读回（GPU Blit 采样 +
        /// ReadPixels，编辑器/运行时均可用），sRGB 标志与源纹理一致以保证字节值不变。
        /// </summary>
        private static Color32[] GetPixels32Any(Texture2D tex)
        {
            if (tex.isReadable) return tex.GetPixels32();
            bool srgb = UnityEngine.Experimental.Rendering.GraphicsFormatUtility.IsSRGBFormat(tex.graphicsFormat);
            var rt = RenderTexture.GetTemporary(tex.width, tex.height, 0, RenderTextureFormat.ARGB32,
                srgb ? RenderTextureReadWrite.sRGB : RenderTextureReadWrite.Linear);
            var prev = RenderTexture.active;
            Graphics.Blit(tex, rt);
            RenderTexture.active = rt;
            var tmp = new Texture2D(tex.width, tex.height, TextureFormat.RGBA32, false, false);
            tmp.ReadPixels(new Rect(0, 0, tex.width, tex.height), 0, 0);
            tmp.Apply(false);
            RenderTexture.active = prev;
            RenderTexture.ReleaseTemporary(rt);
            var px = tmp.GetPixels32();
            // 编辑模式（烘焙）只能 DestroyImmediate；运行时（Terrain 模式）用 Destroy
            if (Application.isPlaying) UnityEngine.Object.Destroy(tmp);
            else UnityEngine.Object.DestroyImmediate(tmp);
            return px;
        }

        // ---------------- 构建：Baked（Resources 加载） ----------------

        public static ClipmapSourceRuntime FromBaked(string resourcesRoot)
        {
            var manifestAsset = Resources.Load<TextAsset>(resourcesRoot + "/manifest");
            if (manifestAsset == null) return null;
            var m = JsonUtility.FromJson<ClipmapManifest>(manifestAsset.text);

            var s = new ClipmapSourceRuntime { Mode = ClipmapStripJob.MODE_BAKED };
            s._bakedOrigin = new Vector2(m.originX, m.originZ);
            s._bakedSize = m.terrainSize;
            s._l7Res = m.l7Res; s._l8Res = m.l8Res; s._blockRes = m.blockRes;
            s._bakedRoot = resourcesRoot;
            s._tileRes = m.tileRes; s._tileWorld0 = m.tileWorld0; s._tileLevels = m.tileLevels;
            s._hasPyramid = m.tileLevels > 0 && m.tileRes > 0;
            if (m.texFormat != 0) s.BakedTexFormatA = (TextureFormat)m.texFormat;

            if (s._hasPyramid)
            {
                // 直拷模式（§7.1）：更新走 Graphics.CopyTexture，Job 采样永不执行——
                // 跳过 L7 全图与高精块的 CPU 像素加载（2040² 下约 33MB+ 死内存）
                s._l7A = new NativeArray<Color32>(0, Allocator.Persistent);
                s._l7N = new NativeArray<Color32>(0, Allocator.Persistent);
                s._l8A = new NativeArray<Color32>(0, Allocator.Persistent);
                s._l8N = new NativeArray<Color32>(0, Allocator.Persistent);
                s._blockRects = new NativeArray<float4>(0, Allocator.Persistent);
                s._blockA = new NativeArray<Color32>(0, Allocator.Persistent);
                s._blockN = new NativeArray<Color32>(0, Allocator.Persistent);
                return s;
            }

            s._l7A = LoadTexPixels(resourcesRoot + "/l7_albedo", true);
            s._l7N = LoadTexPixels(resourcesRoot + "/l7_normal", false);
            // L8 纹理由 Controller 直接加载使用（不经 Job）
            s._l8A = new NativeArray<Color32>(0, Allocator.Persistent);
            s._l8N = new NativeArray<Color32>(0, Allocator.Persistent);

            int bc = Mathf.Min(MaxBlocks, m.blockRects.Length); // 原型上限 MaxBlocks 块
            s._blockRects = new NativeArray<float4>(bc, Allocator.Persistent);
            int totalPx = bc * m.blockRes * m.blockRes;
            s._blockA = new NativeArray<Color32>(totalPx, Allocator.Persistent);
            s._blockN = new NativeArray<Color32>(totalPx, Allocator.Persistent);
            for (int i = 0; i < bc; ++i)
            {
                var r = m.blockRects[i];
                s._blockRects[i] = new float4(r.x, r.y, r.width, r.height);
                var pa = LoadTexPixels($"{resourcesRoot}/block_{i}_albedo", true);
                var pn = LoadTexPixels($"{resourcesRoot}/block_{i}_normal", false);
                NativeArray<Color32>.Copy(pa, 0, s._blockA, i * m.blockRes * m.blockRes, pa.Length);
                NativeArray<Color32>.Copy(pn, 0, s._blockN, i * m.blockRes * m.blockRes, pn.Length);
                pa.Dispose(); pn.Dispose();
            }
            return s;
        }

        /// <summary>
        /// 加载 .bytes 纹理为 Texture2D（供 L8 直接使用）。
        /// 带 mipmaps 时 .bytes 必须已含离线生成的完整 Mip 链（§4.1：勿依赖运行时 glGenerateMipmap，
        /// ASTC 压缩纹理运行时无法重建 mip），故 Apply(false) 仅上传不重建。
        /// </summary>
        public static Texture2D LoadBakedTexture(string resourcesPath, bool sRGB, bool mipmaps)
        {
            var asset = Resources.Load<TextAsset>(resourcesPath);
            if (asset == null) return null;
            var bytes = asset.bytes;
            int w = BitConverter.ToInt32(bytes, 0);
            int h = BitConverter.ToInt32(bytes, 4);
            var fmt = (TextureFormat)BitConverter.ToInt32(bytes, 8);
            var tex = new Texture2D(w, h, fmt, mipmaps, !sRGB);
            var payload = new byte[bytes.Length - 12];
            Buffer.BlockCopy(bytes, 12, payload, 0, payload.Length);
            tex.LoadRawTextureData(payload);
            tex.Apply(false);
            tex.filterMode = mipmaps ? FilterMode.Trilinear : FilterMode.Bilinear;
            tex.wrapMode = TextureWrapMode.Clamp;
            return tex;
        }

        /// <summary>
        /// 加载 .bytes 纹理（静态层 L8 用）：文件格式当前平台可采样则直接加载；
        /// 否则 CPU 软件解码 mip0 为 RGBA32 并按需重建 mip 链（ASTC 在不支持的平台由 Unity 软件解压）。
        /// 与环面/瓦片的格式协商策略一致，避免压缩烘焙数据在不支持的平台上静态层加载失败。
        /// </summary>
        public static Texture2D LoadBakedTextureChecked(string resourcesPath, bool sRGB, bool mipmaps)
        {
            var asset = Resources.Load<TextAsset>(resourcesPath);
            if (asset == null) return null;
            var bytes = asset.bytes;
            if (bytes == null || bytes.Length < 12) return null;
            int w = BitConverter.ToInt32(bytes, 0);
            int h = BitConverter.ToInt32(bytes, 4);
            var fmt = (TextureFormat)BitConverter.ToInt32(bytes, 8);
            var gf = UnityEngine.Experimental.Rendering.GraphicsFormatUtility.GetGraphicsFormat(fmt, sRGB);
            if (SystemInfo.IsFormatSupported(gf, UnityEngine.Experimental.Rendering.FormatUsage.Sample))
                return LoadBakedTexture(resourcesPath, sRGB, mipmaps);

            // 平台不支持该压缩格式：解码 mip0 → RGBA32（mip 链运行时重建）
            var payload = new byte[bytes.Length - 12];
            Buffer.BlockCopy(bytes, 12, payload, 0, payload.Length);
            var tmp = new Texture2D(w, h, fmt, false, !sRGB);
            tmp.LoadRawTextureData(payload);
            tmp.Apply(false);
            var px = tmp.GetPixels32();
            UnityEngine.Object.Destroy(tmp);
            var tex = new Texture2D(w, h, TextureFormat.RGBA32, mipmaps, !sRGB);
            tex.SetPixels32(px);
            tex.Apply(mipmaps);
            tex.filterMode = mipmaps ? FilterMode.Trilinear : FilterMode.Bilinear;
            tex.wrapMode = TextureWrapMode.Clamp;
            return tex;
        }

        /// <summary>加载 .bytes 纹理并取出 CPU 像素（ASTC 在不支持的平台由 Unity 软件解压）。</summary>
        private static NativeArray<Color32> LoadTexPixels(string resourcesPath, bool sRGB)
        {
            var tex = LoadBakedTexture(resourcesPath, sRGB, false);
            if (tex == null) return new NativeArray<Color32>(0, Allocator.Persistent);
            var px = tex.GetPixels32();
            var arr = new NativeArray<Color32>(px, Allocator.Persistent);
            UnityEngine.Object.Destroy(tex);
            return arr;
        }

        // ---------------- 瓦片金字塔流式（Baked 模式，GPU 纹理驻留 + 直拷） ----------------

        /// <summary>该级瓦片网格边数（地形边缘按 ceil，末列/行为部分覆盖）。</summary>
        public int TileGridCount(int level)
        {
            float tw = _tileWorld0 * (1 << level);
            return Mathf.Max(1, Mathf.CeilToInt(_bakedSize / tw));
        }

        /// <summary>请求瓦片：未驻留且未在途则触发 Resources.LoadAsync；已驻留则刷新 LRU 时间。</summary>
        public void EnsureTileRequested(int level, int tx, int tz)
        {
            if (!_hasPyramid) return;
            GetOrLoadTile(level, tx, tz);
        }

        /// <summary>瓦片是否已驻留（纹理已创建）。</summary>
        public bool IsTileResident(int level, int tx, int tz)
        {
            return _tiles.TryGetValue(TileKey(level, tx, tz), out var e) && e.texA != null;
        }

        /// <summary>取驻留瓦片纹理（用于 CopyTexture 直拷）；未驻留返回 false。</summary>
        public bool TryGetTile(int level, int tx, int tz, out Texture2D albedo, out Texture2D normal)
        {
            if (_tiles.TryGetValue(TileKey(level, tx, tz), out var e) && e.texA != null)
            {
                e.lastUsedFrame = Time.frameCount;
                albedo = e.texA; normal = e.texN;
                return true;
            }
            albedo = null; normal = null;
            return false;
        }

        /// <summary>每帧轮询异步瓦片加载（由 MTClipmapController.Update 驱动）。</summary>
        public void Update()
        {
            for (int i = _loading.Count - 1; i >= 0; --i)
            {
                var e = _loading[i];
                if (!e.reqA.isDone || !e.reqN.isDone) continue;
                _loading.RemoveAt(i);
                FinishLoadTile(e);
            }
        }

        private static long TileKey(int level, int tx, int tz)
            => ((long)level << 40) | ((long)tx << 20) | (uint)tz;

        private TileEntry GetOrLoadTile(int level, int tx, int tz)
        {
            long key = TileKey(level, tx, tz);
            if (_tiles.TryGetValue(key, out var e))
            {
                if (e.texA != null) e.lastUsedFrame = Time.frameCount;
                else if (e.failed && Time.frameCount >= e.nextRetryFrame) IssueTileRequest(e); // 失败后退避重试
                return e;
            }
            if (_residentTiles >= _tileBudget) EvictLRU();
            e = new TileEntry { level = level, tx = tx, tz = tz };
            _tiles.Add(key, e);
            IssueTileRequest(e);
            return e;
        }

        private void IssueTileRequest(TileEntry e)
        {
            e.failed = false;
            string basePath = $"{_bakedRoot}/tiles/l{e.level}_{e.tx}_{e.tz}";
            e.reqA = Resources.LoadAsync<TextAsset>(basePath + "_albedo");
            e.reqN = Resources.LoadAsync<TextAsset>(basePath + "_normal");
            _loading.Add(e);
        }

        private void FinishLoadTile(TileEntry e)
        {
            var ta = e.reqA.asset as TextAsset;
            var tn = e.reqN.asset as TextAsset;
            e.reqA = null; e.reqN = null;
            if (ta == null || tn == null) { MarkLoadFailed(e); return; }
            e.texA = CreateTileTexture(ta.bytes, true, TileFormatA);
            e.texN = CreateTileTexture(tn.bytes, false, TileFormatN);
            Resources.UnloadAsset(ta);
            Resources.UnloadAsset(tn);
            if (e.texA == null || e.texN == null)
            {
                if (e.texA != null) { UnityEngine.Object.Destroy(e.texA); e.texA = null; }
                if (e.texN != null) { UnityEngine.Object.Destroy(e.texN); e.texN = null; }
                MarkLoadFailed(e);
                return;
            }
            e.failCount = 0;
            e.lastUsedFrame = Time.frameCount;
            _residentTiles++;
        }

        /// <summary>加载失败：线性退避后重试（1s 起、上限 10s），避免单文件缺失导致所属层永久失效。</summary>
        private void MarkLoadFailed(TileEntry e)
        {
            e.failed = true;
            e.failCount++;
            int delayFrames = 60 * Mathf.Min(e.failCount, 10);
            e.nextRetryFrame = Time.frameCount + delayFrames;
            if (e.failCount <= 3)
                Debug.LogWarning($"[ClipmapSource] 瓦片 l{e.level}_{e.tx}_{e.tz} 加载失败（第 {e.failCount} 次），{delayFrames / 60f:0}s 后重试");
        }

        /// <summary>
        /// 从 .bytes 创建瓦片纹理。expectFmt != RGBA32（ASTC/BC7）：保持压缩直接上传（零 CPU 解码），
        /// 格式不符返回 null；expectFmt == RGBA32：任意格式 CPU 解码一次，与 RGBA32 环面格式兼容。
        /// </summary>
        private Texture2D CreateTileTexture(byte[] bytes, bool sRGB, TextureFormat expectFmt)
        {
            if (bytes == null || bytes.Length < 12) return null;
            int w = BitConverter.ToInt32(bytes, 0);
            int h = BitConverter.ToInt32(bytes, 4);
            var fmt = (TextureFormat)BitConverter.ToInt32(bytes, 8);
            var payload = new byte[bytes.Length - 12];
            Buffer.BlockCopy(bytes, 12, payload, 0, payload.Length);
            if (expectFmt != TextureFormat.RGBA32)
            {
                if (fmt != expectFmt)
                {
                    Debug.LogError($"[ClipmapSource] 瓦片格式 {fmt} 与环面目标 {expectFmt} 不一致（请按目标平台重新烘焙）");
                    return null;
                }
                var t = new Texture2D(w, h, fmt, false, !sRGB);
                t.LoadRawTextureData(payload);
                t.Apply(false);
                t.wrapMode = TextureWrapMode.Clamp;
                t.filterMode = FilterMode.Bilinear;
                return t;
            }
            var tmp = new Texture2D(w, h, fmt, false, !sRGB);
            tmp.LoadRawTextureData(payload);
            tmp.Apply(false);
            if (fmt == TextureFormat.RGBA32) { tmp.wrapMode = TextureWrapMode.Clamp; return tmp; }
            var px = tmp.GetPixels32(); // CPU 解码（ASTC/BC7 在不支持的平台由 Unity 软件解压）
            UnityEngine.Object.Destroy(tmp);
            var t2 = new Texture2D(w, h, TextureFormat.RGBA32, false, !sRGB);
            t2.SetPixels32(px);
            t2.Apply(false);
            t2.wrapMode = TextureWrapMode.Clamp;
            return t2;
        }

        /// <summary>淘汰最久未用的驻留瓦片（本帧在用或加载中的跳过；都在用则暂时超支）。</summary>
        private void EvictLRU()
        {
            TileEntry victim = null;
            long victimKey = 0;
            foreach (var kv in _tiles)
            {
                var e = kv.Value;
                if (e.texA == null || e.lastUsedFrame >= Time.frameCount) continue;
                if (victim == null || e.lastUsedFrame < victim.lastUsedFrame) { victim = e; victimKey = kv.Key; }
            }
            if (victim == null) return;
            UnityEngine.Object.Destroy(victim.texA);
            UnityEngine.Object.Destroy(victim.texN);
            _tiles.Remove(victimKey);
            _residentTiles--;
        }

        // ---------------- Job 工厂 ----------------

        public ClipmapStripJob CreateJob(int layer, bool albedo, float2 worldMin, float texel,
            int stripW, NativeArray<Color32> output, int dstStride, int dstOffset)
        {
            var j = new ClipmapStripJob
            {
                mode = Mode,
                layer = layer,
                albedo = albedo,
                worldMin = worldMin,
                texel = texel,
                stripW = stripW,
                dstStride = dstStride,
                dstOffset = dstOffset,
                output = output,
                // terrain
                terrainPos = _terrainPos,
                terrainDataSize = _terrainSize3,
                alphamaps = _alphamaps,
                alphaRes = _alphaRes,
                alphaLayers = _alphaLayers,
                layerPix0 = _layerPix0, layerPix1 = _layerPix1,
                layerPix2 = _layerPix2, layerPix3 = _layerPix3,
                layerSize = _layerW, layerSizeH = _layerH,
                tileSizeX = _tileX, tileSizeY = _tileY,
                tileOffX = _tileOffX, tileOffY = _tileOffY,
                layerPix4 = _layerPix4, layerPix5 = _layerPix5,
                layerPix6 = _layerPix6, layerPix7 = _layerPix7,
                layerSizeB = _layerWB, layerSizeHB = _layerHB,
                tileSizeXB = _tileXB, tileSizeYB = _tileYB,
                tileOffXB = _tileOffXB, tileOffYB = _tileOffYB,
                heights = _heights,
                heightRes = _heightRes,
                // baked
                l7Albedo = _l7A, l7Normal = _l7N, l7Res = _l7Res,
                l8Albedo = _l8A, l8Normal = _l8N, l8Res = _l8Res,
                blockRects = _blockRects,
                blockAlbedo = _blockA, blockNormal = _blockN,
                blockRes = _blockRes,
                blockCount = _blockRects.IsCreated ? _blockRects.Length : 0,
                bakedOriginXZ = _bakedOrigin,
                bakedTerrainSize = _bakedSize,
            };
            if (Mode == ClipmapStripJob.MODE_TEST)
            {
                float hue = layer / 8f;
                Color a = Color.HSVToRGB(hue, 0.55f, 0.85f);
                Color b = a * 0.55f;
                j.testColorA = new float4(a.r, a.g, a.b, 1f);
                j.testColorB = new float4(b.r, b.g, b.b, 1f);
                j.testCell = texel * 8f;
            }
            return j;
        }

        public void Dispose()
        {
            DisposeIf(ref _alphamaps); DisposeIf(ref _heights);
            DisposeIf(ref _layerPix0); DisposeIf(ref _layerPix1);
            DisposeIf(ref _layerPix2); DisposeIf(ref _layerPix3);
            DisposeIf(ref _layerPix4); DisposeIf(ref _layerPix5);
            DisposeIf(ref _layerPix6); DisposeIf(ref _layerPix7);
            DisposeIf(ref _l7A); DisposeIf(ref _l7N); DisposeIf(ref _l8A); DisposeIf(ref _l8N);
            DisposeIf(ref _blockRects); DisposeIf(ref _blockA); DisposeIf(ref _blockN);
            foreach (var kv in _tiles)
            {
                if (kv.Value.texA != null) UnityEngine.Object.Destroy(kv.Value.texA);
                if (kv.Value.texN != null) UnityEngine.Object.Destroy(kv.Value.texN);
            }
            _tiles.Clear();
            _loading.Clear();
        }

        private static void DisposeIf<T>(ref NativeArray<T> arr) where T : struct
        {
            if (arr.IsCreated) arr.Dispose();
        }
    }
}
