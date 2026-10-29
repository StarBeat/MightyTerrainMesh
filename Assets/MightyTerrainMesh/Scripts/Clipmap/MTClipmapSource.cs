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
    /// 纹理 .bytes 仅含 GetRawTextureData 字节流；尺寸由各级分辨率字段提供，格式由 texFormat/texFormatN 提供。
    /// </summary>
    [Serializable]
    public class ClipmapManifest
    {
        public const int CurrentTextureDataVersion = 2;
        public int textureDataVersion;
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
        // albedo / normal 的 TextureFormat 枚举值（ASTC 6x6/5x5、BC7/BC7 或 RGBA32/RGBA32）。
        public int texFormat;
        public int texFormatN;
        // 1 = Albedo.A 为 Metallic，Normal.B/A 为 Roughness/AO；0 = 旧数据 Albedo.A 无材质语义。
        public int materialChannels;
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
        public bool PackedMaterial { get; private set; }

        // Terrain 模式数据
        private NativeArray<float> _alphamaps;
        private NativeArray<Color32> _layerPix0, _layerPix1, _layerPix2, _layerPix3;
        private NativeArray<Color32> _layerPix4, _layerPix5, _layerPix6, _layerPix7;
        private NativeArray<Color32> _normalPix0, _normalPix1, _normalPix2, _normalPix3;
        private NativeArray<Color32> _normalPix4, _normalPix5, _normalPix6, _normalPix7;
        private NativeArray<Color32> _maskPix0, _maskPix1, _maskPix2, _maskPix3;
        private NativeArray<Color32> _maskPix4, _maskPix5, _maskPix6, _maskPix7;
        private NativeArray<float> _heights;
        private int _alphaRes, _alphaLayers, _heightRes;
        private int4 _layerW, _layerH, _layerWB, _layerHB;
        private int4 _normalW, _normalH, _normalWB, _normalHB;
        private int4 _maskW, _maskH, _maskWB, _maskHB;
        private float4 _tileX, _tileY, _tileXB, _tileYB;
        private float4 _tileOffX, _tileOffY, _tileOffXB, _tileOffYB;
        private float4 _normalScale, _normalScaleB, _metallic, _metallicB, _smoothness, _smoothnessB;
        private Vector3 _terrainPos, _terrainSize3;

        // Baked 模式数据
        private NativeArray<Color32> _l7A, _l7N, _l8A, _l8N;
        private NativeArray<float4> _blockRects;
        private NativeArray<Color32> _blockA, _blockN;
        private int _l7Res, _l8Res, _blockRes;
        private Vector2 _bakedOrigin;
        private float _bakedSize;

        // 瓦片金字塔流式缓存（Baked 模式可选）——瓦片以 GPU 纹理驻留，
        // 加载不解码，更新 = Graphics.CopyTexture 矩形直拷，CPU 只做簿记。
        private bool _hasPyramid;
        private string _bakedRoot;
        private int _tileRes;
        private float _tileWorld0;
        private int _tileLevels;
        private const int MaxInFlightTiles = 4;
        private int _tileBudget = 64;   // 驻留、排队和在途条目的总预算（对）
        private readonly Dictionary<long, TileEntry> _tiles = new Dictionary<long, TileEntry>();
        private readonly List<TileEntry> _loading = new List<TileEntry>();
        private readonly Queue<TileEntry> _pendingLoads = new Queue<TileEntry>();

        private class TileEntry
        {
            public int level, tx, tz;
            public Texture2D texA, texN;            // 非 null = 已驻留（格式 = 环面目标格式）
            public ResourceRequest reqA, reqN;
            public int lastUsedFrame;
            public bool requested;      // 已排队或在途，防止重复请求
            public bool failed;
            public int failCount;       // 连续失败次数（退避重试用）
            public int nextRetryFrame;  // 允许重新发起加载的最早帧
        }

        public TextureFormat TileFormatA { get; set; } = TextureFormat.RGBA32;
        public TextureFormat TileFormatN { get; set; } = TextureFormat.RGBA32;

        public TextureFormat BakedTexFormatA { get; private set; } = TextureFormat.RGBA32;
        public TextureFormat BakedTexFormatN { get; private set; } = TextureFormat.RGBA32;

        /// <summary>瓦片驻留预算（对）。超出后 LRU 淘汰最久未用瓦片。</summary>
        public int TileBudget { get => _tileBudget; set => _tileBudget = Mathf.Max(8, value); }

        public bool HasPyramid => _hasPyramid;
        public int TileRes => _tileRes;
        public int L8Res => _l8Res;
        public float TileWorld0 => _tileWorld0;
        public int Levels => _tileLevels;

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
            _normalPix0 = new NativeArray<Color32>(0, Allocator.Persistent);
            _normalPix1 = new NativeArray<Color32>(0, Allocator.Persistent);
            _normalPix2 = new NativeArray<Color32>(0, Allocator.Persistent);
            _normalPix3 = new NativeArray<Color32>(0, Allocator.Persistent);
            _normalPix4 = new NativeArray<Color32>(0, Allocator.Persistent);
            _normalPix5 = new NativeArray<Color32>(0, Allocator.Persistent);
            _normalPix6 = new NativeArray<Color32>(0, Allocator.Persistent);
            _normalPix7 = new NativeArray<Color32>(0, Allocator.Persistent);
            _maskPix0 = new NativeArray<Color32>(0, Allocator.Persistent);
            _maskPix1 = new NativeArray<Color32>(0, Allocator.Persistent);
            _maskPix2 = new NativeArray<Color32>(0, Allocator.Persistent);
            _maskPix3 = new NativeArray<Color32>(0, Allocator.Persistent);
            _maskPix4 = new NativeArray<Color32>(0, Allocator.Persistent);
            _maskPix5 = new NativeArray<Color32>(0, Allocator.Persistent);
            _maskPix6 = new NativeArray<Color32>(0, Allocator.Persistent);
            _maskPix7 = new NativeArray<Color32>(0, Allocator.Persistent);
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
            var s = new ClipmapSourceRuntime { Mode = ClipmapStripJob.MODE_TERRAIN, PackedMaterial = true };
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
                NativeArray<Color32> np = new NativeArray<Color32>(0, Allocator.Persistent);
                NativeArray<Color32> mp = new NativeArray<Color32>(0, Allocator.Persistent);
                int w = 0, h = 0, nw = 0, nh = 0, mw = 0, mh = 0;
                float tx = 1, ty = 1, ox = 0, oy = 0;
                float normalScale = 1, metallic = 0, smoothness = 0.5f;
                if (l < td.terrainLayers.Length)
                {
                    var layer = td.terrainLayers[l];
                    tx = layer.tileSize.x; ty = layer.tileSize.y;
                    ox = layer.tileOffset.x; oy = layer.tileOffset.y;
                    normalScale = layer.normalScale;
                    metallic = layer.metallic;
                    smoothness = layer.smoothness;
                    if (layer.diffuseTexture == null)
                    {
                        Debug.LogWarning($"[ClipmapSource] 地形层 {l} ({layer.name}) 无 diffuse 纹理，烘焙回退为灰色");
                    }
                    else
                    {
                        // 非可读纹理经 RenderTexture 中转读回（GPU Blit，无需修改导入设置）
                        px = GetPixels32Any(layer.diffuseTexture);
                        w = layer.diffuseTexture.width; h = layer.diffuseTexture.height;
                    }
                    if (layer.normalMapTexture != null)
                    {
                        np = GetNormalPixels32Any(layer.normalMapTexture);
                        nw = layer.normalMapTexture.width; nh = layer.normalMapTexture.height;
                    }
                    if (layer.maskMapTexture != null)
                    {
                        mp = GetPixels32Any(layer.maskMapTexture);
                        mw = layer.maskMapTexture.width; mh = layer.maskMapTexture.height;
                    }
                }
                switch (l)
                {
                    case 0: s._layerPix0 = px; s._normalPix0 = np; s._maskPix0 = mp; break;
                    case 1: s._layerPix1 = px; s._normalPix1 = np; s._maskPix1 = mp; break;
                    case 2: s._layerPix2 = px; s._normalPix2 = np; s._maskPix2 = mp; break;
                    case 3: s._layerPix3 = px; s._normalPix3 = np; s._maskPix3 = mp; break;
                    case 4: s._layerPix4 = px; s._normalPix4 = np; s._maskPix4 = mp; break;
                    case 5: s._layerPix5 = px; s._normalPix5 = np; s._maskPix5 = mp; break;
                    case 6: s._layerPix6 = px; s._normalPix6 = np; s._maskPix6 = mp; break;
                    default: s._layerPix7 = px; s._normalPix7 = np; s._maskPix7 = mp; break;
                }
                if (l < 4)
                {
                    s._layerW[l] = w; s._layerH[l] = h;
                    s._normalW[l] = nw; s._normalH[l] = nh;
                    s._maskW[l] = mw; s._maskH[l] = mh;
                    s._tileX[l] = tx; s._tileY[l] = ty;
                    s._tileOffX[l] = ox; s._tileOffY[l] = oy;
                    s._normalScale[l] = normalScale;
                    s._metallic[l] = metallic;
                    s._smoothness[l] = smoothness;
                }
                else
                {
                    int l4 = l - 4;
                    s._layerWB[l4] = w; s._layerHB[l4] = h;
                    s._normalWB[l4] = nw; s._normalHB[l4] = nh;
                    s._maskWB[l4] = mw; s._maskHB[l4] = mh;
                    s._tileXB[l4] = tx; s._tileYB[l4] = ty;
                    s._tileOffXB[l4] = ox; s._tileOffYB[l4] = oy;
                    s._normalScaleB[l4] = normalScale;
                    s._metallicB[l4] = metallic;
                    s._smoothnessB[l4] = smoothness;
                }
            }
            if (td.alphamapLayers > 8)
                Debug.LogWarning($"[ClipmapSource] 地形共 {td.alphamapLayers} 层，原型仅烘焙前 8 层，其余层将丢失");
            return s;
        }

        /// <summary>
        /// 读取纹理像素到 Persistent NativeArray。RGBA32 直接复制原生像素；压缩格式保留必要解码。
        /// 非可读纹理由 RenderTexture 中转，sRGB 标志与源纹理一致以保证字节值不变。
        /// </summary>
        private static NativeArray<Color32> GetPixels32Any(Texture2D tex)
        {
            if (tex.isReadable)
            {
                if (tex.format == TextureFormat.RGBA32)
                    return new NativeArray<Color32>(tex.GetPixelData<Color32>(0), Allocator.Persistent);
                return new NativeArray<Color32>(tex.GetPixels32(), Allocator.Persistent);
            }
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
            var px = new NativeArray<Color32>(tmp.GetPixelData<Color32>(0), Allocator.Persistent);
            // 编辑模式（烘焙）只能 DestroyImmediate；运行时（Terrain 模式）用 Destroy
            if (Application.isPlaying) UnityEngine.Object.Destroy(tmp);
            else UnityEngine.Object.DestroyImmediate(tmp);
            return px;
        }

        private static NativeArray<Color32> GetNormalPixels32Any(Texture2D tex)
        {
            var px = GetPixels32Any(tex);
            var format = tex.graphicsFormat;
            if (format == UnityEngine.Experimental.Rendering.GraphicsFormat.RGBA_DXT5_UNorm ||
                format == UnityEngine.Experimental.Rendering.GraphicsFormat.RGBA_DXT5_SRGB)
            {
                for (int i = 0; i < px.Length; ++i)
                {
                    var c = px[i];
                    px[i] = new Color32(c.a, c.g, 255, 255);
                }
            }
            return px;
        }

        // ---------------- 构建：Baked（Resources 加载） ----------------

        public static ClipmapSourceRuntime FromBaked(string resourcesRoot)
        {
            var manifestAsset = Resources.Load<TextAsset>(resourcesRoot + "/manifest");
            if (manifestAsset == null) return null;
            var m = JsonUtility.FromJson<ClipmapManifest>(manifestAsset.text);
            if (m == null) return null;
            if (m.textureDataVersion != ClipmapManifest.CurrentTextureDataVersion)
            {
                Debug.LogError($"[ClipmapSource] 纹理数据版本 {m.textureDataVersion} 与所需 {ClipmapManifest.CurrentTextureDataVersion} 不一致，请重新烘焙纯 raw 数据");
                return null;
            }

            var s = new ClipmapSourceRuntime { Mode = ClipmapStripJob.MODE_BAKED, PackedMaterial = m.materialChannels >= 1 };
            s._bakedOrigin = new Vector2(m.originX, m.originZ);
            s._bakedSize = m.terrainSize;
            s._l7Res = m.l7Res; s._l8Res = m.l8Res; s._blockRes = m.blockRes;
            s._bakedRoot = resourcesRoot;
            s._tileRes = m.tileRes; s._tileWorld0 = m.tileWorld0; s._tileLevels = m.tileLevels;
            s._hasPyramid = m.tileLevels > 0 && m.tileRes > 0;
            s.BakedTexFormatA = (TextureFormat)m.texFormat;
            s.BakedTexFormatN = (TextureFormat)m.texFormatN;

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

            s._l7A = LoadTexPixels(resourcesRoot + "/l7_albedo", true, m.l7Res, s.BakedTexFormatA);
            s._l7N = LoadTexPixels(resourcesRoot + "/l7_normal", false, m.l7Res, s.BakedTexFormatN);
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
                var pa = LoadTexPixels($"{resourcesRoot}/block_{i}_albedo", true, m.blockRes, s.BakedTexFormatA);
                var pn = LoadTexPixels($"{resourcesRoot}/block_{i}_normal", false, m.blockRes, s.BakedTexFormatN);
                NativeArray<Color32>.Copy(pa, 0, s._blockA, i * m.blockRes * m.blockRes, pa.Length);
                NativeArray<Color32>.Copy(pn, 0, s._blockN, i * m.blockRes * m.blockRes, pn.Length);
                pa.Dispose(); pn.Dispose();
            }
            return s;
        }

        /// <summary>
        /// 按 manifest 提供的尺寸和格式加载纯 raw 纹理。带 mipmaps 时数据必须已含完整离线 Mip 链，
        /// Apply(false) 仅上传，不在运行时重建压缩 Mip。
        /// </summary>
        public static Texture2D LoadBakedTexture(string resourcesPath, bool sRGB, bool mipmaps,
            int width, int height, TextureFormat fmt)
        {
            var asset = Resources.Load<TextAsset>(resourcesPath);
            if (asset == null) return null;
            var tex = new Texture2D(width, height, fmt, mipmaps, !sRGB);
            LoadRawPayload(tex, asset.bytes);
            tex.Apply(false);
            tex.filterMode = mipmaps ? FilterMode.Trilinear : FilterMode.Bilinear;
            tex.wrapMode = TextureWrapMode.Clamp;
            return tex;
        }

        /// <summary>加载纯 raw 纹理并取出 CPU 像素，供无金字塔时的 Job 采样使用。</summary>
        private static NativeArray<Color32> LoadTexPixels(string resourcesPath, bool sRGB, int res, TextureFormat fmt)
        {
            var tex = LoadBakedTexture(resourcesPath, sRGB, false, res, res, fmt);
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

        /// <summary>每帧最多完成一对纹理，并补足在途请求，限制上传尖峰和 IO 并发。</summary>
        public void Update()
        {
            for (int i = _loading.Count - 1; i >= 0; --i)
            {
                var e = _loading[i];
                if (!e.reqA.isDone || !e.reqN.isDone) continue;
                _loading.RemoveAt(i);
                FinishLoadTile(e);
                break;
            }
            PumpTileRequests();
        }

        private static long TileKey(int level, int tx, int tz)
            => ((long)level << 40) | ((long)tx << 20) | (uint)tz;

        private TileEntry GetOrLoadTile(int level, int tx, int tz)
        {
            long key = TileKey(level, tx, tz);
            if (_tiles.TryGetValue(key, out var e))
            {
                if (e.texA != null) e.lastUsedFrame = Time.frameCount;
                else if (e.failed && !e.requested && Time.frameCount >= e.nextRetryFrame) QueueTileRequest(e);
                return e;
            }
            if (_tiles.Count >= _tileBudget) EvictLRU();
            e = new TileEntry { level = level, tx = tx, tz = tz };
            _tiles.Add(key, e);
            QueueTileRequest(e);
            return e;
        }

        private void QueueTileRequest(TileEntry e)
        {
            if (e.requested || e.texA != null) return;
            e.failed = false;
            e.requested = true;
            _pendingLoads.Enqueue(e);
            PumpTileRequests();
        }

        private void PumpTileRequests()
        {
            while (_loading.Count < MaxInFlightTiles && _pendingLoads.Count > 0)
            {
                var e = _pendingLoads.Dequeue();
                if (e.texA != null) { e.requested = false; continue; }
                string basePath = $"{_bakedRoot}/tiles/l{e.level}_{e.tx}_{e.tz}";
                e.reqA = Resources.LoadAsync<TextAsset>(basePath + "_albedo");
                e.reqN = Resources.LoadAsync<TextAsset>(basePath + "_normal");
                _loading.Add(e);
            }
        }

        private void FinishLoadTile(TileEntry e)
        {
            var ta = e.reqA.asset as TextAsset;
            var tn = e.reqN.asset as TextAsset;
            e.reqA = null; e.reqN = null;
            e.requested = false;
            try
            {
                if (ta == null || tn == null) { MarkLoadFailed(e); return; }
                e.texA = CreateTileTexture(ta.bytes, true, TileFormatA);
                e.texN = CreateTileTexture(tn.bytes, false, TileFormatN);
                if (e.texA == null || e.texN == null)
                {
                    if (e.texA != null) { UnityEngine.Object.Destroy(e.texA); e.texA = null; }
                    if (e.texN != null) { UnityEngine.Object.Destroy(e.texN); e.texN = null; }
                    MarkLoadFailed(e);
                    return;
                }
                e.failCount = 0;
                e.lastUsedFrame = Time.frameCount;
            }
            finally
            {
                if (ta != null) Resources.UnloadAsset(ta);
                if (tn != null) Resources.UnloadAsset(tn);
            }
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

        /// <summary>按 manifest 的分辨率和格式从纯 raw 数据创建瓦片纹理，目标平台须支持所选格式。</summary>
        private Texture2D CreateTileTexture(byte[] bytes, bool sRGB, TextureFormat expectFmt)
        {
            if (bytes == null || bytes.Length == 0) return null;
            Texture2D t = null;
            try
            {
                t = new Texture2D(_tileRes, _tileRes, expectFmt, false, !sRGB);
                LoadRawPayload(t, bytes);
                t.Apply(false);
                t.wrapMode = TextureWrapMode.Clamp;
                t.filterMode = FilterMode.Bilinear;
                return t;
            }
            catch (Exception e)
            {
                if (t != null) UnityEngine.Object.Destroy(t);
                Debug.LogWarning($"[ClipmapSource] 瓦片纹理创建失败：{e.Message}");
                return null;
            }
        }

        private static unsafe void LoadRawPayload(Texture2D tex, byte[] bytes)
        {
            fixed (byte* data = bytes)
                tex.LoadRawTextureData((IntPtr)data, bytes.Length);
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
                normalPix0 = _normalPix0, normalPix1 = _normalPix1,
                normalPix2 = _normalPix2, normalPix3 = _normalPix3,
                maskPix0 = _maskPix0, maskPix1 = _maskPix1,
                maskPix2 = _maskPix2, maskPix3 = _maskPix3,
                layerSize = _layerW, layerSizeH = _layerH,
                normalSize = _normalW, normalSizeH = _normalH,
                maskSize = _maskW, maskSizeH = _maskH,
                tileSizeX = _tileX, tileSizeY = _tileY,
                tileOffX = _tileOffX, tileOffY = _tileOffY,
                layerPix4 = _layerPix4, layerPix5 = _layerPix5,
                layerPix6 = _layerPix6, layerPix7 = _layerPix7,
                normalPix4 = _normalPix4, normalPix5 = _normalPix5,
                normalPix6 = _normalPix6, normalPix7 = _normalPix7,
                maskPix4 = _maskPix4, maskPix5 = _maskPix5,
                maskPix6 = _maskPix6, maskPix7 = _maskPix7,
                layerSizeB = _layerWB, layerSizeHB = _layerHB,
                normalSizeB = _normalWB, normalSizeHB = _normalHB,
                maskSizeB = _maskWB, maskSizeHB = _maskHB,
                tileSizeXB = _tileXB, tileSizeYB = _tileYB,
                tileOffXB = _tileOffXB, tileOffYB = _tileOffYB,
                normalScale = _normalScale, normalScaleB = _normalScaleB,
                layerMetallic = _metallic, layerMetallicB = _metallicB,
                layerSmoothness = _smoothness, layerSmoothnessB = _smoothnessB,
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
            DisposeIf(ref _normalPix0); DisposeIf(ref _normalPix1);
            DisposeIf(ref _normalPix2); DisposeIf(ref _normalPix3);
            DisposeIf(ref _normalPix4); DisposeIf(ref _normalPix5);
            DisposeIf(ref _normalPix6); DisposeIf(ref _normalPix7);
            DisposeIf(ref _maskPix0); DisposeIf(ref _maskPix1);
            DisposeIf(ref _maskPix2); DisposeIf(ref _maskPix3);
            DisposeIf(ref _maskPix4); DisposeIf(ref _maskPix5);
            DisposeIf(ref _maskPix6); DisposeIf(ref _maskPix7);
            DisposeIf(ref _l7A); DisposeIf(ref _l7N); DisposeIf(ref _l8A); DisposeIf(ref _l8N);
            DisposeIf(ref _blockRects); DisposeIf(ref _blockA); DisposeIf(ref _blockN);
            foreach (var kv in _tiles)
            {
                if (kv.Value.texA != null) UnityEngine.Object.Destroy(kv.Value.texA);
                if (kv.Value.texN != null) UnityEngine.Object.Destroy(kv.Value.texN);
            }
            _tiles.Clear();
            _loading.Clear();
            _pendingLoads.Clear();
        }

        private static void DisposeIf<T>(ref NativeArray<T> arr) where T : struct
        {
            if (arr.IsCreated) arr.Dispose();
        }
    }
}
