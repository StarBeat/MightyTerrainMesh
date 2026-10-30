namespace MightyTerrainMesh.EditorTools
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using UnityEditor;
    using UnityEngine;

    /// <summary>
    /// Clipmap 烘焙产物可视化查看器：
    /// - 直接读取 .bytes 文件（不经过 Resources/AssetDatabase，烘焙落盘即可查看）；
    /// - 支持 l7/l8/block_* 的 albedo 与 normal，以及 tiles/ 瓦片金字塔（网格地图点选、纹理懒加载）；
    /// - 格式支持 RGBA32 / BC7（§10 桌面）/ ASTC_6x6 / ASTC_5x5，RGB/R/G/B/A 通道隔离；
    /// - albedo.A=Metallic；normal 可解码为法线可视化（RG=法线XZ → 重建 Up 分量；B=粗糙度、A=AO 走通道隔离查看）；
    /// - 含 mip 链的纹理（L8）可逐级查看，验证离线 mip 是否正确生成；
    /// - l7/l8 预览上叠加 manifest 高精块区域框（旧数据）；鼠标位置 → UV/texel/世界坐标读数。
    /// </summary>
    public class MTClipmapBakedViewer : EditorWindow
    {
        [MenuItem("MightyTerrainMesh/Clipmap Baked Viewer")]
        static void Open() => OpenWith("Assets/Resources/ClipmapBaked");

        public static void OpenWith(string dir)
        {
            var w = GetWindow<MTClipmapBakedViewer>("Clipmap 烘焙查看器");
            w._dir = dir;
            w.Reload();
        }

        private class Entry
        {
            public string name;
            public string path;             // 逐文件模式；null = 归档模式（用 bin* 字段）
            public string binPath;
            public long binOffset;
            public int binLen;
            public bool sRGB;
            public bool isNormal;
            public bool isFullTerrain;          // l7/l8：覆盖全地形（用于块区域叠加与世界坐标读数）
            public int blockIndex = -1;         // block_{i}_* → i
            public bool isTile;                 // tiles/l{lvl}_{x}_{z}_*：金字塔瓦片（纹理懒加载）
            public int tileLevel, tileX, tileZ;
            public Texture2D tex;               // 原始数据（含 mip 链）
            public Texture2D display;           // 按当前选项生成的显示纹理
            public Color32[] dispPx;            // 当前 mip 的原始像素（供鼠标读数 + 通道切换缓存）
            public int dispMip = -1;            // dispPx 对应的 mip（-1 = 未缓存）
            public int dispW, dispH;
            public string displayKey = "";
            public int width, height, mipCount = 1;
            public TextureFormat format;
            public long fileSize;
            public string error;
        }

        // 金字塔瓦片引用：仅登记位置，点选时才加载纹理解码（全量模式 L0 可达数万片）。
        // 两种来源：逐文件（pathA/pathN，旧版 tiles/ 目录）或归档（binPath + offset/lenA/lenN）。
        private class TileRef
        {
            public int level, tx, tz;
            public string pathA, pathN; // albedo / normal 文件路径（逐文件模式）
            public string binPath;      // tiles.bin（归档模式）
            public long offset;
            public int lenA, lenN;
        }

        private string _dir = "Assets/Resources/ClipmapBaked";
        private readonly List<Entry> _entries = new List<Entry>();
        private int _selected = -1;
        private ClipmapManifest _manifest;
        private Vector2 _scrollList;
        private Rect _previewRect;
        private string _loadError;

        // 瓦片金字塔浏览状态
        private readonly Dictionary<int, List<TileRef>> _tiles = new Dictionary<int, List<TileRef>>();
        private readonly List<int> _tileLevels = new List<int>();
        private int _tileLevelIdx;
        private bool _tileMode;          // true = 预览区显示瓦片而非根目录条目
        private TileRef _curTile;
        private Entry _tileEntry;        // 当前瓦片的动态条目（选中即建、切换即毁）
        private bool _tileAlbedo = true;

        private enum ChannelMode { RGB, R, G, B, A }
        private static readonly string[] ChannelNames = { "RGB", "R", "G", "B", "A" };
        private ChannelMode _channel = ChannelMode.RGB;
        private bool _decodeNormal = true;
        private int _mip = 0;
        private bool _pointFilter = true;
        private bool _showBlocks = true;

        void OnEnable() { if (string.IsNullOrEmpty(_loadError)) Reload(); }

        void OnDisable() => ClearEntries();

        // ---------------- 加载 ----------------

        void ClearEntries()
        {
            foreach (var e in _entries) DestroyEntry(e);
            _entries.Clear();
            _selected = -1;
            DestroyEntry(_tileEntry);
            _tileEntry = null;
            _curTile = null;
            _tileMode = false;
            _tiles.Clear();
            _tileLevels.Clear();
            _tileLevelIdx = 0;
            if (_tileMapTex != null) { DestroyImmediate(_tileMapTex); _tileMapTex = null; }
            _tileMapLevel = -1;
        }

        static void DestroyEntry(Entry e)
        {
            if (e == null) return;
            if (e.tex != null) DestroyImmediate(e.tex);
            if (e.display != null) DestroyImmediate(e.display);
        }

        void Reload()
        {
            ClearEntries();
            _manifest = null;
            _loadError = null;
            string full = MTTerrainBakePipeline.ToFullPath(_dir);
            if (!Directory.Exists(full))
            {
                _loadError = "目录不存在: " + full;
                return;
            }
            string mf = Path.Combine(full, "manifest.json");
            if (!File.Exists(mf))
            {
                _loadError = "缺少 manifest.json，无法确定纹理尺寸和格式";
                return;
            }
            try { _manifest = JsonUtility.FromJson<ClipmapManifest>(File.ReadAllText(mf)); }
            catch (Exception e) { _loadError = "manifest 解析失败: " + e.Message; return; }
            if (_manifest == null || _manifest.textureDataVersion != ClipmapManifest.CurrentTextureDataVersion)
            {
                _loadError = "纹理数据版本不匹配，请重新烘焙纯 raw 数据";
                return;
            }
            foreach (var f in Directory.GetFiles(full, "*.bytes"))
            {
                var name = Path.GetFileNameWithoutExtension(f);
                var e = new Entry
                {
                    name = name,
                    path = f,
                    sRGB = name.EndsWith("_albedo"),
                    isNormal = name.EndsWith("_normal"),
                    isFullTerrain = name.StartsWith("l7_") || name.StartsWith("l8_"),
                    fileSize = new FileInfo(f).Length,
                };
                if (name.StartsWith("block_"))
                {
                    var parts = name.Split('_');
                    if (parts.Length >= 3) int.TryParse(parts[1], out e.blockIndex);
                }
                LoadEntry(e);
                _entries.Add(e);
            }
            _entries.Sort((a, b) =>
            {
                int ka = SortKey(a), kb = SortKey(b);
                return ka != kb ? ka - kb : string.CompareOrdinal(a.name, b.name);
            });
            ScanTiles(full);
            if (_entries.Count == 0 && _tileLevels.Count == 0)
                _loadError = "目录中没有 .bytes 烘焙产物: " + full;
            else if (_entries.Count > 0)
                _selected = 0;
        }

        // 扫描金字塔瓦片：旧版 tiles/ 子目录（只解析文件名）+ StreamingAssets 归档索引（tiles.idx），
        // 两者互补，同键以逐文件条目为准
        void ScanTiles(string full)
        {
            var byLevel = new Dictionary<int, Dictionary<long, TileRef>>();
            string tileDir = Path.Combine(full, "tiles");
            if (Directory.Exists(tileDir))
            foreach (var f in Directory.GetFiles(tileDir, "*.bytes"))
            {
                var p = Path.GetFileNameWithoutExtension(f).Split('_');
                if (p.Length != 4 || p[0].Length < 2 || p[0][0] != 'l') continue;
                if (!int.TryParse(p[0].Substring(1), out int lvl)) continue;
                if (!int.TryParse(p[1], out int tx) || !int.TryParse(p[2], out int tz)) continue;
                bool albedo = p[3] == "albedo";
                if (!albedo && p[3] != "normal") continue;
                if (!byLevel.TryGetValue(lvl, out var map))
                {
                    map = new Dictionary<long, TileRef>();
                    byLevel.Add(lvl, map);
                }
                long key = ((long)tx << 32) | (uint)tz;
                if (!map.TryGetValue(key, out var tr))
                {
                    tr = new TileRef { level = lvl, tx = tx, tz = tz };
                    map.Add(key, tr);
                }
                if (albedo) tr.pathA = f; else tr.pathN = f;
            }
            // 归档：与输出根对应的 StreamingAssets 目录下的 tiles.idx（绝对路径，避免依赖 CWD）
            string idxPath = Path.Combine(
                MTTerrainBakePipeline.ToFullPath(MTTerrainBakePipeline.StreamingRootOf(_dir)), "tiles.idx");
            if (File.Exists(idxPath)) ScanArchive(idxPath, byLevel);
            foreach (var kv in byLevel)
            {
                var list = new List<TileRef>(kv.Value.Values);
                list.Sort((a, b) => a.tz != b.tz ? a.tz - b.tz : a.tx - b.tx);
                _tiles.Add(kv.Key, list);
                _tileLevels.Add(kv.Key);
            }
            _tileLevels.Sort();
        }

        void ScanArchive(string idxPath, Dictionary<int, Dictionary<long, TileRef>> byLevel)
        {
            try
            {
                var bytes = File.ReadAllBytes(idxPath);
                if (bytes.Length < 12 || BitConverter.ToInt32(bytes, 0) != 0x4954544D) return; // 'MTTI'
                int p = 8; // magic + 版本（仅 1 个版本，字段固定）
                int count = BitConverter.ToInt32(bytes, p); p += 4;
                string binPath = Path.Combine(Path.GetDirectoryName(idxPath), "tiles.bin");
                for (int i = 0; i < count; ++i)
                {
                    int lvl = BitConverter.ToInt32(bytes, p); p += 4;
                    int tx = BitConverter.ToInt32(bytes, p); p += 4;
                    int tz = BitConverter.ToInt32(bytes, p); p += 4;
                    long off = BitConverter.ToInt64(bytes, p); p += 8;
                    int lenA = BitConverter.ToInt32(bytes, p); p += 4;
                    int lenN = BitConverter.ToInt32(bytes, p); p += 4;
                    if (!byLevel.TryGetValue(lvl, out var map))
                    {
                        map = new Dictionary<long, TileRef>();
                        byLevel.Add(lvl, map);
                    }
                    long key = ((long)tx << 32) | (uint)tz;
                    if (map.ContainsKey(key)) continue; // 同键以逐文件条目为准
                    map.Add(key, new TileRef { level = lvl, tx = tx, tz = tz,
                        binPath = binPath, offset = off, lenA = lenA, lenN = lenN });
                }
            }
            catch (Exception e) { Debug.LogWarning("[ClipmapViewer] tiles.idx 解析失败: " + e.Message); }
        }

        static int SortKey(Entry e)
        {
            if (e.name.StartsWith("l7_")) return 0;
            if (e.name.StartsWith("l8_")) return 10;
            if (e.blockIndex >= 0) return 20 + e.blockIndex;
            return 100;
        }

        void LoadEntry(Entry e)
        {
            try
            {
                int res = e.isTile ? _manifest.tileRes
                    : e.name.StartsWith("l7_") ? _manifest.l7Res
                    : e.name.StartsWith("l8_") ? _manifest.l8Res
                    : e.blockIndex >= 0 ? _manifest.blockRes : 0;
                if (res <= 0) { e.error = "manifest 中没有该条目的有效分辨率"; return; }
                e.width = e.height = res;
                e.format = (TextureFormat)(e.isNormal ? _manifest.texFormatN : _manifest.texFormat);
                var bytes = e.path != null ? File.ReadAllBytes(e.path) : ReadSegment(e.binPath, e.binOffset, e.binLen);
                // 依据纯 raw 数据长度判断是否含完整 mip 链。
                int sizeNoMip = MipDataSize(e.width, e.height, e.format, false);
                int sizeFull = MipDataSize(e.width, e.height, e.format, true);
                bool hasMips;
                if (bytes.Length == sizeFull && sizeFull != sizeNoMip) hasMips = true;
                else if (bytes.Length == sizeNoMip) hasMips = false;
                else
                {
                    e.error = $"数据长度 {bytes.Length}B 与 {e.format} {e.width}×{e.height} 不匹配" +
                              $"（无 mip 应 {sizeNoMip}B，含 mip 链应 {sizeFull}B）";
                    return;
                }
                var tex = new Texture2D(e.width, e.height, e.format, hasMips, !e.sRGB);
                tex.LoadRawTextureData(bytes);
                tex.Apply(false);
                tex.wrapMode = TextureWrapMode.Clamp;
                e.tex = tex;
                e.mipCount = hasMips ? tex.mipmapCount : 1;
            }
            catch (Exception ex) { e.error = ex.Message; }
        }

        static byte[] ReadSegment(string path, long offset, int len)
        {
            using (var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                fs.Seek(offset, SeekOrigin.Begin);
                var buf = new byte[len];
                int read = 0;
                while (read < len)
                {
                    int r = fs.Read(buf, read, len - read);
                    if (r <= 0) throw new EndOfStreamException("tiles.bin 数据截断");
                    read += r;
                }
                return buf;
            }
        }

        static int MipDataSize(int w, int h, TextureFormat fmt, bool fullChain)
        {
            long total = 0;
            while (true)
            {
                total += SingleMipSize(w, h, fmt);
                if (!fullChain || (w == 1 && h == 1)) break;
                w = Mathf.Max(1, w >> 1);
                h = Mathf.Max(1, h >> 1);
            }
            return (int)total;
        }

        static int SingleMipSize(int w, int h, TextureFormat fmt)
        {
            switch (fmt)
            {
                case TextureFormat.RGBA32: return w * h * 4;
                case TextureFormat.BC7: return ((w + 3) / 4) * ((h + 3) / 4) * 16; // 4×4 块 × 16B（§10 桌面格式）
                case TextureFormat.ASTC_6x6: return ((w + 5) / 6) * ((h + 5) / 6) * 16;
                case TextureFormat.ASTC_5x5: return ((w + 4) / 5) * ((h + 4) / 5) * 16;
                default: throw new NotSupportedException("查看器暂不支持格式: " + fmt);
            }
        }

        // ---------------- 显示纹理生成 ----------------

        string DisplayKey() => $"{_channel}|{_decodeNormal}|{_mip}|{_pointFilter}";

        void RebuildDisplay(Entry e)
        {
            if (e.display != null) { DestroyImmediate(e.display); e.display = null; }
            if (e.tex == null) return;
            int mip = Mathf.Min(_mip, e.mipCount - 1);
            // ASTC/BC7 的 GetPixels32 走编辑器软解，2040² 耗时可观：
            // 按 mip 缓存解码结果，切通道/滤波/解码开关时不再重复软解
            Color32[] px;
            if (e.dispPx != null && e.dispMip == mip) px = e.dispPx;
            else
            {
                px = e.tex.GetPixels32(mip); // 压缩格式由编辑器软件解码；sRGB 纹理返回原始存储值
                e.dispPx = px;
                e.dispMip = mip;
            }
            int w = Mathf.Max(1, e.width >> mip);
            int h = Mathf.Max(1, e.height >> mip);
            var outPx = new Color32[px.Length];
            bool decode = _channel == ChannelMode.RGB && _decodeNormal && e.isNormal;
            for (int i = 0; i < px.Length; ++i)
            {
                var c = px[i];
                Color32 o;
                switch (_channel)
                {
                    case ChannelMode.R: o = new Color32(c.r, c.r, c.r, 255); break;
                    case ChannelMode.G: o = new Color32(c.g, c.g, c.g, 255); break;
                    case ChannelMode.B: o = new Color32(c.b, c.b, c.b, 255); break;
                    case ChannelMode.A: o = new Color32(c.a, c.a, c.a, 255); break;
                    default: o = new Color32(c.r, c.g, c.b, 255); break;
                }
                if (decode)
                {
                    // RG = 法线 XZ（0.5 编码）→ 重建 Up 分量，按 (x, up, z) 显示
                    float nx = c.r / 255f * 2f - 1f;
                    float nz = c.g / 255f * 2f - 1f;
                    float ny = Mathf.Sqrt(Mathf.Max(0f, 1f - nx * nx - nz * nz));
                    o = new Color32(
                        (byte)((nx * 0.5f + 0.5f) * 255f),
                        (byte)((ny * 0.5f + 0.5f) * 255f),
                        (byte)((nz * 0.5f + 0.5f) * 255f), 255);
                }
                outPx[i] = o;
            }
            // 显示纹理用 linear：原始字节值直显（数据查看器要 WYSIWYG，不做 sRGB 二次解释）
            var d = new Texture2D(w, h, TextureFormat.RGBA32, false, true);
            d.SetPixelData(outPx, 0);
            d.Apply(false);
            d.filterMode = _pointFilter ? FilterMode.Point : FilterMode.Bilinear;
            d.wrapMode = TextureWrapMode.Clamp;
            e.display = d;
            e.dispPx = px;   // 保留当前 mip 原始像素供读数
            e.dispW = w;
            e.dispH = h;
            e.displayKey = DisplayKey();
        }

        // ---------------- 瓦片金字塔浏览 ----------------

        // 点选瓦片：销毁上一片纹理（2040² RGBA32 达 33MB，不缓存多片），按当前 albedo/normal 选择加载对应文件
        void SelectTile(TileRef t)
        {
            _tileMode = true;
            _selected = -1;
            _curTile = t;
            DestroyEntry(_tileEntry);
            _tileEntry = null;
            string path = _tileAlbedo ? t.pathA : t.pathN;
            if (path == null && t.binPath == null) return; // 无逐文件也无归档（理论上不发生，防御）
            _tileEntry = new Entry
            {
                name = $"l{t.level}_{t.tx}_{t.tz}_{(_tileAlbedo ? "albedo" : "normal")}",
                path = path,
                binPath = t.binPath,
                binOffset = t.offset + (_tileAlbedo ? 0 : t.lenA), // 归档内 albedo 在前、normal 在后
                binLen = _tileAlbedo ? t.lenA : t.lenN,
                sRGB = _tileAlbedo,
                isNormal = !_tileAlbedo,
                isTile = true,
                tileLevel = t.level,
                tileX = t.tx,
                tileZ = t.tz,
                fileSize = path != null ? new FileInfo(path).Length : (_tileAlbedo ? t.lenA : t.lenN),
            };
            LoadEntry(_tileEntry);
            _mip = 0;
        }

        private Texture2D _tileMapTex;   // 网格地图缓存（按 level 重建）
        private int _tileMapLevel = -1;
        private int _tileMapRes;         // 纹理分辨率（聚合桶边长）

        // 层级网格地图：色块 = 已烘焙瓦片，黄框 = 当前选中，点击选取（替代数万项的列表）。
        // 全量模式 L0 可达数万片，逐格 GUI.DrawTexture 不可行 → 聚合成小纹理一次绘制。
        void DrawTileMap(int level)
        {
            var list = _tiles[level];
            int gc = 1;
            foreach (var t in list) { gc = Mathf.Max(gc, t.tx + 1); gc = Mathf.Max(gc, t.tz + 1); }
            EditorGUILayout.LabelField(
                $"L{level}：{list.Count} 片（网格 {gc}×{gc}" +
                (_manifest != null && _manifest.tileWorld0 > 0
                    ? $"，单片 {_manifest.tileWorld0 * (1 << level):F0}m" : "") + "）",
                EditorStyles.miniLabel);
            if (_tileMapTex == null || _tileMapLevel != level)
            {
                if (_tileMapTex != null) DestroyImmediate(_tileMapTex);
                _tileMapRes = Mathf.Min(gc, 160);
                var px = new Color32[_tileMapRes * _tileMapRes];
                var bg = new Color32(51, 51, 51, 255);
                var on = new Color32(115, 179, 115, 255);
                for (int i = 0; i < px.Length; ++i) px[i] = bg;
                foreach (var t in list)
                    px[t.tz * _tileMapRes / gc * _tileMapRes + t.tx * _tileMapRes / gc] = on;
                _tileMapTex = new Texture2D(_tileMapRes, _tileMapRes, TextureFormat.RGBA32, false, true);
                _tileMapTex.SetPixelData(px, 0);
                _tileMapTex.Apply(false);
                _tileMapTex.filterMode = FilterMode.Point;
                _tileMapLevel = level;
            }
            Rect map = GUILayoutUtility.GetRect(160, 160, GUILayout.Width(160), GUILayout.Height(160));
            if (Event.current.type == EventType.Repaint)
            {
                // GUI.DrawTexture 中 texel(0,0) 位于左下：px 行序即 tz 桶序，tz=0（地形原点侧）自然落在底部
                GUI.DrawTexture(map, _tileMapTex, ScaleMode.StretchToFill);
                if (_tileMode && _curTile != null && _curTile.level == level)
                {
                    float cw = map.width / gc, ch = map.height / gc;
                    GUI.color = Color.yellow;
                    float bx = map.x + _curTile.tx * cw, by = map.yMax - (_curTile.tz + 1) * ch;
                    GUI.DrawTexture(new Rect(bx, by, Mathf.Max(2f, cw), 1.5f), EditorGUIUtility.whiteTexture);
                    GUI.DrawTexture(new Rect(bx, by + Mathf.Max(2f, ch) - 1.5f, Mathf.Max(2f, cw), 1.5f), EditorGUIUtility.whiteTexture);
                    GUI.DrawTexture(new Rect(bx, by, 1.5f, Mathf.Max(2f, ch)), EditorGUIUtility.whiteTexture);
                    GUI.DrawTexture(new Rect(bx + Mathf.Max(2f, cw) - 1.5f, by, 1.5f, Mathf.Max(2f, ch)), EditorGUIUtility.whiteTexture);
                    GUI.color = Color.white;
                }
            }
            var mp = Event.current.mousePosition;
            if (Event.current.type == EventType.MouseDown && map.Contains(mp))
            {
                int btx = Mathf.Clamp((int)((mp.x - map.x) / map.width * _tileMapRes), 0, _tileMapRes - 1);
                int btz = Mathf.Clamp((int)((map.yMax - mp.y) / map.height * _tileMapRes), 0, _tileMapRes - 1);
                // 聚合桶内取第一片（gc ≤ 160 时桶即格，无精度损失）
                var hit = list.Find(t => t.tx * _tileMapRes / gc == btx && t.tz * _tileMapRes / gc == btz);
                if (hit != null) { SelectTile(hit); Event.current.Use(); }
            }
        }

        // ---------------- GUI ----------------

        void OnGUI()
        {
            EditorGUILayout.BeginHorizontal(EditorStyles.toolbar);
            GUILayout.Label("目录", GUILayout.Width(30));
            _dir = EditorGUILayout.TextField(_dir);
            if (GUILayout.Button("加载 / 刷新", EditorStyles.toolbarButton, GUILayout.Width(80))) Reload();
            EditorGUILayout.EndHorizontal();

            if (_manifest != null)
            {
                int blockCount = _manifest.blockRects != null ? _manifest.blockRects.Length : 0;
                int tileCount = 0;
                foreach (var list in _tiles.Values) tileCount += list.Count;
                EditorGUILayout.LabelField(
                    $"manifest: origin=({_manifest.originX:F1}, {_manifest.originZ:F1})  size={_manifest.terrainSize:F0}  " +
                    $"L7={_manifest.l7Res}  L8={_manifest.l8Res}  金字塔={_manifest.tileLevels}级/{tileCount}片" +
                    (blockCount > 0 ? $"  高精块={blockCount}" : ""),
                    EditorStyles.miniLabel);
            }
            if (_loadError != null)
            {
                EditorGUILayout.HelpBox(_loadError, MessageType.Warning);
                return;
            }

            EditorGUILayout.BeginHorizontal();
            // 左侧列表
            _scrollList = EditorGUILayout.BeginScrollView(_scrollList, GUILayout.Width(170), GUILayout.MaxWidth(170));
            for (int i = 0; i < _entries.Count; ++i)
            {
                var e = _entries[i];
                string label = e.error != null ? e.name + "  ⚠" : e.name;
                if (GUILayout.Toggle(!_tileMode && _selected == i, label, "Button") && (_tileMode || _selected != i))
                {
                    _selected = i;
                    _tileMode = false;
                    _mip = 0;
                }
            }
            // 瓦片金字塔（tiles/）：层级选择 + 网格地图点选 + albedo/normal 切换
            if (_tileLevels.Count > 0)
            {
                GUILayout.Space(6);
                GUILayout.Label("─ 瓦片金字塔 ─", EditorStyles.centeredGreyMiniLabel);
                _tileLevelIdx = Mathf.Clamp(_tileLevelIdx, 0, _tileLevels.Count - 1);
                var levelNames = new string[_tileLevels.Count];
                for (int i = 0; i < levelNames.Length; ++i) levelNames[i] = "L" + _tileLevels[i];
                _tileLevelIdx = EditorGUILayout.Popup(_tileLevelIdx, levelNames, GUILayout.Width(160));
                bool albedo = GUILayout.Toggle(_tileAlbedo, "albedo", "Button", GUILayout.Width(78));
                // 同一行放不下两个按钮，纵排即可
                bool normal = GUILayout.Toggle(!_tileAlbedo, "normal", "Button", GUILayout.Width(78));
                if (normal) albedo = false;
                if (albedo != _tileAlbedo)
                {
                    _tileAlbedo = albedo;
                    if (_curTile != null) SelectTile(_curTile); // 重载对应通道文件
                }
                DrawTileMap(_tileLevels[_tileLevelIdx]);
            }
            EditorGUILayout.EndScrollView();

            // 右侧预览
            EditorGUILayout.BeginVertical();
            DrawPreview();
            EditorGUILayout.EndVertical();
            EditorGUILayout.EndHorizontal();
        }

        void DrawPreview()
        {
            Entry e = _tileMode ? _tileEntry : (_selected >= 0 && _selected < _entries.Count ? _entries[_selected] : null);
            if (e == null)
            {
                EditorGUILayout.HelpBox(
                    _tileMode && _curTile != null
                        ? $"该瓦片缺少 {(_tileAlbedo ? "albedo" : "normal")} 文件"
                        : "左侧选择要查看的纹理或瓦片",
                    MessageType.Info);
                return;
            }
            if (e.error != null)
            {
                EditorGUILayout.HelpBox(e.error, MessageType.Error);
                return;
            }

            EditorGUILayout.BeginHorizontal();
            _channel = (ChannelMode)GUILayout.Toolbar((int)_channel, ChannelNames, GUILayout.Width(220));
            if (e.isNormal)
                _decodeNormal = EditorGUILayout.ToggleLeft("解码法线", _decodeNormal, GUILayout.Width(70));
            _pointFilter = EditorGUILayout.ToggleLeft("Point", _pointFilter, GUILayout.Width(55));
            if (e.isFullTerrain && _manifest != null)
                _showBlocks = EditorGUILayout.ToggleLeft("块区域", _showBlocks, GUILayout.Width(60));
            EditorGUILayout.EndHorizontal();
            if (e.mipCount > 1)
                _mip = EditorGUILayout.IntSlider($"Mip（共 {e.mipCount} 级）", _mip, 0, e.mipCount - 1);
            else
                _mip = 0;

            if (e.display == null || e.displayKey != DisplayKey())
                RebuildDisplay(e);
            if (e.display == null) return;

            // 预览区：等比适配剩余空间
            Rect r = GUILayoutUtility.GetRect(100, 100, GUILayout.ExpandWidth(true), GUILayout.ExpandHeight(true));
            float s = Mathf.Min(r.width / e.display.width, r.height / e.display.height);
            var pr = new Rect(
                r.x + (r.width - e.display.width * s) * 0.5f,
                r.y + (r.height - e.display.height * s) * 0.5f,
                e.display.width * s, e.display.height * s);
            GUI.DrawTexture(pr, e.display, ScaleMode.StretchToFill);
            _previewRect = pr;

            // 高精块区域叠加（仅全地形纹理；v2.5 起高精块已被瓦片金字塔取代，旧数据保留兼容）
            if (_showBlocks && e.isFullTerrain && _manifest != null && _manifest.blockRects != null
                && _manifest.blockRects.Length > 0 && Event.current.type == EventType.Repaint)
            {
                for (int i = 0; i < _manifest.blockRects.Length; ++i)
                {
                    var br = _manifest.blockRects[i];
                    float u0 = (br.x - _manifest.originX) / _manifest.terrainSize;
                    float v0 = (br.y - _manifest.originZ) / _manifest.terrainSize;
                    var brRect = UvToRect(pr, u0, v0,
                        (br.xMax - _manifest.originX) / _manifest.terrainSize,
                        (br.yMax - _manifest.originZ) / _manifest.terrainSize);
                    DrawRectOutline(brRect, Color.yellow);
                    var old = GUI.color;
                    GUI.color = Color.yellow;
                    GUI.Label(new Rect(brRect.x + 3, brRect.y + 2, 40, 16), i.ToString(), EditorStyles.miniBoldLabel);
                    GUI.color = old;
                }
            }

            // 鼠标读数：UV / texel / 世界坐标
            var mp = Event.current.mousePosition;
            string readout = " ";
            if (pr.Contains(mp))
            {
                float u = Mathf.InverseLerp(pr.x, pr.xMax, mp.x);
                float v = 1f - Mathf.InverseLerp(pr.y, pr.yMax, mp.y); // GUI y 向下 → 纹理 v 向上
                int tx = Mathf.Clamp((int)(u * e.display.width), 0, e.display.width - 1);
                int ty = Mathf.Clamp((int)(v * e.display.height), 0, e.display.height - 1);
                readout = $"uv=({u:F4}, {v:F4})  texel(mip{_mip})=({tx}, {ty})";
                if (e.dispPx != null && e.dispW > 0)
                {
                    var c = e.dispPx[ty * e.dispW + tx];
                    readout += $"  rgba=({c.r},{c.g},{c.b},{c.a})  #{c.r:X2}{c.g:X2}{c.b:X2}{c.a:X2}";
                }
                if (_manifest != null)
                {
                    float wx, wz;
                    if (e.isFullTerrain)
                    {
                        wx = _manifest.originX + u * _manifest.terrainSize;
                        wz = _manifest.originZ + v * _manifest.terrainSize;
                    }
                    else if (e.isTile && _manifest.tileWorld0 > 0)
                    {
                        float tw = _manifest.tileWorld0 * (1 << e.tileLevel);
                        wx = _manifest.originX + (e.tileX + u) * tw;
                        wz = _manifest.originZ + (e.tileZ + v) * tw;
                    }
                    else if (e.blockIndex >= 0 && _manifest.blockRects != null && e.blockIndex < _manifest.blockRects.Length)
                    {
                        var br = _manifest.blockRects[e.blockIndex];
                        wx = br.x + u * br.width;
                        wz = br.y + v * br.height;
                    }
                    else { wx = wz = float.NaN; }
                    if (!float.IsNaN(wx)) readout += $"  world=({wx:F2}, {wz:F2})";
                }
                if (Event.current.type == EventType.MouseMove) Repaint();
            }
            else if (Event.current.type == EventType.MouseMove) Repaint(); // 移出预览区及时清读数
            EditorGUILayout.LabelField(readout, EditorStyles.miniLabel);
            string tileInfo = "";
            if (e.isTile && _manifest != null && _manifest.tileWorld0 > 0)
            {
                float tw = _manifest.tileWorld0 * (1 << e.tileLevel);
                tileInfo = $"  世界范围=({_manifest.originX + e.tileX * tw:F0}, {_manifest.originZ + e.tileZ * tw:F0})~({_manifest.originX + (e.tileX + 1) * tw:F0}, {_manifest.originZ + (e.tileZ + 1) * tw:F0})m";
            }
            EditorGUILayout.LabelField(
                $"{e.name}: {e.width}×{e.height} {e.format}  mip={e.mipCount}  文件={e.fileSize / 1024}KB" +
                (e.sRGB ? "  sRGB" : "  linear") + tileInfo,
                EditorStyles.miniLabel);
        }

        static Rect UvToRect(Rect pr, float u0, float v0, float u1, float v1)
        {
            float x0 = pr.x + u0 * pr.width;
            float x1 = pr.x + u1 * pr.width;
            float y0 = pr.yMax - v1 * pr.height; // v 向上 → GUI y 向下
            float y1 = pr.yMax - v0 * pr.height;
            return Rect.MinMaxRect(x0, y0, x1, y1);
        }

        static void DrawRectOutline(Rect r, Color c)
        {
            var old = GUI.color;
            GUI.color = c;
            GUI.DrawTexture(new Rect(r.x, r.y, r.width, 1), EditorGUIUtility.whiteTexture);
            GUI.DrawTexture(new Rect(r.x, r.yMax - 1, r.width, 1), EditorGUIUtility.whiteTexture);
            GUI.DrawTexture(new Rect(r.x, r.y, 1, r.height), EditorGUIUtility.whiteTexture);
            GUI.DrawTexture(new Rect(r.xMax - 1, r.y, 1, r.height), EditorGUIUtility.whiteTexture);
            GUI.color = old;
        }
    }
}
