namespace MightyTerrainMesh.EditorTools
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using UnityEditor;
    using UnityEngine;

    /// <summary>
    /// Clipmap 烘焙产物可视化查看器（v3：PNG 源图）：
    /// - 直接解码 PNG 文件（Texture2D.LoadImage，不经导入器，烘焙落盘即可查看，无平台压缩干扰）；
    /// - 支持 l7/l8 的 albedo 与 normal，以及 tiles/ 瓦片金字塔（网格地图点选、纹理懒加载）；
    /// - RGB/R/G/B/A 通道隔离；albedo.A=Metallic；normal 可解码为法线可视化
    ///   （RG=法线XZ → 重建 Up 分量；B=粗糙度、A=AO 走通道隔离查看）；
    /// - 平台压缩后的实际效果（BC7/ASTC 质量）请在导入产物上用 Inspector 查看，此处恒为无损源图；
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
            public string path;             // PNG 文件路径
            public bool sRGB;
            public bool isNormal;
            public bool isFullTerrain;          // l7/l8：覆盖全地形（用于块区域叠加与世界坐标读数）
            public int blockIndex = -1;         // block_{i}_* → i（旧数据，v3 不再产出）
            public bool isTile;                 // tiles/l{lvl}_{x}_{z}_*：金字塔瓦片（纹理懒加载）
            public int tileLevel, tileX, tileZ;
            public Texture2D tex;               // PNG 解码纹理（RGBA32）
            public Texture2D display;           // 按当前选项生成的显示纹理
            public Color32[] dispPx;            // 原始像素（供鼠标读数 + 通道切换缓存）
            public int dispW, dispH;
            public string displayKey = "";
            public int width, height;
            public long fileSize;
            public string error;
        }

        // 金字塔瓦片引用：仅登记 PNG 路径，点选时才解码（全量模式 L0 可达数万片）。
        private class TileRef
        {
            public int level, tx, tz;
            public string pathA, pathN; // albedo / normal PNG 路径
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
                _loadError = "纹理数据版本不匹配，请重新烘焙（v3 = PNG 源图）";
                return;
            }
            foreach (var f in Directory.GetFiles(full, "*.png"))
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
                _loadError = "目录中没有 .png 烘焙产物: " + full;
            else if (_entries.Count > 0)
                _selected = 0;
        }

        // 扫描金字塔瓦片：tiles/ 子目录的 PNG（只解析文件名，点选时才解码）
        void ScanTiles(string full)
        {
            var byLevel = new Dictionary<int, Dictionary<long, TileRef>>();
            string tileDir = Path.Combine(full, "tiles");
            if (Directory.Exists(tileDir))
            foreach (var f in Directory.GetFiles(tileDir, "*.png"))
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
            foreach (var kv in byLevel)
            {
                var list = new List<TileRef>(kv.Value.Values);
                list.Sort((a, b) => a.tz != b.tz ? a.tz - b.tz : a.tx - b.tx);
                _tiles.Add(kv.Key, list);
                _tileLevels.Add(kv.Key);
            }
            _tileLevels.Sort();
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
                // PNG 直解码（Lossless 源图）：不经导入器，烘焙落盘即可查看；尺寸/内容自描述
                var tex = new Texture2D(2, 2, TextureFormat.RGBA32, false, true);
                if (!tex.LoadImage(File.ReadAllBytes(e.path)))
                {
                    DestroyImmediate(tex);
                    e.error = "PNG 解码失败: " + e.path;
                    return;
                }
                tex.wrapMode = TextureWrapMode.Clamp;
                e.tex = tex;
                e.width = tex.width;
                e.height = tex.height;
            }
            catch (Exception ex) { e.error = ex.Message; }
        }

        // ---------------- 显示纹理生成 ----------------

        string DisplayKey() => $"{_channel}|{_decodeNormal}|{_pointFilter}";

        void RebuildDisplay(Entry e)
        {
            if (e.display != null) { DestroyImmediate(e.display); e.display = null; }
            if (e.tex == null) return;
            // PNG 源图解码结果按条目缓存：切通道/滤波/解码开关时不再重复解码
            Color32[] px = e.dispPx ?? (e.dispPx = e.tex.GetPixels32(0));
            int w = e.width;
            int h = e.height;
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
            e.dispPx = px;   // 保留原始像素供读数
            e.dispW = w;
            e.dispH = h;
            e.displayKey = DisplayKey();
        }

        // ---------------- 瓦片金字塔浏览 ----------------

        // 点选瓦片：销毁上一片纹理（2040² RGBA32 达 33MB，不缓存多片），按当前 albedo/normal 选择解码对应 PNG
        void SelectTile(TileRef t)
        {
            _tileMode = true;
            _selected = -1;
            _curTile = t;
            DestroyEntry(_tileEntry);
            _tileEntry = null;
            string path = _tileAlbedo ? t.pathA : t.pathN;
            if (path == null) return;
            _tileEntry = new Entry
            {
                name = $"l{t.level}_{t.tx}_{t.tz}_{(_tileAlbedo ? "albedo" : "normal")}",
                path = path,
                sRGB = _tileAlbedo,
                isNormal = !_tileAlbedo,
                isTile = true,
                tileLevel = t.level,
                tileX = t.tx,
                tileZ = t.tz,
                fileSize = new FileInfo(path).Length,
            };
            LoadEntry(_tileEntry);
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
                readout = $"uv=({u:F4}, {v:F4})  texel=({tx}, {ty})";
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
                $"{e.name}: {e.width}×{e.height} PNG 源图（平台压缩见导入器）  文件={e.fileSize / 1024}KB" +
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
