namespace MightyTerrainMesh.EditorTools
{
    using System.Collections.Generic;
    using UnityEditor;
    using UnityEngine;

    /// <summary>
    /// 一键烘焙窗口：单个 Terrain 一次完成
    ///   ① MT 几何数据（mesh pack / 四叉树 / 高度图 / 细节 / 材质 / MTData 头）
    ///   ② ClipmapVT 纹理源（L7/L8 全地形 + 区域高精块 + manifest）
    ///   ③ 场景布置（MTLoader + VTCreator + MTClipmapController 自动挂载连线）
    /// 输出目录遵循项目约定：
    ///   MTData  → {mtDataRoot}/{terrain.name}/（资产引用，位置任意）
    ///   mesh pack → Resources/MeshData/（运行时装载机硬编码 Resources.Load("MeshData/...")）
    ///   Clipmap → Resources/ClipmapBaked/（MTClipmapController.bakedResourcesRoot）
    /// 重复烘焙幂等：覆盖同名资产、MTData 保留 GUID、场景节点只更新字段。
    /// </summary>
    public class MTOneClickBakeWindow : EditorWindow
    {
        [MenuItem("MightyTerrainMesh/One-Click Baker (MT + Clipmap)")]
        static void Open() => GetWindow<MTOneClickBakeWindow>("MT 一键烘焙");

        // ---- 源 ----
        private Terrain _terrain;
        private Terrain _lastTerrain;

        // ---- MT 几何 ----
        private int _quadTreeDepth = 2;
        private MTLODSetting[] _lodSettings = new MTLODSetting[0];
        private int _lodCount = 1;
        private int _datapack = 64; // 与项目现有 mesh pack 一致（64 mesh/包）
        private bool _genUV2 = false;

        // ---- Clipmap 纹理 ----
        private bool _bakeClipmap = true;
        private int _l7Res = 512;
        private int _l8Res = 2040;
        private MTTerrainBakePipeline.ClipmapBakeFormat _bakeFormat = MTTerrainBakePipeline.ClipmapBakeFormat.ASTC;
        private bool _bakePyramid = true;
        private bool _pyramidFull = false; // D8：默认区域高精包（regions 内 L0 高精，粗级全量）
        private int _coarseFullFrom = 3;   // 粗级全量起始级（j≥此级全网格直接烘焙）
        private List<Rect> _regions = new List<Rect>();

        // ---- 输出目录 ----
        private string _mtDataRoot = "Assets/ArtRes/BakedData";
        private string _meshDataDir = "Assets/Resources/MeshData";
        private string _clipmapDir = "Assets/Resources/ClipmapBaked";

        // ---- 场景布置 ----
        private bool _placeInScene = true;
        private MTLODPolicy _lodPolicy;
        private Camera _camera;
        private bool _addClipmapController = true;
        private bool _disableTerrainRendering = true;

        private Vector2 _scroll;

        void OnEnable()
        {
            if (_lodPolicy == null)
                _lodPolicy = AssetDatabase.LoadAssetAtPath<MTLODPolicy>("Assets/MightyTerrainMesh/Res/LODPolicy.asset");
            if (_camera == null)
                _camera = Camera.main;
            if (_terrain == null && Selection.activeGameObject != null)
                _terrain = Selection.activeGameObject.GetComponent<Terrain>();
        }

        void OnGUI()
        {
            _scroll = EditorGUILayout.BeginScrollView(_scroll);

            _terrain = (Terrain)EditorGUILayout.ObjectField("源地形 (Terrain)", _terrain, typeof(Terrain), true);
            // 地形变更时自动按尺寸推荐 Clipmap 分辨率（§5.1/§7.1 密度契约）
            if (_terrain != _lastTerrain)
            {
                _lastTerrain = _terrain;
                if (_terrain != null)
                {
                    float ts = _terrain.terrainData.size.x;
                    MTTerrainBakePipeline.RecommendClipmapResolutions(ts, out _l7Res, out _l8Res);
                    // 区域模式默认预填地形中心 1/4 区域（D8 高精包；空区域 = 无 L0 高精）
                    if (_regions.Count == 0)
                    {
                        var p = _terrain.transform.position;
                        _regions.Add(new Rect(p.x + ts * 0.25f, p.z + ts * 0.25f, ts * 0.5f, ts * 0.5f));
                    }
                }
            }
            if (_terrain != null)
            {
                var td0 = _terrain.terrainData;
                EditorGUILayout.HelpBox(
                    $"尺寸 {td0.size.x:F0}×{td0.size.z:F0} m，位置 ({_terrain.transform.position.x:F0}, {_terrain.transform.position.z:F0})，" +
                    $"高度图 {td0.heightmapResolution}，Alphamap {td0.alphamapWidth}，层数 {td0.terrainLayers.Length}",
                    MessageType.None);
            }

            // ---------------- MT 几何 ----------------
            EditorGUILayout.Space();
            EditorGUILayout.LabelField("MT 几何数据", EditorStyles.boldLabel);
            int curSlice = 1 << _quadTreeDepth;
            int slice = EditorGUILayout.IntField("切片数 (NxN)", curSlice);
            if (slice != curSlice)
            {
                slice = Mathf.Max(1, Mathf.NextPowerOfTwo(slice));
                _quadTreeDepth = Mathf.FloorToInt(Mathf.Log(slice, 2));
            }
            if (_lodCount != _lodSettings.Length)
            {
                var old = _lodSettings;
                _lodSettings = new MTLODSetting[_lodCount];
                for (int i = 0; i < Mathf.Min(_lodCount, old.Length); ++i) _lodSettings[i] = old[i];
                for (int i = old.Length; i < _lodCount; ++i) { _lodSettings[i] = new MTLODSetting(); _lodSettings[i].Subdivision = 4; }
            }
            _lodCount = Mathf.Max(1, EditorGUILayout.IntField("LOD 层数", _lodSettings.Length));
            EditorGUI.indentLevel++;
            for (int i = 0; i < _lodSettings.Length; ++i) _lodSettings[i].OnGUIDraw(i);
            EditorGUI.indentLevel--;
            _datapack = Mathf.Max(1, EditorGUILayout.IntField("Mesh 打包 (个/文件)", _datapack));
            _genUV2 = EditorGUILayout.ToggleLeft("生成 UV2", _genUV2);

            // ---------------- Clipmap 纹理 ----------------
            EditorGUILayout.Space();
            _bakeClipmap = EditorGUILayout.ToggleLeft("烘焙 ClipmapVT 纹理源", _bakeClipmap, EditorStyles.boldLabel);
            if (_bakeClipmap)
            {
                EditorGUI.indentLevel++;
                _l7Res = EditorGUILayout.IntField("L7 源分辨率（Job 回退用）", _l7Res);
                _l8Res = EditorGUILayout.IntField("静态层 L_K 分辨率", _l8Res);
                if (_terrain != null)
                {
                    float ts = _terrain.terrainData.size.x;
                    int k = Mathf.Max(1, Mathf.RoundToInt(Mathf.Log(Mathf.Max(32f, ts), 2f)) - 5);
                    EditorGUILayout.LabelField(
                        $"§3.1.1：K={k}（环面 L0..L{k - 1} + 静态层 L{k}），静态层 texel {ts / _l8Res:F3} m" +
                        $"（契约 S/2040），L0 层 1.57cm（32m/2040）",
                        EditorStyles.miniLabel);
                    if (GUILayout.Button("重新按地形计算"))
                        MTTerrainBakePipeline.RecommendClipmapResolutions(ts, out _l7Res, out _l8Res);
                }
                _bakeFormat = (MTTerrainBakePipeline.ClipmapBakeFormat)EditorGUILayout.EnumPopup(
                    new GUIContent("压缩格式（§10）", "ASTC=移动端（编辑器 D3D11 无硬件 ASTC 会解码回退，2040² 下内存大）；BC7=DX/桌面（编辑器测试建议）；RGBA32=未压缩"),
                    _bakeFormat);
                _bakePyramid = EditorGUILayout.ToggleLeft(
                    "瓦片金字塔（§7.1 流式契约：L0 = 32m@2040²，运行时异步流式 + 直拷）", _bakePyramid);
                if (_bakePyramid && _terrain != null)
                {
                    EditorGUI.indentLevel++;
                    _pyramidFull = EditorGUILayout.ToggleLeft(
                        new GUIContent("全区域覆盖（警告：2040² 片 ≈4.5MB(ASTC)，1024m 全量 ≈6GB；D8 建议关闭走区域包）", ""),
                        _pyramidFull);
                    if (!_pyramidFull)
                        _coarseFullFrom = EditorGUILayout.IntSlider(
                            new GUIContent("粗级全量起始级", "j≥此级的粗瓦片全网格直接烘焙（保证顶层环面/静态层可供流式直拷）"),
                            _coarseFullFrom, 1, 6);
                    float regionArea = 0f;
                    foreach (var r in _regions) regionArea += r.width * r.height;
                    MTTerrainBakePipeline.EstimatePyramid(_terrain.terrainData.size.x, 32f,
                        _pyramidFull, regionArea, _coarseFullFrom, out int plv, out long pt);
                    float mbPerTile = _bakeFormat == MTTerrainBakePipeline.ClipmapBakeFormat.ASTC ? 4.5f
                        : _bakeFormat == MTTerrainBakePipeline.ClipmapBakeFormat.BC7 ? 8.3f : 33.3f;
                    EditorGUILayout.LabelField(
                        $"金字塔：{plv} 级，约 {pt} 片 × {mbPerTile:F1}MB ≈ {pt * mbPerTile / 1024f:F2}GB（{_bakeFormat}）",
                        EditorStyles.miniLabel);
                    EditorGUI.indentLevel--;
                }
                EditorGUILayout.LabelField(_pyramidFull ? "高精区域（全区域模式下忽略）" : "高精区域（世界坐标 Rect，米；区域包仅烘这些区域）");
                EditorGUI.indentLevel++;
                for (int i = 0; i < _regions.Count; ++i)
                {
                    EditorGUILayout.BeginHorizontal();
                    _regions[i] = EditorGUILayout.RectField(_regions[i]);
                    if (GUILayout.Button("×", GUILayout.Width(22))) { _regions.RemoveAt(i); --i; }
                    EditorGUILayout.EndHorizontal();
                }
                if (GUILayout.Button("添加区域")) _regions.Add(new Rect(0, 0, 128, 128));
                EditorGUI.indentLevel--;
                EditorGUI.indentLevel--;
            }

            // ---------------- 输出目录 ----------------
            EditorGUILayout.Space();
            EditorGUILayout.LabelField("输出目录", EditorStyles.boldLabel);
            _mtDataRoot = EditorGUILayout.TextField("MTData 根目录", _mtDataRoot);
            _meshDataDir = EditorGUILayout.TextField("Mesh pack 目录", _meshDataDir);
            if (_bakeClipmap)
            {
                EditorGUILayout.BeginHorizontal();
                _clipmapDir = EditorGUILayout.TextField("Clipmap 目录", _clipmapDir);
                if (GUILayout.Button("查看产物", GUILayout.Width(60)))
                    MTClipmapBakedViewer.OpenWith(_clipmapDir);
                EditorGUILayout.EndHorizontal();
            }
            if (_terrain != null)
                EditorGUILayout.HelpBox(
                    $"MTData  →  {_mtDataRoot}/{_terrain.name}/\n" +
                    $"Mesh    →  {_meshDataDir}/{_terrain.name}_*.bytes" +
                    (_bakeClipmap ? $"\nClipmap →  {_clipmapDir}/（manifest + l7/l8/block）" : ""),
                    MessageType.None);

            // ---------------- 场景布置 ----------------
            EditorGUILayout.Space();
            _placeInScene = EditorGUILayout.ToggleLeft("布置到当前场景", _placeInScene, EditorStyles.boldLabel);
            if (_placeInScene)
            {
                EditorGUI.indentLevel++;
                _lodPolicy = (MTLODPolicy)EditorGUILayout.ObjectField("LOD Policy", _lodPolicy, typeof(MTLODPolicy), false);
                _camera = (Camera)EditorGUILayout.ObjectField("相机", _camera, typeof(Camera), true);
                if (_bakeClipmap)
                    _addClipmapController = EditorGUILayout.Toggle("挂 MTClipmapController", _addClipmapController);
                _disableTerrainRendering = EditorGUILayout.Toggle("关闭源地形渲染", _disableTerrainRendering);
                EditorGUI.indentLevel--;
            }

            // ---------------- 执行 ----------------
            EditorGUILayout.Space();
            GUI.enabled = _terrain != null;
            if (GUILayout.Button("一键烘焙并布置", GUILayout.Height(32)))
                Run();
            GUI.enabled = true;

            EditorGUILayout.EndScrollView();
        }

        void Run()
        {
            if (_lodSettings == null || _lodSettings.Length == 0)
            {
                MTLog.LogError("no lod setting");
                return;
            }
            if (!ValidateDirs()) return;

            string dataFolder = $"{_mtDataRoot}/{_terrain.name}";

            // ① MT 几何
            var header = MTTerrainBakePipeline.BakeMT(_terrain, _quadTreeDepth, _lodSettings, _datapack, _genUV2,
                dataFolder, _meshDataDir);
            if (header == null) return;

            // ② Clipmap 纹理
            if (_bakeClipmap)
                MTTerrainBakePipeline.BakeClipmap(_terrain, _l7Res, _l8Res, _bakeFormat, _clipmapDir,
                    _regions, _bakePyramid, _pyramidFull, _coarseFullFrom);

            // ③ 场景布置
            if (_placeInScene)
            {
                if (_lodPolicy == null)
                    Debug.LogWarning("[MTBake] 未指定 LODPolicy，MTLoader.lodPolicy 为空");
                MTTerrainBakePipeline.PlaceInScene(_terrain, header, _lodPolicy, _camera,
                    _bakeClipmap && _addClipmapController, ToResourcesRelative(_clipmapDir), _disableTerrainRendering);
            }

            Debug.Log($"[MTBake] 一键烘焙完成：{_terrain.name}\n" +
                      $"  MTData  → {dataFolder}/{_terrain.name}.asset\n" +
                      $"  Mesh    → {_meshDataDir}\n" +
                      (_bakeClipmap ? $"  Clipmap → {_clipmapDir}\n" : "") +
                      (_placeInScene ? $"  场景节点 → {_terrain.name}_MT" : ""));
        }

        bool ValidateDirs()
        {
            if (!_mtDataRoot.StartsWith("Assets"))
            {
                Debug.LogError("[MTBake] MTData 根目录必须在 Assets 下: " + _mtDataRoot);
                return false;
            }
            if (!_meshDataDir.Replace("\\", "/").Contains("/Resources/") || !_meshDataDir.EndsWith("/MeshData"))
            {
                Debug.LogError("[MTBake] Mesh pack 目录必须是某 Resources 下的 MeshData（运行时装载机硬编码 Resources.Load(\"MeshData/...\")）: " + _meshDataDir);
                return false;
            }
            if (_bakeClipmap && ToResourcesRelative(_clipmapDir) == null)
            {
                Debug.LogError("[MTBake] Clipmap 目录必须位于某 Resources 文件夹下: " + _clipmapDir);
                return false;
            }
            return true;
        }

        /// <summary>"Assets/.../Resources/ClipmapBaked" → "ClipmapBaked"（Resources.Load 相对路径）。</summary>
        static string ToResourcesRelative(string assetPath)
        {
            var p = assetPath.Replace("\\", "/");
            int i = p.LastIndexOf("/Resources/");
            if (i < 0) return null;
            return p.Substring(i + "/Resources/".Length);
        }
    }
}
