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
        private MTTerrainBakePipeline.ClipmapBakeFormat _bakeFormat = MTTerrainBakePipeline.ClipmapBakeFormat.Auto;
        private TextureCompressionQuality _fineQuality = TextureCompressionQuality.Fast;
        private bool _bakePyramid = true;
        private bool _pyramidFull = false; // D8：默认区域高精包（regions 内 L0 高精，粗级全量）
        private int _coarseFullFrom = 3;   // 粗级全量起始级（j≥此级全网格直接烘焙）
        private bool _customL0Density = false;      // 自定义 L0 纹素密度（footprint 保持 32m 契约，反推分辨率）
        private float _l0TexelCm = 1.57f;           // L0 纹素密度（厘米/像素），默认 1.57 ≡ 2040²
        private bool _chMetallic = true;            // 材质通道：金属度（Albedo.A）
        private bool _chAO = true;                  // 材质通道：AO（NormalAttr.A）
        private bool _chSmoothness = true;          // 材质通道：光滑度（NormalAttr.B）
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
                    int k = Mathf.Max(1, Mathf.CeilToInt(Mathf.Log(ts / 32f, 2f) - 1e-3f));
                    int l0Res = EffectiveTileRes();
                    EditorGUILayout.LabelField(
                        $"§3.1.1：K={k}（环面 L0..L{k - 1} + 静态层 L{k}），静态层 texel {ts / _l8Res:F3} m" +
                        $"（契约 S/2040），L0 层 {3200f / l0Res:F2}cm（32m/{l0Res}）",
                        EditorStyles.miniLabel);
                    if (GUILayout.Button("重新按地形计算"))
                        MTTerrainBakePipeline.RecommendClipmapResolutions(ts, out _l7Res, out _l8Res);
                }
                _bakeFormat = (MTTerrainBakePipeline.ClipmapBakeFormat)EditorGUILayout.EnumPopup(
                    new GUIContent("压缩模式（§10）", "产物为 PNG 源图，实际压缩由导入管线按平台完成：\n" +
                        "Auto = Standalone BC7 / 移动端 ASTC 6x6/5x5（推荐）；\n" +
                        "ASTC/BC7 = 全平台强制（目标平台须支持）；RGBA32 = 不压缩（联调）"),
                    _bakeFormat);
                _fineQuality = (TextureCompressionQuality)EditorGUILayout.EnumPopup(
                    new GUIContent("导入压缩质量", "写入 manifest，导入器平台覆盖的 compressionQuality。BC7 Best 极慢，建议 Fast/Normal"),
                    _fineQuality);
                EditorGUILayout.LabelField("材质通道（关闭的通道不烘焙，shader 用常量替代）", EditorStyles.miniBoldLabel);
                EditorGUI.indentLevel++;
                _chMetallic = EditorGUILayout.ToggleLeft("金属度（Albedo.A，关闭恒 0）", _chMetallic);
                _chAO = EditorGUILayout.ToggleLeft("环境光遮蔽 AO（NormalAttr.A，关闭恒 1）", _chAO);
                _chSmoothness = EditorGUILayout.ToggleLeft("光滑度（NormalAttr.B，关闭恒 0.5）", _chSmoothness);
                EditorGUI.indentLevel--;
                _bakePyramid = EditorGUILayout.ToggleLeft(
                    "瓦片金字塔（§7.1 流式契约：默认 L0 = 32m@2040²，可对齐 splat 密度；运行时异步流式 + 直拷）", _bakePyramid);
                if (_bakePyramid && _terrain != null)
                {
                    EditorGUI.indentLevel++;
                    _customL0Density = EditorGUILayout.ToggleLeft(
                        new GUIContent("自定义 L0 纹素密度", "L0 footprint 保持 32m 契约，分辨率 = 3200 / 密度（cm/px），" +
                            "按 60 对齐（ASTC/BC7 块对齐）并钳制到 [60, 4096]；最细可达 ≈0.78cm（4096²）"),
                        _customL0Density);
                    if (_customL0Density)
                    {
                        EditorGUI.indentLevel++;
                        _l0TexelCm = EditorGUILayout.Slider(
                            new GUIContent("L0 纹素密度 (cm/像素)", "每个 L0 texel 覆盖的地面尺寸（厘米）。参考最细 splat 层密度，低于它无意义"),
                            _l0TexelCm, 0.5f, 8f);
                        float st = MTTerrainBakePipeline.FinestSplatTexelSize(_terrain);
                        int r0 = EffectiveTileRes();
                        float achieved = 3200f / r0;
                        EditorGUILayout.LabelField(
                            $"→ L0 32m@{r0}²（实际 texel {achieved:F2}cm）" +
                            (Mathf.Abs(achieved - _l0TexelCm) > 0.01f ? "（块对齐/4096² 上限截断）" : "") +
                            (st > 0f ? $"；最细 splat texel {st * 100f:F2}cm" : ""),
                            EditorStyles.miniLabel);
                        EditorGUI.indentLevel--;
                    }
                    _pyramidFull = EditorGUILayout.ToggleLeft(
                        new GUIContent("全区域覆盖（警告：2040² 全量片数多、PNG 源体积大；D8 建议关闭走区域包）", ""),
                        _pyramidFull);
                    if (!_pyramidFull)
                        _coarseFullFrom = EditorGUILayout.IntSlider(
                            new GUIContent("粗级全量起始级", "j≥此级的粗瓦片全网格直接烘焙（保证顶层环面/静态层可供流式直拷）"),
                            _coarseFullFrom, 1, 6);
                    float regionArea = 0f;
                    foreach (var r in _regions) regionArea += r.width * r.height;
                    MTTerrainBakePipeline.EstimatePyramid(_terrain.terrainData.size.x, 32f,
                        _pyramidFull, regionArea, _coarseFullFrom, out int plv, out long pt);
                    int effRes = EffectiveTileRes();
                    double texels = effRes * (double)effRes; // 每片一对 albedo+normal 的 texel 数基数
                    double bc7Mb = texels * 2.0 / (1024 * 1024);
                    double astcMb = texels * (16.0 / 36.0 + 16.0 / 25.0) / (1024 * 1024);
                    double rgbaMb = texels * 8.0 / (1024 * 1024);
                    string sizeHint = _bakeFormat == MTTerrainBakePipeline.ClipmapBakeFormat.RGBA32 ? $"{rgbaMb:F1}MB/片"
                        : _bakeFormat == MTTerrainBakePipeline.ClipmapBakeFormat.BC7 ? $"{bc7Mb:F1}MB/片"
                        : _bakeFormat == MTTerrainBakePipeline.ClipmapBakeFormat.ASTC ? $"{astcMb:F1}MB/片"
                        : $"{bc7Mb:F1}MB(PC)/{astcMb:F1}MB(移动) 每片";
                    EditorGUILayout.LabelField(
                        $"金字塔：{plv} 级，约 {pt} 片；PNG 源 ≈ {pt * 10f * texels / (2040.0 * 2040.0) / 1024f:F1}GB（不入库），包体压缩后 {sizeHint}",
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
                    (_bakeClipmap ? $"\nClipmap →  {_clipmapDir}/（manifest + l7/l8 + tiles/*.png，导入时按平台压缩）" : ""),
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
                    _regions, _bakePyramid, _pyramidFull, _coarseFullFrom, _fineQuality,
                    _customL0Density ? _l0TexelCm : 0f, ChannelMask());

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

        /// <summary>材质通道位掩码：bit0 金属度 / bit1 AO / bit2 光滑度。</summary>
        int ChannelMask() => (_chMetallic ? 1 : 0) | (_chAO ? 2 : 0) | (_chSmoothness ? 4 : 0);

        /// <summary>当前生效的金字塔 L0 分辨率：自定义密度开启时由 3200/texelCm 反推（footprint 恒为 32m），否则契约默认 2040。</summary>
        int EffectiveTileRes()
        {
            if (!_customL0Density || _terrain == null)
                return 2040;
            return MTTerrainBakePipeline.L0TileResForTexelCm(_l0TexelCm);
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
