namespace MightyTerrainMesh.EditorTools
{
    using System.Collections.Generic;
    using UnityEditor;
    using UnityEngine;

    /// <summary>
    /// Clipmap 离线烘焙器（对应 ClipmapVT.md §7.1）：
    /// - 全地形粗烘焙：L7 源（默认 2048²）与 L8 源（默认 2048²，含离线 Mip 链）；
    /// - 区域高精包：指定世界矩形 → 32m×32m 块，每块 blockRes²；
    /// - 输出 ASTC 压缩字节流（albedo 6×6 / normal 5×5）+ manifest.json 到 Resources；
    /// - 运行时由 MTClipmapController 的 Baked 模式装配条带，不做运行时 ASTC 编码。
    /// 注意：原型分辨率低于文档 1.56cm 契约，由 blockRes/l7Res 控制，可逐步逼近。
    /// 烘焙逻辑已下沉到 <see cref="MTTerrainBakePipeline.BakeClipmap"/>，本类仅为 UI。
    /// </summary>
    public class MTClipmapBaker : EditorWindow
    {
        private Terrain _terrain;
        private Terrain _lastTerrain;
        private int _l7Res = 512;
        private int _l8Res = 2040;
        private MTTerrainBakePipeline.ClipmapBakeFormat _bakeFormat = MTTerrainBakePipeline.ClipmapBakeFormat.ASTC;
        private TextureCompressionQuality _fineQuality = TextureCompressionQuality.Fast;
        private bool _bakePyramid = true;
        private bool _pyramidFull = false; // D8：默认区域高精包
        private int _coarseFullFrom = 3;
        private string _outputRoot = "Assets/Resources/ClipmapBaked";
        private List<Rect> _regions = new List<Rect>();
        private Vector2 _scroll;

        [MenuItem("MightyTerrainMesh/Clipmap Baker")]
        static void Open() => GetWindow<MTClipmapBaker>("Clipmap Baker");

        void OnGUI()
        {
            _scroll = EditorGUILayout.BeginScrollView(_scroll);
            _terrain = (Terrain)EditorGUILayout.ObjectField("Terrain", _terrain, typeof(Terrain), true);
            // 地形变更时自动按尺寸推荐分辨率（§5.1/§7.1 密度契约）
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
                var td = _terrain.terrainData;
                EditorGUILayout.HelpBox(
                    $"尺寸 {td.size.x:F0}×{td.size.z:F0} m，位置 ({_terrain.transform.position.x:F0}, {_terrain.transform.position.z:F0})\n" +
                    $"高度图 {td.heightmapResolution}，Alphamap {td.alphamapWidth}，层数 {td.terrainLayers.Length}",
                    MessageType.None);
            }
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
                new GUIContent("压缩格式（§10）", "ASTC=移动端；BC7=DX/桌面（编辑器测试建议）；RGBA32=未压缩。" +
                    "ASTC/BC7 分别可由 astcenc/Compressonator 外部编码器并行加速（MightyTerrainMesh 菜单下载）"),
                _bakeFormat);
            _fineQuality = (TextureCompressionQuality)EditorGUILayout.EnumPopup(
                new GUIContent("L0 压缩质量", "ASTC 压缩耗时大头：Fast ≈ Normal 的 1/4~1/6；最终资产可换 Normal 重烘（≥L1 恒为 Fast）"),
                _fineQuality);
            _bakePyramid = EditorGUILayout.ToggleLeft(
                "瓦片金字塔（§7.1 流式契约：L0 = 32m@2040²，运行时异步流式 + 直拷）", _bakePyramid);
            if (_bakePyramid && _terrain != null)
            {
                EditorGUI.indentLevel++;
                _pyramidFull = EditorGUILayout.ToggleLeft("全区域覆盖（警告：2040² 全量体积巨大，D8 建议关闭）", _pyramidFull);
                if (!_pyramidFull)
                    _coarseFullFrom = EditorGUILayout.IntSlider("粗级全量起始级", _coarseFullFrom, 1, 6);
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
            _outputRoot = EditorGUILayout.TextField("输出目录 (Resources 下)", _outputRoot);

            EditorGUILayout.LabelField(_pyramidFull ? "高精区域（全区域模式下忽略）" : "高精区域（世界坐标 Rect，米；区域包仅烘这些区域）", EditorStyles.boldLabel);
            for (int i = 0; i < _regions.Count; ++i)
            {
                EditorGUILayout.BeginHorizontal();
                _regions[i] = EditorGUILayout.RectField(_regions[i]);
                if (GUILayout.Button("×", GUILayout.Width(22))) { _regions.RemoveAt(i); --i; }
                EditorGUILayout.EndHorizontal();
            }
            if (GUILayout.Button("添加区域")) _regions.Add(new Rect(0, 0, 128, 128));

            EditorGUILayout.Space();
            GUI.enabled = _terrain != null;
            if (GUILayout.Button("烘焙"))
                MTTerrainBakePipeline.BakeClipmap(_terrain, _l7Res, _l8Res, _bakeFormat, _outputRoot,
                    _regions, _bakePyramid, _pyramidFull, _coarseFullFrom, _fineQuality);
            GUI.enabled = true;
            EditorGUILayout.EndScrollView();
        }
    }
}
