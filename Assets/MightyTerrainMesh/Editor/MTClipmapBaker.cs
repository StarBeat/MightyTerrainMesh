namespace MightyTerrainMesh.EditorTools
{
    using System.Collections.Generic;
    using UnityEditor;
    using UnityEngine;

    /// <summary>
    /// Clipmap 离线烘焙器（对应 ClipmapVT.md §7.1）：
    /// - 全地形粗烘焙：L7 源（默认 2048²）与 L8 源（默认 2048²，mip 链由导入器生成）；
    /// - 瓦片金字塔：L0 footprint 固定 32m，分辨率默认 2040²（可指定 L0 纹素密度 cm/px 反推），区域高精包（D8）；
    /// - 产物为无损 PNG 源图 + manifest.json 到 Resources；平台压缩（BC7/ASTC）由
    ///   MTClipmapBakedImporter 在导入时按 manifest 配置，运行时 Resources.LoadAsync 直拷进环面。
    /// 烘焙逻辑已下沉到 <see cref="MTTerrainBakePipeline.BakeClipmap"/>，本类仅为 UI。
    /// </summary>
    public class MTClipmapBaker : EditorWindow
    {
        private Terrain _terrain;
        private Terrain _lastTerrain;
        private int _l7Res = 512;
        private int _l8Res = 2040;
        private MTTerrainBakePipeline.ClipmapBakeFormat _bakeFormat = MTTerrainBakePipeline.ClipmapBakeFormat.Auto;
        private TextureCompressionQuality _fineQuality = TextureCompressionQuality.Fast;
        private bool _bakePyramid = true;
        private bool _pyramidFull = false; // D8：默认区域高精包
        private int _coarseFullFrom = 3;
        private bool _customL0Density = false;      // 自定义 L0 纹素密度（footprint 保持 32m 契约，反推分辨率）
        private float _l0TexelCm = 1.57f;           // L0 纹素密度（厘米/像素），默认 1.57 ≡ 2040²
        private bool _chMetallic = true;            // 材质通道：金属度（Albedo.A）
        private bool _chAO = true;                  // 材质通道：AO（NormalAttr.A）
        private bool _chSmoothness = true;          // 材质通道：光滑度（NormalAttr.B）
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
                _pyramidFull = EditorGUILayout.ToggleLeft("全区域覆盖（警告：2040² 全量体积巨大，D8 建议关闭）", _pyramidFull);
                if (!_pyramidFull)
                    _coarseFullFrom = EditorGUILayout.IntSlider("粗级全量起始级", _coarseFullFrom, 1, 6);
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
                    _regions, _bakePyramid, _pyramidFull, _coarseFullFrom, _fineQuality,
                    _customL0Density ? _l0TexelCm : 0f, ChannelMask());
            GUI.enabled = true;
            EditorGUILayout.EndScrollView();
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
    }
}
