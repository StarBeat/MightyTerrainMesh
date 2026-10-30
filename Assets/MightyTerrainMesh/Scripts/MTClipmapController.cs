namespace MightyTerrainMesh
{
    using System.Collections.Generic;
    using Unity.Collections;
    using Unity.Jobs;
    using Unity.Mathematics;
    using UnityEngine;
    using UnityEngine.Rendering;

    /// <summary>
    /// ClipmapVT 控制器 v6（对齐 ClipmapVT.md v2.5 §3.1/§5/§6.5/§7.1/§8.1/§10）：
    /// - 层分辨率 2040 = 68×30 = 340×6 = 408×5（ASTC 6x6/5x5 与 BC7 4×4(Q=60) 均块对齐，§3.1）；
    /// - 动态层数 K = ceil(log2(S/baseFootprint))（§3.1.1 推广：默认 base=32m 即 K = log2(S)−5，
    ///   1024m→K=5 环面 L0..L4 + 静态层 L5 覆盖全地形 @2040²，
    ///   texel = S/2040，离线完整 Mip 链）；shader 端 _ClipLevelCount 全局同步；
    /// - 固定图集寻址：采样 uv = frac(localXZ / footprint_k)，origin 仅作更新簿记；
    /// - 更新量子 Q：ASTC=30 / BC7=60（§5.1）；条带矩形由 Q 对齐性自动块对齐；
    /// - 直拷模式（Baked + 金字塔）：烘焙 texel ≡ 环面 texel（1:1），更新 = 纯矩形簿记 +
    ///   Graphics.CopyTexture（ASTC/BC7 压缩直拷），CPU 零逐像素工作；
    /// - 环面格式（§10/D12）：= 烘焙瓦片资产的平台压缩格式（导入器按 manifest.formatMode 配置：
    ///   Auto → Standalone BC7 / 移动端 ASTC 6x6/5x5；另有全平台强制与 RGBA32 联调档），
    ///   运行时探针顶级瓦片（L8 兜底）读取实际格式，CopyTexture 要求环面与瓦片同格式自动满足；
    /// - 瓦片未到位 → 不写入兜底内容，shader 按驻留掩码 _ClipValidMask 逐层向上扫描（§6.5，
    ///   支持非连续层失效；粗级全量烘焙保证收敛，静态层恒有效）；
    /// - 带宽门闸（§8.1）：帧时 EMA 持续超预算 → ① 混合带 25%→20%（_ClipBlendStart 0.75→0.80）
    ///   → ② 静态层 Aniso 2→1；
    /// - 通道契约（§4.1/D14）：splat 完全烘焙、运行时无 splat/detail 采样；
    ///   Albedo.A = Metallic，NormalAttr B/A = 粗糙度/AO（旧 manifest 保持非金属）；
    /// - 上传字节预算（§8.2）：每帧 uploadBudgetBytes（默认 1MB）；整层重填按预算分横带
    ///   多帧完成（RGBA32 联调模式下单层 ≈33MB 不再单帧突发），期间采样端按掩码回退粗层；
    /// - Job 重采样路径保留：Terrain（实时 splat）/ Test 模式，及无金字塔的旧烘焙数据；
    /// - 传送：阈值 footprint(L0) = 32m，全层失效后恢复循环按 粗→细 分带重填（§5.4）。
    /// </summary>
    [DefaultExecutionOrder(-200)]
    public class MTClipmapController : MonoBehaviour
    {
        public static bool Active { get; private set; }
        public static Material ClipmapMaterial { get; private set; }

        [Header("引用")]
        public MTLoader loader;
        public Camera viewCamera;
        public Terrain sourceTerrain;

        [Header("世界契约（与 CDLOD.md §1.3 一致）")]
        public Vector2 terrainOriginXZ = Vector2.zero;
        public float terrainSize = 8192f;

        [Header("Clipmap 参数")]
        [Tooltip("层分辨率（§3.1：2040 = 68×30，ASTC/BC7 均块对齐）。烘焙金字塔存在时自动以 manifest 为准")]
        public int layerResolution = 2040;
        [Tooltip("L0 层 footprint（米，§3.1：默认 32m → L0 texel 1.57cm；splat 密度对齐烘焙时以 manifest 为准）。烘焙金字塔存在时自动以 manifest 为准")]
        public float baseFootprint = 32f;
        /// <summary>层数上限（8192m 地形）；运行时层数 K = log2(S)−5，见 _levelCount。</summary>
        public const int MaxLevelCount = 8;
        [Tooltip("更新量子 Q（texel 数，§5.1：ASTC 30 / BC7 自动切 60）；Q_world_k = Q·texel_k，逐层翻倍")]
        public int updateQuantumTexels = 30;

        [Header("烘焙数据（§7.1）")]
        [Tooltip("Resources 下的烘焙数据目录；存在 manifest 时优先使用")]
        public string bakedResourcesRoot = "ClipmapBaked";
        [Tooltip("瓦片金字塔驻留预算（对，albedo+normal 各 2040²）；窗口上限 K×4，超出后 LRU 淘汰")]
        public int tileBudget = 24;

        [Header("更新预算")]
        [Tooltip("每帧上传字节预算（§8.2：1MB）。整层重填按预算分横带多帧完成；条带按层原子提交；本帧尚无上传时允许单个最小带/单层条带超限（防死锁）")]
        public long uploadBudgetBytes = 1 << 20;
        [Tooltip("条带分块上限（texel）：staging 尺寸 = chunk×res / res×chunk，限制单次 Apply 上传量")]
        public int stripChunkTexels = 120;

        [Header("带宽门闸（§8.1）")]
        [Tooltip("帧时 EMA 预算(ms)；连续超限触发降级：① 混合带 25%→20% ② 静态层 Aniso 2→1；持续低于 80% 预算则逐级恢复")]
        public float frameBudgetMs = 33f;
        [Tooltip("预测性预取前瞻（秒，0=关闭）：按相机速度提前请求前方窗口瓦片，减少未驻留导致的粗层回退")]
        public float prefetchAheadSeconds = 0.5f;

        public enum DebugLodMode { Off, Requested, Actual }

        [Header("调试")]
        [Tooltip("按 viewCamera 的选层结果显示（像素足迹与距离取 max），不受 Scene 相机影响；Requested：请求 LOD；Actual：包含瓦片未驻留时的粗层回退")]
        public DebugLodMode debugLodMode = DebugLodMode.Requested;
        [Tooltip("Scene 视图绘制 clipmap 区块：各层环面窗口（色相按层级，半透明 = 层失效重填中）、混合带内框、静态层全地形外框")]
        public bool showGizmos = true;

        private enum StripKind { Col, Row, Full }

        private struct StripOp
        {
            public int layer;
            public StripKind kind;
            public RectInt dst;      // 目标纹理矩形（已 wrap 拆分，保证界内）
            public float2 worldMin;  // 内容世界坐标最小角（含 TerrainOrigin）
            public int w, h;
        }

        private Texture2DArray _albedoArray, _normalArray;
        private Texture2D _albedoL8, _normalL8;
        private bool _l8AssetA, _l8AssetN; // 烘焙模式下 L8 是 Resources 资产（只能 UnloadAsset，不能 Destroy）
        // staging 分级：竖条带 chunk×res / 横条带 res×chunk / 整层 res×res（§8.2 上传带宽）
        private Texture2D _stgColA, _stgColN, _stgRowA, _stgRowN, _stgFullA, _stgFullN;
        private Vector2[] _origins;      // [_levelCount]，Awake 分配
        private Vector2[] _prevOrigins;
        private Vector2 _prevCamPos;
        private bool[] _layerValid;
        private int[] _refillRow;        // 整层重填进度（已写入行数，Q 的倍数；0 = 未开始）
        private Vector2[] _refillOrigin; // 重填起始帧的吸附原点快照（分带跨帧期间相机可能继续移动）
        private ClipmapSourceRuntime _source;
        private readonly List<StripOp> _ops = new List<StripOp>();
        private int _levelCount;        // K = log2(S)−5（§3.1.1），Awake 计算
        private bool _useCopy;          // 直拷模式：Baked + 金字塔（更新 = 矩形簿记 + CopyTexture）
        private TextureFormat _ringFmtA = TextureFormat.RGBA32, _ringFmtN = TextureFormat.RGBA32; // 环面格式（§10）
        private int _effectiveQ = 30;   // ASTC=30 / BC7=60
        private int _blockA = 1, _blockN = 1; // 直拷块对齐约束（ASTC 6/5，BC7 4/4，RGBA32 1=无约束）
        private double _bptPair;        // 双图每 texel 字节数（ASTC≈1.084 / BC7=2 / RGBA32=8），字节预算用
        private bool _alignWarned;
        private int _l8Aniso = 2;       // §6.4：静态层 Aniso 2x（门闸降级②可降为 1，恢复时回升）
        private float _emaMs = 16f;     // 帧时 EMA（带宽门闸）
        private int _overFrames, _underFrames, _governorStep;
        private Vector2 _camVel;        // 相机水平速度（米/秒，预测性预取用；传送帧清零）

        // ---------------------------------------------------------

        void Awake()
        {
            Active = true;
            if (viewCamera == null) viewCamera = Camera.main;

            // 数据源：烘焙 > Terrain > 测试图案
            _source = ClipmapSourceRuntime.FromBaked(bakedResourcesRoot);
            if (_source != null)
            {
                terrainOriginXZ = _source.BakedOrigin;
                terrainSize = _source.BakedSize;
                Debug.Log("[MTClipmap] 使用烘焙数据源（§7.1）");
            }
            else if (sourceTerrain != null)
            {
                _source = ClipmapSourceRuntime.FromTerrain(sourceTerrain);
                Debug.Log("[MTClipmap] 使用 Terrain 实时采样源");
            }
            else
            {
                _source = new ClipmapSourceRuntime(); // MODE_TEST
                Debug.Log("[MTClipmap] 使用测试图案源");
            }
            _source.TileBudget = tileBudget;

            // 直拷模式：与烘焙金字塔密度 1:1 自对齐（以 manifest 为准，避免序列化旧值失配）
            _useCopy = _source.Mode == ClipmapStripJob.MODE_BAKED && _source.HasPyramid;
            if (_useCopy)
            {
                layerResolution = _source.TileRes;
                baseFootprint = _source.TileWorld0;
            }

            // 动态层数 K = ceil(log2(S/baseFootprint))（§3.1.1 推广：默认 base=32m 时即 K = log2(S)−5，
            // 1024m→K=5 环面 L0..L4 + 静态层 L5 覆盖全地形 @2040²；splat 密度对齐烘焙后 base 以 manifest 为准）。
            // Ceil 保证顶层环面 footprint(K−1) ≥ S/2（覆盖要求）；2 幂边界用 epsilon 防止浮点误差多出一层
            int log2s = Mathf.RoundToInt(Mathf.Log(Mathf.Max(32f, terrainSize), 2f));
            if (Mathf.Abs(terrainSize - (1 << log2s)) > 0.5f)
                Debug.LogWarning($"[MTClipmap] 地形尺寸 {terrainSize}m 非 2 幂，层数按最近 2 幂估算（§3.1.1 契约建议 1024/2048/4096）");
            int kFull = Mathf.Max(1, Mathf.CeilToInt(Mathf.Log(terrainSize / baseFootprint, 2f) - 1e-3f));
            _levelCount = Mathf.Clamp(kFull, 1, MaxLevelCount);
            if (kFull > MaxLevelCount)
                Debug.LogWarning($"[MTClipmap] 地形 {terrainSize}m 所需层数 K={kFull} 超上限 {MaxLevelCount}，已截断——顶层环面覆盖不足区域由静态层兜底（采样密度降低，功能正确）");
            _origins = new Vector2[_levelCount];
            _prevOrigins = new Vector2[_levelCount];
            _layerValid = new bool[_levelCount];
            _refillRow = new int[_levelCount];
            _refillOrigin = new Vector2[_levelCount];

            // 环面直接使用 manifest 格式；目标平台须支持所选 ASTC/BC7。
            if (_useCopy)
            {
                _ringFmtA = _source.BakedTexFormatA;
                _ringFmtN = _source.BakedTexFormatN;
            }
            _source.TileFormatA = _ringFmtA;
            _source.TileFormatN = _ringFmtN;
            _effectiveQ = _ringFmtA == TextureFormat.BC7 ? 60 : updateQuantumTexels; // §5.1：BC7 4×4 块 → Q=60
            _blockA = _ringFmtA == TextureFormat.ASTC_6x6 ? 6 : _ringFmtA == TextureFormat.BC7 ? 4 : 1;
            _blockN = _ringFmtN == TextureFormat.ASTC_5x5 ? 5 : _ringFmtN == TextureFormat.BC7 ? 4 : 1;
            _bptPair = ClipmapSourceRuntime.BytesPerTexel(_ringFmtA) + ClipmapSourceRuntime.BytesPerTexel(_ringFmtN);

            // 块对齐：层分辨率与条带分块必须为 Q 的倍数（所有条带矩形自动块对齐）；
            // Job 回退路径（Terrain/Test/旧烘焙）上限 1020，控制 staging/环面内存
            if (!_useCopy) layerResolution = Mathf.Min(layerResolution, 1020);
            layerResolution = Mathf.Max(_effectiveQ, layerResolution / _effectiveQ * _effectiveQ);
            stripChunkTexels = Mathf.Max(_effectiveQ, stripChunkTexels / _effectiveQ * _effectiveQ);

            BuildTextures();
            var shader = Shader.Find("MT/TerrainClipmap");

            ClipmapMaterial = new Material(shader);
            // 材质通道关键字：manifest channelMask 关闭的通道由 shader 常量替代（烘焙期已写中性值，
            // Job 不再采样 mask）；用 multi_compile 变体，打包不裁剪
            SetChannelKeyword(ClipmapMaterial, "_CLIP_NO_METALLIC", (_source.ChannelMask & 1) == 0);
            SetChannelKeyword(ClipmapMaterial, "_CLIP_NO_AO", (_source.ChannelMask & 2) == 0);
            SetChannelKeyword(ClipmapMaterial, "_CLIP_NO_SMOOTHNESS", (_source.ChannelMask & 4) == 0);
            Shader.SetGlobalVector("_TerrainOrigin", new Vector4(terrainOriginXZ.x, 0, terrainOriginXZ.y, 0));
            Shader.SetGlobalFloat("_TerrainSize", terrainSize);
            Shader.SetGlobalFloat("_ClipBaseFootprint", baseFootprint);
            Shader.SetGlobalFloat("_ClipLayerRes", layerResolution);
            Shader.SetGlobalInt("_ClipLevelCount", _levelCount);
            Shader.SetGlobalFloat("_ClipBlendStart", 0.75f); // §6.2：外环 25% 混合带（门闸降级①收窄至 20%）
            Shader.SetGlobalVector("_ClipViewCameraPos", viewCamera != null ? viewCamera.transform.position : Vector3.zero);
            Shader.SetGlobalInt("_ClipDebugLOD", (int)debugLodMode);
            Shader.SetGlobalInt("_ClipPackedMaterial", _source.PackedMaterial ? 1 : 0);

            // 关闭原 VT 系统，避免冲突
            foreach (var vt in FindObjectsOfType<MTVTCreator>()) vt.enabled = false;

            // L8 静态层：烘焙 PNG 资产（导入器已按平台压缩 + 生成完整 mip 链），直接使用。
            if (_source.Mode == ClipmapStripJob.MODE_BAKED)
            {
                var a = Resources.Load<Texture2D>(bakedResourcesRoot + "/l8_albedo");
                var n = Resources.Load<Texture2D>(bakedResourcesRoot + "/l8_normal");
                if (a != null) { Destroy(_albedoL8); _albedoL8 = a; _l8AssetA = true; }
                if (n != null) { Destroy(_normalL8); _normalL8 = n; _l8AssetN = true; }
                SetupL8Sampler(_albedoL8);
                SetupL8Sampler(_normalL8);
            }
            else
            {
                FillStaticLayerViaJob();
            }

            // 环面与静态层纹理全局绑定：纹理对象生命周期内不变，只设一次（不再每帧下发）
            Shader.SetGlobalTexture("_ClipAlbedoArray", _albedoArray);
            Shader.SetGlobalTexture("_ClipNormalArray", _normalArray);
            Shader.SetGlobalTexture("_ClipAlbedoL8", _albedoL8);
            Shader.SetGlobalTexture("_ClipNormalL8", _normalL8);

            // 初始：所有层无效（_layerValid 默认 false）→ Update 恢复循环按 粗→细 分带重填
            _prevCamPos = viewCamera != null
                ? new Vector2(viewCamera.transform.position.x, viewCamera.transform.position.z)
                : Vector2.zero;
        }

        void OnDestroy()
        {
            Active = false;
            Shader.SetGlobalInt("_ClipDebugLOD", 0);
            _source?.Dispose();
            if (ClipmapMaterial != null) Destroy(ClipmapMaterial);
            // 环面/staging 均为运行时创建，正常 Destroy；烘焙模式的 L8 是 Resources 资产，只能 UnloadAsset
            foreach (var t in new Object[] { _albedoArray, _normalArray,
                                             _stgColA, _stgColN, _stgRowA, _stgRowN, _stgFullA, _stgFullN })
                if (t != null) Destroy(t);
            if (_albedoL8 != null) { if (_l8AssetA) Resources.UnloadAsset(_albedoL8); else Destroy(_albedoL8); }
            if (_normalL8 != null) { if (_l8AssetN) Resources.UnloadAsset(_normalL8); else Destroy(_normalL8); }
        }

        // ---------------------------------------------------------

        void Update()
        {
            Shader.SetGlobalVector("_ClipViewCameraPos", viewCamera != null ? viewCamera.transform.position : Vector3.zero);
            Shader.SetGlobalInt("_ClipDebugLOD", (int)debugLodMode);
            _source.Update(); // 轮询瓦片异步加载（到位后由恢复循环自然拾取）
            TickGovernor();   // 带宽门闸（§8.1）

            Vector2 cam = viewCamera != null
                ? new Vector2(viewCamera.transform.position.x, viewCamera.transform.position.z)
                : _prevCamPos;
            Vector2 camLocal = cam - terrainOriginXZ;

            for (int k = 0; k < _levelCount; ++k)
                _origins[k] = ComputeOrigin(camLocal, k);

            // 传送检测（§5.4：阈值 = footprint(L0) = 32m）→ 全层失效，恢复循环按 粗→细 分带重填
            bool teleported = (cam - _prevCamPos).sqrMagnitude > baseFootprint * baseFootprint;
            // 相机水平速度（预测性预取用）；传送帧位移不是速度，清零避免一次性请求远端瓦片
            _camVel = !teleported && Time.unscaledDeltaTime > 1e-5f
                ? (cam - _prevCamPos) / Time.unscaledDeltaTime : Vector2.zero;
            if (teleported)
                for (int k = 0; k < _levelCount; ++k)
                {
                    _prevOrigins[k] = _origins[k];
                    InvalidateLayer(k);
                }
            _prevCamPos = cam;

            var ops = _ops;
            ops.Clear();
            // 自适应上传预算（§8.2）：帧时宽裕放宽 1.5×（加速流式/恢复），紧张收紧 0.5×（让出主线程）
            long frameBudget = uploadBudgetBytes;
            if (_emaMs > frameBudgetMs) frameBudget >>= 1;
            else if (_emaMs < frameBudgetMs * 0.7f) frameBudget += frameBudget >> 1;
            long budgetLeft = frameBudget;
            long collected = 0;

            // 无效层恢复（粗→细优先）：整层按字节预算分横带、每帧每层最多一带、多帧完成（§8.2）；
            // 完成前该层保持无效，采样端按驻留掩码向上回退（§6.5 语义）。
            // 直拷模式需窗口瓦片全部驻留；未驻留则发起异步请求并等待。
            for (int k = _levelCount - 1; k >= 0; --k)
            {
                if (_layerValid[k]) continue;
                Vector2 winOrigin = _refillRow[k] == 0 ? _origins[k] : _refillOrigin[k];
                if (!WindowTilesReady(k, winOrigin)) { RequestWindowTiles(k, winOrigin); continue; }
                int bandH = BandRowsForBudget(budgetLeft, _refillRow[k]);
                if (bandH <= 0)
                {
                    if (collected > 0) continue; // 本帧预算已用完，下帧继续
                    bandH = Mathf.Min(_effectiveQ, layerResolution - _refillRow[k]); // 防死锁：放行最小带
                }
                CollectRefillBand(k, bandH, ops);
                long bytes = (long)(layerResolution * (double)bandH * _bptPair);
                budgetLeft -= bytes; collected += bytes;
                _refillRow[k] += bandH;
                if (_refillRow[k] >= layerResolution)
                {
                    _refillRow[k] = 0;
                    _prevOrigins[k] = _refillOrigin[k]; // 与快照对齐；分带期间相机位移由下帧条带追平
                    _layerValid[k] = true;
                }
            }

            // 有效层增量条带（细→粗；同一层的条带原子提交，不做跨帧拆分）
            for (int k = 0; k < _levelCount && (budgetLeft > 0 || collected == 0); ++k)
            {
                if (!_layerValid[k]) continue;
                long bytes = EstimateStripBytes(k, _origins[k], out bool needsFull);
                if (needsFull) { InvalidateLayer(k); continue; } // 移动超窗：整层作废，交分带重填
                if (bytes == 0) continue;
                if (bytes > budgetLeft && collected > 0) continue; // 预算不足：下帧再收（origin 未推进，无内容丢失）
                int before = ops.Count;
                CollectStrips(k, _origins[k], ops);
                if (ops.Count == before) continue;
                if (OpsTilesReady(ops, before))
                {
                    budgetLeft -= bytes; collected += bytes;
                }
                else
                {
                    ops.RemoveRange(before, ops.Count - before);
                    InvalidateLayer(k); // 内容已部分过期 → 交由恢复循环分带重填
                }
            }

            // ---- 执行：直拷 = 矩形簿记 + CopyTexture；否则 Job 采样 + staging ----
            // 直拷失败（非块对齐等异常）会把对应层标记失效，故掩码在执行之后计算
            foreach (var op in ops) ExecuteOp(op);

            // 驻留掩码（bit k = 环面层 k 有效；bit K = 静态层恒有效）→ shader 逐层向上扫描
            int mask = 1 << _levelCount;
            for (int k = 0; k < _levelCount; ++k) if (_layerValid[k]) mask |= 1 << k;
            Shader.SetGlobalInt("_ClipValidMask", mask);

            // 预测性预取：按当前速度请求前瞻位置的各层窗口瓦片（已驻留/在途自动去重，仅排队不进预算）
            if (_useCopy && prefetchAheadSeconds > 0f && _camVel.sqrMagnitude > 0.25f)
            {
                Vector2 aheadLocal = camLocal + _camVel * prefetchAheadSeconds;
                for (int k = 0; k < _levelCount; ++k)
                    RequestWindowTiles(k, ComputeOrigin(aheadLocal, k));
            }
        }

        // ---------------------------------------------------------

        float Footprint(int k) => baseFootprint * (1 << k);
        float Texel(int k) => Footprint(k) / layerResolution;
        float QWorld(int k) => _effectiveQ * Texel(k); // §5.1：Q_world_k = Q·texel_k（ASTC Q=30 / BC7 Q=60）

        static void SetChannelKeyword(Material m, string keyword, bool off)
        {
            if (off) m.EnableKeyword(keyword); else m.DisableKeyword(keyword);
        }

        // ---------------- 带宽门闸（§8.1 降级序列） ----------------

        void TickGovernor()
        {
            _emaMs = Mathf.Lerp(_emaMs, Time.unscaledDeltaTime * 1000f, 0.05f);
            if (_emaMs > frameBudgetMs)
            {
                _underFrames = 0;
                if (++_overFrames >= 90 && _governorStep < 2)
                    ApplyGovernorStep(++_governorStep);
            }
            else
            {
                _overFrames = 0;
                // 恢复（带回滞，避免在阈值附近来回抖）：帧时持续低于 80% 预算 600 帧 → 回退一级降级
                if (_governorStep > 0 && _emaMs < frameBudgetMs * 0.8f)
                {
                    if (++_underFrames >= 600)
                        RevertGovernorStep(--_governorStep);
                }
                else _underFrames = 0;
            }
        }

        void RevertGovernorStep(int step)
        {
            _underFrames = 0;
            if (step == 1) // 从②回到①：静态层 Aniso 恢复 2x
            {
                _l8Aniso = 2;
                if (_albedoL8 != null) _albedoL8.anisoLevel = 2;
                if (_normalL8 != null) _normalL8.anisoLevel = 2;
                Debug.Log("[MTClipmap] 带宽门闸恢复：静态层 Aniso 回到 2（§8.1）");
            }
            else // 从①回到无降级：混合带恢复 25%
            {
                Shader.SetGlobalFloat("_ClipBlendStart", 0.75f);
                Debug.Log("[MTClipmap] 带宽门闸恢复：混合带回到 25%（§8.1）");
            }
        }

        void ApplyGovernorStep(int step)
        {
            _overFrames = 0;
            if (step == 1) // 降级①：混合带收窄 25%→20%
            {
                Shader.SetGlobalFloat("_ClipBlendStart", 0.80f);
                Debug.LogWarning("[MTClipmap] 带宽门闸：帧时持续超预算，混合带收窄 25%→20%（§8.1）");
            }
            else // 降级②：静态层 Aniso 2→1
            {
                _l8Aniso = 1;
                if (_albedoL8 != null) _albedoL8.anisoLevel = 1;
                if (_normalL8 != null) _normalL8.anisoLevel = 1;
                Debug.LogWarning("[MTClipmap] 带宽门闸：静态层 Aniso 降为 1（§8.1）");
            }
        }

        Vector2 ComputeOrigin(Vector2 camLocal, int k)
        {
            float fp = Footprint(k);
            float q = QWorld(k);
            float ox = Mathf.Floor((camLocal.x - fp * 0.5f) / q) * q;
            float oz = Mathf.Floor((camLocal.y - fp * 0.5f) / q) * q;
            float maxO = terrainSize - fp;
            return new Vector2(Mathf.Clamp(ox, 0, maxO), Mathf.Clamp(oz, 0, maxO));
        }

        /// <summary>层失效：重置分带重填进度（快照在 CollectRefillBand 重新取）。</summary>
        void InvalidateLayer(int k)
        {
            _layerValid[k] = false;
            _refillRow[k] = 0;
        }

        /// <summary>字节预算内的重附带高（行数，吸附 Q 网格保证 dst 块对齐；≤0 表示预算不足）。</summary>
        int BandRowsForBudget(long budgetLeft, int rowDone)
        {
            double rowBytes = layerResolution * _bptPair;
            int rows = (int)(budgetLeft / rowBytes) / _effectiveQ * _effectiveQ;
            return Mathf.Min(rows, layerResolution - rowDone);
        }

        // ---------------- 瓦片驻留检查 / 请求（直拷模式） ----------------

        /// <summary>世界矩形覆盖的金字塔瓦片范围（层 k ↔ 级 min(k, 顶级)，矩形 ≤ 瓦片边长 → ≤2×2）。</summary>
        void TileRange(int layer, float2 w0, float2 w1,
            out int lvl, out int tx0, out int tz0, out int tx1, out int tz1, out float tw)
        {
            lvl = Mathf.Min(layer, _source.Levels - 1);
            tw = _source.TileWorld0 * (1 << lvl);
            int gc = _source.TileGridCount(lvl);
            tx0 = Mathf.Clamp(Mathf.FloorToInt((w0.x - terrainOriginXZ.x) / tw), 0, gc - 1);
            tz0 = Mathf.Clamp(Mathf.FloorToInt((w0.y - terrainOriginXZ.y) / tw), 0, gc - 1);
            tx1 = Mathf.Clamp(Mathf.FloorToInt((w1.x - 0.001f - terrainOriginXZ.x) / tw), 0, gc - 1);
            tz1 = Mathf.Clamp(Mathf.FloorToInt((w1.y - 0.001f - terrainOriginXZ.y) / tw), 0, gc - 1);
        }

        bool WindowTilesReady(int k, Vector2 origin)
        {
            if (!_useCopy) return true; // Job 路径（Terrain/Test/旧烘焙）无瓦片依赖
            float2 w0 = WorldOf(origin.x, origin.y);
            float2 w1 = w0 + new float2(Footprint(k), Footprint(k));
            TileRange(k, w0, w1, out int lvl, out int tx0, out int tz0, out int tx1, out int tz1, out _);
            for (int tz = tz0; tz <= tz1; ++tz)
            for (int tx = tx0; tx <= tx1; ++tx)
                if (!_source.IsTileResident(lvl, tx, tz)) return false;
            return true;
        }

        void RequestWindowTiles(int k, Vector2 origin)
        {
            if (!_useCopy) return;
            float2 w0 = WorldOf(origin.x, origin.y);
            float2 w1 = w0 + new float2(Footprint(k), Footprint(k));
            TileRange(k, w0, w1, out int lvl, out int tx0, out int tz0, out int tx1, out int tz1, out _);
            for (int tz = tz0; tz <= tz1; ++tz)
            for (int tx = tx0; tx <= tx1; ++tx)
                _source.EnsureTileRequested(lvl, tx, tz);
        }

        /// <summary>检查 ops[start..] 覆盖的瓦片全部驻留；缺失的顺便发起请求。</summary>
        bool OpsTilesReady(List<StripOp> ops, int start)
        {
            if (!_useCopy) return true;
            bool ready = true;
            for (int i = start; i < ops.Count; ++i)
            {
                var op = ops[i];
                float2 w1 = op.worldMin + new float2(op.w, op.h) * Texel(op.layer);
                TileRange(op.layer, op.worldMin, w1, out int lvl, out int tx0, out int tz0, out int tx1, out int tz1, out _);
                for (int tz = tz0; tz <= tz1; ++tz)
                for (int tx = tx0; tx <= tx1; ++tx)
                {
                    if (_source.IsTileResident(lvl, tx, tz)) continue;
                    _source.EnsureTileRequested(lvl, tx, tz);
                    ready = false;
                }
            }
            return ready;
        }

        /// <summary>估算层 k 从上次吸附原点移到 cur 的条带字节量（角区已裁剪，无双写）；超窗移动置 needsFull。</summary>
        long EstimateStripBytes(int k, Vector2 cur, out bool needsFull)
        {
            needsFull = false;
            Vector2 prev = _prevOrigins[k];
            if (cur == prev) return 0;
            float texel = Texel(k);
            int res = layerResolution;
            int dx = Mathf.RoundToInt((cur.x - prev.x) / texel);
            int dz = Mathf.RoundToInt((cur.y - prev.y) / texel);
            if (dx == 0 && dz == 0) return 0;
            if (Mathf.Abs(dx) >= res || Mathf.Abs(dz) >= res) { needsFull = true; return 0; }
            double texels = (double)Mathf.Abs(dx) * res + (double)(res - Mathf.Abs(dx)) * Mathf.Abs(dz);
            return (long)(texels * _bptPair);
        }

        void CollectStrips(int k, Vector2 cur, List<StripOp> ops)
        {
            Vector2 prev = _prevOrigins[k];
            if (cur == prev) return;
            _prevOrigins[k] = cur;

            float texel = Texel(k);
            int res = layerResolution;
            int dx = Mathf.RoundToInt((cur.x - prev.x) / texel);
            int dz = Mathf.RoundToInt((cur.y - prev.y) / texel);
            if (dx == 0 && dz == 0) return; // 地形边缘钳制导致的亚 texel 抖动，忽略
            if (Mathf.Abs(dx) >= res || Mathf.Abs(dz) >= res) return; // EstimateStripBytes 已拦截（整层作废走分带重填）

            if (dx != 0)
            {
                // 竖条带：新内容 = [prev+fp, cur+fp)（+x）或 [cur, prev)（-x）× 全高（§5.2）
                int wTotal = Mathf.Abs(dx);
                float regionMinX = dx > 0 ? cur.x + Footprint(k) - wTotal * texel : cur.x;
                // 固定图集：行 j ↔ localZ ≡ j·texel (mod fp)，全高条带从窗口 z 最小角对应行写起
                int dstY = WrapTexel(cur.y / texel, res);
                int done = 0;
                while (done < wTotal)
                {
                    int cw = Mathf.Min(stripChunkTexels, wTotal - done);
                    float wx = regionMinX + done * texel;
                    AddWrapOps(ops, k, StripKind.Col, WrapTexel(wx / texel, res), dstY, cw, res,
                        WorldOf(wx, cur.y), texel);
                    done += cw;
                }
            }
            if (dz != 0)
            {
                // 横条带：角区 |dx| 列已由同批竖条带写入（同层条带原子提交、竖条带在前），裁剪掉避免双写
                int hTotal = Mathf.Abs(dz);
                int clipX = Mathf.Abs(dx);
                float regionMinZ = dz > 0 ? cur.y + Footprint(k) - hTotal * texel : cur.y;
                float rowMinX = dx < 0 ? cur.x + clipX * texel : cur.x;
                int rowW = res - clipX;
                int dstX = WrapTexel(rowMinX / texel, res);
                int done = 0;
                while (done < hTotal)
                {
                    int ch = Mathf.Min(stripChunkTexels, hTotal - done);
                    float wz = regionMinZ + done * texel;
                    AddWrapOps(ops, k, StripKind.Row, dstX, WrapTexel(wz / texel, res), rowW, ch,
                        WorldOf(rowMinX, wz), texel);
                    done += ch;
                }
            }
        }

        /// <summary>
        /// 固定图集 wrap 拆分：条带在纹理边缘处拆为 ≤2 段/轴（§5.2 v2.3.2 实现补充），
        /// 保证 CopyTexture 目标矩形不越界；每段内容保持世界坐标仿射连续。
        /// </summary>
        void AddWrapOps(List<StripOp> ops, int k, StripKind kind, int dstX, int dstY,
            int w, int h, float2 worldMin, float texel)
        {
            int res = layerResolution;
            int x = dstX; float wx = worldMin.x; int rw = w;
            while (rw > 0)
            {
                int cw = Mathf.Min(rw, res - x);
                int y = dstY; float wz = worldMin.y; int rh = h;
                while (rh > 0)
                {
                    int ch = Mathf.Min(rh, res - y);
                    ops.Add(new StripOp
                    {
                        layer = k, kind = kind,
                        dst = new RectInt(x, y, cw, ch),
                        worldMin = new float2(wx, wz),
                        w = cw, h = ch,
                    });
                    rh -= ch; y = 0; wz += ch * texel;
                }
                rw -= cw; x = 0; wx += cw * texel;
            }
        }

        float2 WorldOf(float localX, float localZ)
            => new float2(localX + terrainOriginXZ.x, localZ + terrainOriginXZ.y);

        /// <summary>
        /// 整层重填的一个横带（res × bandH）：首带开始时快照吸附原点，后续带沿用——
        /// 分带跨帧期间相机继续移动时，完成后由常规条带从快照原点追平（§5.4 粗→细重填）。
        /// </summary>
        void CollectRefillBand(int k, int bandH, List<StripOp> ops)
        {
            if (_refillRow[k] == 0) _refillOrigin[k] = _origins[k];
            float texel = Texel(k);
            int res = layerResolution;
            Vector2 o = _refillOrigin[k];
            float localZ0 = o.y + _refillRow[k] * texel;
            // 固定图集：带内容 = 窗口 [o, o+fp)² 的行区间，从对应 texel 起 wrap 拆分
            AddWrapOps(ops, k, StripKind.Full,
                WrapTexel(o.x / texel, res), WrapTexel(localZ0 / texel, res),
                res, bandH, WorldOf(o.x, localZ0), texel);
        }

        static int WrapTexel(float t, int res)
        {
            int i = Mathf.RoundToInt(t) % res;
            return i < 0 ? i + res : i;
        }

        // ---------------------------------------------------------

        void ExecuteOp(StripOp op)
        {
            if (_useCopy) { ExecuteOpCopy(op); return; }
            ExecuteOpJob(op);
        }

        /// <summary>
        /// 直拷执行：条带世界矩形与覆盖瓦片求交（≤4），每交集一块 CopyTexture。
        /// CPU 仅矩形簿记——烘焙 texel 即环面 texel（1:1），无任何重采样。
        /// </summary>
        void ExecuteOpCopy(StripOp op)
        {
            float texel = Texel(op.layer);
            float2 w1 = op.worldMin + new float2(op.w, op.h) * texel;
            TileRange(op.layer, op.worldMin, w1, out int lvl, out int tx0, out int tz0, out int tx1, out int tz1, out float tw);
            for (int tz = tz0; tz <= tz1; ++tz)
            for (int tx = tx0; tx <= tx1; ++tx)
            {
                if (!_source.TryGetTile(lvl, tx, tz, out var ta, out var tn))
                {
                    // 瓦片在收集后被 LRU 淘汰等异常：内容已不可靠，整层失效交恢复循环分带重填
                    InvalidateLayer(op.layer);
                    continue;
                }
                float tileMinX = terrainOriginXZ.x + tx * tw;
                float tileMinZ = terrainOriginXZ.y + tz * tw;
                float ix0 = Mathf.Max(op.worldMin.x, tileMinX);
                float iz0 = Mathf.Max(op.worldMin.y, tileMinZ);
                float ix1 = Mathf.Min(w1.x, tileMinX + tw);
                float iz1 = Mathf.Min(w1.y, tileMinZ + tw);
                if (ix1 <= ix0 || iz1 <= iz0) continue;
                int sx = Mathf.RoundToInt((ix0 - tileMinX) / texel);
                int sz = Mathf.RoundToInt((iz0 - tileMinZ) / texel);
                int dx = op.dst.x + Mathf.RoundToInt((ix0 - op.worldMin.x) / texel);
                int dz = op.dst.y + Mathf.RoundToInt((iz0 - op.worldMin.y) / texel);
                int cw = Mathf.RoundToInt((ix1 - ix0) / texel);
                int ch = Mathf.RoundToInt((iz1 - iz0) / texel);
                // 地形边缘钳制会破坏 origin 的 Q 对齐，产生分数 texel 偏移，RoundToInt 可能越出
                // 纹理边界（sz=res）：源/目标矩形同步钳制进 [0,res]，内容对齐关系保持不变
                ClampCopyRect(ref sx, ref dx, ref cw);
                ClampCopyRect(ref sz, ref dz, ref ch);
                if (cw <= 0 || ch <= 0) continue;
                // 压缩直拷需块对齐（res 与 Q 已保证对齐；仅地形边缘钳制等异常会触发）
                bool aligned = (sx % _blockA | sz % _blockA | dx % _blockA | dz % _blockA | cw % _blockA | ch % _blockA) == 0
                            && (sx % _blockN | sz % _blockN | dx % _blockN | dz % _blockN | cw % _blockN | ch % _blockN) == 0;
                if (!aligned)
                {
                    // 层内容已不可靠：标记失效，掩码降级到粗层，下帧恢复循环按对齐窗口分带重填
                    InvalidateLayer(op.layer);
                    if (!_alignWarned)
                    {
                        _alignWarned = true;
                        Debug.LogWarning("[MTClipmap] 非块对齐矩形，放弃本次压缩直拷——该层已标记失效并回退粗层显示，下帧起分带重填");
                    }
                    continue;
                }
                Graphics.CopyTexture(ta, 0, 0, sx, sz, cw, ch, _albedoArray, op.layer, 0, dx, dz);
                Graphics.CopyTexture(tn, 0, 0, sx, sz, cw, ch, _normalArray, op.layer, 0, dx, dz);
            }
        }

        /// <summary>源矩形一维钳制进 [0,res]，目标坐标同步平移、长度同步收窄，保持内容对齐。</summary>
        void ClampCopyRect(ref int src, ref int dst, ref int len)
        {
            int s0 = Mathf.Clamp(src, 0, layerResolution);
            int s1 = Mathf.Clamp(src + len, 0, layerResolution);
            dst += s0 - src;
            len = s1 - s0;
            src = s0;
        }

        /// <summary>Job 重采样路径（Terrain/Test 模式，及无金字塔的旧烘焙数据）。</summary>
        void ExecuteOpJob(StripOp op)
        {
            float texel = Texel(op.layer);
            Texture2D stgA, stgN;
            switch (op.kind)
            {
                case StripKind.Col:
                    EnsureStaging(ref _stgColA, ref _stgColN, stripChunkTexels, layerResolution);
                    stgA = _stgColA; stgN = _stgColN; break;
                case StripKind.Row:
                    EnsureStaging(ref _stgRowA, ref _stgRowN, layerResolution, stripChunkTexels);
                    stgA = _stgRowA; stgN = _stgRowN; break;
                default:
                    EnsureStaging(ref _stgFullA, ref _stgFullN, layerResolution, layerResolution);
                    stgA = _stgFullA; stgN = _stgFullN; break;
            }
            var rawA = stgA.GetRawTextureData<Color32>();
            var rawN = stgN.GetRawTextureData<Color32>();

            var jobA = _source.CreateJob(op.layer, true, op.worldMin, texel, op.w, rawA, stgA.width, 0);
            var jobN = _source.CreateJob(op.layer, false, op.worldMin, texel, op.w, rawN, stgN.width, 0);
            var hA = jobA.Schedule(op.w * op.h, 64);
            var hN = jobN.Schedule(op.w * op.h, 64);
            JobHandle.CompleteAll(ref hA, ref hN);

            stgA.Apply(false);
            stgN.Apply(false);
            Graphics.CopyTexture(stgA, 0, 0, 0, 0, op.w, op.h,
                                 _albedoArray, op.layer, 0, op.dst.x, op.dst.y);
            Graphics.CopyTexture(stgN, 0, 0, 0, 0, op.w, op.h,
                                 _normalArray, op.layer, 0, op.dst.x, op.dst.y);
        }

        void FillStaticLayerViaJob()
        {
            var rawA = _albedoL8.GetRawTextureData<Color32>();
            var rawN = _normalL8.GetRawTextureData<Color32>();
            float texel = terrainSize / layerResolution;
            var w0 = WorldOf(0, 0);
            // layer=8 是 Job 内部约定（>= 8 选 l8 全地形源），与 shader 动态层数 K 无关
            var jobA = _source.CreateJob(8, true, w0, texel, layerResolution, rawA, layerResolution, 0);
            var jobN = _source.CreateJob(8, false, w0, texel, layerResolution, rawN, layerResolution, 0);
            var hA = jobA.Schedule(layerResolution * layerResolution, 64);
            var hN = jobN.Schedule(layerResolution * layerResolution, 64);
            JobHandle.CompleteAll(ref hA, ref hN);
            _albedoL8.Apply(true);
            _normalL8.Apply(true);
        }

        // ---------------------------------------------------------

        void BuildTextures()
        {
            if (_ringFmtA != TextureFormat.RGBA32)
            {
                // §4.1/§10：压缩环面（ASTC 6x6 sRGB / ASTC 5x5 linear，或 BC7）——与瓦片同格式，CopyTexture 直拷。
                // 压缩纹理无法 CPU 初始化：未写入层由驻留掩码保证不被采样，无需清底。
                _albedoArray = new Texture2DArray(layerResolution, layerResolution, _levelCount,
                    _ringFmtA, false, false) { wrapMode = TextureWrapMode.Repeat };
                _normalArray = new Texture2DArray(layerResolution, layerResolution, _levelCount,
                    _ringFmtN, false, true) { wrapMode = TextureWrapMode.Repeat };
            }
            else
            {
                // §4.1：Albedo = sRGB / NormalAttr = linear；环面层无 Mip + Repeat
                _albedoArray = new Texture2DArray(layerResolution, layerResolution, _levelCount,
                    TextureFormat.RGBA32, false, false) { wrapMode = TextureWrapMode.Repeat };
                _normalArray = new Texture2DArray(layerResolution, layerResolution, _levelCount,
                    TextureFormat.RGBA32, false, true) { wrapMode = TextureWrapMode.Repeat };

                var neutralA = new Color32(128, 128, 128, 255);
                var neutralN = new Color32(128, 128, 216, 255);
                var fillA = new Color32[layerResolution * layerResolution];
                var fillN = new Color32[layerResolution * layerResolution];
                for (int i = 0; i < fillA.Length; ++i) { fillA[i] = neutralA; fillN[i] = neutralN; }
                for (int k = 0; k < _levelCount; ++k)
                {
                    _albedoArray.SetPixelData(fillA, 0, k);
                    _normalArray.SetPixelData(fillN, 0, k);
                }
                _albedoArray.Apply(false);
                _normalArray.Apply(false);
            }
            // 静态层 L_K：独立 Texture2D，完整 Mip 链（D5；烘焙数据含离线 Mip 链，Job 路径运行时生成）
            _albedoL8 = new Texture2D(layerResolution, layerResolution, TextureFormat.RGBA32, true, false);
            _normalL8 = new Texture2D(layerResolution, layerResolution, TextureFormat.RGBA32, true, true);
            SetupL8Sampler(_albedoL8);
            SetupL8Sampler(_normalL8);
        }

        void SetupL8Sampler(Texture2D t)
        {
            if (t == null) return;
            t.wrapMode = TextureWrapMode.Clamp;   // §4.1：静态层用 Clamp（环面才用 Repeat）
            t.filterMode = FilterMode.Trilinear;  // 完整 Mip 链
            t.anisoLevel = _l8Aniso;              // §6.4：静态层 Aniso 2x（门闸降级②可降为 1）
        }

        static void EnsureStaging(ref Texture2D stgA, ref Texture2D stgN, int w, int h)
        {
            // sRGB 标志与目标 array 一致（CopyTexture 要求格式兼容）：albedo sRGB / normal linear
            if (stgA == null) stgA = new Texture2D(w, h, TextureFormat.RGBA32, false, false);
            if (stgN == null) stgN = new Texture2D(w, h, TextureFormat.RGBA32, false, true);
        }

        // ---------------------------------------------------------
        // Gizmos：clipmap 区块可视化（运行时 = 实时窗口吸附状态；编辑模式 = 按序列化参数预估）

        void OnDrawGizmos()
        {
            if (!showGizmos) return;
            // 混合带起点与 shader 全局值同步（带宽门闸降级①收窄 0.75→0.80）
            float blendStart = _governorStep >= 1 ? 0.80f : 0.75f;
            Vector3 camPos = viewCamera != null ? viewCamera.transform.position
                : new Vector3(terrainOriginXZ.x + terrainSize * 0.5f, 0f, terrainOriginXZ.y + terrainSize * 0.5f);

            // 运行时以 Awake/Update 的实际状态为准（层数 K 与 baseFootprint 已被 manifest 校正）；
            // 编辑模式 _origins 为空，按序列化参数估算 K = ceil(log2(S/base))，窗口以相机为中心近似
            bool runtime = _origins != null && _levelCount > 0;
            int kCount = runtime ? _levelCount : Mathf.Clamp(
                Mathf.Max(1, Mathf.CeilToInt(Mathf.Log(terrainSize / Mathf.Max(1f, baseFootprint), 2f) - 1e-3f)),
                1, MaxLevelCount);
            var camLocal = new Vector2(camPos.x - terrainOriginXZ.x, camPos.z - terrainOriginXZ.y);

            for (int k = kCount - 1; k >= 0; --k) // 粗层先画，细层后画不被盖住
            {
                float fp = baseFootprint * (1 << k);
                Vector2 o = runtime ? _origins[k] : ComputeOrigin(camLocal, k);
                float w0x = terrainOriginXZ.x + o.x, w0z = terrainOriginXZ.y + o.y;
                bool valid = !runtime || _layerValid[k];
                // 贴合地表：取窗口中心地形高度（无高度图时回退相机高度），逐层抬高避免共面闪烁
                float y = GroundY(w0x + fp * 0.5f, w0z + fp * 0.5f, camPos.y) + 0.5f + 0.2f * k;

                var c = Color.HSVToRGB((float)k / Mathf.Max(1, kCount), 0.85f, 1f);
                c.a = valid ? 1f : 0.25f; // 失效重填中的层半透明
                Gizmos.color = c;
                DrawRectXZ(w0x, w0z, fp, y);
                // 混合带内框：r ≤ blendStart·R_k 区域采本层，外环向粗层混合（§6.2）
                c.a *= 0.5f;
                Gizmos.color = c;
                float half = fp * 0.5f * blendStart;
                DrawRectXZ(w0x + fp * 0.5f - half, w0z + fp * 0.5f - half, half * 2f, y);
#if UNITY_EDITOR
                c.a = 1f;
                UnityEditor.Handles.color = c;
                UnityEditor.Handles.Label(new Vector3(w0x, y, w0z),
                    $"L{k}: {fp:F0}m, texel {fp / layerResolution * 100f:F2}cm" +
                    (runtime && !valid ? "（重填中）" : runtime ? "" : "（预估）"));
#endif
            }
            // 实际驻留的烘焙瓦片划分（直拷模式）：青色边界 + 坐标/尺寸标签
            if (runtime && _source != null && _source.HasPyramid)
            {
                foreach (var (level, tx, tz) in _source.ResidentTiles)
                {
                    float tw = _source.TileWorld0 * (1 << level);
                    float wx = terrainOriginXZ.x + tx * tw;
                    float wz = terrainOriginXZ.y + tz * tw;
                    float yt = GroundY(wx + tw * 0.5f, wz + tw * 0.5f, camPos.y) + 1.0f;
                    Gizmos.color = new Color(0f, 1f, 1f, 0.8f);
                    DrawRectXZ(wx, wz, tw, yt);
#if UNITY_EDITOR
                    UnityEditor.Handles.color = new Color(0f, 1f, 1f, 0.9f);
                    UnityEditor.Handles.Label(new Vector3(wx, yt, wz), $"瓦片 l{level}_{tx}_{tz}  {tw:F0}m");
#endif
                }
            }
            // 静态层 L_K：覆盖全地形（白色外框）
            float ys = GroundY(terrainOriginXZ.x + terrainSize * 0.5f, terrainOriginXZ.y + terrainSize * 0.5f, camPos.y) + 0.1f;
            Gizmos.color = new Color(1f, 1f, 1f, 0.4f);
            DrawRectXZ(terrainOriginXZ.x, terrainOriginXZ.y, terrainSize, ys);
#if UNITY_EDITOR
            UnityEditor.Handles.color = Color.white;
            UnityEditor.Handles.Label(new Vector3(terrainOriginXZ.x, ys, terrainOriginXZ.y),
                $"静态层 L{kCount}: 全地形 {terrainSize:F0}m, texel {terrainSize / layerResolution:F2}m");
#endif
        }

        /// <summary>地表高度采样（高度图未注册时回退 fallback）。</summary>
        float GroundY(float wx, float wz, float fallback)
        {
            float h = 0f;
            return MTHeightMap.GetHeightInterpolated(new Vector3(wx, 0f, wz), ref h) ? h : fallback;
        }

        static void DrawRectXZ(float x, float z, float size, float y)
        {
            var a = new Vector3(x, y, z);
            var b = new Vector3(x + size, y, z);
            var c = new Vector3(x + size, y, z + size);
            var d = new Vector3(x, y, z + size);
            Gizmos.DrawLine(a, b); Gizmos.DrawLine(b, c);
            Gizmos.DrawLine(c, d); Gizmos.DrawLine(d, a);
        }
    }
}
