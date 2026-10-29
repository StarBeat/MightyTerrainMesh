namespace MightyTerrainMesh
{
    using Unity.Burst;
    using Unity.Collections;
    using Unity.Jobs;
    using Unity.Mathematics;
    using UnityEngine;

    /// <summary>
    /// Clipmap 条带采样 Job（Burst）。逐 texel 计算世界坐标 → 按模式采样 → 写入目标缓冲。
    /// 目标缓冲通常是 staging Texture2D 的 GetRawTextureData 视图（stride = 纹理宽），
    /// 也可以是整块 NativeArray（烘焙时）。
    /// </summary>
    [BurstCompile]
    public struct ClipmapStripJob : IJobParallelFor
    {
        public const int MODE_TEST = 0;
        public const int MODE_TERRAIN = 1;
        public const int MODE_BAKED = 2;

        public int mode;
        public int layer;           // 采样目标层（8 = L8 静态层）
        public bool albedo;         // true=albedo，false=normalAttr
        public float2 worldMin;     // 条带区域世界最小角（世界坐标，含 TerrainOrigin）
        public float texel;         // 世界米/texel

        public int stripW;          // 条带宽（texel）
        public int dstStride;       // 目标缓冲行宽
        public int dstOffset;       // 目标缓冲起始偏移（texel 数）
        [NativeDisableParallelForRestriction]
        [WriteOnly] public NativeArray<Color32> output;

        public float4 testColorA;
        public float4 testColorB;
        public float testCell;

        public float3 terrainPos;
        public float3 terrainDataSize;
        [ReadOnly] public NativeArray<float> alphamaps;   // (y*res+x)*layers + l
        public int alphaRes;
        public int alphaLayers;
        [ReadOnly] public NativeArray<Color32> layerPix0;
        [ReadOnly] public NativeArray<Color32> layerPix1;
        [ReadOnly] public NativeArray<Color32> layerPix2;
        [ReadOnly] public NativeArray<Color32> layerPix3;
        [ReadOnly] public NativeArray<Color32> normalPix0;
        [ReadOnly] public NativeArray<Color32> normalPix1;
        [ReadOnly] public NativeArray<Color32> normalPix2;
        [ReadOnly] public NativeArray<Color32> normalPix3;
        [ReadOnly] public NativeArray<Color32> maskPix0;
        [ReadOnly] public NativeArray<Color32> maskPix1;
        [ReadOnly] public NativeArray<Color32> maskPix2;
        [ReadOnly] public NativeArray<Color32> maskPix3;
        public int4 layerSize;      // 各层纹理宽（0 = 无）
        public int4 layerSizeH;     // 各层纹理高
        public int4 normalSize;
        public int4 normalSizeH;
        public int4 maskSize;
        public int4 maskSizeH;
        public float4 tileSizeX;
        public float4 tileSizeY;
        public float4 tileOffX;     // TerrainLayer.tileOffset（tile 单位）
        public float4 tileOffY;
        [ReadOnly] public NativeArray<Color32> layerPix4;   // 层 4–7
        [ReadOnly] public NativeArray<Color32> layerPix5;
        [ReadOnly] public NativeArray<Color32> layerPix6;
        [ReadOnly] public NativeArray<Color32> layerPix7;
        [ReadOnly] public NativeArray<Color32> normalPix4;
        [ReadOnly] public NativeArray<Color32> normalPix5;
        [ReadOnly] public NativeArray<Color32> normalPix6;
        [ReadOnly] public NativeArray<Color32> normalPix7;
        [ReadOnly] public NativeArray<Color32> maskPix4;
        [ReadOnly] public NativeArray<Color32> maskPix5;
        [ReadOnly] public NativeArray<Color32> maskPix6;
        [ReadOnly] public NativeArray<Color32> maskPix7;
        public int4 layerSizeB;
        public int4 layerSizeHB;
        public int4 normalSizeB;
        public int4 normalSizeHB;
        public int4 maskSizeB;
        public int4 maskSizeHB;
        public float4 tileSizeXB;
        public float4 tileSizeYB;
        public float4 tileOffXB;
        public float4 tileOffYB;
        public float4 normalScale;
        public float4 normalScaleB;
        public float4 layerMetallic;
        public float4 layerMetallicB;
        public float4 layerSmoothness;
        public float4 layerSmoothnessB;
        [ReadOnly] public NativeArray<float> heights;     // 归一化 [0,1]，y*res+x
        public int heightRes;

        [ReadOnly] public NativeArray<Color32> l7Albedo;
        [ReadOnly] public NativeArray<Color32> l7Normal;
        public int l7Res;
        [ReadOnly] public NativeArray<Color32> l8Albedo;
        [ReadOnly] public NativeArray<Color32> l8Normal;
        public int l8Res;
        [ReadOnly] public NativeArray<float4> blockRects;   // xy=min.xz（世界），z=w，w=h（米）
        [ReadOnly] public NativeArray<Color32> blockAlbedo; // 块 i 起始 = i*blockRes²
        [ReadOnly] public NativeArray<Color32> blockNormal;
        public int blockRes;
        public int blockCount;
        public float2 bakedOriginXZ;
        public float bakedTerrainSize;

        // ---------------------------------------------------------

        public void Execute(int index)
        {
            int x = index % stripW;
            int y = index / stripW;
            float2 world = worldMin + new float2(x, y) * texel;
            float4 c = albedo ? SampleAlbedo(world) : SampleNormalAttr(world);
            output[dstOffset + y * dstStride + x] = ToColor32(c);
        }

        static Color32 ToColor32(float4 c)
        {
            return new Color32(
                (byte)(math.saturate(c.x) * 255f),
                (byte)(math.saturate(c.y) * 255f),
                (byte)(math.saturate(c.z) * 255f),
                (byte)(math.saturate(c.w) * 255f));
        }

        static float4 Bilinear(NativeArray<Color32> px, int res, float2 uv)
        {
            float2 p = math.saturate(uv) * (res - 1);
            int2 i0 = (int2)math.floor(p);
            int2 i1 = math.min(i0 + 1, res - 1);
            float2 t = p - i0;
            float4 c00 = ToFloat4(px[i0.y * res + i0.x]);
            float4 c10 = ToFloat4(px[i0.y * res + i1.x]);
            float4 c01 = ToFloat4(px[i1.y * res + i0.x]);
            float4 c11 = ToFloat4(px[i1.y * res + i1.x]);
            return math.lerp(math.lerp(c00, c10, t.x), math.lerp(c01, c11, t.x), t.y);
        }

        static float4 ToFloat4(Color32 c)
        {
            return new float4(c.r / 255f, c.g / 255f, c.b / 255f, c.a / 255f);
        }

        // ---------------- Albedo ----------------

        float4 SampleAlbedo(float2 w)
        {
            switch (mode)
            {
                case MODE_TERRAIN: return SplatAlbedo(w);
                case MODE_BAKED: return BakedAlbedo(w);
                default: return TestAlbedo(w);
            }
        }

        float4 TestAlbedo(float2 w)
        {
            int cx = (int)math.floor(w.x / testCell);
            int cy = (int)math.floor(w.y / testCell);
            return ((cx + cy) & 1) == 0 ? testColorA : testColorB;
        }

        float4 SplatAlbedo(float2 w)
        {
            int aBase = AlphaBase(w);
            float3 rgb = float3.zero;
            float metallic = 0;
            for (int l = 0; l < 8 && l < alphaLayers; ++l)
            {
                float weight = alphamaps[aBase + l];
                if (weight < 0.003f) continue;
                float4 diffuse = LayerTexel(l, w, 0, new float4(0.5f, 0.5f, 0.5f, 1f));
                float4 mask = LayerTexel(l, w, 2, new float4(LayerMetallic(l), 1f, 0f, 0f));
                rgb += diffuse.xyz * weight;
                metallic += mask.x * weight;
            }
            return new float4(rgb, metallic);
        }

        int AlphaBase(float2 w)
        {
            float u = math.saturate((w.x - terrainPos.x) / terrainDataSize.x);
            float v = math.saturate((w.y - terrainPos.z) / terrainDataSize.z);
            int ax = math.min(alphaRes - 1, (int)(u * alphaRes));
            int ay = math.min(alphaRes - 1, (int)(v * alphaRes));
            return (ay * alphaRes + ax) * alphaLayers;
        }

        float4 LayerTexel(int l, float2 w, int kind, float4 fallback)
        {
            NativeArray<Color32> px;
            int lw, lh;
            float tx, ty, ox, oy;
            if (l < 4)
            {
                if (kind == 0)
                {
                    px = l == 0 ? layerPix0 : l == 1 ? layerPix1 : l == 2 ? layerPix2 : layerPix3;
                    lw = layerSize[l]; lh = layerSizeH[l];
                }
                else if (kind == 1)
                {
                    px = l == 0 ? normalPix0 : l == 1 ? normalPix1 : l == 2 ? normalPix2 : normalPix3;
                    lw = normalSize[l]; lh = normalSizeH[l];
                }
                else
                {
                    px = l == 0 ? maskPix0 : l == 1 ? maskPix1 : l == 2 ? maskPix2 : maskPix3;
                    lw = maskSize[l]; lh = maskSizeH[l];
                }
                tx = tileSizeX[l]; ty = tileSizeY[l]; ox = tileOffX[l]; oy = tileOffY[l];
            }
            else
            {
                int l4 = l - 4;
                if (kind == 0)
                {
                    px = l4 == 0 ? layerPix4 : l4 == 1 ? layerPix5 : l4 == 2 ? layerPix6 : layerPix7;
                    lw = layerSizeB[l4]; lh = layerSizeHB[l4];
                }
                else if (kind == 1)
                {
                    px = l4 == 0 ? normalPix4 : l4 == 1 ? normalPix5 : l4 == 2 ? normalPix6 : normalPix7;
                    lw = normalSizeB[l4]; lh = normalSizeHB[l4];
                }
                else
                {
                    px = l4 == 0 ? maskPix4 : l4 == 1 ? maskPix5 : l4 == 2 ? maskPix6 : maskPix7;
                    lw = maskSizeB[l4]; lh = maskSizeHB[l4];
                }
                tx = tileSizeXB[l4]; ty = tileSizeYB[l4]; ox = tileOffXB[l4]; oy = tileOffYB[l4];
            }
            if (lw <= 0 || lh <= 0 || px.Length == 0) return fallback;
            // Unity 地形层平铺以地形原点为基准：uv = (worldXZ - terrainPosXZ) / tileSize + tileOffset
            float rx = w.x - terrainPos.x;
            float ry = w.y - terrainPos.z;
            float lu = (rx / math.max(0.01f, tx) + ox) % 1f; if (lu < 0) lu += 1f;
            float lv = (ry / math.max(0.01f, ty) + oy) % 1f; if (lv < 0) lv += 1f;
            int lx = math.min(lw - 1, (int)(lu * lw));
            int ly = math.min(lh - 1, (int)(lv * lh));
            return ToFloat4(px[ly * lw + lx]);
        }

        float LayerNormalScale(int l) => l < 4 ? normalScale[l] : normalScaleB[l - 4];
        float LayerMetallic(int l) => l < 4 ? layerMetallic[l] : layerMetallicB[l - 4];
        float LayerSmoothness(int l) => l < 4 ? layerSmoothness[l] : layerSmoothnessB[l - 4];

        float4 BakedAlbedo(float2 w)
        {
            if (layer >= 8 && l8Albedo.Length > 0)
                return Bilinear(l8Albedo, l8Res, (w - bakedOriginXZ) / bakedTerrainSize);
            // 区域高精块优先（§7.1 有包区域，legacy）
            for (int i = 0; i < blockCount; ++i)
            {
                float4 r = blockRects[i];
                if (w.x >= r.x && w.y >= r.y && w.x < r.x + r.z && w.y < r.y + r.w)
                {
                    float2 uv = (w - r.xy) / r.zw;
                    return Bilinear(blockAlbedo, blockRes, uv, i);
                }
            }
            // 无包区域：L7 源上采样
            return Bilinear(l7Albedo, l7Res, (w - bakedOriginXZ) / bakedTerrainSize);
        }

        // 块数据双线性（块沿一维数组顺序平铺）
        static float4 Bilinear(NativeArray<Color32> blocks, int blockRes, float2 uv, int blockIdx)
        {
            int baseIdx = blockIdx * blockRes * blockRes;
            float2 p = math.saturate(uv) * (blockRes - 1);
            int2 i0 = (int2)math.floor(p);
            int2 i1 = math.min(i0 + 1, blockRes - 1);
            float2 t = p - i0;
            float4 c00 = ToFloat4(blocks[baseIdx + i0.y * blockRes + i0.x]);
            float4 c10 = ToFloat4(blocks[baseIdx + i0.y * blockRes + i1.x]);
            float4 c01 = ToFloat4(blocks[baseIdx + i1.y * blockRes + i0.x]);
            float4 c11 = ToFloat4(blocks[baseIdx + i1.y * blockRes + i1.x]);
            return math.lerp(math.lerp(c00, c10, t.x), math.lerp(c01, c11, t.x), t.y);
        }

        // ---------------- NormalAttr ----------------
        // §4.1 通道契约：RG = 法线 XY（TS，x→worldX，y→worldZ），B = 粗糙度，A = AO

        float4 SampleNormalAttr(float2 w)
        {
            switch (mode)
            {
                case MODE_TERRAIN: return SplatNormalAttr(w);
                case MODE_BAKED: return BakedNormalAttr(w);
                default: return new float4(0.5f, 0.5f, 0.85f, 1f);
            }
        }

        float4 SplatNormalAttr(float2 w)
        {
            float e = terrainDataSize.x / (heightRes - 1);
            float hL = HeightAt(w + new float2(-e, 0));
            float hR = HeightAt(w + new float2(e, 0));
            float hD = HeightAt(w + new float2(0, -e));
            float hU = HeightAt(w + new float2(0, e));
            float3 heightNormal = math.normalize(new float3(hL - hR, hD - hU, 2f * e));

            int aBase = AlphaBase(w);
            float3 detailNormal = float3.zero;
            float smoothness = 0, occlusion = 0;
            for (int l = 0; l < 8 && l < alphaLayers; ++l)
            {
                float weight = alphamaps[aBase + l];
                if (weight < 0.003f) continue;
                float4 packedNormal = LayerTexel(l, w, 1, new float4(0.5f, 0.5f, 1f, 1f));
                float2 xy = (packedNormal.xy * 2f - 1f) * LayerNormalScale(l);
                float z = math.sqrt(math.saturate(1f - math.dot(xy, xy)));
                detailNormal += new float3(xy, z) * weight;

                float4 diffuse = LayerTexel(l, w, 0, new float4(0.5f, 0.5f, 0.5f, 1f));
                float defaultSmoothness = diffuse.w * LayerSmoothness(l);
                float4 mask = LayerTexel(l, w, 2, new float4(LayerMetallic(l), 1f, 0f, defaultSmoothness));
                smoothness += mask.w * weight;
                occlusion += mask.y * weight;
            }
            detailNormal.z += 1e-5f;
            detailNormal = math.normalize(detailNormal);
            float3 combined = math.normalize(new float3(
                heightNormal.xy + detailNormal.xy,
                heightNormal.z * detailNormal.z));
            return new float4(combined.x * 0.5f + 0.5f, combined.y * 0.5f + 0.5f,
                1f - smoothness, occlusion);
        }

        float HeightAt(float2 w)
        {
            float u = math.saturate((w.x - terrainPos.x) / terrainDataSize.x) * (heightRes - 1);
            float v = math.saturate((w.y - terrainPos.z) / terrainDataSize.z) * (heightRes - 1);
            int x0 = math.min(heightRes - 2, (int)u);
            int y0 = math.min(heightRes - 2, (int)v);
            float tx = u - x0, ty = v - y0;
            float h00 = heights[y0 * heightRes + x0];
            float h10 = heights[y0 * heightRes + x0 + 1];
            float h01 = heights[(y0 + 1) * heightRes + x0];
            float h11 = heights[(y0 + 1) * heightRes + x0 + 1];
            return math.lerp(math.lerp(h00, h10, tx), math.lerp(h01, h11, tx), ty) * terrainDataSize.y;
        }

        float4 BakedNormalAttr(float2 w)
        {
            if (layer >= 8 && l8Normal.Length > 0)
                return Bilinear(l8Normal, l8Res, (w - bakedOriginXZ) / bakedTerrainSize);
            for (int i = 0; i < blockCount; ++i)
            {
                float4 r = blockRects[i];
                if (w.x >= r.x && w.y >= r.y && w.x < r.x + r.z && w.y < r.y + r.w)
                {
                    float2 uv = (w - r.xy) / r.zw;
                    return Bilinear(blockNormal, blockRes, uv, i);
                }
            }
            return Bilinear(l7Normal, l7Res, (w - bakedOriginXZ) / bakedTerrainSize);
        }
    }
}
