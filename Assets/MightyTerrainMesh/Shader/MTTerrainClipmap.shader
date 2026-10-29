// ClipmapVT shader：按 ClipmapVT.md v2.5 §6 采样规范实现（固定图集寻址 + 动态层数 K）
// - L0–L(K-1)：Texture2DArray 环面层（Repeat，无 Mip）；uv = frac(localXZ / footprint_k)，origin 仅 CPU 更新簿记
// - L_K（静态层）：独立 Texture2D（Clamp + 完整 Mip 链 + Trilinear/Aniso 2x）；globalUV 连续无折叠，硬件导数自动选 mip
// - 选层：max(像素足迹 F → log2(F·layerRes/baseFootprint)，距离下界 ceil(log2(2r/baseFootprint)))，
//   clamp 到 [0, K]（_ClipLevelCount）；距离下界保证所层环面窗口覆盖采样点（r ≤ R_k）
// - 过渡：外环混合带（_ClipBlendStart，默认 25%，带宽门闸收窄至 20%），法线混合后 renormalize（§6.3）
Shader "MT/TerrainClipmap"
{
	Properties{
		[HideInInspector]_Diffuse("unused", 2D) = "grey" {}
		[HideInInspector]_Normal("unused", 2D) = "grey" {}
	}
	SubShader
	{
		Tags { "Queue" = "Geometry-100" "RenderType" = "Opaque" "RenderPipeline" = "UniversalPipeline" "IgnoreProjector" = "False"}

		Pass
		{
			Name "ForwardLit"
			Tags { "LightMode" = "UniversalForward" }
			HLSLPROGRAM
			#pragma prefer_hlslcc gles
			#pragma exclude_renderers d3d11_9x
			#pragma target 3.5

			#pragma vertex vert
			#pragma fragment frag

			#pragma multi_compile _ _MAIN_LIGHT_SHADOWS
			#pragma multi_compile _ _MAIN_LIGHT_SHADOWS_CASCADE
			#pragma multi_compile _ _ADDITIONAL_LIGHTS_VERTEX _ADDITIONAL_LIGHTS
			#pragma multi_compile _ _ADDITIONAL_LIGHT_SHADOWS
			#pragma multi_compile _ _SHADOWS_SOFT
			#pragma multi_compile _ _MIXED_LIGHTING_SUBTRACTIVE

			#pragma multi_compile _ DIRLIGHTMAP_COMBINED
			#pragma multi_compile _ LIGHTMAP_ON
			#pragma multi_compile_fog
			#pragma multi_compile_instancing

			#include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
			#include "Packages/com.unity.render-pipelines.core/ShaderLibrary/CommonMaterial.hlsl"
			#include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/SurfaceInput.hlsl"
			#include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"

			// ---- Clipmap 全局输入（由 MTClipmapController 设置）----
			TEXTURE2D_ARRAY(_ClipAlbedoArray);  SAMPLER(sampler_ClipAlbedoArray);   // L0–L7，Repeat，无 Mip
			TEXTURE2D_ARRAY(_ClipNormalArray);  SAMPLER(sampler_ClipNormalArray);   // §4.1：RG=法线XY, B=粗糙度, A=AO
			TEXTURE2D(_ClipAlbedoL8);           SAMPLER(sampler_ClipAlbedoL8);      // Clamp + Mip + Aniso
			TEXTURE2D(_ClipNormalL8);           SAMPLER(sampler_ClipNormalL8);

			float4 _TerrainOrigin;      // xz = 地形世界原点
			float  _TerrainSize;        // 默认 8192
			float  _ClipBaseFootprint;  // 32
			float  _ClipLayerRes;       // 层分辨率（选层偏移与之联动）
			int    _ClipValidMask;      // §6.5 驻留掩码：bit k = 环面层 k 有效，bit K = 静态层恒有效
			int    _ClipLevelCount;     // §3.1.1 动态层数 K = log2(S)−5（静态层 = L_K）
			float  _ClipBlendStart;     // §6.2/§8.1 外环混合带起点（占 footprint 半径比例，默认 0.75，门闸降级①调至 0.80）
			float3 _ClipViewCameraPos;  // 固定的主相机位置，Scene 窗口不使用 Scene 相机替代
			int    _ClipDebugLOD;       // 0：关闭；1：请求层（足迹与距离取 max）；2：包含驻留回退
			int    _ClipPackedMaterial; // 1：Albedo.A=Metallic；0：旧烘焙固定非金属

			struct Attributes
			{
				float4 positionOS : POSITION;
				float3 normalOS : NORMAL;
				float2 texcoord : TEXCOORD0;
				float2 texcoord1 : TEXCOORD1;
			};

			struct Varyings
			{
				float4 uvMainAndLM              : TEXCOORD0; // zw: lightmap
				float3 normalWS                 : TEXCOORD1;
				float3 positionWS               : TEXCOORD2;
				half3 vertexSH                  : TEXCOORD3;
				half4 fogFactorAndVertexLight   : TEXCOORD4; // x: fogFactor, yzw: vertex light
#if defined(REQUIRES_VERTEX_SHADOW_COORD_INTERPOLATOR)
				float4 shadowCoord              : TEXCOORD6;
#endif
				float4 clipPos                  : SV_POSITION;
			};

			Varyings vert(Attributes v)
			{
				Varyings o = (Varyings)0;
				VertexPositionInputs posInputs = GetVertexPositionInputs(v.positionOS.xyz);

				o.uvMainAndLM.xy = v.texcoord;
				o.uvMainAndLM.zw = v.texcoord1 * unity_LightmapST.xy + unity_LightmapST.zw;
				o.normalWS = TransformObjectToWorldNormal(v.normalOS);
				o.positionWS = posInputs.positionWS;
				o.vertexSH = SampleSH(o.normalWS);
				o.fogFactorAndVertexLight.x = ComputeFogFactor(posInputs.positionCS.z);
				o.fogFactorAndVertexLight.yzw = VertexLighting(posInputs.positionWS, o.normalWS);
				o.clipPos = posInputs.positionCS;
#if defined(REQUIRES_VERTEX_SHADOW_COORD_INTERPOLATOR)
				o.shadowCoord = GetShadowCoord(posInputs);
#endif
				return o;
			}

			float ClipFootprint(int k) { return _ClipBaseFootprint * exp2((float)k); } // 32·2^k 米

			// §6.5：从 k 向上找第一个已驻留层（瓦片未到位时采样时降级，支持非连续失效；bit K 恒有效保证收敛）
			int FinestValidLevel(int k)
			{
				while (k < _ClipLevelCount && ((_ClipValidMask >> k) & 1) == 0) ++k;
				return k;
			}

			half3 LodDebugColor(int level)
			{
				switch (level)
				{
					case 0: return half3(1.0h, 0.1h, 0.1h); // 红
					case 1: return half3(1.0h, 0.5h, 0.0h); // 橙
					case 2: return half3(1.0h, 1.0h, 0.0h); // 黄
					case 3: return half3(0.1h, 0.9h, 0.1h); // 绿
					case 4: return half3(0.0h, 0.9h, 0.9h); // 青
					case 5: return half3(0.1h, 0.3h, 1.0h); // 蓝
					case 6: return half3(0.6h, 0.2h, 1.0h); // 紫
					case 7: return half3(1.0h, 0.1h, 0.7h); // 品红
					default: return half3(1.0h, 1.0h, 1.0h); // 静态层：白
				}
			}

			// 环面层采样：uv = frac(localXZ / footprint_k)，显式 wrap（Repeat 等价）
			// 固定图集寻址（v2.3.2）：origin 不参与采样地址，仅用于 CPU 侧更新簿记
			float4 SampleClipAlbedo(int k, float2 localXZ)
			{
				float fp = ClipFootprint(k);
				float2 uv = frac(localXZ / fp);
				return SAMPLE_TEXTURE2D_ARRAY(_ClipAlbedoArray, sampler_ClipAlbedoArray, uv, k);
			}
			float4 SampleClipNormal(int k, float2 localXZ)
			{
				float fp = ClipFootprint(k);
				float2 uv = frac(localXZ / fp);
				return SAMPLE_TEXTURE2D_ARRAY(_ClipNormalArray, sampler_ClipNormalArray, uv, k);
			}
			// L8：全局连续 UV，硬件导数选 mip（trilinear + aniso），无需显式 LOD/bias
			float4 SampleL8Albedo(float2 globalUV)
			{
				return SAMPLE_TEXTURE2D(_ClipAlbedoL8, sampler_ClipAlbedoL8, globalUV);
			}
			float4 SampleL8Normal(float2 globalUV)
			{
				return SAMPLE_TEXTURE2D(_ClipNormalL8, sampler_ClipNormalL8, globalUV);
			}

			// NormalAttr 解包：RG = 法线 XY（TS，x→worldX，y→worldZ），Z 重建
			half3 UnpackNormalTS(float4 na)
			{
				half2 xy = na.rg * 2.0h - 1.0h;
				half z = sqrt(saturate(1.0h - dot(xy, xy)));
				return half3(xy, z);
			}

			half4 frag(Varyings IN) : SV_TARGET
			{
				// ---- §6.2 解析 LOD（导数只用全局连续 UV）----
				float2 localXZ  = IN.positionWS.xz - _TerrainOrigin.xz;
				float2 globalUV = localXZ / _TerrainSize;
				float2 camLocalXZ = _ClipViewCameraPos.xz - _TerrainOrigin.xz;
				float  r = length(localXZ - camLocalXZ);

				float2 ddxUV = ddx(globalUV) * _TerrainSize;   // 米/像素
				float2 ddyUV = ddy(globalUV) * _TerrainSize;
				float  F  = sqrt(max(dot(ddxUV, ddxUV), dot(ddyUV, ddyUV)));
				// texel_k = baseFootprint·2^k / layerRes → lodF = log2(F / texel_0)
				float  lodF   = log2(max(F, 1e-5) * _ClipLayerRes / _ClipBaseFootprint);
				// 距离下界：层 k 环面窗口只覆盖 r ≤ R_k（R_k = fp_k/2）。俯视/高空相机下 F 不随 r 增长，
				// 仅按 lodF 选层会选出窗口覆盖不到该点的层，混合带采到 frac wrap 后的别处内容（位置错位）。
				int    ringK  = (int)ceil(log2(max(r * 2.0 / _ClipBaseFootprint, 1.0)));
				int    requestedLevel = clamp(max((int)floor(lodF), ringK), 0, _ClipLevelCount);
				int    level  = FinestValidLevel(requestedLevel); // §6.5：向上扫描到已驻留层

				[branch] if (_ClipDebugLOD != 0)
				{
					int debugLevel = _ClipDebugLOD == 2 ? level : requestedLevel;
					half3 debugColor = debugLevel >= _ClipLevelCount ? half3(1.0h, 1.0h, 1.0h) : LodDebugColor(debugLevel);
					return half4(debugColor, 1.0h);
				}

				// ---- §6.3 外环混合带（水平 XZ 距离）----

				float4 albedo4;
				float4 na;   // 混合后的 NormalAttr（§4.1：RG=法线XY，B=粗糙度，A=AO）
				half3  normalTS;
				if (level >= _ClipLevelCount)
				{
					// k=K（静态层）：无混合带，内部 Mip 链处理 minification
					albedo4  = SampleL8Albedo(globalUV);
					na       = SampleL8Normal(globalUV);
					normalTS = UnpackNormalTS(na);
				}
				else
				{
					float Rk = ClipFootprint(level) * 0.5;
					float b = smoothstep(_ClipBlendStart * Rk, Rk, r); // 外环 25% → 粗层（带宽门闸收窄至 20%）
					float4 a0 = SampleClipAlbedo(level, localXZ);
					float4 n0 = SampleClipNormal(level, localXZ);
					// §8.1 带宽口径：b=0（层内 75% 区域）只采样本层（2 次 fetch）；仅混合带内加采相邻粗层（4 次）。
					// b=1（r > R_k）时本层 wrap fetch 仍执行、结果被 lerp 丢弃（§6.3：勿对该 fetch 做分支剔除假设）。
					[branch] if (b > 0.0)
					{
						int levelNext = FinestValidLevel(level + 1); // 混合目标同样需已驻留（可能跳到静态层）
						float4 a1, n1;
						if (levelNext >= _ClipLevelCount)
						{
							a1 = SampleL8Albedo(globalUV);
							n1 = SampleL8Normal(globalUV);
						}
						else
						{
							a1 = SampleClipAlbedo(levelNext, localXZ);
							n1 = SampleClipNormal(levelNext, localXZ);
						}
						albedo4   = lerp(a0, a1, b);
						na        = lerp(n0, n1, b);
						// §6.3：法线混合后必须 renormalize
						normalTS  = normalize(lerp(UnpackNormalTS(n0), UnpackNormalTS(n1), b));
					}
					else
					{
						albedo4  = a0;
						na       = n0;
						normalTS = UnpackNormalTS(n0);
					}
				}

				// ---- 解码：Albedo.A=Metallic；NormalAttr RG=法线XY，B=粗糙度，A=AO ----
				half3 albedo = albedo4.rgb;
				half roughness  = na.b;
				half occlusion  = na.a;
				half smoothness = 1.0h - roughness;
				half metallic   = albedo4.a * _ClipPackedMaterial;

				// ---- URP PBR（复用插件的切线框架构造）----
				float3 normalWS = SafeNormalize(IN.normalWS);
				float4 vertexTangent = float4(cross(float3(0, 0, 1), normalWS), 1.0);
				VertexNormalInputs normalInput = GetVertexNormalInputs(normalWS, vertexTangent);
				float3 bumpWS = TransformTangentToWorld(normalTS,
					half3x3(-normalInput.tangentWS, normalInput.bitangentWS, normalInput.normalWS));

				InputData inputData = (InputData)0;
				inputData.positionWS = IN.positionWS;
				inputData.normalWS = SafeNormalize(bumpWS);
				inputData.viewDirectionWS = SafeNormalize(GetCameraPositionWS() - IN.positionWS);
#if defined(REQUIRES_VERTEX_SHADOW_COORD_INTERPOLATOR)
				inputData.shadowCoord = IN.shadowCoord;
#elif defined(MAIN_LIGHT_CALCULATE_SHADOWS)
				inputData.shadowCoord = TransformWorldToShadowCoord(IN.positionWS);
#else
				inputData.shadowCoord = float4(0, 0, 0, 0);
#endif
				inputData.fogCoord = IN.fogFactorAndVertexLight.x;
				inputData.vertexLighting = IN.fogFactorAndVertexLight.yzw;
#if defined(LIGHTMAP_ON)
				inputData.bakedGI = SAMPLE_GI(IN.uvMainAndLM.zw, IN.vertexSH, inputData.normalWS);
#else
				inputData.bakedGI = IN.vertexSH;
#endif

				half4 color = UniversalFragmentPBR(inputData, albedo, metallic, half3(0.0h, 0.0h, 0.0h), smoothness, occlusion, half3(0, 0, 0), 1.0h);
				color.rgb = MixFog(color.rgb, inputData.fogCoord);
				return half4(color.rgb, 1.0h);
			}
			ENDHLSL
		}

		// 阴影/深度 pass：几何由地形网格携带，无需采样 Clipmap；
		// 缺失这些 pass 会导致地形不投影、不进相机深度/法线纹理（SSAO/DOF/软粒子失效）
		Pass
		{
			Name "ShadowCaster"
			Tags{ "LightMode" = "ShadowCaster" }
			ZWrite On
			ZTest LEqual
			ColorMask 0
			Cull Back

			HLSLPROGRAM
			#pragma exclude_renderers d3d11_9x
			#pragma target 3.5
			#pragma vertex ShadowVert
			#pragma fragment ShadowFrag
			#pragma multi_compile_vertex _ _CASTING_PUNCTUAL_LIGHT_SHADOW
			#pragma multi_compile_instancing

			#include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
			// core 14 将 LerpWhiteTo 移至 CommonMaterial.hlsl（core 12 在 Common.hlsl），本工程 URP 本地锁定 12.1.2 与 core 14 混用，Shadows.hlsl:264 依赖该符号
			#include "Packages/com.unity.render-pipelines.core/ShaderLibrary/CommonMaterial.hlsl"
			#include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Shadows.hlsl"

			float3 _LightDirection;
			float3 _LightPosition;

			struct ShadowAttributes
			{
				float4 positionOS : POSITION;
				float3 normalOS : NORMAL;
				UNITY_VERTEX_INPUT_INSTANCE_ID
			};

			struct ShadowVaryings
			{
				float4 positionCS : SV_POSITION;
			};

			ShadowVaryings ShadowVert(ShadowAttributes input)
			{
				ShadowVaryings o = (ShadowVaryings)0;
				UNITY_SETUP_INSTANCE_ID(input);
				float3 positionWS = TransformObjectToWorld(input.positionOS.xyz);
				float3 normalWS = TransformObjectToWorldNormal(input.normalOS);
			#if _CASTING_PUNCTUAL_LIGHT_SHADOW
				float3 lightDirectionWS = normalize(_LightPosition - positionWS);
			#else
				float3 lightDirectionWS = _LightDirection;
			#endif
				float4 positionCS = TransformWorldToHClip(ApplyShadowBias(positionWS, normalWS, lightDirectionWS));
			#if UNITY_REVERSED_Z
				positionCS.z = min(positionCS.z, UNITY_NEAR_CLIP_VALUE * positionCS.w);
			#else
				positionCS.z = max(positionCS.z, UNITY_NEAR_CLIP_VALUE * positionCS.w);
			#endif
				o.positionCS = positionCS;
				return o;
			}

			half4 ShadowFrag(ShadowVaryings input) : SV_TARGET
			{
				return 0;
			}
			ENDHLSL
		}

		Pass
		{
			Name "DepthOnly"
			Tags{ "LightMode" = "DepthOnly" }
			ZWrite On
			ZTest LEqual
			ColorMask 0
			Cull Back

			HLSLPROGRAM
			#pragma exclude_renderers d3d11_9x
			#pragma target 3.5
			#pragma vertex DepthVert
			#pragma fragment DepthFrag
			#pragma multi_compile_instancing

			#include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

			struct DepthAttributes
			{
				float4 positionOS : POSITION;
				UNITY_VERTEX_INPUT_INSTANCE_ID
			};

			struct DepthVaryings
			{
				float4 positionCS : SV_POSITION;
			};

			DepthVaryings DepthVert(DepthAttributes input)
			{
				DepthVaryings o = (DepthVaryings)0;
				UNITY_SETUP_INSTANCE_ID(input);
				o.positionCS = TransformObjectToHClip(input.positionOS.xyz);
				return o;
			}

			half4 DepthFrag(DepthVaryings input) : SV_TARGET
			{
				return 0;
			}
			ENDHLSL
		}

		Pass
		{
			Name "DepthNormals"
			Tags{ "LightMode" = "DepthNormals" }
			ZWrite On
			ZTest LEqual
			Cull Back

			HLSLPROGRAM
			#pragma exclude_renderers d3d11_9x
			#pragma target 3.5
			#pragma vertex DepthNormalsVert
			#pragma fragment DepthNormalsFrag
			#pragma multi_compile_instancing

			#include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
			#include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Packing.hlsl"

			struct DepthNormalsAttributes
			{
				float4 positionOS : POSITION;
				float3 normalOS : NORMAL;
				UNITY_VERTEX_INPUT_INSTANCE_ID
			};

			struct DepthNormalsVaryings
			{
				float4 positionCS : SV_POSITION;
				float3 normalWS : TEXCOORD0;
			};

			DepthNormalsVaryings DepthNormalsVert(DepthNormalsAttributes input)
			{
				DepthNormalsVaryings o = (DepthNormalsVaryings)0;
				UNITY_SETUP_INSTANCE_ID(input);
				o.positionCS = TransformObjectToHClip(input.positionOS.xyz);
				o.normalWS = TransformObjectToWorldNormal(input.normalOS);
				return o;
			}

			half4 DepthNormalsFrag(DepthNormalsVaryings input) : SV_TARGET
			{
				// 几何法线即可满足 SSAO/深度法线重建（Clipmap 法线属于着色细节）
				float3 normalWS = SafeNormalize(input.normalWS);
				return half4(PackNormalOctRectEncode(TransformWorldToViewDir(normalWS, true)), 0.0, 0.0);
			}
			ENDHLSL
		}
	}
	Fallback "Hidden/Universal Render Pipeline/FallbackError"
}
