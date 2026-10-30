namespace MightyTerrainMesh.EditorTools
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using UnityEditor;
    using UnityEngine;

    /// <summary>
    /// Clipmap 烘焙产物（v3：PNG 源图）的导入策略，首次导入与手动重导均生效：
    /// 目录（瓦片则为父目录）内含 v3 manifest.json 的 l7_/l8_/l{j}_{x}_{z}_{albedo|normal} PNG
    /// 一律按 manifest 的 formatMode / importQuality 配置 TextureImporter：
    /// - NPOT 不缩放（2040 必须原样保留，否则环面/瓦片 1:1 契约破坏）；albedo=sRGB / normal=linear；
    /// - 瓦片 Clamp 无 mip；l8 完整 mip 链 + Trilinear；l7 开 Read/Write（Job 回退路径 CPU 采样）；
    /// - 平台覆盖：Auto → Standalone=BC7、Android/iOS=ASTC 6x6(albedo)/5x5(normal)；
    ///   ASTC/BC7 = 全平台强制；RGBA32 = 不压缩（联调）。
    /// 设置写入各 PNG 的 .meta：源图不入库时（.gitignore），每台机器重烘后导入仍自动正确。
    /// </summary>
    public class MTClipmapBakedImporter : AssetPostprocessor
    {
        private class Cfg
        {
            public ClipmapManifest manifest;
            public DateTime stamp;
        }

        private static readonly Dictionary<string, Cfg> _cache = new Dictionary<string, Cfg>();

        void OnPreprocessTexture()
        {
            string path = assetImporter.assetPath.Replace('\\', '/');
            if (!path.EndsWith(".png", StringComparison.OrdinalIgnoreCase)) return;
            string file = Path.GetFileNameWithoutExtension(path);
            bool isTile = IsTileName(file);
            bool isL7 = file.StartsWith("l7_");
            bool isL8 = file.StartsWith("l8_");
            if (!isTile && !isL7 && !isL8) return;

            string dir = Path.GetDirectoryName(path).Replace('\\', '/');
            // 瓦片在 {root}/tiles/ 下，manifest 在父级；l7/l8 与 manifest 同级
            var m = FindManifest(isTile ? Path.GetDirectoryName(dir).Replace('\\', '/') : dir);
            if (m == null) return;

            bool albedo = file.EndsWith("_albedo");
            var mode = (MTTerrainBakePipeline.ClipmapBakeFormat)m.formatMode;
            int quality = Mathf.Clamp(m.importQuality, 0, 2); // TextureCompressionQuality: Fast/Normal/Best

            var ti = (TextureImporter)assetImporter;
            ti.textureType = TextureImporterType.Default;
            ti.npotScale = TextureImporterNPOTScale.None; // 2040 非 2 幂，默认 ToNearest 会改成 2048
            ti.sRGBTexture = albedo;
            ti.alphaIsTransparency = false;
            ti.mipmapEnabled = isL8;
            ti.isReadable = isL7;
            ti.wrapMode = TextureWrapMode.Clamp;
            ti.filterMode = isL8 ? FilterMode.Trilinear : FilterMode.Bilinear;
            if (mode == MTTerrainBakePipeline.ClipmapBakeFormat.RGBA32)
            {
                ti.textureCompression = TextureImporterCompression.Uncompressed;
                return; // 联调模式：不设平台覆盖，各平台均为 RGBA32
            }
            ti.textureCompression = TextureImporterCompression.Compressed;
            Override(ti, "Standalone", FormatFor(mode, true, albedo), quality);
            Override(ti, "Android", FormatFor(mode, false, albedo), quality);
            Override(ti, "iPhone", FormatFor(mode, false, albedo), quality);
        }

        private static void Override(TextureImporter ti, string platform,
            TextureImporterFormat format, int quality)
        {
            var ps = ti.GetPlatformTextureSettings(platform);
            ps.overridden = true;
            ps.maxTextureSize = 8192;
            ps.format = format;
            ps.compressionQuality = quality;
            ti.SetPlatformTextureSettings(ps);
        }

        private static TextureImporterFormat FormatFor(
            MTTerrainBakePipeline.ClipmapBakeFormat mode, bool standalone, bool albedo)
        {
            switch (mode)
            {
                case MTTerrainBakePipeline.ClipmapBakeFormat.BC7:
                    return TextureImporterFormat.BC7;
                case MTTerrainBakePipeline.ClipmapBakeFormat.Auto:
                    return standalone ? TextureImporterFormat.BC7
                        : albedo ? TextureImporterFormat.ASTC_6x6 : TextureImporterFormat.ASTC_5x5;
                default: // ASTC 全平台
                    return albedo ? TextureImporterFormat.ASTC_6x6 : TextureImporterFormat.ASTC_5x5;
            }
        }

        /// <summary>tiles/ 瓦片文件名：l{level}_{tx}_{tz}_{albedo|normal}。</summary>
        private static bool IsTileName(string f)
        {
            var p = f.Split('_');
            return p.Length == 4 && p[0].Length > 1 && p[0][0] == 'l'
                && int.TryParse(p[0].Substring(1), out _)
                && int.TryParse(p[1], out _) && int.TryParse(p[2], out _)
                && (p[3] == "albedo" || p[3] == "normal");
        }

        /// <summary>读目录下的 manifest.json（按写入时间缓存，重烘后自动失效）；非 v3 返回 null。</summary>
        private static ClipmapManifest FindManifest(string dir)
        {
            if (string.IsNullOrEmpty(dir)) return null;
            string mp = dir + "/manifest.json";
            if (!File.Exists(mp)) return null;
            var stamp = File.GetLastWriteTimeUtc(mp);
            if (_cache.TryGetValue(dir, out var c) && c.stamp == stamp) return c.manifest;
            ClipmapManifest m = null;
            try
            {
                m = JsonUtility.FromJson<ClipmapManifest>(File.ReadAllText(mp));
                if (m != null && m.textureDataVersion != ClipmapManifest.CurrentTextureDataVersion) m = null;
            }
            catch (Exception) { m = null; }
            _cache[dir] = new Cfg { manifest = m, stamp = stamp };
            return m;
        }
    }
}
