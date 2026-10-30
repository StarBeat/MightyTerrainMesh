namespace MightyTerrainMesh.EditorTools
{
    using System;
    using System.IO;
    using System.Linq;
    using UnityEditor;
    using UnityEngine;

    /// <summary>
    /// AMD Compressonator CLI（GPUOpen，BC7 多线程编码）的定位、下载与运行封装。
    /// 输入/输出均为 DDS（约定严格、无 TGA 原点歧义）：管线自写 128 字节头未压缩 DDS 输入，
    /// 输出剥 DDS 头（128/148 字节）取纯 BC7 块。进程复用 <see cref="MTAstcEncoder.RunProcess"/>。
    /// 定位顺序：随库内置 Assets/MightyTerrainMesh/Editor/Tools/compressonator/（Editor 目录，不打包）
    /// → Library/compressonator/ → Program Files\Compressonator* → PATH。
    /// </summary>
    public static class MTBc7Encoder
    {
        // Compressonator release 资产名可能随版本变化；下载失败时按提示手动安装即可（不影响烘焙，回退 Unity 压缩）
        private const string ReleasePage = "https://github.com/GPUOpen-Tools/compressonator/releases";
        private static readonly string[] DownloadUrls =
        {
            "https://github.com/GPUOpen-Tools/compressonator/releases/download/V4.6.0/compressonatorcli-4.6.0-win64.zip",
            "https://gh-proxy.com/https://github.com/GPUOpen-Tools/compressonator/releases/download/V4.6.0/compressonatorcli-4.6.0-win64.zip",
        };

        private static string _cached;

        /// <summary>随库内置目录（git 追踪，Editor 目录天然不打包）。</summary>
        public static string BundledDir =>
            Path.Combine("Assets", "MightyTerrainMesh", "Editor", "Tools", "compressonator");

        public static string LocalDir => Path.Combine("Library", "compressonator");

        /// <summary>定位 CompressonatorCLI 可执行文件。找不到返回 null。</summary>
        public static string Locate()
        {
            if (_cached != null) return _cached.Length > 0 ? _cached : null;
            string found = FindIn(BundledDir) ?? FindIn(LocalDir) ?? FindInProgramFiles() ?? FindOnPath();
            _cached = found ?? "";
            return found;
        }

        static string FindIn(string dir)
        {
            if (!Directory.Exists(dir)) return null;
            try
            {
                return Directory.GetFiles(dir, "CompressonatorCLI*", SearchOption.AllDirectories)
                    .Where(IsExecutable)
                    .FirstOrDefault();
            }
            catch (Exception) { return null; }
        }

        static string FindInProgramFiles()
        {
            foreach (var pf in new[]
            {
                Environment.GetEnvironmentVariable("ProgramFiles"),
                Environment.GetEnvironmentVariable("ProgramFiles(x86)"),
            })
            {
                if (string.IsNullOrEmpty(pf) || !Directory.Exists(pf)) continue;
                try
                {
                    foreach (var d in Directory.GetDirectories(pf, "Compressonator*"))
                    {
                        var hit = FindIn(d);
                        if (hit != null) return hit;
                    }
                }
                catch (Exception) { /* 无权限等，跳过 */ }
            }
            return null;
        }

        static string FindOnPath()
        {
            var pathEnv = Environment.GetEnvironmentVariable("PATH");
            if (string.IsNullOrEmpty(pathEnv)) return null;
            foreach (var d in pathEnv.Split(Path.PathSeparator))
            {
                if (string.IsNullOrEmpty(d)) continue;
                var hit = FindIn(d);
                if (hit != null) return hit;
            }
            return null;
        }

        static bool IsExecutable(string p)
        {
            string ext = Path.GetExtension(p).ToLowerInvariant();
            return ext == ".exe" || ext == "";
        }

        [MenuItem("MightyTerrainMesh/下载 Compressonator（BC7 编码器）")]
        public static void DownloadEncoder()
        {
            Directory.CreateDirectory(LocalDir);
            string zip = Path.Combine(LocalDir, "compressonator.zip");
            Exception lastErr = null;
            bool ok = false;
            foreach (var url in DownloadUrls)
            {
                try
                {
                    using (var wc = new System.Net.WebClient())
                    {
                        Debug.Log("[BC7] 下载中: " + url);
                        wc.DownloadFile(url, zip);
                    }
                    var head = File.ReadAllBytes(zip);
                    if (head.Length < 4 || head[0] != 'P' || head[1] != 'K')
                        throw new Exception("下载内容不是 zip（可能是镜像错误页或资产名已变更）");
                    ok = true;
                    break;
                }
                catch (Exception e) { lastErr = e; }
            }
            if (!ok)
            {
                try { if (File.Exists(zip)) File.Delete(zip); } catch (Exception) { }
                Debug.LogError($"[BC7] 全部下载源失败（最后错误：{lastErr?.Message}）\n" +
                               $"请从 {ReleasePage} 手动下载安装，或把 CompressonatorCLI.exe（含同目录 DLL）放到 {Path.GetFullPath(LocalDir)}");
                return;
            }
            try
            {
                System.IO.Compression.ZipFile.ExtractToDirectory(zip, LocalDir);
                File.Delete(zip);
                _cached = null;
                string exe = Locate();
                if (exe != null) Debug.Log("[BC7] 就绪: " + Path.GetFullPath(exe));
                else Debug.LogWarning("[BC7] 解压后未找到 CompressonatorCLI，请检查 " + Path.GetFullPath(LocalDir));
            }
            catch (Exception e)
            {
                Debug.LogError($"[BC7] 解压失败：{e.Message}\n请手动解压 {zip} 到 {Path.GetFullPath(LocalDir)}");
            }
        }

        /// <summary>压缩一张未压缩 DDS 为 BC7 DDS。4.6 起已移除 -numthreads（线程数自动），jobs 参数保留仅为接口一致。</summary>
        public static bool Run(string inputDds, string outputDds, int jobs, out string error)
        {
            string exe = Locate();
            if (exe == null) { error = "CompressonatorCLI 不存在"; return false; }
            return MTAstcEncoder.RunProcess(exe,
                $"-fd BC7 \"{inputDds}\" \"{outputDds}\"",
                outputDds, out error);
        }
    }
}
