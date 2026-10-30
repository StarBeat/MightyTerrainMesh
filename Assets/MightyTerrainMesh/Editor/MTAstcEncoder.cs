namespace MightyTerrainMesh.EditorTools
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using UnityEditor;
    using UnityEngine;

    /// <summary>
    /// astcenc 官方命令行编码器（ARM astc-encoder，Apache 2.0）的定位、下载与进程封装。
    /// Unity 内置 EditorUtility.CompressTexture 为主线程单线程，是瓦片烘焙的最大瓶颈；
    /// astcenc 多线程 + SIMD，外部进程池可并行压缩多片瓦片（同等质量约 10×）。
    /// 定位顺序：随库内置 Assets/MightyTerrainMesh/Editor/Tools/astcenc/（Editor 目录，不打包）
    /// → Library/astcenc/（菜单下载的目标位置）→ PATH。找不到时烘焙回退 Unity 内置压缩。
    /// </summary>
    public static class MTAstcEncoder
    {
        private const string ReleasePath =
            "ARM-software/astc-encoder/releases/download/5.7.0/astcenc-5.7.0-windows-x64.zip";
        // 直连 GitHub 失败时依次尝试镜像
        private static readonly string[] DownloadUrls =
        {
            "https://github.com/" + ReleasePath,
            "https://gh-proxy.com/https://github.com/" + ReleasePath,
            "https://ghproxy.net/https://github.com/" + ReleasePath,
        };

        private static string _cached; // null=未探测；""=探测过但不存在
        private static readonly List<System.Diagnostics.Process> _procs = new List<System.Diagnostics.Process>();

        /// <summary>随库内置目录（git 追踪，Editor 目录天然不打包）。</summary>
        public static string BundledDir =>
            Path.Combine("Assets", "MightyTerrainMesh", "Editor", "Tools", "astcenc");

        /// <summary>本地 astcenc 目录（菜单下载解压位置；Library 不入库，按机器隔离）。</summary>
        public static string LocalDir => Path.Combine("Library", "astcenc");

        /// <summary>定位编码器可执行文件（优先 AVX2 &gt; SSE4.1 &gt; SSE2）。找不到返回 null。</summary>
        public static string Locate()
        {
            if (_cached != null) return _cached.Length > 0 ? _cached : null;
            string found = FindIn(BundledDir) ?? FindIn(LocalDir) ?? FindOnPath();
            _cached = found ?? "";
            if (found == null)
                Debug.LogWarning("[astcenc] 未找到编码器（内置 Tools/astcenc、Library/astcenc 或 PATH），" +
                                 "可用菜单 MightyTerrainMesh/下载 astcenc 编码器 自动下载；烘焙将回退 Unity 内置压缩");
            return found;
        }

        static string FindIn(string dir)
        {
            if (!Directory.Exists(dir)) return null;
            return Directory.GetFiles(dir, "astcenc*", SearchOption.AllDirectories)
                .Where(IsExecutable)
                .OrderByDescending(IsaScore)
                .FirstOrDefault();
        }

        static string FindOnPath()
        {
            var pathEnv = Environment.GetEnvironmentVariable("PATH");
            if (string.IsNullOrEmpty(pathEnv)) return null;
            foreach (var d in pathEnv.Split(Path.PathSeparator))
            {
                if (string.IsNullOrEmpty(d) || !Directory.Exists(d)) continue;
                try
                {
                    var hit = Directory.GetFiles(d, "astcenc*")
                        .Where(IsExecutable)
                        .OrderByDescending(IsaScore)
                        .FirstOrDefault();
                    if (hit != null) return hit;
                }
                catch (Exception) { /* 无权限等，跳过 */ }
            }
            return null;
        }

        static bool IsExecutable(string p)
        {
            string ext = Path.GetExtension(p).ToLowerInvariant();
            return ext == ".exe" || ext == ""; // Windows 用 .exe；无扩展名兼容类 Unix 放置
        }

        static int IsaScore(string p)
        {
            string n = Path.GetFileNameWithoutExtension(p).ToLowerInvariant();
            return n.Contains("avx2") ? 4 : n.Contains("sse4") ? 3 : n.Contains("sse2") ? 2 : 1;
        }

        [MenuItem("MightyTerrainMesh/下载 astcenc 编码器（GitHub，约 6MB）")]
        public static void DownloadEncoder()
        {
            Directory.CreateDirectory(LocalDir);
            string zip = Path.Combine(LocalDir, "astcenc.zip");
            Exception lastErr = null;
            bool ok = false;
            foreach (var url in DownloadUrls)
            {
                try
                {
                    using (var wc = new System.Net.WebClient())
                    {
                        Debug.Log("[astcenc] 下载中: " + url);
                        wc.DownloadFile(url, zip);
                    }
                    // 镜像故障时常返回小体积错误页而非报错，校验 zip 头（PK）
                    var head = File.ReadAllBytes(zip);
                    if (head.Length < 4 || head[0] != 'P' || head[1] != 'K')
                        throw new Exception("下载内容不是 zip（可能是镜像错误页）");
                    ok = true;
                    break;
                }
                catch (Exception e) { lastErr = e; }
            }
            if (!ok)
            {
                try { if (File.Exists(zip)) File.Delete(zip); } catch (Exception) { }
                Debug.LogError($"[astcenc] 全部下载源失败（最后错误：{lastErr?.Message}）\n" +
                               $"请手动下载 https://github.com/{ReleasePath} 并解压到 {Path.GetFullPath(LocalDir)}");
                return;
            }
            try
            {
                System.IO.Compression.ZipFile.ExtractToDirectory(zip, LocalDir);
                File.Delete(zip);
                _cached = null; // 重新探测
                string exe = Locate();
                if (exe != null) Debug.Log("[astcenc] 就绪: " + Path.GetFullPath(exe));
                else Debug.LogWarning("[astcenc] 解压后未找到可执行文件，请检查 " + Path.GetFullPath(LocalDir));
            }
            catch (Exception e)
            {
                Debug.LogError($"[astcenc] 解压失败：{e.Message}\n请手动解压 {zip} 到 {Path.GetFullPath(LocalDir)}");
            }
        }

        /// <summary>
        /// 压缩一张 TGA（32bpp BGRA，左下原点——与 Unity raw 行序一致）为 .astc 文件。
        /// block 例 "6x6"/"5x5"；srgb=true 用 -cl（sRGB 误差加权，albedo 用），false 用 -cL（linear，normal 用）。
        /// </summary>
        public static bool Run(string inputTga, string outputAstc, string block, bool srgb,
            TextureCompressionQuality quality, int jobs, out string error)
        {
            string exe = Locate();
            if (exe == null) { error = "astcenc 不存在"; return false; }
            string q = quality == TextureCompressionQuality.Best ? "-thorough"
                     : quality == TextureCompressionQuality.Normal ? "-medium" : "-fast";
            return RunProcess(exe,
                $"{(srgb ? "-cl" : "-cL")} \"{inputTga}\" \"{outputAstc}\" {block} {q} -j {jobs} -silent",
                outputAstc, out error);
        }

        /// <summary>
        /// 运行外部编码进程（各类编码器共享）：重定向输出（小输出假设），登记全局杀进程表，
        /// 以退出码 + 输出文件存在判定成败。进程注册表跨编码器共享，KillAllProcesses 一并清理。
        /// </summary>
        public static bool RunProcess(string exe, string arguments, string expectOutFile, out string error)
        {
            error = null;
            var psi = new System.Diagnostics.ProcessStartInfo
            {
                FileName = exe,
                Arguments = arguments,
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
            };
            System.Diagnostics.Process p = null;
            try
            {
                p = System.Diagnostics.Process.Start(psi);
                if (p == null) { error = "进程启动失败"; return false; }
                lock (_procs) _procs.Add(p);
                // 编码器均带静默/小输出，顺序读两个流不会死锁
                string so = p.StandardOutput.ReadToEnd();
                string se = p.StandardError.ReadToEnd();
                p.WaitForExit();
                if (p.ExitCode != 0 || (expectOutFile != null && !File.Exists(expectOutFile)))
                {
                    error = $"exit={p.ExitCode} {(so + " " + se).Trim()}";
                    return false;
                }
                return true;
            }
            catch (Exception e)
            {
                error = e.Message;
                return false;
            }
            finally
            {
                if (p != null)
                {
                    lock (_procs) _procs.Remove(p);
                    p.Dispose();
                }
            }
        }

        /// <summary>杀掉所有在途编码进程（烘焙取消/域重载时调用）。</summary>
        public static void KillAllProcesses()
        {
            lock (_procs)
            {
                foreach (var p in _procs)
                {
                    try { if (!p.HasExited) p.Kill(); } catch (Exception) { /* 已退出 */ }
                }
                _procs.Clear();
            }
        }
    }
}
