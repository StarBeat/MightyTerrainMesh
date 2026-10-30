namespace MightyTerrainMesh
{
    using System;
    using System.IO;
    using System.Collections.Generic;
    using UnityEngine;
    using UnityEngine.Rendering;

    internal class MTRuntimeMeshPool
    {
        class MeshStreamCache
        {
            public string Path { get; private set; }
            public bool Obseleted
            {
                get { return offsets == null || usedCount == offsets.Length; }
            }
            private MemoryStream memStream;
            private int[] offsets;
            private int usedCount = 0;
            public MeshStreamCache(string path, int pack, byte[] data)
            {
                Path = path;
                memStream = new MemoryStream(data);
                offsets = new int[pack];
                for (int i = 0; i < pack; ++i)
                {
                    offsets[i] = MTFileUtils.ReadInt(memStream);
                }
            }
            public MTRenderMesh GetMesh(int meshId)
            {
                int offset_stride = meshId % offsets.Length;
                int offset = offsets[offset_stride];
                var rm = new MTRenderMesh();
                memStream.Position = offset;
                MTMeshUtils.Deserialize(memStream, rm);
                ++usedCount;
                return rm;
            }
            public void Clear()
            {
                memStream.Close();
            }
        }
        private MTData rawData;
        private class MeshEntry
        {
            public MTRenderMesh mesh;
            public int lastUsedFrame;
        }
        private Dictionary<int, MeshEntry> parsedMesh = new Dictionary<int, MeshEntry>();
        //memory data cache, once all data parsed into mesh, it will be destroied
        private Dictionary<string, MeshStreamCache> dataStreams = new Dictionary<string, MeshStreamCache>();
        private IMeshDataLoader loader;
        private readonly Func<int, bool> inUse; // 活跃 patch 判定（驱逐保护）
        private readonly int capacity;
        public MTRuntimeMeshPool(MTData data, IMeshDataLoader ld, Func<int, bool> inUsePred = null, int capacity = 1024)
        {
            rawData = data;
            loader = ld;
            inUse = inUsePred;
            this.capacity = Mathf.Max(64, capacity);
        }
        public MTRenderMesh PopMesh(int meshId)
        {
            if (!parsedMesh.ContainsKey(meshId) && meshId >= 0)
            {
                int startMeshId = meshId / rawData.MeshDataPack * rawData.MeshDataPack;
                var path = string.Format("{0}_{1}", rawData.MeshPrefix, startMeshId);
                if (!dataStreams.ContainsKey(path))
                {
                    var meshbytes = loader.LoadMeshData(path);
                    var cache = new MeshStreamCache(path, rawData.MeshDataPack, meshbytes);
                    dataStreams.Add(path, cache);
                }
                var streamCache = dataStreams[path];
                var rm = streamCache.GetMesh(meshId);
                parsedMesh.Add(meshId, new MeshEntry { mesh = rm, lastUsedFrame = Time.frameCount });
                if (parsedMesh.Count > capacity) EvictLRU(); // roaming 大地形防无限增长
                if (streamCache.Obseleted)
                {
                    dataStreams.Remove(streamCache.Path);
                    loader.UnloadAsset(streamCache.Path);
                    streamCache.Clear();
                }
            }
            if (parsedMesh.TryGetValue(meshId, out var e))
            {
                e.lastUsedFrame = Time.frameCount;
                return e.mesh;
            }
            return null;
        }
        /// <summary>淘汰最久未用的已解析 mesh（活跃 patch 跳过；全部活跃时暂时超支）。</summary>
        private void EvictLRU()
        {
            int victimKey = -1;
            int best = int.MaxValue;
            MeshEntry victim = null;
            foreach (var kv in parsedMesh)
            {
                if (inUse != null && inUse(kv.Key)) continue;
                if (kv.Value.lastUsedFrame < best) { best = kv.Value.lastUsedFrame; victim = kv.Value; victimKey = kv.Key; }
            }
            if (victim == null) return;
            victim.mesh.Clear();
            parsedMesh.Remove(victimKey);
        }
        public void Clear()
        {
            foreach (var cache in dataStreams.Values)
            {
                loader.UnloadAsset(cache.Path);
                cache.Clear();
            }
            dataStreams.Clear();
            foreach (var m in parsedMesh.Values)
            {
                m.mesh.Clear();
            }
            parsedMesh.Clear();
        }
    }

    public class MTLoader : MonoBehaviour
    {
        public MTData header;
        public MTLODPolicy lodPolicy;
        public Camera cullCamera;
        public GameObject VTCreatorGo;
        public bool receiveShadow = true;
        public float detailDrawDistance = 80;
        public bool showDebug = false;
        [Tooltip("已解析 mesh 的缓存上限（个，LRU 淘汰，活跃 patch 不驱逐）；roaming 大地形时防内存无限增长")]
        public int meshCacheCapacity = 1024;
        //
        private MTRuntimeMeshPool meshPool;
        private MTQuadTreeUtil quadtree;
        private MTHeightMap heightMap;
        private MTArray<MTQuadTreeNode> activeCmd;
        private MTArray<MTQuadTreeNode> deactiveCmd;
        private Dictionary<int, MTPooledRenderMesh> activeMeshes = new Dictionary<int, MTPooledRenderMesh>();
        private IVTCreator vtCreator;
        private MTDetailRenderer detailRenderer;
        private Matrix4x4 projM;
        private Matrix4x4 detailProjM;
        private Matrix4x4 prevWorld2Cam;
        private Plane[] detailCullPlanes = new Plane[6];
        private void ActiveMesh(MTQuadTreeNode node)
        {
            MTPooledRenderMesh patch = MTPooledRenderMesh.Pop();
            var m = meshPool.PopMesh(node.meshIdx);
            patch.Reset(header, vtCreator, m, transform.position);
            activeMeshes.Add(node.meshIdx, patch);
        }
        private void DeactiveMesh(MTQuadTreeNode node)
        {
            var p = activeMeshes[node.meshIdx];
            activeMeshes.Remove(node.meshIdx);
            MTPooledRenderMesh.Push(p);
        }
        private void Awake()
        {
            IMeshDataLoader loader = new MeshDataResLoader();
            quadtree = new MTQuadTreeUtil(header.TreeData.bytes, transform.position);
            heightMap = new MTHeightMap(quadtree.Bound, header.HeightmapResolution, header.HeightmapScale, header.HeightMap.bytes);
            activeCmd = new MTArray<MTQuadTreeNode>(quadtree.NodeCount);
            deactiveCmd = new MTArray<MTQuadTreeNode>(quadtree.NodeCount);
            meshPool = new MTRuntimeMeshPool(header, loader, idx => activeMeshes.ContainsKey(idx), meshCacheCapacity);
            vtCreator = VTCreatorGo.GetComponent<IVTCreator>();
            detailRenderer = new MTDetailRenderer(header, quadtree.Bound, receiveShadow);
            prevWorld2Cam = Matrix4x4.identity;
            projM = Matrix4x4.Perspective(cullCamera.fieldOfView, cullCamera.aspect, cullCamera.nearClipPlane, cullCamera.farClipPlane);
            detailProjM = Matrix4x4.Perspective(cullCamera.fieldOfView, cullCamera.aspect, cullCamera.nearClipPlane, detailDrawDistance);
            RenderPipelineManager.beginFrameRendering += OnFrameRendering;
        }
        void OnFrameRendering(ScriptableRenderContext context, Camera[] cameras)
        {
            if (quadtree != null && cullCamera != null)
            {
                Matrix4x4 world2Cam = cullCamera.worldToCameraMatrix;
                if (prevWorld2Cam != world2Cam)
                {
                    prevWorld2Cam = world2Cam;
                    activeCmd.Reset();
                    deactiveCmd.Reset();
                    quadtree.CullQuadtree(cullCamera.transform.position, cullCamera.fieldOfView, Screen.height, Screen.width, world2Cam, projM,
                        activeCmd, deactiveCmd, lodPolicy);
                    for (int i = 0; i < activeCmd.Length; ++i)
                    {
                        ActiveMesh(activeCmd.Data[i]);
                    }
                    for (int i = 0; i < deactiveCmd.Length; ++i)
                    {
                        DeactiveMesh(deactiveCmd.Data[i]);
                    }
                    if (quadtree.ActiveNodes.Length > 0)
                    {
                        for (int i=0; i< quadtree.ActiveNodes.Length; ++i)
                        {
                            var node = quadtree.ActiveNodes.Data[i];
                            var p = activeMeshes[node.meshIdx];
                            p.UpdatePatch(cullCamera.transform.position, cullCamera.fieldOfView, Screen.height, Screen.width);
                        }
                    }
                    GeometryUtility.CalculateFrustumPlanes(detailProjM * world2Cam, detailCullPlanes);                    
                    detailRenderer.Cull(detailCullPlanes);
                }
                detailRenderer.OnUpdate(cullCamera);
            }
        }
        private void OnDestroy()
        {
            RenderPipelineManager.beginFrameRendering -= OnFrameRendering;
            detailRenderer.Clear();
            meshPool.Clear();
            MTPooledRenderMesh.Clear();
            MTHeightMap.UnregisterMap(heightMap);
        }
        private void OnDrawGizmos()
        {
            if (!showDebug)
                return;
            if (detailRenderer != null)
                detailRenderer.DrawDebug();
        }
    }
}
