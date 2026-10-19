using UnityEngine;
using UnityEditor;
using MightyTerrainMesh;

public class MTDataEditor : EditorWindow
{
    [MenuItem("Assets/Create/MightyTerrainMesh/LOD Policy")]
    static void CreateLodPolicyData()
    {
        UnityEngine.Object folder = Selection.activeObject;
        if (folder)
        {
            var path = AssetDatabase.GetAssetPath(folder);
            path = string.Format("{0}/{1}", path, "LODPolicy.asset");
            var asset = ScriptableObject.CreateInstance<MTLODPolicy>();
            AssetDatabase.CreateAsset(asset, path);
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
        }
    }
    [MenuItem("MightyTerrainMesh/DataCreator")]
    private static void ShowWindow()
    {
        EditorWindow.CreateWindow<MTDataEditor>();
    }
    //properties
    private int QuadTreeDepth = 2;
    private MTLODSetting[] LODSettings = new MTLODSetting[0];
    private Terrain terrainTarget;
    private bool genUV2 = false;
    private int lodCount = 1;
    private int datapack = 1;//mesh count per data pack file
    //
    private void OnGUI()
    {
        Terrain curentTarget = EditorGUILayout.ObjectField("Convert Target", terrainTarget, typeof(Terrain), true) as Terrain;
        if (curentTarget != terrainTarget)
        {
            terrainTarget = curentTarget;
        }
        int curSliceCount = Mathf.FloorToInt(1 << QuadTreeDepth);
        int sliceCount = EditorGUILayout.IntField("Slice Count(NxN)", curSliceCount);
        if (sliceCount != curSliceCount)
        {
            curSliceCount = Mathf.NextPowerOfTwo(sliceCount);
            QuadTreeDepth = Mathf.FloorToInt(Mathf.Log(curSliceCount, 2));
        }
        if (lodCount != LODSettings.Length)
        {
            MTLODSetting[] old = LODSettings;
            LODSettings = new MTLODSetting[lodCount];
            for (int i = 0; i < Mathf.Min(lodCount, old.Length); ++i)
            {
                LODSettings[i] = old[i];
            }
            for (int i = old.Length; i < lodCount; ++i)
            {
                LODSettings[i] = new MTLODSetting();
                LODSettings[i].Subdivision = 4;
            }
        }
        lodCount = EditorGUILayout.IntField("LOD Count", LODSettings.Length);
        if (LODSettings.Length > 0)
        {
            EditorGUI.indentLevel++;
            for (int i = 0; i < LODSettings.Length; ++i)
                LODSettings[i].OnGUIDraw(i);
            EditorGUI.indentLevel--;
        }
        datapack = EditorGUILayout.IntField("Data Pack", datapack);
        genUV2 = EditorGUILayout.ToggleLeft("Generate UV2", genUV2);
        if (GUILayout.Button("Generate"))
        {
            if (LODSettings == null || LODSettings.Length == 0)
            {
                MTLog.LogError("no lod setting");
                return;
            }
            if (terrainTarget == null)
            {
                MTLog.LogError("no target terrain");
                return;
            }
            // Bake logic lives in MTTerrainBakePipeline; keep legacy output paths here.
            // NOTE: must be fully qualified - this file is in the global namespace with
            // "using UnityEditor;", and the bare name "EditorTools" collides with the
            // UnityEditor.EditorTools namespace.
            global::MightyTerrainMesh.EditorTools.MTTerrainBakePipeline.BakeMT(terrainTarget, QuadTreeDepth, LODSettings, datapack, genUV2,
                string.Format("Assets/{0}", terrainTarget.name), "Assets/MeshData");
        }
    }
}
