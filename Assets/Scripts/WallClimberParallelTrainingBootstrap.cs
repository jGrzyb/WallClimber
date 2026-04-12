using UnityEngine;

/// <summary>
/// Groups the WallClimber scene objects into a single training-area root and attaches
/// <see cref="WallClimberHorizontalAreaReplicator"/> so the number of parallel agents
/// matches <c>env_settings.num_areas</c> in your training YAML.
/// </summary>
/// <remarks>
/// <para>
/// When training with <c>mlagents-learn</c>, the Python side sends <c>env_settings.num_areas</c>
/// from your YAML; <see cref="WallClimberHorizontalAreaReplicator"/> uses
/// <c>Academy.Instance.NumAreas</c> after the handshake. <see cref="DefaultNumAreas"/> and the
/// serialized <c>numAreas</c> field are fallbacks for Play-in-Editor without a trainer (match
/// your YAML to avoid confusion in that mode).
/// </para>
/// <para>
/// Place one instance of this component on an empty GameObject in the training scene,
/// or leave the scene without it: on play, a bootstrap object is created automatically
/// unless a <see cref="WallClimberParallelTrainingBootstrap"/> already exists.
/// Scenes with <see cref="WallClimberInferenceScene"/> skip auto-bootstrap (single agent, inference).
/// </para>
/// </remarks>
[DefaultExecutionOrder(-20)]
public class WallClimberParallelTrainingBootstrap : MonoBehaviour
{
    /// <summary>Fallback when no Python trainer is attached (e.g. Play in Editor). Prefer matching YAML.</summary>
    public const int DefaultNumAreas = 1;

    [Tooltip("Copies of the training area (original + replicas). When training with Python, " +
             "WallClimberHorizontalAreaReplicator uses Academy.NumAreas from env_settings.num_areas instead.")]
    [SerializeField]
    int numAreas = DefaultNumAreas;

    [Tooltip("Distance between adjacent area centres in the replicator grid.")]
    [SerializeField]
    float separation = 48f;

    [Tooltip("If true, areas are also replicated when training from the Unity Editor (recommended).")]
    [SerializeField]
    bool replicateInEditor = true;

    static WallClimberParallelTrainingBootstrap s_instance;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    static void AutoCreateIfNeeded()
    {
        if (s_instance != null)
            return;

        var existing = FindObjectsByType<WallClimberParallelTrainingBootstrap>(
            FindObjectsInactive.Include,
            FindObjectsSortMode.None);
        if (existing.Length > 0)
            return;

        if (!TryFindTrainingContent(out _, out _, out _))
            return;

        var inference = FindObjectsByType<WallClimberInferenceScene>(
            FindObjectsInactive.Include,
            FindObjectsSortMode.None);
        if (inference.Length > 0)
            return;

        var go = new GameObject(nameof(WallClimberParallelTrainingBootstrap));
        go.AddComponent<WallClimberParallelTrainingBootstrap>();
    }

    void Awake()
    {
        if (s_instance != null && s_instance != this)
        {
            Destroy(gameObject);
            return;
        }

        s_instance = this;
        SetupReplication();
    }

    void OnDestroy()
    {
        if (s_instance == this)
            s_instance = null;
    }

    void SetupReplication()
    {
        if (GameObject.Find("TrainingArea") != null)
            return;

        if (!TryFindTrainingContent(out var ground, out var grid, out var climber))
        {
            Debug.LogWarning(
                "[WallClimberParallelTrainingBootstrap] Ground / GripGrid / Climber not all found — skipping parallel setup.");
            return;
        }

        var areaRoot = new GameObject("TrainingArea");
        areaRoot.transform.SetPositionAndRotation(Vector3.zero, Quaternion.identity);

        ground.transform.SetParent(areaRoot.transform, true);
        grid.transform.SetParent(areaRoot.transform, true);
        climber.transform.SetParent(areaRoot.transform, true);

        // Scene background and every limb/body sprite are named "Square". GameObject.Find("Square")
        // is undefined which one you get — it can grab a forearm sprite and rip it off the rig.
        ReparentSceneBackgroundSquaresOnly(areaRoot.transform, climber.transform);

        var replicatorGo = new GameObject("WallClimberHorizontalAreaReplicator");
        replicatorGo.transform.SetParent(transform, false);
        replicatorGo.SetActive(false);

        var replicator = replicatorGo.AddComponent<WallClimberHorizontalAreaReplicator>();
        replicator.baseArea = areaRoot;
        replicator.numAreas = numAreas;
        replicator.separation = separation;
        replicator.buildOnly = !replicateInEditor;

        replicatorGo.SetActive(true);
    }

    static bool TryFindTrainingContent(out GameObject ground, out GameObject grid, out GameObject climber)
    {
        ground = GameObject.Find("Ground");
        grid = GameObject.Find("GripGrid");
        climber = GameObject.Find("Climber");
        return ground != null && grid != null && climber != null;
    }

    /// <summary>
    /// Moves only "Square" objects that are not part of the climber hierarchy (body + arms + forearms)
    /// under <paramref name="areaRoot"/> so we never steal a limb sprite from its rigidbody parent.
    /// </summary>
    static void ReparentSceneBackgroundSquaresOnly(Transform areaRoot, Transform climber)
    {
        foreach (var t in FindObjectsByType<Transform>(
                     FindObjectsInactive.Include,
                     FindObjectsSortMode.None))
        {
            if (t.name != "Square")
                continue;
            if (t.IsChildOf(climber))
                continue;
            t.SetParent(areaRoot, true);
        }
    }
}
