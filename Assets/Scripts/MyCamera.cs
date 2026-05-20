using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Assertions;

public enum WallClimberCameraMode
{
    /// <summary>Follow one or average multiple climbers (inference / single-area).</summary>
    FollowAgent = 0,

    /// <summary>Fixed orthographic framing of all climbers (multi-area training).</summary>
    StaticTrainingOverview = 1,
}

/// <summary>
/// Training: use <see cref="WallClimberCameraMode.StaticTrainingOverview"/> to keep the camera
/// fixed and frame every parallel area. Inference: use <see cref="WallClimberCameraMode.FollowAgent"/>.
/// </summary>
public class MyCamera : MonoBehaviour
{
    [SerializeField]
    WallClimberCameraMode mode = WallClimberCameraMode.FollowAgent;

    [Tooltip("Extra world units around the climber bounds when framing (training static view).")]
    [SerializeField]
    float framingPadding = 8f;

    List<Transform> targets;

    [SerializeField]
    float offsetY = 2f;

    [SerializeField]
    float followSpeed = 10f;

    [SerializeField]
    GameObject grid;

    Camera _cam;

    void Awake()
    {
        _cam = GetComponent<Camera>();
        // Hard-disable legacy OnMouse_ raycasting from this camera
        // This instantly stops Unity's hidden SendMouseEvents from eating CPU 
        // regardless of project settings or physics components.
        _cam.eventMask = 0;
    }

    void Start()
    {
        if (mode == WallClimberCameraMode.StaticTrainingOverview)
        {
            ApplyStaticTrainingFraming();
            return;
        }

        targets = FindObjectsByType<Climber>(FindObjectsSortMode.None).Select(c => c.transform).ToList();
        Assert.IsTrue(
            targets.Count > 0,
            "No Climber found. Assign inference scene or ensure a Climber is present.");
    }

    /// <summary>Recompute static framing (e.g. after late-spawned agents). Call is optional.</summary>
    public void ApplyStaticTrainingFraming()
    {
        if (_cam == null)
            _cam = GetComponent<Camera>();

        var climbers = FindObjectsByType<Climber>(FindObjectsSortMode.None);
        if (climbers.Length == 0)
            return;

        if (!_cam.orthographic)
        {
            Debug.LogWarning("[MyCamera] StaticTrainingOverview expects an orthographic camera.");
            return;
        }

        var b = new Bounds(climbers[0].transform.position, Vector3.zero);
        foreach (var c in climbers)
            b.Encapsulate(c.transform.position);

        b.Expand(framingPadding);

        float aspect = Mathf.Max(_cam.aspect, 0.01f);
        float halfW = b.extents.x;
        float halfH = b.extents.y;
        float ortho = Mathf.Max(halfH, halfW / aspect);

        _cam.orthographicSize = ortho;

        var p = b.center;
        p.y += offsetY;
        p.z = transform.position.z;
        transform.position = p;
    }

    void LateUpdate()
    {
        if (mode == WallClimberCameraMode.StaticTrainingOverview)
            return;

        if (targets == null || targets.Count == 0)
            return;

        Vector3 newPosition = targets.Aggregate(Vector3.zero, (acc, t) => acc + t.position) / targets.Count;
        newPosition.y += offsetY;
        newPosition.z = transform.position.z;
        transform.position = Vector3.Lerp(transform.position, newPosition, Time.deltaTime * followSpeed);
        if (grid != null)
        {
            Vector3 newGridPosition = Vector3Int.FloorToInt(transform.position / 4f) * 4;
            newGridPosition.z = grid.transform.position.z;
            grid.transform.position = newGridPosition;
        }
    }
}
