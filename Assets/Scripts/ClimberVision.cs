using System.Collections.Generic;
using Unity.MLAgents.Sensors;
using UnityEngine;

/// <summary>
/// Angular-bin 2D vision sensor for the Climber agent.
/// </summary>
public class ClimberVision : MonoBehaviour
{
    [Tooltip("Number of angular bins that span 360°. Must match the delta added to VectorObservationSize.")]
    [SerializeField] private int rayCount = 32;

    [Tooltip("Maximum sensing distance. A good starting point is 1.5 × grid cell spacing.")]
    [SerializeField] private float viewRange = 4.5f;

    [Tooltip("Draw thin spokes in the Scene view showing where each bin boundary lies.")]
    [SerializeField] private bool showBinDividers = true;

    public int ObservationCount => rayCount;

    // ── Internal state ───────────────────────────────────────────────────────
    private float[] _bins;
    private float _lastRefreshTime = -1f;

#if UNITY_EDITOR
    // Slower tracked list only exists and allocates when inside the Unity Editor for Gizmos
    private readonly List<(Vector2 pos, float normDist)> _visible = new();
#endif

    public void CollectVisionObservations(VectorSensor sensor)
    {
        EnsureRefreshed();
        // Since we know the exact size, a standard for-loop is slightly faster than foreach
        for (int i = 0; i < rayCount; i++)
        {
            sensor.AddObservation(_bins[i]);
        }
    }

#if UNITY_EDITOR
    private void FixedUpdate()
    {
        // Only run here if we want Gizmos to look smooth in the editor while playing.
        // Otherwise, it gets executed exactly when ML-agents requests it.
        if (showBinDividers)
            EnsureRefreshed();
    }
#endif

    private void EnsureRefreshed()
    {
        // Tie refresh to physics time, as ML-agents steps on physics frames
        if (Mathf.Approximately(Time.fixedTime, _lastRefreshTime)) return;
        _lastRefreshTime = Time.fixedTime;
        RefreshBins();
    }

    private void RefreshBins()
    {
        if (_bins == null || _bins.Length != rayCount)
            _bins = new float[rayCount];
        else
            System.Array.Clear(_bins, 0, rayCount);

#if UNITY_EDITOR
        _visible.Clear();
#endif
        var origin = (Vector2)transform.position;

        // Caching values natively speeds up the inner loop
        float viewRangeSqr = viewRange * viewRange;
        float invViewRange = 1f / viewRange; 
        float invBinAngle  = rayCount / 360f; // Multiplying by this is same as dividing by (360/rayCount)

        using var _ = UnityEngine.Pool.ListPool<GripPoint>.Get(out List<GripPoint> nearbyGrips);
        GripPoint.GetGripsInRadius(origin, viewRange, nearbyGrips); 

        // Iterator runs over ONLY grips mathematically inside the chunk boundaries 
        // Eliminates analyzing the remaining 95% of world scale points.
        int gripCount = nearbyGrips.Count;
        for (int i = 0; i < gripCount; i++)
        {
            var gp = nearbyGrips[i];
            var pos = (Vector2)gp.transform.position; // Can be cached to lower Component lookup overhead!
            var delta = pos - origin;
            
            // Optimization: Filter strictly by squared distance to avoid Math.Sqrt entirely
            var sqrDist = delta.sqrMagnitude;
            if (sqrDist > viewRangeSqr || sqrDist < 0.000001f) continue;

            // Math.Sqrt is only evaluated for the ~5% of grips that are ACTUALLY near the agent
            var dist = Mathf.Sqrt(sqrDist);
            var normDist = dist * invViewRange;

#if UNITY_EDITOR
            if (showBinDividers)
                _visible.Add((pos, normDist)); // 'pos' is used here implicitly from cache!
#endif

            // Map world angle [0, 360) → bin index
            var angleDeg = Mathf.Atan2(delta.y, delta.x) * Mathf.Rad2Deg;
            if (angleDeg < 0f) angleDeg += 360f;

            // Direct int cast is faster than Mathf.FloorToInt. 
            // Clamp is occasionally needed if angleDeg hits 360 exactly due to float imprecision.
            var bin = (int)(angleDeg * invBinAngle);
            if (bin >= rayCount) bin = rayCount - 1;

            // Closest grip per bin wins
            if (_bins[bin] == 0f || normDist < _bins[bin])
                _bins[bin] = normDist;
        }
    }

#if UNITY_EDITOR
    // ── Gizmos ───────────────────────────────────────────────────────────────
    private void OnDrawGizmos()
    {
        DrawRangeCircle();
        if (showBinDividers) DrawBinSpokes();
        if (Application.isPlaying && _visible.Count > 0) DrawGripLines();
    }

    private void DrawRangeCircle()
    {
        Gizmos.color = new Color(0.2f, 0.9f, 0.2f, 0.55f);
        const int segs = 64;
        var c = transform.position;
        for (var i = 0; i < segs; i++)
        {
            var a0 = i * (2f * Mathf.PI / segs);
            var a1 = (i + 1) * (2f * Mathf.PI / segs);
            Gizmos.DrawLine(
                c + new Vector3(Mathf.Cos(a0), Mathf.Sin(a0)) * viewRange,
                c + new Vector3(Mathf.Cos(a1), Mathf.Sin(a1)) * viewRange
            );
        }
    }

    private void DrawBinSpokes()
    {
        Gizmos.color = new Color(0.2f, 0.9f, 0.2f, 0.12f);
        var c = transform.position;
        for (var i = 0; i < rayCount; i++)
        {
            var a = i * (2f * Mathf.PI / rayCount);
            Gizmos.DrawLine(c, c + new Vector3(Mathf.Cos(a), Mathf.Sin(a)) * viewRange);
        }
    }

    private void DrawGripLines()
    {
        foreach (var (pos, normDist) in _visible)
        {
            Gizmos.color = Color.Lerp(new Color(1f, 0.25f, 0.1f), new Color(1f, 0.9f, 0.1f), normDist);
            Gizmos.DrawLine(transform.position, (Vector3)pos);
        }
    }
#endif
}