using System.Collections.Generic;
using Unity.MLAgents.Sensors;
using UnityEngine;

/// <summary>
/// Angular-bin 2D vision sensor for the Climber agent.
///
/// HOW IT WORKS
/// ─────────────
/// The full 360° around the agent is divided into <see cref="rayCount"/> equal angular bins.
/// Each frame, every active GripPoint is measured:
///   • If it is within <see cref="viewRange"/>, its polar angle determines which bin it occupies.
///   • The bin's observation value is set to   dist / viewRange   (0 = at the agent, 1 = at range edge).
///   • When multiple grips fall in the same bin, only the closest one is recorded.
///   • Bins with no grip in range keep the value 0.
///
/// The resulting float[] of length <see cref="rayCount"/> is appended to the agent's VectorSensor
/// via <see cref="CollectVisionObservations"/>.
///
/// GIZMOS (always visible in Scene view)
/// ────────────────────────────────────
///   Green ring        : the view range boundary.
///   Faint green spokes: bin boundaries showing the angular resolution.
///   Coloured lines    : visible grips (red = close → yellow = far edge of range).
///                       Only drawn during Play mode once observations have been collected.
/// </summary>
public class ClimberVision : MonoBehaviour
{
    // ── Inspector ────────────────────────────────────────────────────────────

    [Tooltip("Number of angular bins that span 360°. Must match the delta added to VectorObservationSize.")]
    [SerializeField] private int rayCount = 32;

    [Tooltip("Maximum sensing distance. A good starting point is 1.5 × grid cell spacing.")]
    [SerializeField] private float viewRange = 4.5f;

    [Tooltip("Draw thin spokes in the Scene view showing where each bin boundary lies.")]
    [SerializeField] private bool showBinDividers = true;

    // ── Public API ───────────────────────────────────────────────────────────

    /// <summary>Number of floats this sensor adds to the VectorSensor.</summary>
    public int ObservationCount => rayCount;

    /// <summary>
    /// Appends <see cref="rayCount"/> floats to <paramref name="sensor"/>.
    /// Call this from <c>Agent.CollectObservations</c>.
    /// </summary>
    public void CollectVisionObservations(VectorSensor sensor)
    {
        EnsureRefreshed();
        foreach (var v in _bins)
            sensor.AddObservation(v);
    }

    // ── Internal state ───────────────────────────────────────────────────────

    private float[] _bins;

    // All GripPoints currently inside viewRange — used only for gizmo drawing.
    private readonly List<(Vector2 pos, float normDist)> _visible = new();

    private int _lastRefreshFrame = -1;

    // ── Unity messages ───────────────────────────────────────────────────────

    private void LateUpdate() => EnsureRefreshed();

    // ── Core logic ───────────────────────────────────────────────────────────

    private void EnsureRefreshed()
    {
        if (Time.frameCount == _lastRefreshFrame) return;
        _lastRefreshFrame = Time.frameCount;
        RefreshBins();
    }

    private void RefreshBins()
    {
        if (_bins == null || _bins.Length != rayCount)
            _bins = new float[rayCount];
        else
            System.Array.Clear(_bins, 0, rayCount);

        _visible.Clear();

        var origin    = (Vector2)transform.position;
        var binAngle  = 360f / rayCount;      // degrees per bin

        foreach (var gp in GripPoint.ActiveGrips)
        {
            var delta    = (Vector2)gp.transform.position - origin;
            var dist     = delta.magnitude;
            if (dist > viewRange || dist < 0.001f) continue;

            var normDist = dist / viewRange;
            _visible.Add(((Vector2)gp.transform.position, normDist));

            // Map world angle [0, 360) → bin index
            var angleDeg = Mathf.Atan2(delta.y, delta.x) * Mathf.Rad2Deg;
            if (angleDeg < 0f) angleDeg += 360f;
            var bin = Mathf.Clamp(Mathf.FloorToInt(angleDeg / binAngle), 0, rayCount - 1);

            // Closest grip per bin wins
            if (_bins[bin] == 0f || normDist < _bins[bin])
                _bins[bin] = normDist;
        }
    }

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
            var a0 = i       * (2f * Mathf.PI / segs);
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
            // Red (close) → yellow (near range edge)
            Gizmos.color = Color.Lerp(new Color(1f, 0.25f, 0.1f), new Color(1f, 0.9f, 0.1f), normDist);
            Gizmos.DrawLine(transform.position, (Vector3)pos);
        }
    }
}
