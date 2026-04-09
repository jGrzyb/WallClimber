using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Kinematic anchor for the hand hinge. No collider — the climber does not physically push off grips.
/// Gripping is allowed when the forearm hand position lies inside this grip's circle (see <see cref="ContainsHandPoint"/>).
/// </summary>
[RequireComponent(typeof(Rigidbody2D))]
public class GripPoint : MonoBehaviour
{
    [SerializeField] private float worldRadius = 0.35f;

    public Rigidbody2D Body { get; private set; }
    public float WorldRadius => worldRadius;

    static readonly List<GripPoint> Active = new();

    /// <summary>Read-only view of all currently enabled grip points (used by ClimberVision).</summary>
    public static IReadOnlyList<GripPoint> ActiveGrips => Active;

    public static bool TryFindBestForHand(Vector2 handWorld, out GripPoint best)
    {
        best = null;
        var bestDistSq = float.PositiveInfinity;
        foreach (var gp in Active)
        {
            if (!gp.ContainsHandPoint(handWorld))
                continue;
            var d = ((Vector2)gp.transform.position - handWorld).sqrMagnitude;
            if (d < bestDistSq)
            {
                bestDistSq = d;
                best = gp;
            }
        }

        return best != null;
    }

    public bool ContainsHandPoint(Vector2 handWorld)
    {
        var c = (Vector2)transform.position;
        return (handWorld - c).sqrMagnitude <= worldRadius * worldRadius;
    }

    public void Configure(float radiusWorld)
    {
        worldRadius = radiusWorld;
    }

    void Awake()
    {
        Body = GetComponent<Rigidbody2D>();
    }

    void OnEnable()
    {
        Active.Add(this);
    }

    void OnDisable()
    {
        Active.Remove(this);
    }
}
