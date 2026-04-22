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

    // ── Global collections ──
    static readonly List<GripPoint> Active = new();
    public static IReadOnlyList<GripPoint> ActiveGrips => Active;

    // ── Spatial Partitioning Grid ──
    const float GridCellSize = 5f;
    static readonly Dictionary<Vector2Int, List<GripPoint>> SpatialGrid = new();
    private Vector2Int _currentCell;

    /// <summary>
    /// Returns all GripPoints residing in the grid cell overlapping the given coordinates,
    /// plus the immediately adjacent 8 cells (9 cells total).
    /// </summary>
    public static void GetGripsInRadius(Vector2 center, float radius, List<GripPoint> results)
    {
        results.Clear();
        int minX = Mathf.FloorToInt((center.x - radius) / GridCellSize);
        int maxX = Mathf.FloorToInt((center.x + radius) / GridCellSize);
        int minY = Mathf.FloorToInt((center.y - radius) / GridCellSize);
        int maxY = Mathf.FloorToInt((center.y + radius) / GridCellSize);

        for (int x = minX; x <= maxX; x++)
        {
            for (int y = minY; y <= maxY; y++)
            {
                var cell = new Vector2Int(x, y);
                if (SpatialGrid.TryGetValue(cell, out var list))
                {
                    results.AddRange(list);
                }
            }
        }
    }

    private Vector2Int GetCell()
    {
        var pos = transform.position;
        return new Vector2Int(
            Mathf.FloorToInt(pos.x / GridCellSize),
            Mathf.FloorToInt(pos.y / GridCellSize)
        );
    }

    public static bool TryFindBestForHand(Vector2 handWorld, out GripPoint best)
    {
        best = null;
        var bestDistSq = float.PositiveInfinity;
        
        using var _ = UnityEngine.Pool.ListPool<GripPoint>.Get(out List<GripPoint> localGrips);
        GetGripsInRadius(handWorld, 2.0f, localGrips); // Small radius assumption for hand interaction. Feel free to raise it if hand interaction range is larger!

        var list = localGrips.Count > 0 ? localGrips : Active;
        int count = list.Count;
        for (int i = 0; i < count; i++)
        {
            var gp = list[i];
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
        AddToGrid();
    }

    void OnDisable()
    {
        Active.Remove(this);
        RemoveFromGrid();
    }

    private void AddToGrid()
    {
        _currentCell = GetCell();
        if (!SpatialGrid.TryGetValue(_currentCell, out var list))
        {
            list = new();
            SpatialGrid[_currentCell] = list;
        }
        list.Add(this);
    }

    private void RemoveFromGrid()
    {
        if (SpatialGrid.TryGetValue(_currentCell, out var list))
        {
            list.Remove(this);
            if (list.Count == 0)
                SpatialGrid.Remove(_currentCell); // Keep memory footprint small
        }
    }

    void FixedUpdate() // Or Update, but kinematic bodies sync nicely here
    {
        // Dynamic re-parenting across grid cells if this object moves (only runs if pos actually changes significantly)
        var newCell = GetCell();
        if (newCell != _currentCell)
        {
            RemoveFromGrid();
            AddToGrid();
        }
    }
}
