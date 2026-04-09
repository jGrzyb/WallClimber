using UnityEngine;

/// <summary>
/// Spawns a uniform grid of circular grip holds at runtime (Awake).
/// Holds have no colliders — they are visual + kinematic anchors only.
/// </summary>
public class GripGridPlacer : MonoBehaviour
{
    [SerializeField] private int columns = 12;
    [SerializeField] private int rows = 8;
    [SerializeField] private Vector2 cellSpacing = new Vector2(1.5f, 1.5f);
    [SerializeField] private Vector3 gridOrigin = new Vector3(-9f, -4f, 0f);
    [SerializeField] private float circleRadius = 0.35f;
    [SerializeField] private Sprite circleSprite;
    [SerializeField] private int sortingOrder = 5;

    void Awake()
    {
        if (circleSprite == null)
        {
            Debug.LogError("[GripGridPlacer] Assign a circle Sprite (e.g. built-in Circle).", this);
            enabled = false;
            return;
        }

        for (var y = 0; y < rows; y++)
        {
            for (var x = 0; x < columns; x++)
            {
                var local = gridOrigin + new Vector3(x * cellSpacing.x, y * cellSpacing.y, 0f);
                CreateGrip(local, x, y);
            }
        }
    }

    void CreateGrip(Vector3 localPosition, int ix, int iy)
    {
        var go = new GameObject($"Grip_{ix}_{iy}");
        go.transform.SetParent(transform, false);
        go.transform.localPosition = localPosition;
        go.transform.localRotation = Quaternion.identity;
        var diameter = circleRadius * 2f;
        go.transform.localScale = new Vector3(diameter, diameter, 1f);

        var sr = go.AddComponent<SpriteRenderer>();
        sr.sprite = circleSprite;
        sr.sortingOrder = sortingOrder;
        sr.color = new Color(0.85f, 0.85f, 0.2f, 1f);

        var rb = go.AddComponent<Rigidbody2D>();
        rb.bodyType = RigidbodyType2D.Kinematic;
        rb.simulated = true;
        rb.gravityScale = 0f;

        var gp = go.AddComponent<GripPoint>();
        gp.Configure(circleRadius);
    }
}
