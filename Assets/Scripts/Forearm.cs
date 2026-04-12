using UnityEngine;

public class Forearm : Arm
{
    [SerializeField]
    [Tooltip(
        "Draw grip debug at the hand (red = gripping, gray = not). " +
        "Game view: enable the Gizmos toggle in the Game tab toolbar or use Scene view.")]
    bool showGripGizmo = true;

    [SerializeField]
    [Tooltip("World-space radius of the grip gizmo.")]
    float gripGizmoRadius = 0.22f;

    static readonly Color UngrippedGizmoColor = new Color(0.7f, 0.7f, 0.7f, 1f);

    /// <summary>
    /// Optional second hinge (hand ↔ grip). Upper arms reuse this script but only have the
    /// shoulder hinge — in that case there is no grip joint and all grip APIs no-op.
    /// </summary>
    HingeJoint2D _gripHinge;
    bool _gripHingeResolved;

    HingeJoint2D GripHinge
    {
        get
        {
            if (!_gripHingeResolved)
            {
                var joints = GetComponents<HingeJoint2D>();
                _gripHinge = joints.Length >= 2 ? joints[1] : null;
                _gripHingeResolved = true;
            }

            return _gripHinge;
        }
    }

    void OnDrawGizmos()
    {
        if (!showGripGizmo)
            return;
        var hj = GripHinge;
        if (hj == null)
            return;
        var handWorld = transform.TransformPoint(hj.anchor);
        var c = IsGripping ? Color.red : UngrippedGizmoColor;
        Gizmos.color = c;
        Gizmos.DrawWireSphere(handWorld, gripGizmoRadius);
        // Cross in the XY plane so the marker stays readable from side-on 2D cameras (wire sphere can look like a line).
        float r = gripGizmoRadius;
        Gizmos.DrawLine(handWorld + new Vector3(-r, 0f, 0f), handWorld + new Vector3(r, 0f, 0f));
        Gizmos.DrawLine(handWorld + new Vector3(0f, -r, 0f), handWorld + new Vector3(0f, r, 0f));
    }

    public bool IsGripping
    {
        get
        {
            var hj = GripHinge;
            return hj != null && hj.enabled;
        }
    }

    public void SetGrip(bool isGripping)
    {
        if (!isGripping)
        {
            ReleaseGrip();
            return;
        }

        var hj = GripHinge;
        if (hj == null)
            return;

        // Use the joint's own anchor as the hand position so the grip-detection point
        // matches the physical attachment point exactly, regardless of the forearm's
        // local rotation (left and right forearms are mirrored via the prefab).
        var handWorld = (Vector2)transform.TransformPoint(hj.anchor);
        if (!GripPoint.TryFindBestForHand(handWorld, out var grip))
        {
            ReleaseGrip();
            return;
        }

        if (hj.enabled && hj.connectedBody == grip.Body)
            return;

        hj.connectedBody = grip.Body;
        hj.autoConfigureConnectedAnchor = true;
        hj.enabled = true;
    }

    void ReleaseGrip()
    {
        var hj = GripHinge;
        if (hj == null)
            return;
        hj.connectedBody = null;
        hj.enabled = false;
    }

    public new void Reset()
    {
        ReleaseGrip();
        base.Reset();
    }
}
