using UnityEngine;

public class Forearm : Arm
{
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
