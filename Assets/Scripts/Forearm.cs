using UnityEngine;

public class Forearm : Arm
{
    [SerializeField] private Vector2 handLocalAnchor = new Vector2(0f, 1f);

    public bool IsGripping => handJoint.enabled;
    private HingeJoint2D _handJoint;
    private HingeJoint2D handJoint => _handJoint ??= GetComponents<HingeJoint2D>()[1];

    public void SetGrip(bool isGripping)
    {
        if (!isGripping)
        {
            ReleaseGrip();
            return;
        }

        var handWorld = (Vector2)transform.TransformPoint(handLocalAnchor);
        if (!GripPoint.TryFindBestForHand(handWorld, out var grip))
        {
            ReleaseGrip();
            return;
        }

        if (handJoint.enabled && handJoint.connectedBody == grip.Body)
            return;

        handJoint.connectedBody = grip.Body;
        handJoint.autoConfigureConnectedAnchor = true;
        handJoint.enabled = true;
    }

    void ReleaseGrip()
    {
        handJoint.connectedBody = null;
        handJoint.enabled = false;
    }

    public new void Reset()
    {
        ReleaseGrip();
        base.Reset();
    }
}
