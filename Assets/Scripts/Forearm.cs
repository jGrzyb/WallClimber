using UnityEngine;

public class Forearm : Arm
{
    public bool IsGripping  => handJoint.enabled;
    private HingeJoint2D _handJoint;
    private HingeJoint2D handJoint => _handJoint ??= GetComponents<HingeJoint2D>()[1];
    
    public void SetGrip(bool isGripping) {
        handJoint.enabled = isGripping;
    }
}
