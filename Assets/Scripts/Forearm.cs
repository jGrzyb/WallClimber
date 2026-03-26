using UnityEngine;

public class Forearm : Arm
{
    private HingeJoint2D _handJoint;
    public HingeJoint2D handJoint => _handJoint ??= GetComponents<HingeJoint2D>()[1];
    
    public void Grip(bool isGripping) {
        handJoint.enabled = isGripping;
    }
}
