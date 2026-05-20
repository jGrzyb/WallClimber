using UnityEngine;

public class Arm : MonoBehaviour
{
    public float Angle => armHinge.jointAngle;
    public float MinAngle => armHinge.limits.min;
    public float MaxAngle => armHinge.limits.max;
    public float ReactionForce => armHinge.reactionForce.magnitude;

    private Rigidbody2D _rb;
    private Rigidbody2D rb => _rb ??= GetComponent<Rigidbody2D>();
    private HingeJoint2D _armHinge;
    private HingeJoint2D armHinge => _armHinge ??= GetComponent<HingeJoint2D>();
    private Vector3 initPos;
    private Quaternion initRot;

    void Awake() {
        initPos = transform.localPosition;
        initRot = transform.localRotation;
    }

    public void SetMotor(float speed) {
        JointMotor2D motor = armHinge.motor;
        if (Angle > MaxAngle - 1f && speed > 0)
            motor.motorSpeed = 0;
        else if (Angle < MinAngle + 1f && speed < 0)
            motor.motorSpeed = 0;
        else
            motor.motorSpeed = speed;
        armHinge.motor = motor;
    }

    public void Reset() {
        transform.localPosition = initPos;
        transform.localRotation = initRot;
        rb.linearVelocity = Vector2.zero;
        rb.angularVelocity = 0f;
    }
}
