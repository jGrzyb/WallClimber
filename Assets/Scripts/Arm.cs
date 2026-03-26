using UnityEngine;

public class Arm : MonoBehaviour
{
    protected Rigidbody2D _rb;
    public Rigidbody2D rb => _rb ??= GetComponent<Rigidbody2D>();
    protected HingeJoint2D _armHinge;
    public HingeJoint2D armHinge => _armHinge ??= GetComponents<HingeJoint2D>()[0];
    private Vector3 initPos;
    private Quaternion initRot;

    void Awake() {
        initPos = transform.localPosition;
        initRot = transform.localRotation;
    }

    public void SetMotor(float speed) {
        armHinge.useMotor = true;
        JointMotor2D motor = armHinge.motor;
        if (armHinge.jointAngle > armHinge.limits.max && speed > 0) {
            motor.motorSpeed = 0; 
        } 
        else if (armHinge.jointAngle < armHinge.limits.min && speed < 0) {
            motor.motorSpeed = 0;
        }
        else {
            motor.motorSpeed = speed;
        }
        armHinge.motor = motor;
    }

    public void Reset() {
        transform.localPosition = initPos;
        transform.localRotation = initRot;
        rb.linearVelocity = Vector2.zero;
        rb.angularVelocity = 0f;
    }
}
