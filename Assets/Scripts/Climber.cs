using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.InputSystem;

public class Climber : Agent
{
    [SerializeField] private float motorMultiplier = 100f;
    [Space]
    [SerializeField] private Arm leftArm;
    [SerializeField] private Arm rightArm;
    [SerializeField] private Forearm leftForearm;
    [SerializeField] private Forearm rightForearm;
    private Rigidbody2D rb;
    private Vector3 initPos;
    private Quaternion initRot;

    public override void Initialize() {
        rb = GetComponent<Rigidbody2D>();
        initPos = transform.localPosition;
        initRot = transform.localRotation;
    }

    public override void CollectObservations(VectorSensor sensor) {
        sensor.AddObservation(leftArm.armHinge.jointAngle);
        sensor.AddObservation(rightArm.armHinge.jointAngle);
        sensor.AddObservation(leftForearm.armHinge.jointAngle);
        sensor.AddObservation(rightForearm.armHinge.jointAngle);
        sensor.AddObservation(leftForearm.handJoint.enabled ? 1f : 0f);
        sensor.AddObservation(rightForearm.handJoint.enabled ? 1f : 0f);
        sensor.AddObservation(rb.linearVelocity);
        sensor.AddObservation(rb.angularVelocity);
        sensor.AddObservation(transform.rotation.eulerAngles.z);
    }

    public override void OnActionReceived(ActionBuffers actions) {
        leftArm.SetMotor(actions.ContinuousActions[0] * motorMultiplier);
        rightArm.SetMotor(actions.ContinuousActions[1] * motorMultiplier);
        leftForearm.SetMotor(actions.ContinuousActions[2] * motorMultiplier);
        rightForearm.SetMotor(actions.ContinuousActions[3] * motorMultiplier);
        AddReward(rb.linearVelocityY * 1e-3f);
        // if (transform.rotation.eulerAngles.z > 90 && transform.rotation.eulerAngles.z < 270) {
        //     AddReward(-1e-2f);
        // }

        if (StepCount > MaxStep) {
            EndEpisode();
        }
        if (transform.position.y < -10) {
            AddReward(-1f);
            EndEpisode();
        }

        leftForearm.Grip(actions.DiscreteActions[0] == 1);
        rightForearm.Grip(actions.DiscreteActions[1] == 1);
    }

    public override void OnEpisodeBegin() {
        Reset();
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        Assert.IsNotNull(InputManager.I, "[KUBA] InputManager instance is null. Ensure an InputManager is present in the scene.");
        var continuousActionsOut = actionsOut.ContinuousActions;
        
        continuousActionsOut[0] = InputManager.I.ArmLeft;
        continuousActionsOut[1] = InputManager.I.ArmRight;
        continuousActionsOut[2] = InputManager.I.ForearmLeft;
        continuousActionsOut[3] = InputManager.I.ForearmRight;

        var discreteActionsOut = actionsOut.DiscreteActions;
        discreteActionsOut[0] = InputManager.I.GripLeft ? 1 : 0;
        discreteActionsOut[1] = InputManager.I.GripRight ? 1 : 0;
    }

    private void Reset() {
        transform.localPosition = initPos;
        transform.localRotation = initRot;
        rb.linearVelocity = Vector2.zero;
        rb.angularVelocity = 0f;
        leftArm.Reset();
        rightArm.Reset();
        leftForearm.Reset();
        rightForearm.Reset();
    }
}
