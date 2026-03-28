using System;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
using UnityEngine.Assertions;

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
    private float strain => 
        leftArm.ReactionForce + 
        rightArm.ReactionForce + 
        leftForearm.ReactionForce + 
        rightForearm.ReactionForce;

    public override void Initialize() {
        rb = GetComponent<Rigidbody2D>();
        initPos = transform.localPosition;
        initRot = transform.localRotation;
    }

    public override void CollectObservations(VectorSensor sensor) {
        sensor.AddObservation(leftArm.Angle);
        sensor.AddObservation(rightArm.Angle);
        sensor.AddObservation(leftForearm.Angle);
        sensor.AddObservation(rightForearm.Angle);
        sensor.AddObservation(leftForearm.IsGripping ? 1f : 0f);
        sensor.AddObservation(rightForearm.IsGripping ? 1f : 0f);
        sensor.AddObservation(rb.linearVelocity);
        sensor.AddObservation(rb.angularVelocity);
        sensor.AddObservation(transform.rotation.eulerAngles.z);
    }

    public override void OnActionReceived(ActionBuffers actions) {
        leftArm.SetMotor(actions.ContinuousActions[0] * motorMultiplier);
        rightArm.SetMotor(actions.ContinuousActions[1] * motorMultiplier);
        leftForearm.SetMotor(actions.ContinuousActions[2] * motorMultiplier);
        rightForearm.SetMotor(actions.ContinuousActions[3] * motorMultiplier);

        leftForearm.SetGrip(actions.DiscreteActions[0] == 1);
        rightForearm.SetGrip(actions.DiscreteActions[1] == 1);

        HandleReward();
        HandleLogging();
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

    public override void OnEpisodeBegin() {
        Reset();
    }

    private void HandleReward() {
        AddReward(rb.linearVelocityY * 1e-3f);
        // AddReward(-Mathf.Abs(rb.linearVelocityX) * 1e-4f);
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
        
        if (strain > 5000f) {
            AddReward(-strain * 1e-5f);
        }

        // if (leftArm.Angle < leftArm.MinAngle) {
        //     float angleDiff = Mathf.Abs(leftArm.Angle - leftArm.MinAngle);
        //     AddReward(-angleDiff * 1e-2f);
        //     // Debug.Log($"Left arm angle {angleDiff}");
        // }
        // if(leftArm.Angle > leftArm.MaxAngle) {
        //     float angleDiff = Mathf.Abs(leftArm.Angle - leftArm.MaxAngle);
        //     AddReward(-angleDiff * 1e-2f);
        //     // Debug.Log($"Left arm angle {angleDiff}");
        // }
        // if (rightArm.Angle < rightArm.MinAngle) {
        //     float angleDiff = Mathf.Abs(rightArm.Angle - rightArm.MinAngle);
        //     AddReward(-angleDiff * 1e-2f);
        //     // Debug.Log($"Right arm angle {angleDiff}");
        // }
        // if(rightArm.Angle > rightArm.MaxAngle) {
        //     float angleDiff = Mathf.Abs(rightArm.Angle - rightArm.MaxAngle);
        //     AddReward(-angleDiff * 1e-2f);
        //     // Debug.Log($"Right arm angle {angleDiff}");
        // }
    }

    private void HandleLogging() {
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
