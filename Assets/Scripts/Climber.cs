using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
using UnityEngine.Assertions;

/// <summary>
/// Climber agent with a configurable number of limbs.
///
/// Observation layout (per limb count N):
///   [0..N-1]       arm angles
///   [N..2N-1]      forearm angles
///   [2N..3N-1]     grip states (0/1)
///   [3N]           body velocity x
///   [3N+1]         body velocity y
///   [3N+2]         body angular velocity
///   [3N+3]         body rotation z
///   [3N+4..]       vision (ClimberVision)
///
/// For N=4: 16 proprioception + 32 vision = 48 total.
/// arms[0..1] = LeftArm/RightArm, arms[2..3] = LeftLeg/RightLeg
/// forearms[0..1] = LeftForearm/RightForearm, forearms[2..3] = LeftFoot/RightFoot
///
/// Action layout:
///   continuous[0..N-1]   arm motor speeds
///   continuous[N..2N-1]  forearm motor speeds
///   discrete[0..N-1]     grip toggles (branch size 2 each)
///
/// All knobs below are also exposed as ML-Agents environment_parameters
/// so they can be set from climber_custom.yaml without recompiling.
/// </summary>
public class Climber : Agent
{
    [Header("Limbs")]
    [SerializeField] private Arm[] arms;
    [SerializeField] private Forearm[] forearms;

    [Header("Vision")]
    [SerializeField] private ClimberVision vision;

    [Header("Motor")]
    [Tooltip("Scales the [-1,1] action output to motor speed. Overridden by 'motor_multiplier' env param.")]
    [SerializeField] private float motorMultiplier = 200f;

    [Header("Rewards")]
    [Tooltip("Strain (sum of joint reaction forces) above which penalty applies. Env param: 'strain_threshold'.")]
    [SerializeField] private float strainThreshold = 5000f;
    [Tooltip("Penalty multiplier per unit of excess strain. Env param: 'strain_penalty_scale'.")]
    [SerializeField] private float strainPenaltyScale = 1e-5f;
    [Tooltip("Y position below which the agent is considered to have fallen. Env param: 'fall_y_threshold'.")]
    [SerializeField] private float fallYThreshold = -10f;
    [Tooltip("Negative reward applied on fall. Env param: 'fall_penalty'.")]
    [SerializeField] private float fallPenalty = 1f;

    [Tooltip("0 = velocity reward (default), 1 = position reward. Env param: 'reward_mode'.")]
    [SerializeField] private int rewardMode = 0;
    [Tooltip("Scale applied to linearVelocityY per step (mode 0). Env param: 'velocity_reward_scale'.")]
    [SerializeField] private float velocityRewardScale = 1e-3f;
    [Tooltip("Scale applied to clamped Y position per step (mode 1). Env param: 'position_reward_scale'.")]
    [SerializeField] private float positionRewardScale = 1e-3f;
    [Tooltip("Y position cap before position reward scaling (mode 1). Env param: 'position_height_cap'.")]
    [SerializeField] private float positionHeightCap = 50f;

    [Header("Debug")]
    [SerializeField] private int logInterval = 1000;

    Rigidbody2D rb;
    Vector3 initPos;
    Quaternion initRot;
    long _lastLogStep = -1;

    float strain {
        get {
            float s = 0f;
            foreach (var a in arms)    s += a.ReactionForce;
            foreach (var f in forearms) s += f.ReactionForce;
            return s;
        }
    }

    // ------------------------------------------------------------------ //
    // Lifecycle
    // ------------------------------------------------------------------ //

    public override void Initialize() {
        rb = GetComponent<Rigidbody2D>();
        initPos = transform.localPosition;
        initRot = transform.localRotation;
        if (vision == null) vision = GetComponent<ClimberVision>();
        ApplyEnvironmentParameters();
        RegisterEnvironmentParameterCallbacks();
    }

    public override void OnEpisodeBegin() => Reset();

    // ------------------------------------------------------------------ //
    // Environment parameters from YAML
    // ------------------------------------------------------------------ //

    void ApplyEnvironmentParameters() {
        var ep = Academy.Instance.EnvironmentParameters;

        // DecisionRequester
        var dr = GetComponent<DecisionRequester>();
        if (dr != null) {
            dr.DecisionPeriod = Mathf.Max(1, Mathf.RoundToInt(
                ep.GetWithDefault("decision_period", dr.DecisionPeriod)));
            dr.DecisionStep = Mathf.Clamp(
                Mathf.RoundToInt(ep.GetWithDefault("decision_step", dr.DecisionStep)),
                0, Mathf.Max(0, dr.DecisionPeriod - 1));
            float tabd = ep.GetWithDefault(
                "take_actions_between_decisions",
                dr.TakeActionsBetweenDecisions ? 1f : 0f);
            dr.TakeActionsBetweenDecisions = tabd >= 0.5f;
        }

        MaxStep = Mathf.Max(1, Mathf.RoundToInt(ep.GetWithDefault("agent_max_step", MaxStep)));

        // Motor
        motorMultiplier = ep.GetWithDefault("motor_multiplier", motorMultiplier);

        // Reward shaping
        strainThreshold    = ep.GetWithDefault("strain_threshold",    strainThreshold);
        strainPenaltyScale = ep.GetWithDefault("strain_penalty_scale", strainPenaltyScale);
        fallYThreshold     = ep.GetWithDefault("fall_y_threshold",     fallYThreshold);
        fallPenalty        = ep.GetWithDefault("fall_penalty",         fallPenalty);

        // Reward mode
        rewardMode           = Mathf.RoundToInt(ep.GetWithDefault("reward_mode",            rewardMode));
        velocityRewardScale  = ep.GetWithDefault("velocity_reward_scale", velocityRewardScale);
        positionRewardScale  = ep.GetWithDefault("position_reward_scale", positionRewardScale);
        positionHeightCap    = ep.GetWithDefault("position_height_cap",   positionHeightCap);
    }

    void RegisterEnvironmentParameterCallbacks() {
        var ep = Academy.Instance.EnvironmentParameters;
        void OnChange(float _) => ApplyEnvironmentParameters();
        ep.RegisterCallback("decision_period",               OnChange);
        ep.RegisterCallback("decision_step",                 OnChange);
        ep.RegisterCallback("take_actions_between_decisions", OnChange);
        ep.RegisterCallback("agent_max_step",                OnChange);
        ep.RegisterCallback("motor_multiplier",              OnChange);
        ep.RegisterCallback("strain_threshold",              OnChange);
        ep.RegisterCallback("strain_penalty_scale",          OnChange);
        ep.RegisterCallback("fall_y_threshold",              OnChange);
        ep.RegisterCallback("fall_penalty",                  OnChange);
        ep.RegisterCallback("reward_mode",                   OnChange);
        ep.RegisterCallback("velocity_reward_scale",         OnChange);
        ep.RegisterCallback("position_reward_scale",         OnChange);
        ep.RegisterCallback("position_height_cap",           OnChange);
    }

    // ------------------------------------------------------------------ //
    // Observations  (N=2 → 10 proprioception + 32 vision = 42 total)
    // ------------------------------------------------------------------ //

    public override void CollectObservations(VectorSensor sensor) {
        foreach (var a in arms)     sensor.AddObservation(a.Angle);
        foreach (var f in forearms) sensor.AddObservation(f.Angle);
        foreach (var f in forearms) sensor.AddObservation(f.IsGripping ? 1f : 0f);
        sensor.AddObservation(rb.linearVelocity);
        sensor.AddObservation(rb.angularVelocity);
        sensor.AddObservation(transform.rotation.eulerAngles.z);
        vision.CollectVisionObservations(sensor);
    }

    // ------------------------------------------------------------------ //
    // Actions
    // ------------------------------------------------------------------ //

    public override void OnActionReceived(ActionBuffers actions) {
        var cont = actions.ContinuousActions;
        var disc = actions.DiscreteActions;
        int n = arms.Length;

        for (int i = 0; i < n; i++)
            arms[i].SetMotor(cont[i] * motorMultiplier);
        for (int i = 0; i < forearms.Length; i++)
            forearms[i].SetMotor(cont[n + i] * motorMultiplier);
        for (int i = 0; i < forearms.Length; i++)
            forearms[i].SetGrip(disc[i] == 1);

        HandleReward();
        RecordStats();
        HandleLogging();
    }

    public override void Heuristic(in ActionBuffers actionsOut) {
        Assert.IsNotNull(InputManager.I,
            "[Climber] InputManager instance is null. Ensure an InputManager is present in the scene.");

        var cont = actionsOut.ContinuousActions;
        var disc = actionsOut.DiscreteActions;
        int n = arms.Length;

        // Map first two arms to left/right inputs; rest hold at 0
        if (n > 0) cont[0] = InputManager.I.ArmLeft;
        if (n > 1) cont[1] = InputManager.I.ArmRight;

        // Map first two forearms to left/right inputs; rest hold at 0
        if (forearms.Length > 0) cont[n + 0] = InputManager.I.ForearmLeft;
        if (forearms.Length > 1) cont[n + 1] = InputManager.I.ForearmRight;

        if (forearms.Length > 0) disc[0] = InputManager.I.GripLeft  ? 1 : 0;
        if (forearms.Length > 1) disc[1] = InputManager.I.GripRight ? 1 : 0;
    }

    // ------------------------------------------------------------------ //
    // Reward / episode end
    // ------------------------------------------------------------------ //

    void HandleReward() {
        if (rewardMode == 1) {
            // Position reward: higher Y = more reward each step.
            float h = Mathf.Clamp(transform.position.y, 0f, positionHeightCap);
            AddReward(h * positionRewardScale);
        } else {
            // Velocity reward (default): reward upward movement.
            AddReward(rb.linearVelocityY * velocityRewardScale);
        }

        if (transform.position.y < fallYThreshold) {
            AddReward(-fallPenalty);
            EndEpisode();
            return;
        }

        float excess = strain - strainThreshold;
        if (excess > 0f)
            AddReward(-excess * strainPenaltyScale);
    }

    // ------------------------------------------------------------------ //
    // Stats / logging
    // ------------------------------------------------------------------ //

    void RecordStats() {
        var stats = Academy.Instance.StatsRecorder;
        stats.Add("Climber/Height",      transform.position.y);
        stats.Add("Climber/VelocityY",   rb.linearVelocityY);
        stats.Add("Climber/VelocityX",   rb.linearVelocityX);
        stats.Add("Climber/Strain",      strain);
        for (int i = 0; i < forearms.Length; i++)
            stats.Add($"Climber/Gripping{i}", forearms[i].IsGripping ? 1f : 0f);
    }

    void HandleLogging() {
        var step = Academy.Instance.TotalStepCount;
        if (step - _lastLogStep < logInterval) return;
        _lastLogStep = step;
        Debug.Log($"[Climber] Step {step:N0} | Episode {CompletedEpisodes} | Reward {GetCumulativeReward():F4}");
    }

    // ------------------------------------------------------------------ //
    // Reset
    // ------------------------------------------------------------------ //

    void Reset() {
        transform.localPosition = initPos;
        transform.localRotation = initRot;
        rb.linearVelocity  = Vector2.zero;
        rb.angularVelocity = 0f;
        foreach (var a in arms)    a.Reset();
        foreach (var f in forearms) f.Reset();
    }
}
