using UnityEngine;

public class InputManager : MonoBehaviour
{
    public static InputManager I { get; private set; }
    public InputSystem_Actions InputActions { get; private set; }
    public float ArmLeft => InputActions.Player.ArmLeft.ReadValue<float>();
    public float ArmRight => -InputActions.Player.ArmRight.ReadValue<float>();
    public float ForearmLeft => InputActions.Player.ForearmLeft.ReadValue<float>();
    public float ForearmRight => -InputActions.Player.ForearmRight.ReadValue<float>();
    public bool GripLeft => InputActions.Player.GripLeft.ReadValue<float>() > 0.5f;
    public bool GripRight => InputActions.Player.GripRight.ReadValue<float>() > 0.5f;
    void Awake()
    {
        if (I != null && I != this) {
            Destroy(gameObject);
        } else {
            I = this;
            DontDestroyOnLoad(gameObject);
            InputActions = new InputSystem_Actions();
            InputActions.Enable();
        }
    }
}