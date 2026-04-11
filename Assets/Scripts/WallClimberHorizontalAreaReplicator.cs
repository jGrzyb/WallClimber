using Unity.MLAgents;
using UnityEngine;

/// <summary>
/// Like ML-Agents' <c>TrainingAreaReplicator</c>, but lays out copies in a single row
/// along +X so all areas sit side-by-side (better for one wide orthographic shot).
/// The first area is the existing <see cref="baseArea"/>; additional areas are instantiated
/// at <c>(separation, 0, 0)</c>, <c>(2*separation, 0, 0)</c>, …
/// </summary>
[DefaultExecutionOrder(-5)]
public class WallClimberHorizontalAreaReplicator : MonoBehaviour
{
    public GameObject baseArea;
    public int numAreas = 1;
    public float separation = 48f;
    public bool buildOnly = true;

    string m_TrainingAreaName = "TrainingArea";

    void Awake()
    {
        // ML-Agents 4.x: use IsCommunicatorOn (Communicator was removed from the public API).
        if (Academy.Instance != null && Academy.Instance.IsCommunicatorOn)
            numAreas = Academy.Instance.NumAreas;

        if (baseArea != null)
            m_TrainingAreaName = baseArea.name;
    }

    void OnEnable()
    {
        if (buildOnly)
        {
#if UNITY_STANDALONE && !UNITY_EDITOR
            AddEnvironments();
#endif
            return;
        }

        AddEnvironments();
    }

    void AddEnvironments()
    {
        if (baseArea == null)
            return;

        for (int i = 1; i < numAreas; i++)
        {
            var area = Instantiate(
                baseArea,
                new Vector3(i * separation, 0f, 0f),
                Quaternion.identity);
            area.name = m_TrainingAreaName;
        }
    }
}
