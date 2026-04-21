using System.Collections;
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
        if (baseArea != null)
            m_TrainingAreaName = baseArea.name;
    }

    /// <summary>
    /// Replication must run after the Python trainer connects; in <see cref="Awake"/> /
    /// <see cref="OnEnable"/>, <see cref="Academy.Instance.IsCommunicatorOn"/> is usually still false,
    /// so YAML <c>env_settings.num_areas</c> was never applied and the bootstrap default (previously 4)
    /// stuck.
    /// </summary>
    IEnumerator Start()
    {
        // Wait until the communicator is up so Academy.NumAreas matches env_settings.num_areas.
        var frame = 0;
        while (Academy.Instance != null)
        {
            if (Academy.Instance.IsCommunicatorOn)
            {
                numAreas = Mathf.Max(1, Academy.Instance.NumAreas);
                break;
            }

#if UNITY_EDITOR
            // Play Mode without mlagents-learn: communicator never connects; don't block long.
            if (frame++ >= 120)
                break;
#else
            if (frame++ >= 6000)
                break;
#endif
            yield return null;
        }

        if (Academy.Instance != null && Academy.Instance.IsCommunicatorOn)
            numAreas = Mathf.Max(1, Academy.Instance.NumAreas);

        if (buildOnly)
        {
#if UNITY_STANDALONE && !UNITY_EDITOR
            AddEnvironments();
#endif
            yield break;
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
