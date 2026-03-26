using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Assertions;

public class MyCamera : MonoBehaviour
{
    List<Transform> targets;
    [SerializeField] private float offsetY = 2f;
    [SerializeField] private float followSpeed = 10f;
    [SerializeField] private GameObject grid;

    void Start()
    {
        targets = FindObjectsByType<Climber>(FindObjectsSortMode.None).Select(c => c.transform).ToList();
        Assert.IsTrue(targets.Count > 0, "[KUBA] No Climber found in the scene. Ensure a Climber is present for the camera to follow.");
    }

    void LateUpdate()
    {
        Vector3 newPosition = targets.Aggregate(Vector3.zero, (acc, t) => acc + t.position) / targets.Count;
        newPosition.y += offsetY;
        newPosition.z = transform.position.z;
        transform.position = Vector3.Lerp(transform.position, newPosition, Time.deltaTime * followSpeed);
        if (grid != null) {
            Vector3 newGridPosition = Vector3Int.FloorToInt(transform.position / 4f) * 4;
            newGridPosition.z = grid.transform.position.z;
            grid.transform.position = newGridPosition;
        }
    }
}