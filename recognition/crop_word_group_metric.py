"""Post-hoc oracle evaluation with one output per group AND per original crop."""
from __future__ import annotations

from collections import deque


def grouped_oracle(groups: list[dict], witnesses: dict[int, list[str]],
                   gt_count: int) -> set[int]:
    """Maximum flow: source -> group -> crop -> GT -> sink.

    Split each crop into two nodes of capacity one. Otherwise a shared reading
    could spuriously recognize two spatial instances of the same GT spelling.
    Ground truth is accepted only by this evaluation module, never the grouper.
    """
    if gt_count < 0 or any(not 0 <= gi < gt_count for gi in witnesses):
        raise ValueError("invalid ground-truth population")
    available = {cid for group in groups for cid in group["candidate_ids"]}
    ids = sorted(available.intersection(cid for values in witnesses.values() for cid in values))
    crop_index = {cid: i for i, cid in enumerate(ids)}
    source = 0
    crop_start = 1 + len(groups)
    gt_start = crop_start + 2 * len(ids)
    sink = gt_start + gt_count
    graph = [[] for _ in range(sink + 1)]

    def edge(left, right):
        forward = [right, len(graph[right]), 1]
        backward = [left, len(graph[left]), 0]
        graph[left].append(forward)
        graph[right].append(backward)

    for group_index, group in enumerate(groups, 1):
        edge(source, group_index)
        for cid in sorted(set(group["candidate_ids"]) & set(crop_index)):
            edge(group_index, crop_start + 2 * crop_index[cid])
    for cid, index in crop_index.items():
        edge(crop_start + 2 * index, crop_start + 2 * index + 1)
    for gi, cids in sorted(witnesses.items()):
        for cid in sorted(set(cids) & set(crop_index)):
            edge(crop_start + 2 * crop_index[cid] + 1, gt_start + gi)
    gt_edges = []
    for gi in range(gt_count):
        gt_edges.append((gt_start + gi, len(graph[gt_start + gi])))
        edge(gt_start + gi, sink)
    # Short augmenting paths, deterministic BFS, integral capacities.
    while True:
        previous = {source: None}
        queue = deque([source])
        while queue and sink not in previous:
            current = queue.popleft()
            for index, (target, _, capacity) in enumerate(graph[current]):
                if capacity and target not in previous:
                    previous[target] = (current, index)
                    queue.append(target)
        if sink not in previous:
            break
        current = sink
        while current != source:
            left, index = previous[current]
            forward = graph[left][index]
            forward[2] -= 1
            graph[current][forward[1]][2] += 1
            current = left
    return {gi for gi, (node, index) in enumerate(gt_edges) if graph[node][index][2] == 0}
