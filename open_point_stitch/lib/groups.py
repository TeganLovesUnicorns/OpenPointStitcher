"""Joining items into groups: which walls are one wall, which scans are linked, which surfaces are one surface."""

from __future__ import annotations

from typing import Callable, Iterable, Optional


def connected_groups(n: int, edges: Iterable[tuple[int, int]] = (),
                     linked: Optional[Callable[[int, int], bool]] = None) -> list[list[int]]:
    """Items 0..n-1 joined wherever an edge or linked(i, j) says so, chains included. linked is asked once for each
    pair i < j not already in one group. Groups come in the order of their first item, members in index order."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        parent[find(a)] = find(b)
    if linked is not None:
        for i in range(n):
            for j in range(i + 1, n):
                if find(i) != find(j) and linked(i, j):
                    parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())
