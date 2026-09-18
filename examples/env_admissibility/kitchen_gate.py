"""A domain-specific admissibility gate for a toy cooking gridworld.

A study comparing agents on a kitchen layout can only say something about
coordination if every recipe in the layout can actually be completed. If an
ingredient station is walled off, every agent scores zero on that recipe and a
"no difference" result is an artifact of the environment, not of the agents.
"""

from __future__ import annotations

from collections import deque

from preregister import GateContext, custom_gate

LAYOUTS: dict[str, list[str]] = {
    "open_kitchen": [
        "##########",
        "#S   O  T#",
        "#  ##    #",
        "#O    P  #",
        "#   ##  D#",
        "##########",
    ],
    "split_kitchen": [
        "##########",
        "#S   O #T#",
        "#  ##   ##",
        "#O    P  #",
        "#   ##  D#",
        "##########",
    ],
}

STATIONS = {"O": "onion", "T": "tomato", "P": "pot", "D": "delivery"}

RECIPES: dict[str, set[str]] = {
    "onion_soup": {"onion", "pot", "delivery"},
    "tomato_soup": {"tomato", "pot", "delivery"},
}


def reachable_stations(layout: list[str]) -> set[str]:
    """Stations adjacent to a floor cell reachable from the start cell."""
    grid = [list(row) for row in layout]
    start = next((r, c) for r, row in enumerate(grid) for c, ch in enumerate(row) if ch == "S")
    seen = {start}
    queue = deque([start])
    found: set[str] = set()
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < len(grid) and 0 <= nc < len(grid[nr])):
                continue
            ch = grid[nr][nc]
            if ch in STATIONS:
                found.add(STATIONS[ch])
            elif ch != "#" and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return found


@custom_gate("RecipesCompletable")
def recipes_completable(ctx: GateContext) -> tuple[bool, str]:
    problems = []
    for cname in ctx.prediction.parsed.conditions:
        config = ctx.study.conditions[cname].config or {}
        layout_name = config.get("layout")
        if layout_name not in LAYOUTS:
            problems.append(f"{cname}: unknown layout {layout_name!r}")
            continue
        stations = reachable_stations(LAYOUTS[layout_name])
        for recipe in config.get("recipes", sorted(RECIPES)):
            missing = RECIPES[recipe] - stations
            if missing:
                problems.append(
                    f"{cname}: recipe {recipe} on {layout_name} needs unreachable {sorted(missing)}"
                )
    if problems:
        return False, "; ".join(problems)
    return True, "every recipe in every layout is completable"
