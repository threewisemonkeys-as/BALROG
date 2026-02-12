from __future__ import annotations

import re
from dataclasses import dataclass
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

Pos = Tuple[int, int]   # (x, y)
DPos = Tuple[int, int]  # (dx, dy)

# -------------------------
# Parsing
# -------------------------

def parse_grid(s: str) -> List[List[str]]:
    """
    Parse a fixed-width ASCII grid.

    IMPORTANT (MiniHack/NLE):
    - All-space rows are meaningful (screen padding) and must NOT be stripped.
    - Only strip truly empty lines ("") that can appear due to formatting.
    """
    lines = s.splitlines()
    while lines and lines[0] == "":
        lines.pop(0)
    while lines and lines[-1] == "":
        lines.pop()
    if not lines:
        raise ValueError("Empty grid.")
    w = max(len(x) for x in lines)
    return [list(line.ljust(w)) for line in lines]

def find_char(grid: List[List[str]], ch: str) -> Pos:
    for y, row in enumerate(grid):
        for x, xch in enumerate(row):
            if xch == ch:
                return (x, y)
    raise ValueError(f"Character {ch!r} not found.")

def in_bounds(grid: List[List[str]], p: Pos) -> bool:
    x, y = p
    return 0 <= y < len(grid) and 0 <= x < len(grid[0])

def rel(a: Pos, b: Pos) -> DPos:
    return (b[0] - a[0], b[1] - a[1])  # (dx, dy)

def manhattan(d: DPos) -> int:
    return abs(d[0]) + abs(d[1])

def dir8(d: DPos) -> str:
    dx, dy = d
    v = "N" if dy < 0 else ("S" if dy > 0 else "")
    h = "W" if dx < 0 else ("E" if dx > 0 else "")
    return (v + h) if (v or h) else "HERE"

# -------------------------
# Glyph classes (tune as you like)
# -------------------------

PLAYER = "@"
STAIRS_UP = "<"
STAIRS_DOWN = ">"
LAVA = "}"
CORRIDOR = "#"

WALLS: Set[str] = set("-|")
DOORS: Set[str] = set("+")
FLOOR: Set[str] = set(".#")
BLANK: Set[str] = set(" ")

OBJECTS: Set[str] = set(list(r""")([*%_:!?/=,"\$`0-9&"""))

DIRS4: Dict[str, DPos] = {"N": (0, -1), "S": (0, 1), "W": (-1, 0), "E": (1, 0)}
DIRS4_LIST = list(DIRS4.values())

def is_wall(ch: str) -> bool:
    return ch in WALLS

def is_door(ch: str) -> bool:
    return ch in DOORS

def is_lava(ch: str) -> bool:
    return ch == LAVA

def is_stairs(ch: str) -> bool:
    return ch in (STAIRS_UP, STAIRS_DOWN)

def is_floorlike(ch: str) -> bool:
    return ch in FLOOR or is_stairs(ch) or ch == PLAYER

def is_monster(ch: str) -> bool:
    return ch.isalpha()

def is_object(ch: str) -> bool:
    return ch in OBJECTS

def is_passable(ch: str, doors_block: bool = True) -> bool:
    if doors_block and is_door(ch):
        return False
    if is_wall(ch) or is_lava(ch):
        return False
    if ch in BLANK:
        return False
    return is_floorlike(ch) or is_monster(ch) or is_object(ch) or (is_door(ch) and not doors_block)

def describe_cell(ch: str) -> str:
    if ch == PLAYER:
        return "YOU"
    if ch == CORRIDOR:
        return "corridor"
    if is_lava(ch):
        return "LAVA!"
    if is_wall(ch):
        return "wall"
    if is_door(ch):
        return "door"
    if ch == STAIRS_UP:
        return "stairs_up"
    if ch == STAIRS_DOWN:
        return "stairs_down"
    if is_monster(ch):
        return f"monster({ch})"
    if is_object(ch):
        return f"item({ch})"
    if ch in FLOOR:
        return "floor"
    if ch in BLANK:
        return "empty"
    return f"unknown({ch})"

# -------------------------
# Core scans
# -------------------------

def positions_where(grid: List[List[str]], pred) -> List[Pos]:
    out: List[Pos] = []
    for y, row in enumerate(grid):
        for x, ch in enumerate(row):
            if pred(ch):
                out.append((x, y))
    return out

def bbox_rel(origin: Pos, pts: List[Pos]) -> Optional[Dict[str, int]]:
    if not pts:
        return None
    rels = [rel(origin, p) for p in pts]  # (dx, dy)
    dxs = [d[0] for d in rels]
    dys = [d[1] for d in rels]
    min_dx, max_dx = min(dxs), max(dxs)
    min_dy, max_dy = min(dys), max(dys)
    if min_dx == max_dx and min_dy == max_dy:
        return None  # single-cell -> suppress bbox printing
    return {"min_dx": min_dx, "min_dy": min_dy, "max_dx": max_dx, "max_dy": max_dy}

def nearest_rel(origin: Pos, pts: List[Pos]) -> Optional[Dict[str, object]]:
    if not pts:
        return None
    best = min(pts, key=lambda p: manhattan(rel(origin, p)))
    d = rel(origin, best)
    return {"d": d, "dist": manhattan(d), "dir": dir8(d)}  # keep dist internally, don't print it

# -------------------------
# Lava blobs (contiguous components)
# -------------------------

def flood_blobs(grid: List[List[str]], pred) -> List[List[Pos]]:
    seen: Set[Pos] = set()
    blobs: List[List[Pos]] = []
    H, W = len(grid), len(grid[0])

    for y in range(H):
        for x in range(W):
            p = (x, y)
            if p in seen or not pred(grid[y][x]):
                continue
            q = deque([p])
            seen.add(p)
            blob: List[Pos] = []
            while q:
                cx, cy = q.popleft()
                blob.append((cx, cy))
                for dx, dy in DIRS4_LIST:
                    nx, ny = cx + dx, cy + dy
                    np = (nx, ny)
                    if not in_bounds(grid, np) or np in seen:
                        continue
                    if pred(grid[ny][nx]):
                        seen.add(np)
                        q.append(np)
            blobs.append(blob)
    return blobs

# -------------------------
# Path distances (optional, compact)
# -------------------------

def bfs_dist(grid: List[List[str]], start: Pos, doors_block: bool = True) -> Dict[Pos, int]:
    q = deque([start])
    dist: Dict[Pos, int] = {start: 0}
    while q:
        x, y = q.popleft()
        for dx, dy in DIRS4_LIST:
            nx, ny = x + dx, y + dy
            np = (nx, ny)
            if not in_bounds(grid, np) or np in dist:
                continue
            if not is_passable(grid[ny][nx], doors_block=doors_block):
                continue
            dist[np] = dist[(x, y)] + 1
            q.append(np)
    return dist

def nearest_by_path(grid: List[List[str]], origin: Pos, targets: List[Pos], doors_block: bool = True):
    if not targets:
        return None
    dist = bfs_dist(grid, origin, doors_block=doors_block)
    reachable = [p for p in targets if p in dist]
    if not reachable:
        return {"reachable": False}
    best = min(reachable, key=lambda p: dist[p])
    d = rel(origin, best)
    return {"reachable": True, "steps": dist[best], "d": d, "dir": dir8(d)}

# -------------------------
# Compact percept output
# -------------------------

@dataclass(frozen=True)
class CategorySummary:
    count: int
    nearest: Optional[Dict[str, object]] = None
    bbox: Optional[Dict[str, int]] = None

@dataclass(frozen=True)
class LavaBlobSummary:
    size: int
    nearest: Dict[str, object]
    bbox: Optional[Dict[str, int]]

@dataclass(frozen=True)
class Percepts:
    player: Pos
    stairs_up: CategorySummary
    stairs_down: CategorySummary
    doors: CategorySummary
    walls: CategorySummary
    monsters: CategorySummary
    objects: CategorySummary
    lava: CategorySummary
    lava_blobs: List[LavaBlobSummary]
    path_to_down: Optional[Dict[str, object]] = None
    neighbors: Optional[Dict[str, str]] = None
    passable_dirs: Optional[List[str]] = None
    danger_warnings: Optional[List[str]] = None

def compute_percepts(grid_str: str, *, doors_block: bool = True, max_lava_blobs: int = 3) -> Percepts:
    grid = parse_grid(grid_str)
    origin = find_char(grid, PLAYER)

    up_pts = positions_where(grid, lambda ch: ch == STAIRS_UP)
    down_pts = positions_where(grid, lambda ch: ch == STAIRS_DOWN)
    door_pts = positions_where(grid, is_door)
    wall_pts = positions_where(grid, is_wall)
    monster_pts = positions_where(grid, is_monster)
    object_pts = positions_where(grid, is_object)
    lava_pts = positions_where(grid, is_lava)

    blobs = flood_blobs(grid, is_lava)
    blob_summaries: List[LavaBlobSummary] = []
    for blob in blobs:
        n = nearest_rel(origin, blob)
        b = bbox_rel(origin, blob)
        if n is None:
            continue
        blob_summaries.append(LavaBlobSummary(size=len(blob), nearest=n, bbox=b))
    blob_summaries.sort(key=lambda x: x.nearest["dist"])
    blob_summaries = blob_summaries[:max_lava_blobs]

    path_to_down = nearest_by_path(grid, origin, down_pts, doors_block=doors_block) if down_pts else None

    neighbors: Dict[str, str] = {}
    passable: List[str] = []
    danger_warnings: List[str] = []
    ox, oy = origin
    for name, (dx, dy) in DIRS4.items():
        nx, ny = ox + dx, oy + dy
        if in_bounds(grid, (nx, ny)):
            ch = grid[ny][nx]
            neighbors[name] = ch
            if is_passable(ch, doors_block=doors_block):
                passable.append(name)
            if is_lava(ch):
                danger_warnings.append(f"LAVA to {name}")
        else:
            neighbors[name] = "OUT_OF_BOUNDS"

    return Percepts(
        player=origin,
        stairs_up=CategorySummary(len(up_pts), nearest=nearest_rel(origin, up_pts), bbox=bbox_rel(origin, up_pts)),
        stairs_down=CategorySummary(len(down_pts), nearest=nearest_rel(origin, down_pts), bbox=bbox_rel(origin, down_pts)),
        doors=CategorySummary(len(door_pts), nearest=nearest_rel(origin, door_pts), bbox=bbox_rel(origin, door_pts)),
        walls=CategorySummary(len(wall_pts), nearest=nearest_rel(origin, wall_pts), bbox=bbox_rel(origin, wall_pts)),
        monsters=CategorySummary(len(monster_pts), nearest=nearest_rel(origin, monster_pts), bbox=bbox_rel(origin, monster_pts)),
        objects=CategorySummary(len(object_pts), nearest=nearest_rel(origin, object_pts), bbox=bbox_rel(origin, object_pts)),
        lava=CategorySummary(len(lava_pts), nearest=nearest_rel(origin, lava_pts), bbox=bbox_rel(origin, lava_pts)),
        lava_blobs=blob_summaries,
        path_to_down=path_to_down,
        neighbors=neighbors,
        passable_dirs=passable,
        danger_warnings=danger_warnings if danger_warnings else None,
    )

def pretty(percepts: Percepts, *, cursor_pos: Optional[Pos] = None) -> str:
    def fmt_bbox(b: Optional[Dict[str, int]]) -> str:
        if not b:
            return ""
        return f", bbox=(min_dx={b['min_dx']}, min_dy={b['min_dy']}, max_dx={b['max_dx']}, max_dy={b['max_dy']})"

    def fmt_cat(name: str, cs: CategorySummary, *, prefix: str = "") -> str:
        if cs.count == 0:
            return f"{name}: none"
        n = cs.nearest
        b = cs.bbox
        if not n:
            return f"{name}: count={cs.count}{fmt_bbox(b)}"
        pre = (prefix + " ") if prefix else ""
        return f"{name}: {pre}count={cs.count}, nearest={n['d']} ({n['dir']}){fmt_bbox(b)}"

    if cursor_pos is not None:
        # off = (cursor_pos[0] - percepts.player[0], cursor_pos[1] - percepts.player[1])
        # lines = [
        #     f"player at (x={cursor_pos[0]}, y={cursor_pos[1]}) (cursor), map_idx=(x={percepts.player[0]}, y={percepts.player[1]}), offset=(dx={off[0]}, dy={off[1]})"
        # ]
        lines = [
            f"player at (x={cursor_pos[0]}, y={cursor_pos[1]})"
        ]
    else:
        lines = [f"player at (x={percepts.player[0]}, y={percepts.player[1]})"]

    if percepts.danger_warnings:
        lines.append(f"DANGER: {'; '.join(percepts.danger_warnings)} - DO NOT move there!")

    if percepts.neighbors:
        neigh_parts = []
        for d in ["N", "S", "E", "W"]:
            if d in percepts.neighbors:
                ch = percepts.neighbors[d]
                neigh_parts.append(f"{d}={describe_cell(ch)}")
        lines.append(f"adjacent: {', '.join(neigh_parts)}")

    if percepts.passable_dirs:
        lines.append(f"can move: {', '.join(sorted(percepts.passable_dirs))}")
    else:
        lines.append("can move: (trapped - no passable directions!)")

    lines.append(f"coordinates of objects below are presented relative to the player (dx, dy)")

    lines.append(fmt_cat("stairs_up", percepts.stairs_up))
    lines.append(fmt_cat("stairs_down", percepts.stairs_down))
    lines.append(fmt_cat("doors", percepts.doors))
    lines.append(fmt_cat("monsters", percepts.monsters))
    # lines.append(fmt_cat("objects", percepts.objects))
    lines.append(fmt_cat("lava", percepts.lava))

    if percepts.lava_blobs:
        lines.append("lava_blobs (nearest first):")
        for i, lb in enumerate(percepts.lava_blobs, 1):
            b = fmt_bbox(lb.bbox)
            lines.append(f"  #{i}: size={lb.size}, nearest={lb.nearest['d']} ({lb.nearest['dir']}){b}")

    if percepts.path_to_down is not None:
        p = percepts.path_to_down
        if p.get("reachable") is False:
            if percepts.stairs_down.count > 0 and percepts.stairs_down.nearest:
                ns = percepts.stairs_down.nearest
                lines.append(
                    f"path_to_down: BLOCKED (stairs are {ns.get('dir','?')} at d={ns.get('d','?')}, "
                    f"but path is obstructed - likely by lava, walls, or doors)"
                )
            else:
                lines.append("path_to_down: unreachable (stairs not visible or no path exists)")
        else:
            lines.append(f"path_to_down: steps={p['steps']}, d={p['d']} ({p['dir']})")

    return "\n".join(lines)

# -------------------------
# Robust observation parsing (fix map extraction + align to cursor)
# -------------------------

_CURSOR_RE = re.compile(r"\(x=(\d+),\s*y=(\d+)\)")
_HUD_RE = re.compile(r"^\s*(Agent the\b|Dlvl:)", re.IGNORECASE)
_SECTION_RE = re.compile(r"^\s*(message|cursor|inventory|map with descriptions|tips)\s*:", re.IGNORECASE)

def _parse_cursor_pos(observation_text: str) -> Optional[Pos]:
    for line in observation_text.splitlines():
        m = _CURSOR_RE.search(line)
        if m:
            x = int(m.group(1))
            y = int(m.group(2))
            return (x, y)
    return None

def _extract_map_block(observation_text: str, *, min_rows: int = 21) -> Optional[str]:
    lines = observation_text.split("\n")
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("map:"):
            start = i + 1
            break
    if start is None:
        return None

    map_lines: List[str] = []
    for line in lines[start:]:
        if _SECTION_RE.match(line):
            break
        if _HUD_RE.match(line):
            break
        map_lines.append(line)
        if len(map_lines) >= min_rows:
            break

    return "\n".join(map_lines) if map_lines else None

# -------------------------
# Main perceive function (required by evaluator.py)
# -------------------------

def perceive(observation_text: str) -> str:
    """
    Entry point called by the evaluator.

    Fixes:
    - Extract only the dungeon viewport lines (avoid HUD contamination).
    - Keep all-space padding rows in the grid (avoid row-offset).
    - Prefer printing cursor-aligned player position.
    - Use (x,y) everywhere (map coordinate order).
    """
    cursor_pos = _parse_cursor_pos(observation_text)

    min_rows = 21
    if cursor_pos is not None:
        min_rows = max(min_rows, cursor_pos[1] + 1)  # y + 1

    map_str = _extract_map_block(observation_text, min_rows=min_rows)
    if not map_str:
        return "Could not extract map from observation."

    try:
        percepts = compute_percepts(map_str, doors_block=True)
        return pretty(percepts, cursor_pos=cursor_pos)
    except Exception as e:
        return f"Perception failed: {e}"
