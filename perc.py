from __future__ import annotations
from dataclasses import dataclass, asdict
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

Pos = Tuple[int, int]   # (row, col)
DPos = Tuple[int, int]  # (drow, dcol)

# -------------------------
# Parsing
# -------------------------

def parse_grid(s: str) -> List[List[str]]:
    lines = s.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        raise ValueError("Empty grid.")
    w = max(len(x) for x in lines)
    return [list(line.ljust(w)) for line in lines]

def find_char(grid: List[List[str]], ch: str) -> Pos:
    for r, row in enumerate(grid):
        for c, x in enumerate(row):
            if x == ch:
                return (r, c)
    raise ValueError(f"Character {ch!r} not found.")

def in_bounds(grid: List[List[str]], p: Pos) -> bool:
    r, c = p
    return 0 <= r < len(grid) and 0 <= c < len(grid[0])

def rel(a: Pos, b: Pos) -> DPos:
    return (b[0] - a[0], b[1] - a[1])

def manhattan(d: DPos) -> int:
    return abs(d[0]) + abs(d[1])

def dir8(d: DPos) -> str:
    dr, dc = d
    v = "N" if dr < 0 else ("S" if dr > 0 else "")
    h = "W" if dc < 0 else ("E" if dc > 0 else "")
    return (v + h) if (v or h) else "HERE"

# -------------------------
# Glyph classes (tune as you like)
# -------------------------

PLAYER = "@"
STAIRS_UP = "<"
STAIRS_DOWN = ">"
LAVA = "}"
WALLS: Set[str] = set("-|")
DOORS: Set[str] = set("+")          # treat + as door
FLOOR: Set[str] = set(".")
BLANK: Set[str] = set(" ")

# “Objects”: common NetHack glyphs (not exhaustive, but good default)
OBJECTS: Set[str] = set(list(r""")([*%_:!?/=,"\$`0-9&"""))

DIRS4: Dict[str, DPos] = {"N": (-1, 0), "S": (1, 0), "W": (0, -1), "E": (0, 1)}
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
    # Letters are a good heuristic for monsters/pets in many ASCII roguelikes.
    return ch.isalpha()

def is_object(ch: str) -> bool:
    return ch in OBJECTS

def is_passable(ch: str, doors_block: bool = True) -> bool:
    if doors_block and is_door(ch):
        return False
    if is_wall(ch) or is_lava(ch):
        return False
    # allow floor, stairs, player, monsters, objects (agent can step onto them)
    return ch in BLANK or is_floorlike(ch) or is_monster(ch) or is_object(ch)

# -------------------------
# Core scans
# -------------------------

def positions_where(grid: List[List[str]], pred) -> List[Pos]:
    out: List[Pos] = []
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if pred(ch):
                out.append((r, c))
    return out

def bbox_rel(origin: Pos, pts: List[Pos]) -> Optional[Dict[str, DPos]]:
    if not pts:
        return None
    rels = [rel(origin, p) for p in pts]
    drs = [d[0] for d in rels]
    dcs = [d[1] for d in rels]
    return {
        "min": (min(drs), min(dcs)),
        "max": (max(drs), max(dcs)),
    }

def nearest_rel(origin: Pos, pts: List[Pos]) -> Optional[Dict[str, object]]:
    if not pts:
        return None
    best = min(pts, key=lambda p: manhattan(rel(origin, p)))
    d = rel(origin, best)
    return {"d": d, "dist": manhattan(d), "dir": dir8(d)}

# -------------------------
# Lava blobs (contiguous components)
# -------------------------

def flood_blobs(grid: List[List[str]], pred) -> List[List[Pos]]:
    seen: Set[Pos] = set()
    blobs: List[List[Pos]] = []
    H, W = len(grid), len(grid[0])

    for r in range(H):
        for c in range(W):
            p = (r, c)
            if p in seen or not pred(grid[r][c]):
                continue
            q = deque([p])
            seen.add(p)
            blob: List[Pos] = []
            while q:
                x = q.popleft()
                blob.append(x)
                xr, xc = x
                for dr, dc in DIRS4_LIST:
                    y = (xr + dr, xc + dc)
                    if not in_bounds(grid, y) or y in seen:
                        continue
                    yr, yc = y
                    if pred(grid[yr][yc]):
                        seen.add(y)
                        q.append(y)
            blobs.append(blob)
    return blobs

# -------------------------
# Path distances (optional, compact)
# -------------------------

def bfs_dist(grid: List[List[str]], start: Pos, doors_block: bool = True) -> Dict[Pos, int]:
    q = deque([start])
    dist: Dict[Pos, int] = {start: 0}
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4_LIST:
            nr, nc = r + dr, c + dc
            np = (nr, nc)
            if not in_bounds(grid, np) or np in dist:
                continue
            if not is_passable(grid[nr][nc], doors_block=doors_block):
                continue
            dist[np] = dist[(r, c)] + 1
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
    bbox: Optional[Dict[str, DPos]] = None

@dataclass(frozen=True)
class LavaBlobSummary:
    size: int
    nearest: Dict[str, object]
    bbox: Dict[str, DPos]

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
    # Path-to-goal convenience (optional)
    path_to_down: Optional[Dict[str, object]] = None
    # Immediate surroundings (what's in each cardinal direction)
    neighbors: Optional[Dict[str, str]] = None
    # Action affordances (which directions can be moved to)
    passable_dirs: Optional[List[str]] = None

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

    # Lava blobs summarized by closeness to player; keep only a few
    blobs = flood_blobs(grid, is_lava)
    blob_summaries: List[LavaBlobSummary] = []
    for blob in blobs:
        n = nearest_rel(origin, blob)
        b = bbox_rel(origin, blob)
        if n is None or b is None:
            continue
        blob_summaries.append(LavaBlobSummary(size=len(blob), nearest=n, bbox=b))
    blob_summaries.sort(key=lambda x: x.nearest["dist"])
    blob_summaries = blob_summaries[:max_lava_blobs]

    path_to_down = nearest_by_path(grid, origin, down_pts, doors_block=doors_block) if down_pts else None

    # Compute immediate neighbors (what's in each cardinal direction)
    neighbors = {}
    passable = []
    r, c = origin
    for name, (dr, dc) in DIRS4.items():
        nr, nc = r + dr, c + dc
        if in_bounds(grid, (nr, nc)):
            ch = grid[nr][nc]
            neighbors[name] = ch
            if is_passable(ch, doors_block=doors_block):
                passable.append(name)
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
    )

def pretty(percepts: Percepts) -> str:
    # Compact, human-readable (no long coord lists)
    def fmt_cat(name: str, cs: CategorySummary) -> str:
        if cs.count == 0:
            return f"{name}: none"
        n = cs.nearest
        b = cs.bbox
        return (
            f"{name}: count={cs.count}, nearest={n['d']} ({n['dir']}, L1={n['dist']}), "
            f"bbox=[{b['min']}..{b['max']}]"
        )

    lines = [f"player at {percepts.player}"]

    # Show immediate surroundings and passable directions
    if percepts.neighbors:
        # Format neighbors compactly
        neigh_str = ", ".join(f"{dir}:{ch}" for dir, ch in sorted(percepts.neighbors.items()))
        lines.append(f"adjacent cells: {neigh_str}")

    if percepts.passable_dirs:
        lines.append(f"can move: {', '.join(sorted(percepts.passable_dirs))}")
    else:
        lines.append("can move: (trapped - no passable directions!)")

    lines.append(fmt_cat("stairs_up", percepts.stairs_up))
    lines.append(fmt_cat("stairs_down", percepts.stairs_down))
    lines.append(fmt_cat("doors", percepts.doors))
    lines.append(fmt_cat("monsters", percepts.monsters))
    lines.append(fmt_cat("objects", percepts.objects))
    lines.append(fmt_cat("lava", percepts.lava))

    if percepts.lava_blobs:
        lines.append("lava_blobs (nearest first):")
        for i, lb in enumerate(percepts.lava_blobs, 1):
            lines.append(
                f"  #{i}: size={lb.size}, nearest={lb.nearest['d']} ({lb.nearest['dir']}, L1={lb.nearest['dist']}), "
                f"bbox=[{lb.bbox['min']}..{lb.bbox['max']}]"
            )

    if percepts.path_to_down is not None:
        p = percepts.path_to_down
        if p.get("reachable") is False:
            # Provide more context about why it's unreachable
            if percepts.stairs_down.count > 0 and percepts.stairs_down.nearest:
                nearest_stairs = percepts.stairs_down.nearest
                dir_to_stairs = nearest_stairs.get("dir", "UNKNOWN")
                dist_to_stairs = nearest_stairs.get("dist", "?")
                lines.append(f"path_to_down: BLOCKED (stairs are {dir_to_stairs} at L1={dist_to_stairs}, but path is obstructed - likely by lava, walls, or doors)")
            else:
                lines.append("path_to_down: unreachable (stairs not visible or no path exists)")
        else:
            lines.append(f"path_to_down: steps={p['steps']}, d={p['d']} ({p['dir']})")

    return "\n".join(lines)

# -------------------------
# Example usage:
# p = compute_percepts(screen_text, doors_block=True)
# print(pretty(p))
# -------------------------
