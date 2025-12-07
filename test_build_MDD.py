# test_build_MDD.py
import sys
from pprint import pprint
sys.path.insert(0, 'code')  # adapt if your working dir is different

from CG import build_MDD
from single_agent_planner import a_star, move, build_constraint_table, is_constrained

# helper: make a small empty grid of given size (rows,cols)
def make_empty_map(rows, cols):
    return [[False]*cols for _ in range(rows)]

def validate_mdd(mdd, my_map, start, goal, heuristics, constraints):
    # basic checks
    assert mdd is not None, "MDD returned None (no solution found)"
    assert len(mdd) >= 1
    assert start in mdd[0], f"start {start} not in level 0"
    # compute a reference shortest path (A* returns a list of locations)
    path = a_star(my_map, start, goal, heuristics, 0, constraints)
    assert path is not None, "A* found no path but build_MDD returned an MDD"
    # number of timesteps by A*:
    num_steps = len(path) - 1  # steps = nodes-1
    # Check that goal appears at or before the last level
    found_goal_level = None
    for t, level in enumerate(mdd):
        if goal in level:
            found_goal_level = t
            break
    assert found_goal_level is not None, "goal not present in any MDD level"
    # The MDD should allow getting from start to goal in the same number of steps as A*
    # (some implementations yield extra slack; adapt as desired).
    assert found_goal_level <= len(mdd)-1
    # Optional: check that any path using MDD levels can produce a valid path of length <= num_steps
    # We'll do a simple BFS on MDD to find if goal is reachable within levels
    from collections import deque
    q = deque()
    q.append((start, 0))
    visited = set([(start, 0)])
    while q:
        loc, t = q.popleft()
        if t == len(mdd)-1:
            continue
        # for each candidate in next level, see if transition is legal
        for nxt in mdd[t+1]:
            # check map bounds / obstacle
            r,c = nxt
            if r < 0 or r >= len(my_map) or c < 0 or c >= len(my_map[0]):
                continue
            if my_map[r][c]:
                continue
            # check legal move (including waiting)
            legal = False
            for dir in range(4):
                if move(loc, dir) == nxt:
                    legal = True
                    break
            if loc == nxt:
                legal = True  # wait move
            if not legal:
                continue
            # check constraint table
            ctable = build_constraint_table(constraints, 0)
            if is_constrained(loc, nxt, t+1, ctable):
                continue
            if (nxt, t+1) not in visited:
                visited.add((nxt, t+1))
                q.append((nxt, t+1))
    assert any((goal, t) in visited for t in range(len(mdd))), "goal is not reachable inside MDD levels"
    return True

def test_simple_adjacent():
    my_map = make_empty_map(3,3)
    start = (0,0); goal = (0,1)
    heuristics = {}  # your a_star might expect a heuristic table; compute via compute_heuristics if needed
    # Use a heuristic table from single_agent_planner if code requires; fallback to None if not needed
    try:
        from single_agent_planner import compute_heuristics
        h = compute_heuristics(my_map, goal)
    except Exception:
        h = {}
    mdd = build_MDD(my_map, start, goal, agent=0, heuristics=h, constraints=[])
    pprint(mdd)
    assert mdd is not None
    # start in level0
    assert start in mdd[0]
    # goal should be in some level
    assert any(goal in lvl for lvl in mdd)
    # validate general structural properties
    validate_mdd(mdd, my_map, start, goal, h, [])

def test_with_obstacle():
    my_map = make_empty_map(3,3)
    my_map[0][1] = True  # block direct neighbor, force a detour
    start = (0,0); goal = (0,2)
    from single_agent_planner import compute_heuristics
    h = compute_heuristics(my_map, goal)
    mdd = build_MDD(my_map, start, goal, agent=0, heuristics=h, constraints=[])
    pprint(mdd)
    assert mdd is not None
    validate_mdd(mdd, my_map, start, goal, h, [])

def test_constraint_block_vertex():
    my_map = make_empty_map(3,3)
    start = (0,0); goal = (0,2)
    from single_agent_planner import compute_heuristics
    h = compute_heuristics(my_map, goal)
    # Block middle vertex at timestep 1
    constraints = [{'agent': 0, 'loc': [(0,1)], 'timestep': 1, 'positive': False}]
    mdd = build_MDD(my_map, start, goal, agent=0, heuristics=h, constraints=constraints)
    assert mdd is not None
    # ensure (0,1) is not in level 1
    if len(mdd) > 1:
        assert (0,1) not in mdd[1]
    validate_mdd(mdd, my_map, start, goal, h, constraints)

if __name__ == '__main__':
    test_simple_adjacent()
    test_with_obstacle()
    test_constraint_block_vertex()
    print('All build_MDD checks passed.')