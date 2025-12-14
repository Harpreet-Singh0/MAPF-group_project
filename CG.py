from collections import deque
from single_agent_planner import a_star, is_constrained, build_constraint_table, move


def build_MDD(my_map, start, goal, agent, heuristics, constraints):
    """Build a multi-valued decision diagram (MDD) for `agent` from `start` to `goal` within `max_cost`.

    An MDD is a leveled graph where each level contains the possible locations of the agent at that
    time step, given the constraints. Each node at level t is connected to nodes at level t+1 if
    the agent can move between those locations without violating any constraints.

    Returns the MDD as a list of levels, where each level is a set of locations.
    """
    c_table = build_constraint_table(constraints, agent)
    path = a_star(my_map, start, goal, heuristics, agent, constraints)
    
    if path is None:
        return None
    min_cost = len(path) - 1
    mdd = {'layers': dict(),
           'min_cost': min_cost}
    open_list = deque()
    closed_list = set()
    root = (start, 0)
    open_list.append(root)
    closed_list.add(root)

    # BFS on MDD
    while open_list:
        curr, t = open_list.popleft()
        if t > min_cost:
            continue
        h = heuristics[curr]
        if h+t > min_cost:
             continue
        if t not in mdd['layers']:
            mdd['layers'][t] = []
            mdd['layers'][t].append(curr)
        else:
            if curr not in mdd['layers'][t]:
                mdd['layers'][t].append(curr)
        # check if goal is constrained in future timesteps
        for time in range(t, t + 50):
            if is_constrained(goal, goal, time, c_table):
                goal_constrained = True
                break
            else:
                goal_constrained = False
        # check for termination
        if curr == goal and t == min_cost and not goal_constrained:
            return mdd
        # expand to successors
        for dir in range(4):
            loc = move(curr, dir)
            if loc[0] < 0 or loc[0] >= len(my_map) or \
               loc[1] < 0 or loc[1] >= len(my_map[0]):
                continue
            if my_map[loc[0]][loc[1]]:
                continue
            child_t = t+1
            if not is_constrained(curr, loc, child_t, c_table):
                if (loc, child_t) not in closed_list:
                    open_list.append((loc, child_t))
                    closed_list.add((loc, child_t))
        if not is_constrained(curr, curr, t+1, c_table):
            if (curr, t+1) not in closed_list:
                open_list.append((curr, t+1))
                closed_list.add((curr, t+1))
    return mdd

def get_constraint_for_agent(collision, agent):
    """Return the (negative) constraint that forbids `agent` from participating in `collision`.

    For vertex collisions the loc is the same for both agents. For edge collisions the loc used
    for the second agent is the reversed edge (matching how `standard_splitting` builds constraints).
    """
    # vertex collision
    if len(collision['loc']) == 1:
        loc = collision['loc']
    else:
        # edge collision: constraint for the second agent uses reversed edge
        if agent == collision['a1']:
            loc = collision['loc']
        else:
            loc = collision['loc'][::-1]
    return {'agent': agent,
            'loc': loc,
            'timestep': collision['timestep'],
            'positive': False}

				

def is_cardinal(collision, node, my_map, starts, goals, heuristics):
    """Check if the given `collision` is a cardinal conflict based on whether both agents' path costs
        increase when replanning with the corresponding constraint added.
    """
    a1 = collision['a1']
    a2 = collision['a2']
    
    # current path lengths
    path1_len = len(node['paths'][a1])
    path2_len = len(node['paths'][a2])

    # check for agent a1
    c1 = list(node['constraints']) + [get_constraint_for_agent(collision, a1)]
    new_path1 = a_star(my_map, starts[a1], goals[a1], heuristics[a1], a1, c1)
    if new_path1 is None:
        increased1 = True
    else:
        increased1 = len(new_path1) > path1_len

    # check for agent a2
    c2 = list(node['constraints']) + [get_constraint_for_agent(collision, a2)]
    new_path2 = a_star(my_map, starts[a2], goals[a2], heuristics[a2], a2, c2)
    if new_path2 is None:
        increased2 = True
    else:
        increased2 = len(new_path2) > path2_len
        
    return increased1 and increased2

def is_cardinal_from_mdd(collision, mdd1, mdd2):
    """Check if the given `collision` is a cardinal conflict based on whether both agents are
        constrained to a single location at the collision timestep in their MDDs.
    """
    loc = collision['loc']
    t = collision['timestep']
    
    if len(loc) == 1:
        try:
            if len(mdd1['layers'][t]) == 1 and mdd1['layers'][t] == loc:
                agent1_constrained = True
            else:
                agent1_constrained = False
        except Exception:
            agent1_constrained = True
        try:
            if len(mdd2['layers'][t]) == 1 and mdd2['layers'][t] == loc:
                agent2_constrained = True
            else:
                agent2_constrained = False
        except Exception:
            agent2_constrained = True
        return agent1_constrained and agent2_constrained
    
    else:
        try:
            if len(mdd1['layers'][t]) == 1 and mdd1['layers'][t] == loc[0]:
                agent1_constrained = True
            else:
                agent1_constrained = False
        except Exception:
            agent1_constrained = True
        try:
            if len(mdd2['layers'][t]) == 1 and mdd2['layers'][t] == loc[1]:
                agent2_constrained = True
            else:
                agent2_constrained = False
        except Exception:
            agent2_constrained = True
        return agent1_constrained and agent2_constrained    
    
        

def build_conflict_graph(node, my_map, starts, goals, heuristics):
    """Build the conflict graph (CG) for the high-level `node`."""
    num_agents = len(starts)
    adj = {i: set() for i in range(num_agents)}
    for collision in node['collisions']:
        a1 = collision['a1']
        a2 = collision['a2']
        mdds = dict()
        for agent in range(num_agents):
            constraints = [c for c in node['constraints']] # if c['agent'] == agent]
            mdd = build_MDD(my_map, starts[agent], goals[agent], agent, heuristics[agent], constraints)
            mdds[agent] = mdd
        
        if is_cardinal_from_mdd(collision, mdds[a1], mdds[a2]):
        #if is_cardinal(collision, node, my_map, starts, goals, heuristics):
            adj[a1].add(a2)
            adj[a2].add(a1)
    
    return adj

def greedy_maximal_matching_size(adj):
	"""Compute a (fast) maximal matching and return its size.

	This is a simple greedy algorithm that yields a matching whose size is <= maximum matching size.
	Using this as a lower bound is admissible for a CBS heuristic (it may be weaker than optimal).
	"""
	matched = set()
	matches = 0
	# iterate over vertices in increasing order
	for u in adj:
		if u in matched:
			continue
		for v in adj[u]:
			if v in matched:
				continue
			# take the edge (u,v)
			matched.add(u)
			matched.add(v)
			matches += 1
			break
	return matches
    
def all_edges_covered(covered, edges):
    """Check if all edges are covered by the covered set of vertices."""
    for u, v in edges:
        if u not in covered and v not in covered:
            return False
    return True

def backtrack(covered, edges, idx, min_cover, vertices, adj):
    """Backtracking helper function to compute minimum vertex cover size."""
    ### Base cases
    # If there are no edges left, return the size of the covered set
    if all_edges_covered(covered, edges):
        if len(covered) < len(min_cover):
            min_cover = covered
        return len(min_cover)
    # If all vertices have been considered, return infinity (invalid)
    if idx >= len(vertices):
        return float('inf')
    # If all vertices are covered, return infinity (invalid)
    if len(covered) >= len(min_cover):
        return float('inf')
        
    u = vertices[idx]
    # Option 1: Include u in the vertex cover
    new_covered = covered | {u}
    new_edges = [e for e in edges if u not in e]
    size_with_u = backtrack(new_covered, new_edges, idx + 1, min_cover, vertices, adj)
        
    # Option 2: Exclude u from the vertex cover
    size_without_u = float('inf')
    for v in adj[u]:
        if v not in covered:
            new_covered = covered | {v}
            new_edges = [e for e in edges if v not in e]
            size_without_u = min(size_without_u, backtrack(new_covered, new_edges, idx + 1, min_cover, vertices, adj))
        
    return min(size_with_u, size_without_u)

def get_minimum_vertex_cover_size(adj):
    """Compute the size of a minimum vertex cover from the conflict graph adjacency `adj`.
    
    This function uses a backtracking approach to find the minimum vertex cover.
    """
    vertices = list(adj.keys())
    edges = []
    for u in adj:
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    min_cover = vertices
    return backtrack(set(), edges, 0, min_cover, vertices, adj)

def conflict_graph_heuristic(node, my_map, starts, goals, heuristics, greedy):
    """Compute a heuristic value from the conflict graph for `node`.

    The heuristic can be the size of a greedy maximal matching in the conflict graph of cardinal
    conflicts. This gives a (fast) admissible lower bound on the number of additional cost
    increments required to resolve conflicts in this node.
    Alternatively, the heuristic can be the size of a minimum vertex cover in the conflict graph,
    which is more accurate but computationally more expensive.
    """
    adj = build_conflict_graph(node, my_map, starts, goals, heuristics)
    if greedy:
        return greedy_maximal_matching_size(adj)
    else:
        return get_minimum_vertex_cover_size(adj)
		


