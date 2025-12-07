from single_agent_planner import a_star, is_constrained, build_constraint_table, move

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

def build_MDD(my_map, start, goal, agent, heuristics, constraints):
    """Build a multi-valued decision diagram (MDD) for `agent` from `start` to `goal` within `max_cost`.

    An MDD is a leveled graph where each level contains the possible locations of the agent at that
    time step, given the constraints. Each node at level t is connected to nodes at level t+1 if
    the agent can move between those locations without violating any constraints.

    Returns the MDD as a list of levels, where each level is a set of locations.
    """
    constraint_table = build_constraint_table(constraints, agent)
    min_cost = len(a_star(my_map, start, goal, heuristics, agent, constraints))
    mdd = [set() for _ in range(min_cost + 1)]
    open_list = []
    closed_list = dict()

    root = {'loc': start, 'timestep': 0, 'g': 0, 'h': heuristics[start]}
    open_list.append(root)
    closed_list[(root['loc'], root['timestep'])] = root

    while len(open_list) > 0:
        curr = open_list.pop(0)
        t_step = curr['timestep']
        f = curr['g'] + curr['h']
        if f <= min_cost:
            mdd[t_step].add(curr['loc'])
        
        if curr['loc'] == goal and curr['timestep'] == min_cost:
            return mdd

        for dir in range(4):
            child_loc = move(curr['loc'], dir)
            if child_loc[0] < 0 or child_loc[0] >= len(my_map) or \
               child_loc[1] < 0 or child_loc[1] >= len(my_map[0]):
                continue
            if my_map[child_loc[0]][child_loc[1]]:
                continue
            child_t = t_step + 1
            if is_constrained(curr['loc'], child_loc, child_t, constraint_table):
                continue
            child = {'loc': child_loc, 'timestep': child_t, 'g': curr['g'] + 1, 'h': heuristics[child_loc]}
            if (child['loc'], child['timestep']) not in closed_list:
                closed_list[(child['loc'], child['timestep'])] = child
                open_list.append(child)
		
        child_loc = curr['loc']
        child_t = t_step + 1
        if not is_constrained(curr['loc'], child_loc, child_t, constraint_table):
            child = {'loc': child_loc, 'timestep': child_t, 'g': curr['g'] + 1, 'h': heuristics[child_loc]}
            if (child['loc'], child['timestep']) not in closed_list:
                closed_list[(child['loc'], child['timestep'])] = child
                open_list.append(child)
    return None
				

def is_cardinal(collision, node, my_map, starts, goals, heuristics):
	"""Return True if `collision` is cardinal in the given `node`.

	A collision is considered cardinal if forbidding the collision (i.e. adding the corresponding
	negative constraint) forces both agents involved to take strictly longer paths than their
	current paths in `node['paths']`.
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

def is_cardinal_from_mdd(collision, mdds, a1, a2):
    """Return True if `collision` is cardinal based on the MDDs of agents a1 and a2.

    A collision is considered cardinal if forbidding the collision (i.e. adding the corresponding
    negative constraint) forces both agents involved to take strictly longer paths than their
    current paths represented by their MDDs.
    """
    timestep = collision['timestep']
    locs1 = mdds[a1][timestep]
    locs2 = mdds[a2][timestep]

    if len(locs1) == 1 and len(locs2) == 1:
        return True
    return False

def build_conflict_graph(node, my_map, starts, goals, heuristics):
    
    num_agents = len(starts)
    adj = {i: set() for i in range(num_agents)}
    mdds = dict()
    for agent in range(num_agents):
        constraints = [c for c in node['constraints'] if c['agent'] == agent]
        mdd = build_MDD(my_map, starts[agent], goals[agent], agent, heuristics[agent], constraints)
        print(f"Agent {agent} MDD levels: {[len(level) for level in mdd]}")
        mdds[agent] = mdd
    
    adj = {i: set() for i in range(num_agents)}
    for collision in node['collisions']:
        a1 = collision['a1']
        a2 = collision['a2']
        if is_cardinal_from_mdd(collision, mdds, a1, a2):
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
		


