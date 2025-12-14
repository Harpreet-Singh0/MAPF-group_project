from collections import deque

from single_agent_planner import build_constraint_table, is_constrained, move
from CG import build_MDD, get_minimum_vertex_cover_size

# Module-level caches
#mdd_cache = {}             # key -> MDD for a single agent under constraints
#joint_min_sum_cache = {}   # key -> min sum-of-costs for agent pair
#
#def normalize_constraints_key(constraints, agents=None):
#    """
#    Turn constraints list into a canonical tuple key.
#    If 'agents' is provided (iterable), only include constraints for those agents.
#    """
#    items = []
#    for c in constraints:
#        if agents is not None and c['agent'] not in agents:
#            continue
#        # turn loc list into tuple-of-tuples to be hashable
#        loc_tuple = tuple(tuple(l) for l in c['loc'])
#        items.append((c['agent'], loc_tuple, c['timestep'], bool(c.get('positive', False))))
#    items.sort()
#    return tuple(items)
#
#def mdd_cache_get(agent, start, goal, constraints):
#    key = (agent, start, goal, normalize_constraints_key(constraints, agents=[agent]))
#    return mdd_cache.get(key)
#
#def mdd_cache_set(agent, start, goal, constraints, mdd):
#    key = (agent, start, goal, normalize_constraints_key(constraints, agents=[agent]))
#    mdd_cache[key] = mdd

def extend_mdd(mdd, T):
	"""Extend MDD dict (t->set(locs)) to depth T by repeating last level (goal waits).

	Input `mdd` is a dict mapping timestep->set(loc). Returns new dict with keys 0..T.
	"""
	if mdd is None:
		return None
	max_t = mdd['min_cost']
	extended = {}
	for t in range(0, T + 1):
		if t in mdd['layers']:
			extended[t] = set(mdd['layers'][t])
		else:
			extended[t] = set(mdd['layers'][max_t])
	return extended


def successors_in_mdd(mdd_ext, loc, t, my_map, constraint_table):
	"""Return list of successor locations at time t+1 for `loc` that are in mdd_ext[t+1]."""
	successors = []
	T = max(mdd_ext.keys())
	if t >= T:
		return successors
	for dir in range(4):
		loc2 = move(loc, dir)
		# bounds/obstacles are already filtered when building MDD, but double-check
		if loc2 in mdd_ext[t + 1] and not is_constrained(loc, loc2, t + 1, constraint_table):
			successors.append(loc2)
	# wait
	if loc in mdd_ext[t + 1] and not is_constrained(loc, loc, t + 1, constraint_table):
		successors.append(loc)
	return successors


def pair_has_joint_path(my_map, start1, goal1, start2, goal2, agent1, agent2, heur1, heur2, constraints):
	"""Return True if there exists a collision-free joint path where each agent follows an
	individual shortest-length trajectory (i.e., stays within its MDD levels).

	We build each agent's MDD (using `build_MDD`) under the provided `constraints` and then
	perform a BFS on the product (loc1, loc2, t) for t=0..T (T = max depth).
	"""
	# build MDDs for both agents under the same set of high-level constraints
	#mdd1 = mdd_cache_get(agent1, start1, goal1, constraints)
	#if mdd1 is None:
	#	mdd1 = build_MDD(my_map, start1, goal1, agent1, heur1, constraints)
	#	mdd_cache_set(agent1, start1, goal1, constraints, mdd1)
	#mdd2 = mdd_cache_get(agent2, start2, goal2, constraints)
	#if mdd2 is None:
	#	mdd2 = build_MDD(my_map, start2, goal2, agent2, heur2, constraints)
	#	mdd_cache_set(agent2, start2, goal2, constraints, mdd2)
	mdd1 = build_MDD(my_map, start1, goal1, agent1, heur1, constraints)
	mdd2 = build_MDD(my_map, start2, goal2, agent2, heur2, constraints)
	if mdd1 is None or mdd2 is None:
		return False

	T1 = mdd1['min_cost']
	T2 = mdd2['min_cost']
	T = max(T1, T2)
	m1 = extend_mdd(mdd1, T)
	m2 = extend_mdd(mdd2, T)

	# build constraint tables per agent for quick checks
	ctable1 = build_constraint_table(constraints, agent1)
	ctable2 = build_constraint_table(constraints, agent2)
	joint_MDD = {}
	joint_MDD[0] = []
	joint_MDD[0].append((start1, start2))

	# BFS on product space
	start_pair = (start1, start2, 0)
	open_list = deque([start_pair])
	closed_list = set([start_pair])
	while open_list:
		loc1, loc2, t = open_list.popleft()
		if t == T:
			if loc1 == goal1 and loc2 == goal2:
				return True
			continue
		# iterate over successors for both agents independently, restricted to their MDDs
		successors1 = successors_in_mdd(m1, loc1, t, my_map, ctable1)
		successors2 = successors_in_mdd(m2, loc2, t, my_map, ctable2)
		for next_loc1 in successors1:
			for next_loc2 in successors2:
				# check for pairwise collisions at timestep t+1
				# vertex collision
				if next_loc1 == next_loc2:
					continue
				# edge collision (swap)
				if next_loc1 == loc2 and next_loc2 == loc1:
					continue
				new_state = (next_loc1, next_loc2, t + 1)
				if t+1 not in joint_MDD:
					joint_MDD[t+1] = []
					joint_MDD[t+1].append((next_loc1, next_loc2))
				if new_state not in closed_list:
					closed_list.add(new_state)
					open_list.append(new_state)
	
	return False

def build_dependency_graph(node, my_map, starts, goals, heuristics):
	"""Build the dependency graph (DG) for the high-level `node`.

	Two agents i,j are dependent iff there is NO joint MDD path that allows both to reach
	their goals at their respective shortest-path time horizons under the node's constraints.
	"""
	num_of_agents = len(starts)
	adj = {i: set() for i in range(num_of_agents)}
	constraints = [c for c in node['constraints']]
	for i in range(num_of_agents - 1):
		for j in range(i + 1, num_of_agents):
			dependent = not pair_has_joint_path(my_map, starts[i], goals[i], starts[j], goals[j], i, j,
												heuristics[i], heuristics[j], constraints)
			if dependent:
				adj[i].add(j)
				adj[j].add(i)
	return adj


def dg_heuristic(node, my_map, starts, goals, heuristics):
	"""Return the DG heuristic value: the minimum vertex cover size of the dependency graph."""
	dg = build_dependency_graph(node, my_map, starts, goals, heuristics)
	# reuse the minimum-vertex-cover routine from CG (backtracking)
	return get_minimum_vertex_cover_size(dg)


