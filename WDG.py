import heapq

from single_agent_planner import compute_heuristics, a_star, get_sum_of_cost
from CG import conflict_graph_heuristic
from DG import build_dependency_graph


def _joint_min_sum(node, my_map, start1, goal1, start2, goal2, agent1, agent2, heur1, heur2, constraints):
    """Compute minimal joint sum of costs for agents agent1 and agent2 by running
    a two-agent CBS (using CG greedy matching heuristic) that respects the given
    high-level `constraints` (filtered to these two agents).

    Returns integer sum-of-costs or None if infeasible.
    """
    from cbs import detect_collisions, disjoint_splitting
    # filter constraints for the two agents only
    filtered_constraints = [c for c in node['constraints'] if c['agent'] == agent1 or c['agent'] == agent2]
    # run CBS with CG heuristic in greedy mode
    class TwoAgentCBS:
        def __init__(self, my_map, starts, goals, heuristics):
            self.my_map = my_map
            self.starts = starts
            self.goals = goals
            self.heuristics = heuristics
            self.open_list = []

        def push_node(self, node):
            # compute conflict-graph heuristic and push using f = g + h  
            greedy = True
            h = conflict_graph_heuristic(node, self.my_map, self.starts, self.goals, self.heuristics, greedy)  
            heapq.heappush(self.open_list, (node['cost'] + h, node['cost'], len(node['collisions']), node))

        def find_solution(self, constraints):
            root = {'cost': 0, 'paths': [], 'collisions': [], 'constraints': constraints}
            
            for i in range(2):
                path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i], i, constraints)
                if path is None:
                    return None
                root['paths'].append(path)
            root['cost'] = get_sum_of_cost(root['paths'])
            root['collisions'] = detect_collisions(root['paths'])
            self.push_node(root)

            while len(self.open_list) > 0:
                (_, _, _, node) = heapq.heappop(self.open_list)
                if len(node['collisions']) == 0:
                    return node['cost']
                collision = node['collisions'][0]
                const = disjoint_splitting(collision)
            for c in const:
                child = {'cost': 0, 'paths': list(node['paths']), 'collisions': [], 'constraints': []}
                if c not in node['constraints']:
                    child['constraints'] = list(node['constraints']) + [c]
                else:
                    child['constraints'] = list(node['constraints'])
                agent = c['agent']
                path = a_star(self.my_map, self.starts[agent], self.goals[agent], self.heuristics[agent], agent, child['constraints'])
                if path is not None:
                    child['paths'][agent] = path
                    child['cost'] = get_sum_of_cost(child['paths'])
                    child['collisions'] = detect_collisions(child['paths'])
                if len(child['paths']) == 0:
                    continue
                self.push_node(child)
            return None
                

    cbs = TwoAgentCBS(my_map, [start1, start2], [goal1, goal2], [heur1, heur2])
    return cbs.find_solution(filtered_constraints)
    


def build_weighted_dependency_graph(node, my_map, starts, goals, heuristics):
    """Build the weighted dependency graph (WDG) for the high-level `node`.

    Vertex costs will be derived as the sum of incident edge weights. The function returns
    an adjacency dict where wdg[u][v] = weight(u,v) for dependent pairs only.
    """
    num_of_agents = len(starts)
    dg = build_dependency_graph(node, my_map, starts, goals, heuristics)
    wdg = {i: {} for i in range(num_of_agents)}
    # current sum of costs for pairs
    current_costs = {}
    for i in range(num_of_agents):
        current_costs[i] = len(node['paths'][i]) - 1

    for i in range(num_of_agents - 1):
        for j in range(i + 1, num_of_agents):
            if j not in dg[i]:
                continue
            min_joint = _joint_min_sum(node, my_map, starts[i], goals[i], starts[j], goals[j], i, j,
                                       heuristics[i], heuristics[j], node['constraints'])
            if min_joint is None:
                # treat as high weight (infeasible) — use a large number
                weight = float('inf')
            else:
                weight = min_joint - (current_costs[i] + current_costs[j])
                if weight < 0:
                    weight = 0
            wdg[i][j] = weight
            wdg[j][i] = weight
    return wdg


def _min_cost_vertex_cover_from_wdg(wdg):
    """Compute minimum-cost vertex cover where vertex cost = sum of incident edge weights.

    Uses brute-force/backtracking; returns minimal total vertex cost covering all edges.
    """
    
    vertices = list(wdg.keys())
    edges = []
    v_weight = {v: 0 for v in vertices}
    for u in wdg:
        for v in wdg[u]:
            if u < v:
                edges.append((u, v, wdg[u][v]))
            if u not in v_weight:
                v_weight[u] += wdg[u][v]

    min_cover = vertices

    def all_edges_covered(covered, edges):
        for (u, v, w) in edges:
            if u not in covered and v not in covered:
                return False
        return True

    def backtrack(covered, edges, idx, min_cover, vertices, wdg):
        if all_edges_covered(covered, edges):
            total_weight = sum(v_weight[v] for v in covered)
            if total_weight < sum(v_weight[v] for v in min_cover):
                min_cover.clear()
                min_cover.update(covered)
            return sum(v_weight[v] for v in min_cover)
        # If all vertices have been considered, return infinity (invalid)
        if idx >= len(vertices):
            return float('inf')
        # If all vertices are covered, return infinity (invalid)
        if sum(v_weight[v] for v in covered) >= sum(v_weight[v] for v in min_cover):
            return float('inf')
            
        u = vertices[idx]
        # Option 1: Include u in the vertex cover
        new_covered = covered | {u}
        new_edges = [e for e in edges if u not in e[:2]]
        size_with_u = backtrack(new_covered, new_edges, idx + 1, min_cover, vertices, wdg)
            
        # Option 2: Exclude u from the vertex cover
        size_without_u = float('inf')
        for v in wdg[u]:
            if v not in covered:
                new_covered = covered | {v}
                new_edges = [e for e in edges if v not in e[:2]]
                size_without_u = min(size_without_u, backtrack(new_covered, new_edges, idx + 1, min_cover, vertices, wdg))
            
        return min(size_with_u, size_without_u)
    
    return backtrack(set(), edges, 0, min_cover, vertices, wdg)


def wdg_heuristic(node, my_map, starts, goals, heuristics):
    """
    Return the WDG heuristic value: the minimum-cost vertex cover value derived from WDG.
    """
    wdg = build_weighted_dependency_graph(node, my_map, starts, goals, heuristics)
    return _min_cost_vertex_cover_from_wdg(wdg)