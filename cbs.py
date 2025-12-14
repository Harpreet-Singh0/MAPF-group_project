import time as timer
import heapq
import random
from single_agent_planner import compute_heuristics, a_star, get_location, get_sum_of_cost, a_star_lazy
from CG import conflict_graph_heuristic
from DG import dg_heuristic
from WDG import wdg_heuristic

def detect_collision(path1, path2):
    """Return the first collision between two paths."""
    for t in range(max(len(path1), len(path2))):
         loc1 = get_location(path1, t)
         loc2 = get_location(path2, t)
         # Vertex collision
         if loc1 == loc2:
             return {'loc': [loc1], 'timestep': t}
         # Edge collision
         if t > 0:
             prev_loc1 = get_location(path1, t - 1)
             prev_loc2 = get_location(path2, t - 1)
             if loc1 == prev_loc2 and loc2 == prev_loc1:
                 return {'loc': [prev_loc1, loc1], 'timestep': t}
    return None


def detect_collisions(paths):
    """Return a list of all collisions between pairs of paths."""
    collisions = []
    num_of_agents = len(paths)
    for path1 in range(num_of_agents - 1):
        for path2 in range(path1 + 1, num_of_agents):
            collision = detect_collision(paths[path1], paths[path2])
            if collision is not None:
                collisions.append({'a1': path1,
                                   'a2': path2,
                                   'loc': collision['loc'],
                                   'timestep': collision['timestep']})
    return collisions


def standard_splitting(collision):

    constraints = []
    # Vertex collision
    if len(collision['loc']) == 1:
        constraints.append({'agent': collision['a1'],
                            'loc': collision['loc'],
                            'timestep': collision['timestep'],
                            'positive': False})
        constraints.append({'agent': collision['a2'],
                            'loc': collision['loc'],
                            'timestep': collision['timestep'],
                            'positive': False})
    # Edge collision
    else:
        constraints.append({'agent': collision['a1'],
                            'loc': collision['loc'],
                            'timestep': collision['timestep'],
                            'positive': False})
        constraints.append({'agent': collision['a2'],
                            'loc': collision['loc'][::-1],
                            'timestep': collision['timestep'],
                            'positive': False})
    return constraints


def disjoint_splitting(collision):
    constraints = []
    # Vertex collision
    if len(collision['loc']) == 1:
        agent = random.randint(0,1)
        constraints.append({'agent': collision['a'+str(agent+1)],
                            'loc': collision['loc'],
                            'timestep': collision['timestep'],
                            'positive': True})
        constraints.append({'agent': collision['a'+str(agent+1)],
                            'loc': collision['loc'],
                            'timestep': collision['timestep'],
                            'positive': False})
    # Edge collision
    else:
        agent = random.randint(0,1)
        constraints.append({'agent': collision['a'+str(agent+1)],
                            'loc': collision['loc'],
                            'timestep': collision['timestep'],
                            'positive': True})
        constraints.append({'agent': collision['a'+str(agent+1)],
                            'loc': collision['loc'],
                            'timestep': collision['timestep'],
                            'positive': False})
    return constraints

def paths_violate_constraint(constraint, paths):
    assert constraint['positive'] is True
    rst = []
    for i in range(len(paths)):
        if i == constraint['agent']:
            continue
        curr = get_location(paths[i], constraint['timestep'])
        prev = get_location(paths[i], constraint['timestep'] - 1)
        if len(constraint['loc']) == 1:  # vertex constraint
            if constraint['loc'][0] == curr:
                rst.append(i)
        else:  # edge constraint
            if constraint['loc'][0] == prev or constraint['loc'][1] == curr \
                    or constraint['loc'] == [curr, prev]:
                rst.append(i)
    return rst

# MDD CACHE FOR MEMOIZATION

class MDDCache:
    """Cache for MDD computations."""
    
    def __init__(self):
        self.mdd_cache = {}
    
    def get_mdd_key(self, my_map, start, goal, agent, constraints):
        """Generate a unique key for MDD parameters."""
        constraint_str = str(sorted([str(c) for c in constraints]))
        return f"{str(my_map)}:{start}:{goal}:{agent}:{constraint_str}"
    
    def get_cached_mdd(self, my_map, start, goal, agent, constraints, heuristics):
        """Get cached MDD or compute and cache if not exists."""
        from CG import build_MDD
        key = self.get_mdd_key(my_map, start, goal, agent, constraints)
        
        if key not in self.mdd_cache:
            self.mdd_cache[key] = build_MDD(my_map, start, goal, agent, heuristics, constraints)
        
        return self.mdd_cache[key]
    
    def clear(self):
        """Clear the cache."""
        self.mdd_cache.clear()

# ENHANCED CBS SOLVER WITH LAZY A*

class EnhancedCBSSolver:
    """CBS solver with lazy A* and memoization."""
    
    def __init__(self, my_map, starts, goals):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """
        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)

        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0

        self.open_list = []
        
        # MDD cache for memoization
        self.mdd_cache = MDDCache()

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))
    
    def push_node(self, node, use_wdg=True):
        """Push node to open list with heuristic."""
        h = 0
        
        if use_wdg:
            try:
                h = wdg_heuristic(node, self.my_map, self.starts, self.goals, self.heuristics)
            except:
                # Fall back to zero heuristic if WDG fails
                h = 0
        else:
            # Use conflict graph heuristic with greedy matching
            h = conflict_graph_heuristic(node, self.my_map, self.starts, self.goals, self.heuristics, greedy=True)
               
        heapq.heappush(self.open_list, (node['cost'] + h, node['cost'], len(node['collisions']), self.num_of_generated, node))
        print("Generate node {} (g={}, h={})".format(self.num_of_generated, node.get('cost', 0), h))
        self.num_of_generated += 1
    
    def pop_node(self):
        """Pop node from open list."""
        _, _, _, id, node = heapq.heappop(self.open_list)
        print("Expand node {}".format(id))
        self.num_of_expanded += 1
        return node
    
    def find_solution(self, disjoint=True, use_lazy=True, timeout=30):
        """Enhanced CBS with lazy A* and memoization.
        
        Args:
            disjoint: Whether to use disjoint splitting
            use_lazy: Whether to use lazy A* for low-level search
            timeout: Maximum time in seconds to search for solution
        """
        
        self.start_time = timer.time()
        self.mdd_cache.clear()  # Reset cache for new search

        # Generate the root node
        root = {'cost': 0,
                'constraints': [],
                'paths': [],
                'collisions': []}
        
        # Use lazy A* for initial path finding if enabled
        for i in range(self.num_of_agents):
            if timer.time() - self.start_time > timeout:
                raise BaseException('Timeout: No solutions found within {} seconds'.format(timeout))
                
            if use_lazy:
                path = a_star_lazy(self.my_map, self.starts[i], self.goals[i], 
                                  self.heuristics[i], i, root['constraints'])
            else:
                path = a_star(self.my_map, self.starts[i], self.goals[i], 
                             self.heuristics[i], i, root['constraints'])
            
            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path)

        root['cost'] = get_sum_of_cost(root['paths'])
        root['collisions'] = detect_collisions(root['paths'])
        self.push_node(root)

        while len(self.open_list) > 0:
            # Check timeout
            if timer.time() - self.start_time > timeout:
                print("Timeout: Search exceeded {} seconds".format(timeout))
                return None
            
            node = self.pop_node()
            
            if len(node['collisions']) == 0:
                self.print_results(node)
                return node['paths']
            
            collision = node['collisions'][0]
            
            if disjoint:
                constraints = disjoint_splitting(collision)
            else:
                constraints = standard_splitting(collision)
            
            for constraint in constraints:
                child = {'cost': 0,
                         'constraints': [],
                         'paths': [],
                         'collisions': []}
                
                if constraint not in node['constraints']:
                    child['constraints'] = list(node['constraints']) + [constraint]

                agent = constraint['agent']
                child['paths'] = list(node['paths'])
                
                # Use lazy A* if enabled
                if use_lazy:
                    path = a_star_lazy(self.my_map, self.starts[agent], self.goals[agent], 
                                      self.heuristics[agent], agent, child['constraints'])
                else:
                    path = a_star(self.my_map, self.starts[agent], self.goals[agent], 
                                 self.heuristics[agent], agent, child['constraints'])
                
                if path is not None:
                    child['paths'][agent] = path
                    child['collisions'] = detect_collisions(child['paths'])
                    child['cost'] = get_sum_of_cost(child['paths'])
                    
                    if len(child['paths']) > 0:
                        self.push_node(child)
        
        return None
    
    def print_results(self, node):
        """Print solution results."""
        CPU_time = timer.time() - self.start_time
        print("\n Found a solution! \n")
        print("CPU time (s):    {:.2f}".format(CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(node['paths'])))
        print("Expanded nodes:  {}".format(self.num_of_expanded))
        print("Generated nodes: {}".format(self.num_of_generated))



class CBSSolver(object):
    """The high-level search of CBS."""

    def __init__(self, my_map, starts, goals):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """

        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)

        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0

        self.open_list = []

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def push_node(self, node):
        h = 0
        # compute conflict-graph heuristic and push using f = g + h  
        #h = conflict_graph_heuristic(node, self.my_map, self.starts, self.goals, self.heuristics, greedy=True)  
        #h = conflict_graph_heuristic(node, self.my_map, self.starts, self.goals, self.heuristics, greedy=False)
        #h = dg_heuristic(node, self.my_map, self.starts, self.goals, self.heuristics)
        #h = wdg_heuristic(node, self.my_map, self.starts, self.goals, self.heuristics)
        
        heapq.heappush(self.open_list, (node['cost'] + h, node['cost'], len(node['collisions']), self.num_of_generated, node))
        #print("Generate node {} (g={}, h={})".format(self.num_of_generated, node.get('cost', 0), h))
        self.num_of_generated += 1

        
    def pop_node(self):
        _, _, _, id, node = heapq.heappop(self.open_list)
        #print("Expand node {}".format(id))
        
        self.num_of_expanded += 1
        return node

    def find_solution(self, disjoint=True):
        """ Finds paths for all agents from their start locations to their goal locations

        disjoint    - use disjoint splitting or not
        
        """

        self.start_time = timer.time()

        # Generate the root node
        # constraints   - list of constraints
        # paths         - list of paths, one for each agent
        #               [[(x11, y11), (x12, y12), ...], [(x21, y21), (x22, y22), ...], ...]
        # collisions     - list of collisions in paths
        root = {'cost': 0,
                'constraints': [],
                'paths': [],
                'collisions': []}

        for i in range(self.num_of_agents):  # Find initial path for each agent
            path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                          i, root['constraints'])
            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path)

        root['cost'] = get_sum_of_cost(root['paths'])
        root['collisions'] = detect_collisions(root['paths'])
        self.push_node(root)

        #Testing
        #print(root['collisions'])

        #Testing
        #for collision in root['collisions']:
        #    print(disjoint_splitting(collision))

        while len(self.open_list) > 0:
            node = self.pop_node()
            
            if len(node['collisions']) == 0:
                self.print_results(node)
                return node['paths']
            collision = node['collisions'][0]
            if disjoint:
                constraints = disjoint_splitting(collision)
                
            else:
                constraints = standard_splitting(collision)
            for constraint in constraints:
                child = {'cost': 0,
                         'constraints': [],
                         'paths': [],
                         'collisions': []}
                if constraint not in node['constraints']:
                    child['constraints'] = list(node['constraints']) + [constraint]

                if constraint['positive'] is True:
                    agent = constraint['agent']
                    child['paths'] = list(node['paths'])
                    path = a_star(self.my_map, self.starts[agent], self.goals[agent], self.heuristics[agent],
                                          agent, child['constraints'])
                    if path is not None:
                        child['paths'][agent] = path
                        child['collisions'] = detect_collisions(child['paths'])
                        child['cost'] = get_sum_of_cost(child['paths'])
                                    
                else:
                    agent = constraint['agent']
                    child['paths'] = list(node['paths'])
                    path = a_star(self.my_map, self.starts[agent], self.goals[agent], self.heuristics[agent],
                                          agent, child['constraints'])
                    if path is not None:
                        child['paths'][agent] = path
                        child['collisions'] = detect_collisions(child['paths'])
                        child['cost'] = get_sum_of_cost(child['paths'])
                if len(child['paths']) == 0:
                    continue
                self.push_node(child)
                            

        
        return None


    def print_results(self, node):
        print("\n Found a solution! \n")
        CPU_time = timer.time() - self.start_time
        print("CPU time (s):    {:.2f}".format(CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(node['paths'])))
        print("Expanded nodes:  {}".format(self.num_of_expanded))
        print("Generated nodes: {}".format(self.num_of_generated))
