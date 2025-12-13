import time as timer
import heapq
import random
from single_agent_planner import compute_heuristics, a_star, get_location, get_sum_of_cost
from CG import conflict_graph_heuristic
from DG import dg_heuristic
from WDG import wdg_heuristic

def detect_collision(path1, path2):
    ##############################
    # Task 3.1: Return the first collision that occurs between two robot paths (or None if there is no collision)
    #           There are two types of collisions: vertex collision and edge collision.
    #           A vertex collision occurs if both robots occupy the same location at the same timestep
    #           An edge collision occurs if the robots swap their location at the same timestep.
    #           You should use "get_location(path, t)" to get the location of a robot at time t.
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
    ##############################
    # Task 3.1: Return a list of first collisions between all robot pairs.
    #           A collision can be represented as dictionary that contains the id of the two robots, the vertex or edge
    #           causing the collision, and the timestep at which the collision occurred.
    #           You should use your detect_collision function to find a collision between two robots.
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
    ##############################
    # Task 3.2: Return a list of (two) constraints to resolve the given collision
    #           Vertex collision: the first constraint prevents the first agent to be at the specified location at the
    #                            specified timestep, and the second constraint prevents the second agent to be at the
    #                            specified location at the specified timestep.
    #           Edge collision: the first constraint prevents the first agent to traverse the specified edge at the
    #                          specified timestep, and the second constraint prevents the second agent to traverse the
    #                          specified edge at the specified timestep
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
    ##############################
    # Task 4.1: Return a list of (two) constraints to resolve the given collision
    #           Vertex collision: the first constraint enforces one agent to be at the specified location at the
    #                            specified timestep, and the second constraint prevents the same agent to be at the
    #                            same location at the timestep.
    #           Edge collision: the first constraint enforces one agent to traverse the specified edge at the
    #                          specified timestep, and the second constraint prevents the same agent to traverse the
    #                          specified edge at the specified timestep
    #           Choose the agent randomly
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
        #h = 0
        # compute conflict-graph heuristic and push using f = g + h  

        #h = conflict_graph_heuristic(node, self.my_map, self.starts, self.goals, self.heuristics, greedy=True)  

        #h = conflict_graph_heuristic(node, self.my_map, self.starts, self.goals, self.heuristics, greedy=False)

        #h = dg_heuristic(node, self.my_map, self.starts, self.goals, self.heuristics)

        h = wdg_heuristic(node, self.my_map, self.starts, self.goals, self.heuristics)
               
        heapq.heappush(self.open_list, (node['cost'] + h, node['cost'], len(node['collisions']), self.num_of_generated, node))
        print("Generate node {} (g={}, h={})".format(self.num_of_generated, node.get('cost', 0), h))
        self.num_of_generated += 1

        
    def pop_node(self):
        _, _, _, id, node = heapq.heappop(self.open_list)
        print("Expand node {}".format(id))
        
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

        # Task 3.1: Testing
        print(root['collisions'])

        # Task 3.2: Testing
        for collision in root['collisions']:
            print(disjoint_splitting(collision))

        ##############################
        # Task 3.3: High-Level Search
        #           Repeat the following as long as the open list is not empty:
        #             1. Get the next node from the open list (you can use self.pop_node()
        #             2. If this node has no collision, return solution
        #             3. Otherwise, choose the first collision and convert to a list of constraints (using your
        #                standard_splitting function). Add a new child node to your open list for each constraint
        #           Ensure to create a copy of any objects that your child nodes might inherit
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
