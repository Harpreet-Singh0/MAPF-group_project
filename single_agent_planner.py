import heapq
import functools

def move(loc, dir):
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    return loc[0] + directions[dir][0], loc[1] + directions[dir][1]


def get_sum_of_cost(paths):
    rst = 0
    for path in paths:
        rst += len(path) - 1
    return rst


def compute_heuristics(my_map, goal):
    """Use Dijkstra to build a shortest-path tree rooted at the goal location"""
    open_list = []
    closed_list = dict()
    root = {'loc': goal, 'cost': 0}
    heapq.heappush(open_list, (root['cost'], goal, root))
    closed_list[goal] = root
    while len(open_list) > 0:
        (cost, loc, curr) = heapq.heappop(open_list)
        for dir in range(4):
            child_loc = move(loc, dir)
            child_cost = cost + 1
            if child_loc[0] < 0 or child_loc[0] >= len(my_map) \
               or child_loc[1] < 0 or child_loc[1] >= len(my_map[0]):
               continue
            if my_map[child_loc[0]][child_loc[1]]:
                continue
            child = {'loc': child_loc, 'cost': child_cost}
            if child_loc in closed_list:
                existing_node = closed_list[child_loc]
                if existing_node['cost'] > child_cost:
                    closed_list[child_loc] = child
                    heapq.heappush(open_list, (child_cost, child_loc, child))
            else:
                closed_list[child_loc] = child
                heapq.heappush(open_list, (child_cost, child_loc, child))

    # build the heuristics table
    h_values = dict()
    for loc, node in closed_list.items():
        h_values[loc] = node['cost']
    return h_values


def build_constraint_table(constraints, agent):
    """Build constraint table for the given agent from the list of constraints."""
    constraint_table = dict()
    for constraint in constraints:
        if constraint['agent'] == agent:
            time_step = constraint['timestep']
            if time_step not in constraint_table:
                constraint_table[time_step] = []
            constraint_table[time_step].append(constraint)
    return constraint_table


def get_location(path, time):
    if time < 0:
        return path[0]
    elif time < len(path):
        return path[time]
    else:
        return path[-1]  # wait at the goal location


def get_path(goal_node):
    path = []
    curr = goal_node
    while curr is not None:
        path.append(curr['loc'])
        curr = curr['parent']
    path.reverse()
    return path


def is_constrained(curr_loc, next_loc, next_time, constraint_table):
    if next_time in constraint_table:
        for constraint in constraint_table[next_time]:
            if constraint['loc'] == [next_loc]:
                if constraint['positive'] == True:
                    return False
                else:
                    return True
            if constraint['loc'] == [curr_loc, next_loc]:
                if constraint['positive'] == True:
                    return False
                else:
                    return True
    return False



def push_node(open_list, node):
    heapq.heappush(open_list, (node['g_val'] + node['h_val'], node['h_val'], node['loc'], node))


def pop_node(open_list):
    _, _, _, curr = heapq.heappop(open_list)
    return curr


def compare_nodes(n1, n2):
    """Return true is n1 is better than n2."""
    return n1['g_val'] + n1['h_val'] < n2['g_val'] + n2['h_val']

# LAZY A* IMPLEMENTATION

def memoize(func):
    """Generic memoization decorator."""
    cache = {}
    
    @functools.wraps(func)
    def memoized_func(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    
    return memoized_func


class LazyAStarSolver:
    """A* with lazy heuristic evaluation and memoization."""
    
    def __init__(self, my_map, start_loc, goal_loc, heuristic_generator, agent, constraints):
        self.my_map = my_map
        self.start_loc = start_loc
        self.goal_loc = goal_loc
        self.heuristic_generator = heuristic_generator  # Function to compute heuristics if needed
        self.agent = agent
        self.constraints = constraints
        
        # Cache for computed heuristics
        self.heuristic_cache = {}
        # Cache for constraint tables
        self.constraint_table_cache = None
        
    def get_heuristic(self, loc):
        """Get heuristic value with lazy evaluation and caching."""
        if loc not in self.heuristic_cache:
            # Compute heuristics lazily only when needed
            self.heuristic_cache[loc] = self.heuristic_generator(loc, self.goal_loc)
        return self.heuristic_cache[loc]
    
    def get_constraint_table(self):
        """Get constraint table with caching."""
        if self.constraint_table_cache is None:
            self.constraint_table_cache = build_constraint_table(self.constraints, self.agent)
        return self.constraint_table_cache
    
    def is_constrained_lazy(self, curr_loc, next_loc, next_time):
        """Check constraints using cached constraint table."""
        constraint_table = self.get_constraint_table()
        
        if next_time in constraint_table:
            for constraint in constraint_table[next_time]:
                if constraint['loc'] == [next_loc]:
                    return constraint['positive'] is False
                if constraint['loc'] == [curr_loc, next_loc]:
                    return constraint['positive'] is False
        return False
    
    def lazy_a_star(self):
        """Lazy A* search with memoization."""
        open_list = []
        closed_list = {}
        
        # Initial heuristic is computed lazily
        h_start = self.get_heuristic(self.start_loc)
        root = {
            'loc': self.start_loc,
            'g_val': 0,
            'h_val': h_start,
            'timestep': 0,
            'parent': None
        }
        
        self.push_node_lazy(open_list, root)
        closed_list[(root['loc'], 0)] = root
        
        while open_list:
            curr = self.pop_node_lazy(open_list)
            
            # Goal test with lazy constraint checking
            if curr['loc'] == self.goal_loc:
                # Check if goal is constrained at any future timestep
                goal_constrained = False
                constraint_table = self.get_constraint_table()
                for t in range(curr['timestep'], curr['timestep'] + 50):
                    if t in constraint_table:
                        for constraint in constraint_table[t]:
                            if constraint['loc'] == [self.goal_loc] and constraint['positive'] is False:
                                goal_constrained = True
                                break
                    if goal_constrained:
                        break
                
                if not goal_constrained:
                    return get_path(curr)
            
            # Generate successors with lazy heuristic evaluation
            for dir in range(4):
                child_loc = move(curr['loc'], dir)
                
                # Check bounds and obstacles
                if (child_loc[0] < 0 or child_loc[0] >= len(self.my_map) or
                    child_loc[1] < 0 or child_loc[1] >= len(self.my_map[0]) or
                    self.my_map[child_loc[0]][child_loc[1]]):
                    continue
                
                # Check constraints lazily
                if self.is_constrained_lazy(curr['loc'], child_loc, curr['timestep'] + 1):
                    continue
                
                # Lazy heuristic computation
                h_child = self.get_heuristic(child_loc)
                
                child = {
                    'loc': child_loc,
                    'g_val': curr['g_val'] + 1,
                    'h_val': h_child,
                    'timestep': curr['timestep'] + 1,
                    'parent': curr
                }
                
                self.update_or_add_node(open_list, closed_list, child)
            
            # Wait action with lazy constraint checking
            if not self.is_constrained_lazy(curr['loc'], curr['loc'], curr['timestep'] + 1):
                h_wait = self.get_heuristic(curr['loc'])
                
                wait_child = {
                    'loc': curr['loc'],
                    'g_val': curr['g_val'] + 1,
                    'h_val': h_wait,
                    'timestep': curr['timestep'] + 1,
                    'parent': curr
                }
                
                self.update_or_add_node(open_list, closed_list, wait_child)
        
        return None
    
    def push_node_lazy(self, open_list, node):
        """Push node to open list with lazy f-value computation."""
        f_val = node['g_val'] + node['h_val']
        heapq.heappush(open_list, (f_val, node['h_val'], node['loc'], node['timestep'], node))
    
    def pop_node_lazy(self, open_list):
        """Pop node from open list."""
        _, _, _, _, node = heapq.heappop(open_list)
        return node
    
    def update_or_add_node(self, open_list, closed_list, node):
        """Update existing node or add new node with lazy comparison."""
        key = (node['loc'], node['timestep'])
        
        if key in closed_list:
            existing = closed_list[key]
            # Compare nodes lazily (only when needed)
            if (node['g_val'] + node['h_val']) < (existing['g_val'] + existing['h_val']):
                closed_list[key] = node
                self.push_node_lazy(open_list, node)
        else:
            closed_list[key] = node
            self.push_node_lazy(open_list, node)


def a_star_lazy(my_map, start_loc, goal_loc, h_values, agent, constraints):
    """Lazy A* wrapper that uses the LazyAStarSolver."""
    # Create a heuristic generator function
    def heuristic_generator(loc, goal):
        if loc in h_values:
            return h_values[loc]
        # Fallback: Manhattan distance if not in precomputed heuristics
        return abs(loc[0] - goal[0]) + abs(loc[1] - goal[1])
    
    solver = LazyAStarSolver(
        my_map, 
        start_loc, 
        goal_loc,
        heuristic_generator,
        agent,
        constraints
    )
    return solver.lazy_a_star()


def a_star(my_map, start_loc, goal_loc, h_values, agent, constraints):
    """ my_map      - binary obstacle map
        start_loc   - start position
        goal_loc    - goal position
        agent       - the agent that is being re-planned
        constraints - constraints defining where robot should or cannot go at each timestep
    """
    from CG import conflict_graph_heuristic     # Pulling in here to avoid circular import
    from WDG import wdg_heuristic               #

    open_list = []
    closed_list = dict()
    earliest_goal_timestep = 0
    h_value = h_values[start_loc]
    root = {'loc': start_loc, 
            'g_val': 0, 
            'h_val': h_value, 
            'timestep': earliest_goal_timestep, 
            'parent': None}
    push_node(open_list, root)
    closed_list[(root['loc'], earliest_goal_timestep)] = root
    constraint_table = build_constraint_table(constraints, agent)
    while len(open_list) > 0:
        curr = pop_node(open_list)
        # check if reached goal location
        for t in range(curr['timestep'], curr['timestep'] + 50):
            if is_constrained(goal_loc, goal_loc, t, constraint_table):
                goal_constrained = True
                break
            else:
                goal_constrained = False
                
        if curr['loc'] == goal_loc and not goal_constrained:
            return get_path(curr)
        for dir in range(4):
            child_loc = move(curr['loc'], dir)
            if child_loc[0] < 0 or child_loc[0] >= len(my_map) \
               or child_loc[1] < 0 or child_loc[1] >= len(my_map[0]):
               continue
            if my_map[child_loc[0]][child_loc[1]]:
                continue
            child = {'loc': child_loc,
                    'g_val': curr['g_val'] + 1,
                    'h_val': h_values[child_loc],
                    'timestep': curr['timestep'] + 1,
                    'parent': curr}
            if not is_constrained(curr['loc'], child_loc, child['timestep'], constraint_table):
                if ((child['loc'], child['timestep'])) in closed_list:
                    existing_node = closed_list[(child['loc'], child['timestep'])]
                    if compare_nodes(child, existing_node):
                        closed_list[(child['loc'], child['timestep'])] = child
                        push_node(open_list, child)
                else:
                    closed_list[(child['loc'], child['timestep'])] = child
                    push_node(open_list, child)
        child_loc = curr['loc']  # wait action
        child = {'loc': child_loc,
                'g_val': curr['g_val'] + 1,       
                'h_val': h_values[child_loc],
                'timestep': curr['timestep'] + 1,
                'parent': curr}
        if not is_constrained(curr['loc'], child_loc, child['timestep'], constraint_table):
            if (child['loc'], child['timestep']) in closed_list:
                existing_node = closed_list[(child['loc'], child['timestep'])]
                if compare_nodes(child, existing_node):
                    closed_list[(child['loc'], child['timestep'])] = child
                    push_node(open_list, child)
            else:
                closed_list[(child['loc'], child['timestep'])] = child
                push_node(open_list, child)

    return None  # Failed to find solutions
