import copy
import sys
import queue

# Alon Almog, 319044566. Introduction to AI. Assignment 11.

# This program solves the 15-tile problem. Given a solvable starting state, it runs the problem using the
# algorithms bfs, iddfs, gbfs and a-star, printing the solution path to the goal state and the number of expanded node
# for each algorithm.

grid_length = 4

# Node containing the state, parent, action and path cost.
class Node:
    def __init__(self, state, parent, action, path_cost):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost

    def __lt__(self, other):
        return linear_conflict_heuristic(self) < linear_conflict_heuristic(other)


# Returns the indices of the empty cell as a list.
def find_empty_cell(node):
    for i in range(0, grid_length):
        for j in range(0, grid_length):
            if node.state[i][j] == 0:
                return [i, j]


# Returns a list of all valid movement actions from the current position of the node.
def get_valid_actions(node):
    lower_bound = 0
    upper_bound = 3  # grid length - 1
    empty_cell_indices = find_empty_cell(node)

    if empty_cell_indices[0] == lower_bound:
        if empty_cell_indices[1] == lower_bound:  # case [0, 0] - edge case
            return ['right', 'down']
        if not empty_cell_indices[1] == upper_bound:  # case [0, 1], [0, 2]
            return ['right', 'left', 'down']
        if empty_cell_indices[1] == upper_bound:  # case [0, 3] - edge case
            return ['left', 'down']

    if empty_cell_indices[1] == lower_bound:
        if not empty_cell_indices[0] == upper_bound:  # case [1, 0], [2, 0]
            return ['up', 'right', 'down']
        if empty_cell_indices[0] == upper_bound:  # case [3, 0] - edge case
            return ['up', 'right']

    if empty_cell_indices[1] == upper_bound:
        if not empty_cell_indices[0] == upper_bound:  # case [1, 3], [2, 3]
            return ['up', 'left', 'down']
        if empty_cell_indices[0] == upper_bound:  # case [3, 3] - edge case
            return ['up', 'left']

    if empty_cell_indices[0] == upper_bound:  # all edge cases have already been delt with
        return ['up', 'right', 'left']

    # case the indices aren't located on the edge of the 2d-grid
    return ['up', 'right', 'left', 'down']


# Returns True if 'state' is the goal state and returns False otherwise.
def is_goal_state(state):
    goal_state = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]]

    return goal_state == state


# Returns the solution path from the initial node.
def get_solution_path(node):
    current_node = node
    solution_path = []

    while current_node.parent is not None:
        solution_path.append(current_node.action)
        current_node = current_node.parent

    solution_path.reverse()
    return solution_path


# Returns the list of children nodes
def expand(node):
    children_nodes = []
    empty_cell = find_empty_cell(node)

    for action in get_valid_actions(node):
        deep_copy_state = copy.deepcopy(node.state)

        if action == 'up':
            deep_copy_state[empty_cell[0]][empty_cell[1]] = deep_copy_state[empty_cell[0] - 1][empty_cell[1]]
            deep_copy_state[empty_cell[0] - 1][empty_cell[1]] = 0
        if action == 'right':
            deep_copy_state[empty_cell[0]][empty_cell[1]] = deep_copy_state[empty_cell[0]][empty_cell[1] + 1]
            deep_copy_state[empty_cell[0]][empty_cell[1] + 1] = 0
        if action == 'left':
            deep_copy_state[empty_cell[0]][empty_cell[1]] = deep_copy_state[empty_cell[0]][empty_cell[1] - 1]
            deep_copy_state[empty_cell[0]][empty_cell[1] - 1] = 0
        if action == 'down':
            deep_copy_state[empty_cell[0]][empty_cell[1]] = deep_copy_state[empty_cell[0] + 1][empty_cell[1]]
            deep_copy_state[empty_cell[0] + 1][empty_cell[1]] = 0

        child_node = Node(deep_copy_state, node, action, node.path_cost + 1)
        children_nodes.append(child_node)

    return children_nodes


# Searches for the solution path, scanning each level in the search tree one after another until it finds the goal node,
# at which point it returns the solution path and prints the number of expanded nodes.
def bfs(problem):
    initial_node = Node(problem, None, None, 0)
    expanded_nodes = 0
    print("BFS:")

    if is_goal_state(initial_node.state):
        print("The number of expanded nodes is: ", 0)
        return []

    frontier = queue.Queue()
    explored_nodes = []
    frontier.put(initial_node)
    explored_nodes.append(initial_node)

    while not frontier.empty():
        current_node = frontier.get()

        for child_node in expand(current_node):
            if is_goal_state(child_node.state):
                print("The number of expanded nodes is: ", expanded_nodes)
                return get_solution_path(child_node)

            if child_node not in explored_nodes:
                explored_nodes.append(child_node)
                frontier.put(child_node)

        expanded_nodes += 1


# Runs depth_limited_search with values from 0 to 100 (which is more than what a computer can solve in an 'acceptable'
# amount of time) until it finds a solution, at which point it returns the solution path and prints the number of
# expanded nodes. The cutoff value is returned if no solution was found given a depth limit that is smaller than the max
# depth.
def iddfs(problem):
    max_depth = 100
    cutoff = [0]  # assigned cutoff as a special list.
    number_of_expanded_nodes = [0]
    print("IDDFS:")

    for depth in range(0, max_depth):
        solution_path = depth_limited_search(problem, depth, number_of_expanded_nodes)

        if not solution_path == cutoff:
            print("The number of expanded nodes is: ", number_of_expanded_nodes[0])
            return solution_path


# Checks if frontier already contains such a node (if a cycle is found).
def is_cycle(frontier, node):
    cycle_found = False
    tmp = []

    # Goes through the frontier and checks if the current node creates a cycle.
    while frontier:
        current_node = frontier.pop()
        if current_node.state == node.state:
            cycle_found = True
        tmp.append(current_node)

    # Reverses the frontier to it's original state.
    while tmp:
        current_node = tmp.pop()
        frontier.append(current_node)

    return cycle_found


# Searches for the solution path, scanning as deep as possible along each branch until it reaches the set limit depth,
# at which point it backtracks and continues until it scans the entire tree. Once it finds the goal node, it returns the
# solution path and the number of explored nodes.
def depth_limited_search(problem, max_depth, number_of_expanded_nodes):
    initial_node = Node(problem, None, None, 0)
    frontier = [initial_node]
    solution_path = [-1]  # assigned failure value.

    while frontier:
        current_node = frontier.pop()

        if is_goal_state(current_node.state):
            return get_solution_path(current_node)

        if current_node.path_cost > max_depth:
            solution_path = [0]  # assigned cutoff value.
        elif not is_cycle(frontier, current_node):
            number_of_expanded_nodes[0] += 1
            for child_node in expand(current_node):
                frontier.append(child_node)

    return solution_path


# Returns the goal indices of a tile (containing value) in the form of a list.
def get_goal_indices_of_value(value):
    for i in range(0, grid_length):
        for j in range(0, grid_length):
            if 1 + grid_length*i + j == value:
                return [i, j]


# Returns the Manhattan Distance which is the sum of the distances of the tiles from their goal positions.
def manhattan_distance(node):
    sum_distance = 0

    for i in range(0, grid_length):
        for j in range(0, grid_length):
            tile_value = node.state[i][j]
            goal_indices = get_goal_indices_of_value(tile_value)

            if tile_value != 0:
                sum_distance += abs(goal_indices[0] - i) + abs(goal_indices[1] - j)

    return sum_distance


# Returns the index of the most conflicted tile.
def get_index_of_most_conflicted_tile(conflicts):
    index_of_most_conflicted_tile = 0

    for current_index in range(1, grid_length):
        if conflicts[current_index] > conflicts[index_of_most_conflicted_tile]:
            index_of_most_conflicted_tile = current_index

    return index_of_most_conflicted_tile


# Returns True if there is a none-zero conflict tile, otherwise returns False.
def found_none_zero_conflict(conflicts):
    for conflict in conflicts:
        if conflict != 0:
            return True
    return False


# Determines the number of linear conflicts a tile has in a row.
def determine_row_tile_conflicts(node, row_index, col_index):
    tile_conflicts = 0
    original_tile_value = node.state[row_index][col_index]

    if not original_tile_value == 0:
        original_tile_goal_indices = get_goal_indices_of_value(original_tile_value)

        for k in range(0, grid_length):
            current_tile = node.state[row_index][k]

            # The current tile is not '0'.
            if not current_tile == 0:
                current_tile_goal_indices = get_goal_indices_of_value(current_tile)
                # The current tile and the original tile goal row indices are in the current row.
                if current_tile_goal_indices[0] == original_tile_goal_indices[0] == row_index:
                    # The Original tile is to the right of the current tile and their goals are the opposite to that.
                    if col_index > k and original_tile_goal_indices[1] < current_tile_goal_indices[1]:
                        tile_conflicts += 1

    return tile_conflicts


# Determines the number of linear conflicts a tile has in a column.
def determine_col_tile_conflicts(node, row_index, col_index):
    tile_conflicts = 0
    original_tile_value = node.state[row_index][col_index]

    if not original_tile_value == 0:
        original_tile_goal_indices = get_goal_indices_of_value(original_tile_value)

        for k in range(0, grid_length):
            current_tile = node.state[k][col_index]

            # The current tile is not '0'.
            if not current_tile == 0:
                current_tile_goal_indices = get_goal_indices_of_value(current_tile)
                # The current tile and the original tile goal column indices are in the current column.
                if current_tile_goal_indices[1] == original_tile_goal_indices[1] == col_index:
                    # The Original tile is to the bottom of the current tile and their goals are the opposite to that.
                    if row_index < k and original_tile_goal_indices[0] > current_tile_goal_indices[0]:
                        tile_conflicts += 1

    return tile_conflicts


# Adjusts linear conflicts in all tiles in a row (used only after the number of linear conflict in a tile
# has been nullified).
def adjust_same_row_tile_conflicts(node, tile_row_index, conflicts_per_tile_in_row):
    most_conflicted_tile_col_index = get_index_of_most_conflicted_tile(conflicts_per_tile_in_row)
    most_conflicted_tile = node.state[tile_row_index][most_conflicted_tile_col_index]
    most_conflicted_tile_goal_indices = get_goal_indices_of_value(most_conflicted_tile)

    # After nullifying the number of linear conflicts in the most conflicted tile, adjusts the number
    # of linear conflicts in the rest of the tile in the row.
    for current_tile_col_index in range(0, grid_length):
        current_tile = node.state[tile_row_index][current_tile_col_index]

        # The current tile is not '0' and has conflicts in the row.
        if not current_tile_col_index == 0 and conflicts_per_tile_in_row[current_tile_col_index] != 0:
            current_tile_goal_indices = get_goal_indices_of_value(current_tile_col_index)
            # The current tile and the most conflicted tile goal row indices are in the current row.
            if current_tile_goal_indices[0] == most_conflicted_tile_goal_indices[0] == tile_row_index:
                # The current tile is to the right of the most conflicted tile and their goals are the opposite to that.
                if current_tile_col_index > most_conflicted_tile_col_index and current_tile_goal_indices[1] < \
                        most_conflicted_tile_goal_indices[1]:
                    conflicts_per_tile_in_row[current_tile_col_index] -= 1


# Adjusts linear conflicts in all tiles in a column (used only after the number of linear conflict in a tile
# has been nullified).
def adjust_same_col_tile_conflicts(node, tile_col_index, conflicts_per_tile_in_col):
    most_conflicted_tile_row_index = get_index_of_most_conflicted_tile(conflicts_per_tile_in_col)
    most_conflicted_tile = node.state[most_conflicted_tile_row_index][tile_col_index]
    most_conflicted_tile_goal_indices = get_goal_indices_of_value(most_conflicted_tile)

    # After nullifying the number of linear conflicts in the most conflicted tile, adjusts the number
    # of linear conflicts in the rest of the tiles  in the column.
    for current_tile_row_index in range(0, grid_length):
        current_tile = node.state[current_tile_row_index][tile_col_index]

        # The current tile is not '0' and has conflicts in the column.
        if not current_tile == 0 and conflicts_per_tile_in_col[current_tile_row_index] != 0:
            current_tile_goal_indices = get_goal_indices_of_value(current_tile)
            # The current tile and the most conflicted tile goal column indices are in the current column.
            if current_tile_goal_indices[1] == most_conflicted_tile_goal_indices[1] == tile_col_index:
                # The current tile is to the bottom of the most conflicted tile and their goals are the opposite to that.
                if current_tile_row_index < most_conflicted_tile_row_index and current_tile_goal_indices[0] > \
                        most_conflicted_tile_goal_indices[0]:
                    conflicts_per_tile_in_col[current_tile_row_index] -= 1


# Returns the number linear conflicts in all rows.
def get_number_of_row_conflicts(node):
    sum_of_linear_conflicts = 0
    linear_conflicts = [0, 0, 0, 0]  # Number of linear conflicts in each row.

    for tile_row_index in range(0, grid_length):
        conflicts_per_tile_in_row = [0, 0, 0, 0]  # Number of linear conflicts per tile in the current row.

        # Determines how many linear conflicts each tile has with the rest of the tiles in the row.
        for tile_col_index in range(0, grid_length):
            conflicts_per_tile_in_row[tile_col_index] = determine_row_tile_conflicts(node, tile_row_index, tile_col_index)

        # Calculates the number of linear conflicts in each row.
        while found_none_zero_conflict(conflicts_per_tile_in_row):
            most_conflicted_tile_col_index = get_index_of_most_conflicted_tile(conflicts_per_tile_in_row)
            conflicts_per_tile_in_row[most_conflicted_tile_col_index] = 0
            adjust_same_row_tile_conflicts(node, tile_row_index, conflicts_per_tile_in_row)
            linear_conflicts[tile_row_index] += 1

    # Sums up the number of linear conflicts for all rows.
    for conflict in linear_conflicts:
        sum_of_linear_conflicts += conflict

    return sum_of_linear_conflicts


# Returns the number linear conflicts in all columns.
def get_number_of_col_conflicts(node):
    sum_of_col_conflicts = 0
    linear_conflicts = [0, 0, 0, 0]  # Number of linear conflicts in each column.

    for tile_col_index in range(0, grid_length):
        conflicts_per_tile_in_col = [0, 0, 0, 0]  # Number of linear conflicts per tile in the current column

        # Determines how many linear conflicts each tile has with the rest of the tiles in the column.
        for tile_row_index in range(0, grid_length):
            conflicts_per_tile_in_col[tile_row_index] = determine_col_tile_conflicts(node, tile_row_index, tile_col_index)

        # Calculates the number of linear conflicts in each column.
        while found_none_zero_conflict(conflicts_per_tile_in_col):
            most_conflicted_tile_row_index = get_index_of_most_conflicted_tile(conflicts_per_tile_in_col)
            conflicts_per_tile_in_col[most_conflicted_tile_row_index] = 0
            adjust_same_col_tile_conflicts(node, tile_col_index, conflicts_per_tile_in_col)
            linear_conflicts[tile_col_index] += 1

    # Sums up the number of linear conflicts for all columns.
    for conflict in linear_conflicts:
        sum_of_col_conflicts += conflict

    return sum_of_col_conflicts


# Returns the linear conflict heuristic value of a node.
def linear_conflict_heuristic(node):
    return manhattan_distance(node) + 2*get_number_of_row_conflicts(node) + 2*get_number_of_col_conflicts(node)


# Searches for the solution path, always choosing the closest node to the goal according to the heuristic function.
# Returns the solution path and prints the number of expanded nodes.
def gbfs(problem):
    initial_node = Node(problem, None, None, 0)
    explored_nodes = []
    print("GBFS:")

    frontier = queue.PriorityQueue()
    frontier.put((0, initial_node))

    while not frontier.empty():
        current_node = frontier.get()[1]

        if is_goal_state(current_node.state):
            print("the number of expanded nodes is: ", len(explored_nodes))
            return get_solution_path(current_node)

        explored_nodes.append(current_node)
        for child_node in expand(current_node):
            if child_node not in explored_nodes:
                frontier.put((linear_conflict_heuristic(child_node), child_node))


# Searches for the solution path by always choosing the closest node to the goal, accounting to both heuristic function
# value and path cost which approximates the real distance from the goal state to a better degree.
# Returns the solution path and prints the number of expanded nodes.
def astar(problem):
    initial_node = Node(problem, None, None, 0)
    explored_nodes = []
    print("A-star:")

    frontier = queue.PriorityQueue()
    frontier.put((0, initial_node))

    while not frontier.empty():
        current_node = frontier.get()[1]

        if is_goal_state(current_node.state):
            print("the number of expanded nodes is: ", len(explored_nodes))
            return get_solution_path(current_node)

        explored_nodes.append(current_node)
        for child_node in expand(current_node):
            if child_node not in explored_nodes:
                frontier.put((linear_conflict_heuristic(child_node) + child_node.path_cost, child_node))


if __name__ == '__main__':
    problem = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    # Receives the input in the order requested in the assignment.
    # Assigns the input into the 2d-array: problem.
    for i in range(0, grid_length):
        for j in range(0, grid_length):
            problem[i][j] = int(sys.argv[1+i*grid_length+j])

    # Runs the input on all requested algorithms, printing the solution paths and the numbers of expanded nodes.
    print(bfs(problem))
    print(iddfs(problem))
    print(gbfs(problem))
    print(astar(problem))

