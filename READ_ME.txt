Student name: Alon Almog 
Course: Introduction to AI
Assignment: mmn 11
Note: I have completed the assignment in its previous form (as a 4x4 grid), and will be sending it for grading as been instructed by
	  Dr. Michal in the assignment forum.



Program Design:
The Node object has been written exactly the same as it's written in the book: state, parent, action and path_cost.
State - a 2d-array at the size of a 4x4 grid.
Parent - the parent node.
Action - used to traverse from one state to another: 'Up', 'Right', 'Left', 'Down'.
Path_cost - the cost it has taken to traverse until reaching the current node.



The data structures for each algorithm: (chosen to suit every algorithm)
bfs - queue, dfs - stack (both used as described in the book).
gbfs and a-star - priority queue (both using the heuristic function, linear conflicts heuristic, as a means of comparison. While a-star has also taken into consideration the path_cost).



The Linear Conflict heuristic function: (Using the heuristic function has been approved by the head teacher - dr.michal)
Admissibility - The linear conflict heuristic never overestimates the cost of reaching the goal state. The linear conflict adds a non-negative value to the Manhattan distance, 
				which is a lower bound on the actual cost. Therefore, the linear conflict heuristic remains admissible.
Consistency - The linear conflict heuristic is designed to maintain consistency. The additional cost it introduces is based on the conflicts between tiles in the same row or column, 
			  and this ensures that the heuristic remains consistent with the actual cost of reaching the goal state. Therefore, the linear conflict heuristic is consistent.
			  


Optimality:
bfs - all the edges have the same cost/weight and therefore algorithm returns the optinal solution.
dfs - does not necessarily return the optimal solution. Observe the example: A - start state, B - goal state.
		A ---------------> B
		|				   >
		|				   |
		<				   |
		C -> D -> E -> F-> G
	  depends on how the algorithm is written, the solution might end up being [down, right, right, right, right, up]
	  instead of the optimal solution [right].
gbfs - does not necessarily return the optimal solution. if the heuristic function isn't admissible gbfs might not find the optimal solution.
a-star - returns the optimal solution.



Running description:
As instructed in the assignment, the code is executed by opening the command line in the folder in which the file is stored, and running it so:
"Python Tiles.py 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 0", where the numbers represent the initial/start state.
The numbers are then placed into a 2d-array (called 'problem') in the same order in which they have been received.
The program then runs the algorithms bfs, iddfs, gbfs and a-star, printing the solution path and number of expanded nodes for each algorithm.

Note: All test inputs have been manually written and self checked, with all of them returning the correct solutions, as shown in the added .jpg files.


Fuctions:
find_empty_cell(node) - Returns the indices of the empty cell as a list.
get_valid_actions(node) - Returns a list of all valid movement actions from the current position of the node.
is_goal_state(state) - Returns True if 'state' is the goal state and returns False otherwise.
get_solution_path(node) - Returns the solution path from the initial node.
expand(node) - Returns the list of children nodes
bfs(problem) - Searches for the solution path, scanning each level in the search tree one after another until it finds the goal node, 
			   at which point it returns the solution path and prints the number of expanded nodes.
iddfs(problem) - Runs depth_limited_search with values from 0 to 100 (which is more than what a computer can solve in an 'acceptable'
				 amount of time) until it finds a solution, at which point it returns the solution path and prints the number of
				 expanded nodes. The cutoff value is returned if no solution was found given a depth limit that is smaller than the max depth.
is_cycle(frontier, node) - Checks if frontier already contains such a node (if a cycle is found).
depth_limited_search(problem, max_depth, number_of_expanded_nodes) - Searches for the solution path, scanning as deep as possible along each branch until it reaches the set limit depth,
																	 at which point it backtracks and continues until it scans the entire tree. Once it finds the goal node, it returns the
																	 solution path and the number of explored nodes.
get_goal_indices_of_value(value) - Returns the goal indices of a tile (containing value) in the form of a list.
manhattan_distance(node) - Returns the Manhattan Distance which is the sum of the distances of the tiles from their goal positions.
get_index_of_most_conflicted_tile(conflicts) - Returns the index of the most conflicted tile.
found_none_zero_conflict(conflicts) - Returns True if there is a none-zero conflict tile, otherwise returns False.
determine_row_tile_conflicts(node, row_index, col_index) - Determines the number of linear conflicts a tile has in a row.
determine_col_tile_conflicts(node, row_index, col_index) - Determines the number of linear conflicts a tile has in a column.
adjust_same_row_tile_conflicts(node, tile_row_index, conflicts_per_tile_in_row) - Adjusts linear conflicts in all tiles in a row (used only after the number of linear conflict in 
																				  a tile has been nullified).
adjust_same_col_tile_conflicts(node, tile_col_index, conflicts_per_tile_in_col) - Adjusts linear conflicts in all tiles in a column (used only after the number of linear conflict in 
																				  a tile has been nullified).
get_number_of_row_conflicts(node) - Returns the number linear conflicts in all rows.
get_number_of_col_conflicts(node) - Returns the number linear conflicts in all columns.
linear_conflict_heuristic(node) - Returns the linear conflict heuristic value of a node.
gbfs(problem) - Searches for the solution path, always choosing the closest node to the goal according to the heuristic function.
				Returns the solution path and prints the number of expanded nodes.
astar(problem) - Searches for the solution path by always choosing the closest node to the goal, accounting to both heuristic function
				 value and path cost which approximates the real distance from the goal state to a better degree.
				 Returns the solution path and prints the number of expanded nodes.