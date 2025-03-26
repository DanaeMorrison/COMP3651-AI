import math
import random
import numpy as np


def is_valid_square(pt, n):
    """Checks if pt is a valid square in the sliding tile puzzle of size n."""
    return (pt[0] >= 0) and (pt[0] < n) and (pt[1] >= 0) and (pt[1] < n)

def empty_pos(puzzle):
    """Get the position of the empty square."""
    for i in range(puzzle.shape[0]):
        for j in range(puzzle.shape[1]):
            if puzzle[i, j] == 0:
                return (i, j)

def make_move(puzzle, pos):
    """Moves the tile at position pos to the empty square (no change if the move is invalid)"""
    puzzle_new = np.copy(puzzle)
    empty = empty_pos(puzzle)
    if ((abs(empty[0] - pos[0]) == 1) and (empty[1] == pos[1])) or ((abs(empty[1] - pos[1]) == 1) and (empty[0] == pos[0])):
        puzzle_new[empty[0], empty[1]] = puzzle[pos[0], pos[1]]
        puzzle_new[pos[0], pos[1]] = 0
    return puzzle_new

def init_solved_puzzle(n):
    """Creates a solved initial puzzle of size n"""
    return np.reshape(np.arange(n**2), [n, n])
    
def init_scrambled_puzzle(n, m):
    """Creates a scrambled puzzle of size n by making m moves"""
    puzzle = init_solved_puzzle(n)
    for i in range(m):
        empty = empty_pos(puzzle)
        # Create list of horizontal and vertical moves (positions of tiles that
        # can be moved to the empty square
        candidate_moves = [(empty[0]-1, empty[1]), (empty[0]+1, empty[1]), (empty[0], empty[1]-1), (empty[0], empty[1]+1)]
        # Filter list of candidate moves to eliminate positions outside of the puzzle
        move_list = [x for x in candidate_moves if is_valid_square(x, n)]
        # Select move at random and carry it out
        move = random.sample(move_list, 1)
        move = move[0]
        puzzle = make_move(puzzle, move)
    return puzzle


def is_solved(puzzle):
    """Checks if a puzzle is in the goal state"""
    puzzle_num = 0
    for i in range(puzzle.shape[0]):
        for j in range(puzzle.shape[1]):
            if puzzle[i, j] != puzzle_num:
                return False
            else:
                puzzle_num = puzzle_num + 1
    return True

def num_out_of_order(puzzle):
    """Determines how many puzzle pieces are out of piece in a puzzle"""
    puzzle_num = 0
    wrong_place = 0
    for i in range(puzzle.shape[0]):
        for j in range(puzzle.shape[1]):
            if puzzle[i, j] != puzzle_num and puzzle[i, j] != 0:
                wrong_place += 1
            puzzle_num = puzzle_num + 1
    return wrong_place

def find_total_manhattan_num(puzzle):
    """Determines the total Manhattan distance of all tiles from their proper positions (excluding the empty tile) in a puzzle"""
    puzzle_num = 0
    total_manhattan = 0
    for i in range(puzzle.shape[0]):
        for j in range(puzzle.shape[1]):
            if puzzle[i, j] != puzzle_num and puzzle[i, j] != 0:
                total_manhattan += find_manhattan_num(puzzle, puzzle[i, j], (i, j))
            puzzle_num = puzzle_num + 1
    return total_manhattan

def find_manhattan_num(puzzle, puz_piece, wrong_pos):
    """Determines the Manhattan distance of one tile from its proper position in a puzzle"""
    puz_num = 0
    proper_pos = ()
    for i in range(puzzle.shape[0]):
        for j in range(puzzle.shape[1]):
            if puz_num == puz_piece:
                proper_pos = (i, j)
                break
            else:
                puz_num = puz_num + 1
            
    manhattan_num = abs(wrong_pos[0] - proper_pos[0]) + abs(wrong_pos[1] - proper_pos[1])

    return manhattan_num

def smallest_move(moves_list):
    """Finds the move with the smallest distance from goal + out of place tiles (or total Manhattan distance)"""
    list_len = len(moves_list)
    smallest = 100000000
    smallest_move = []
    index = -1
    for i in range(list_len):
        if (moves_list[i])[3] < smallest:
            smallest = (moves_list[i])[3]
            smallest_move = moves_list[i]
            index = i
    moves_list.pop(index)
    return smallest_move


def solve_puzzle_breadth(puz):
    cur_num_moves_done = 0 #counter for all the expanded/visited moves. Used to keep track of the position of all expanded/visited moves 
    current_lvl = 0 #keeps track of the current level of moves being expanded/visited. Used to help determine when to stop adding moves to num_moves_cur_level
    num_moves_so_far_cur_lvl = 0 #keeps track of the number of moves expanded/visited for a level so far
    num_moves_cur_lvl = 0 #keeps track of the total number of moves generated for a level
    moves_per_level = [0] #keeps track of the generated moves per each level
    puzzle_tree = [] #keeps track of all the moves and their generated puzzles and moves, as well as a note of which number move it was in the number of moves visited and the number of
                    #the move that came before it
    moves = [] #counter for all the generated moves
    path_reverse = [] #the reversed path of the solution
    path = [] #the path for the solution
    full_solution = [] #the path for the solution, the length of it (for total cost of the solution), and the number of states visited
    empty = empty_pos(puz)
    #Each move contains the coordinates of the puzzle piece that is to be slid into the empty space as well as the move number of the state that generated these moves
    candidate_moves = [(empty[0]-1, empty[1], cur_num_moves_done), (empty[0]+1, empty[1], cur_num_moves_done), (empty[0], empty[1]-1, cur_num_moves_done), (empty[0], empty[1]+1, cur_num_moves_done)]
    move_list = [x for x in candidate_moves if is_valid_square(x, 3)]
    num_moves = len(move_list)

    for m in move_list:
            moves.append(m)
    #Each entry in puzzle_tree contains the move that was made, the puzzle that it generated, the number of puzzle pieces that could be slid into its empty space, a track of which number
    #move it was, and the move number of the move that came before it 
    puzzle_tree.append([(-1,-1), puz, num_moves, cur_num_moves_done, -1]) #might not need to store num_moves in puzzle_tree
    moves_per_level.append(num_moves)
    current_lvl += 1
    while is_solved(puz) == False:
        cur_move = moves[cur_num_moves_done]
        cur_num_moves_done += 1
        puz_state_index = cur_move[2]
        puz = (puzzle_tree[puz_state_index])[1]
        new_puz = make_move(puz, cur_move)
        empty = empty_pos(new_puz)
        candidate_moves = [(empty[0]-1, empty[1], cur_num_moves_done), (empty[0]+1, empty[1], cur_num_moves_done), (empty[0], empty[1]-1, cur_num_moves_done), (empty[0], empty[1]+1, cur_num_moves_done)]
        move_list = [x for x in candidate_moves if is_valid_square(x, 3)]
        
        num_moves = len(move_list)

        for m in move_list:
            moves.append(m)

        #conditional that checks if all of the moves generated for the previous level are expanded for adding up the number of moves in the next level
        if moves_per_level[current_lvl] != num_moves_so_far_cur_lvl:
            num_moves_cur_lvl += num_moves
        else:
            moves_per_level.append(num_moves_cur_lvl)
            current_lvl += 1
            num_moves_so_far_cur_lvl = 0
            num_moves_cur_lvl = 0

        num_moves_so_far_cur_lvl += 1
        puz = new_puz
        puzzle_tree.append([(cur_move[0],cur_move[1]), puz, num_moves, cur_num_moves_done, cur_move[2]])


    pos = cur_num_moves_done #this gives us the number of the move that has the goal state

    #this generates the path from the goal state to the root/inital state of the puzzle (specifically just before the root)
    while (puzzle_tree[pos])[3] !=  0:
        path_reverse.append((puzzle_tree[pos])[0])
        pos = (puzzle_tree[pos])[4]

    path_len = len(path_reverse)
    index = path_len

    #this generates the path from the initial state to the goal state by reversing what we found above
    for i in range(path_len):
        path.append(path_reverse[index-1])
        index -= 1

    full_solution = [path, path_len, cur_num_moves_done]
    return full_solution


def solve_puzzle_not_in_place(puz):
    cur_num_moves_done = 0
    current_lvl = 0
    num_moves_so_far_cur_lvl = 0
    num_moves_cur_lvl = 0
    depth = 1 #keeps track of the depth of a move in a tree (i.e., the number of moves done to reach this move/state). Different from current_lvl, which keeps track
                #of the level that moves are currently being expanded for in the bfs algorithm
    moves_per_level = [0]
    puzzle_tree = []
    moves = []
    path_reverse = []
    path = [] 
    full_solution = []
    empty = empty_pos(puz)
    num_out_of_place = num_out_of_order(puz) #finds the total number of out of place tiles (excluding the zero tile) for the puzzle
    #Each move contains the coordinates of the puzzle piece that is to be slid into the empty space, the move number of the state that generated these moves, and
    #the number of tiles out of place for the puzzle (excluding the zero tile) + the number of moves done so far to get to the current move (as described for a a* algorithm)
    candidate_moves = [(empty[0]-1, empty[1], cur_num_moves_done, num_out_of_place + depth), (empty[0]+1, empty[1], cur_num_moves_done, num_out_of_place + depth),
                       (empty[0], empty[1]-1, cur_num_moves_done, num_out_of_place + depth), (empty[0], empty[1]+1, cur_num_moves_done, num_out_of_place + depth)]
    move_list = [x for x in candidate_moves if is_valid_square(x, 3)]
    num_moves = len(move_list)
    depth += 1

    
    for m in move_list:
            moves.append(m)
    
    puzzle_tree.append([(-1,-1), puz, num_moves, cur_num_moves_done, -1])
    moves_per_level.append(num_moves)
    current_lvl += 1
    while is_solved(puz) == False:
        cur_move = smallest_move(moves)# function that returns the least expensive move in a list based on a* algorithm (distance from goal state + heuristic- out of place tiles)
        cur_num_moves_done += 1
        puz_state_index = cur_move[2]
        puz = (puzzle_tree[puz_state_index])[1]
        new_puz = make_move(puz, cur_move)
        empty = empty_pos(new_puz)
        num_out_of_place = num_out_of_order(new_puz)
        candidate_moves = [(empty[0]-1, empty[1], cur_num_moves_done, num_out_of_place + depth), (empty[0]+1, empty[1], cur_num_moves_done, num_out_of_place + depth),
                           (empty[0], empty[1]-1, cur_num_moves_done, num_out_of_place + depth), (empty[0], empty[1]+1, cur_num_moves_done, num_out_of_place + depth)]
        move_list = [x for x in candidate_moves if is_valid_square(x, 3)]
        
        
        num_moves = len(move_list)

        for m in move_list:
            moves.append(m)

        
        if moves_per_level[current_lvl] != num_moves_so_far_cur_lvl:
            num_moves_cur_lvl += num_moves
        else:
            moves_per_level.append(num_moves_cur_lvl)
            current_lvl += 1
            num_moves_so_far_cur_lvl = 0
            num_moves_cur_lvl = 0
            depth += 1

        num_moves_so_far_cur_lvl += 1
        puz = new_puz
        puzzle_tree.append([(cur_move[0],cur_move[1]), puz, num_moves, cur_num_moves_done, cur_move[2]]) #*

    pos = cur_num_moves_done
    
    while (puzzle_tree[pos])[3] !=  0:
        path_reverse.append((puzzle_tree[pos])[0])
        pos = (puzzle_tree[pos])[4]

    path_len = len(path_reverse)
    index = path_len

    for i in range(path_len):
        path.append(path_reverse[index-1])
        index -= 1

    
    full_solution = [path, path_len, cur_num_moves_done]
    return full_solution


def solve_puzzle_manhattan(puz):
    cur_num_moves_done = 0
    current_lvl = 0
    num_moves_so_far_cur_lvl = 0
    num_moves_cur_lvl = 0
    depth = 1
    moves_per_level = [0]
    puzzle_tree = []
    moves = []
    path_reverse = []
    path = []
    full_solution = []
    empty = empty_pos(puz)
    total_manhattan_num = find_total_manhattan_num(puz) #finds the total manhattan distance for the puzzle
    #Each move contains the coordinates of the puzzle piece that is to be slid into the empty space, the move number of the state that generated these moves, and
    #the total manhattan distance of all displaced tiles from their proper positions (excluding the zero-tile) for the puzzle + the number of moves done so far to
    #get to the current move (as described for a a* algorithm)
    candidate_moves = [(empty[0]-1, empty[1], cur_num_moves_done, total_manhattan_num + depth), (empty[0]+1, empty[1], cur_num_moves_done, total_manhattan_num + depth),
                       (empty[0], empty[1]-1, cur_num_moves_done, total_manhattan_num + depth), (empty[0], empty[1]+1, cur_num_moves_done, total_manhattan_num + depth)]
    move_list = [x for x in candidate_moves if is_valid_square(x, 3)]
    num_moves = len(move_list)
    depth += 1

    
    for m in move_list:
            moves.append(m)
    
    puzzle_tree.append([(-1,-1), puz, num_moves, cur_num_moves_done, -1])
    moves_per_level.append(num_moves)
    current_lvl += 1
    while is_solved(puz) == False:
        cur_move = smallest_move(moves) #function that returns the least expensive move in a list based on a* algorithm (distance from goal state + heuristic- manhattan distance)
        cur_num_moves_done += 1
        puz_state_index = cur_move[2]
        puz = (puzzle_tree[puz_state_index])[1]
        new_puz = make_move(puz, cur_move)
        empty = empty_pos(new_puz)
        total_manhattan_num = find_total_manhattan_num(new_puz)
        candidate_moves = [(empty[0]-1, empty[1], cur_num_moves_done, total_manhattan_num + depth), (empty[0]+1, empty[1], cur_num_moves_done, total_manhattan_num + depth),
                           (empty[0], empty[1]-1, cur_num_moves_done, total_manhattan_num + depth), (empty[0], empty[1]+1, cur_num_moves_done, total_manhattan_num + depth)]
        move_list = [x for x in candidate_moves if is_valid_square(x, 3)]
        
        
        num_moves = len(move_list)

        for m in move_list:
            moves.append(m)

        
        if moves_per_level[current_lvl] != num_moves_so_far_cur_lvl:
            num_moves_cur_lvl += num_moves
        else:
            moves_per_level.append(num_moves_cur_lvl)
            current_lvl += 1
            num_moves_so_far_cur_lvl = 0
            num_moves_cur_lvl = 0
            depth += 1

        num_moves_so_far_cur_lvl += 1
        puz = new_puz
        puzzle_tree.append([(cur_move[0],cur_move[1]), puz, num_moves, cur_num_moves_done, cur_move[2]]) #*

    pos = cur_num_moves_done
    
    while (puzzle_tree[pos])[3] !=  0:
        path_reverse.append((puzzle_tree[pos])[0])
        pos = (puzzle_tree[pos])[4]

    path_len = len(path_reverse)
    index = path_len

    for i in range(path_len):
        path.append(path_reverse[index-1])
        index -= 1

    
    full_solution = [path, path_len, cur_num_moves_done]
    return full_solution

def solve_100():
    bfs_total_cost = 0
    a_misplaced_total_cost = 0
    a_manhattan_total_cost = 0

    bfs_total_states = 0
    a_misplaced_total_states = 0
    a_manhattan_total_states = 0

    bfs_total_cost_avg = 0
    a_misplaced_total_cost_avg = 0
    a_manhattan_total_cost_avg = 0

    bfs_total_states_avg = 0
    a_misplaced_total_states_avg = 0
    a_manhattan_total_states_avg = 0

    avg_solution = []
    
    
    for i in range(100):
        ran_puz = init_scrambled_puzzle(3, 7)
        print("Breadth first search")
        solved_puz = solve_puzzle_breadth(ran_puz.copy())
        print()
        print("The total cost:")
        print(solved_puz[1])
        bfs_total_cost += solved_puz[1]
        print()
        print("The number of states visited:")
        print(solved_puz[2])
        bfs_total_states += solved_puz[2]
        print()

        print("A* search (out-of-place tiles)")
        solved_puz2 = solve_puzzle_not_in_place(ran_puz.copy())
        print()
        print("The total cost:")
        print(solved_puz2[1])
        a_misplaced_total_cost += solved_puz2[1]
        print()
        print("The number of states visited:")
        print(solved_puz2[2])
        a_misplaced_total_states += solved_puz2[2]
        print()

        print("A* search (Manhattan distance)")
        solved_puz3 = solve_puzzle_manhattan(ran_puz.copy())
        print()
        print("The total cost:")
        print(solved_puz3[1])
        a_manhattan_total_cost += solved_puz3[1]
        print()
        print("The number of states visited:")
        print(solved_puz3[2])
        a_manhattan_total_states += solved_puz3[2]
        print("- - - - - - - - - - - - - - - - - - - - - - - -")
        print()

    bfs_total_cost_avg = bfs_total_cost/100
    a_misplaced_total_cost_avg = a_misplaced_total_cost/100
    a_manhattan_total_cost_avg = a_manhattan_total_cost/100

    bfs_total_states_avg = bfs_total_states/100
    a_misplaced_total_states_avg = a_misplaced_total_states/100
    a_manhattan_total_states_avg = a_manhattan_total_states/100

    avg_solution = [bfs_total_cost_avg, bfs_total_states_avg, a_misplaced_total_cost_avg, a_misplaced_total_states_avg, a_manhattan_total_cost_avg, a_manhattan_total_states_avg]
    return avg_solution


eg_list = [3, 1, 2, 4, 7, 5, 6, 0, 8]
index = 0
eg_puz = init_solved_puzzle(3)

for i in range(3):
    for j in range(3):
        eg_puz[i, j] = eg_list[index]
        index += 1

print("This is my example puzzle:")
print(eg_puz)
print ()

print("Breadth first search")
solved_puz = solve_puzzle_breadth(eg_puz.copy())
print("The path:")
print(solved_puz[0])
print()
print("The total cost:")
print(solved_puz[1])
print()
print("The number of states visited:")
print(solved_puz[2])
print()

print("A* search (out-of-place tiles)")
solved_puz2 = solve_puzzle_not_in_place(eg_puz.copy())
print("The path:")
print(solved_puz2[0])
print()
print("The total cost:")
print(solved_puz2[1])
print()
print("The number of states visited:")
print(solved_puz2[2])
print()

print("A* search (Manhattan distance)")
solved_puz3 = solve_puzzle_manhattan(eg_puz.copy())
print("The path:")
print(solved_puz3[0])
print()
print("The total cost:")
print(solved_puz3[1])
print()
print("The number of states visited:")
print(solved_puz3[2])
print("- - - - - - - - - - - - - - - - - - - - - - - -")
print()


print("After running an experiment on 100 randomly-generated puzzles of 7 scrambled moves, these are the results:")
print()
all_avg = solve_100()
print("BFS Average Length of Path:", all_avg[0])
print("BFS Average States Visited:", all_avg[1])
print()
print("A* Misplaced Tiles Average Length of Path:", all_avg[2])
print("A* Misplaced Tiles Average States Visited:", all_avg[3])
print()
print("A* Manhattan Distance Average Length of Path:", all_avg[4])
print("A* Manhattan Distance Average States Visited:", all_avg[5])
print()











