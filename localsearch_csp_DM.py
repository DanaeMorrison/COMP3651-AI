import numpy as np
import random

#num_steps = 0

# This function generates a random problem instance
def make_random_problem(n_variables, n_clauses):
    num_clauses = 0
    # This is a helper function to check if a row occurs in a matrix
    def find_row(mx, row):
        for i in range(mx.shape[0]):
            if np.all(mx[i, :] == row):
                return True
        return False
    # This is a helper function to make a random clause (represented as a row
    # in a Numpy matrix)
    def make_random_clause(n_variables):
        # Create a Numpy matrix to store the row
        clause_mx = np.zeros((1, n_variables))
        # Fill in a random clause
        for i in range(n_variables):
            clause_mx[0, i] = random.choice((-1, 0, 1))
        return clause_mx
    # Start with a random row (representing one clause)
    problem_mx = make_random_clause(n_variables)
    num_clauses += 1
    # Add unique, non-empty clauses until the problem reaches the required size
    while problem_mx.shape[0] < n_clauses:
        temp = make_random_clause(n_variables)
        if not find_row(problem_mx, temp) and not np.all(temp == np.zeros((1, n_variables))):
            problem_mx = np.vstack((problem_mx, temp))
            num_clauses += 1
    return problem_mx

# This function makes a random assignment of values to variables
def make_random_assignment(n_variables):
    # Allocate a Numpy array
    assignment_mx = np.zeros((1, n_variables))
    # Assign random truth values to the variables
    for i in range(n_variables):
        assignment_mx[0, i] = random.choice((-1, 1))
    return assignment_mx

def num_constraint_violations(constraint_mx, assignment_mx):
    n_constraint_violate = 0
    
    for i in range(constraint_mx.shape[0]):
            n_false_var_in_clause = 0
            n_var_in_clause = 0
            for j in range(constraint_mx.shape[1]):
                if constraint_mx[i, j] != 0:
                    n_var_in_clause += 1
                    if constraint_mx[i, j] != assignment_mx[0, j]:
                        n_false_var_in_clause += 1

            #checks if the number of contradicting variables between the constraint and the
            #assigned variables is equal to the number of existing variables in a clause.
            #if it is the same, that means that no variable in the candidate solution was
            #proven true in a clause, so that constraint is violated and is added to the
            #variable keeping track of the number of violated constraints
                    
            if n_var_in_clause != 0 and n_false_var_in_clause == n_var_in_clause:
                n_constraint_violate += 1
    return n_constraint_violate

def local_search(n_variables, n_clauses):
    num_steps = 0
    problem = make_random_problem(n_variables, n_clauses)
    solution = (make_random_assignment(n_variables), num_steps)
    n_constraint_violate = num_constraint_violations(problem, solution[0])
    lowest_n_violate = (solution, n_constraint_violate)

    while n_constraint_violate != 0:
        successor_solutions = []
        num_violate = 0
        
        for i in range(n_variables):
            new_solution = solution[0].copy()
            cur_var = new_solution[0, i]
            if cur_var == -1:
                new_solution[0, i] = 1
            else:
                new_solution[0, i] = -1

            num_violate = num_constraint_violations(problem, new_solution)

            if num_violate < lowest_n_violate[1]:
                lowest_n_violate = (new_solution, num_violate)
            
            successor_solutions.append((new_solution, num_violate))

        if n_constraint_violate == lowest_n_violate[1]:
            return False
        else:
            num_steps += 1
            solution = (lowest_n_violate[0], num_steps)
            n_constraint_violate = lowest_n_violate[1]
        
    return solution

def local_search_hundred_times(n_variables, n_clauses):
    average_steps = 0
    total_steps = 0
    solution = 0
    solutions_found = 0
    failed_searches = 0
    results = 0

    for i in range(100):
        solution = local_search(n_variables, n_clauses)
        if solution != False:
            solutions_found += 1
            total_steps += solution[1]
        else:
            failed_searches += 1

    if solutions_found != 0:
        average_steps = total_steps/solutions_found
        
    results = (average_steps, solutions_found, failed_searches)

    print("The average number of steps for each of the", results[1], "problem(s) that had a solution was", results[0], ".")
    print("No solution was found for", results[2]," problem(s).")

    return results

#print("For a csp with two variables and 2 constraints:")
#print()
#test_solution1 = local_search(2, 2)
#print("--------------------------------------------------")
#print()

#print("For a csp with two variables and 5 constraints:")
#print()
#test_solution1 = local_search(2, 5)
#print("--------------------------------------------------")
#print()

#print("For a csp with two variables and 8 constraints:")
#print()
#test_solution1 = local_search(2, 8)
#print("--------------------------------------------------")
#print()

#print("For a csp with two variables and 9 constraints:")
#print()
#test_solution1 = local_search(2, 9)
#print("--------------------------------------------------")
#print("--------------------------------------------------")
#print()
#print()

#print("For a csp with 10 variables and 10 constraints:")
#print()
#test_solution1 = local_search(10, 10)
#print("--------------------------------------------------")
#print()

#print("For a csp with 10 variables and 50 constraints:")
#print()
#test_solution1 = local_search(10, 50)
#print("--------------------------------------------------")
#print()

#print("For a csp with 10 variables and 100 constraints:")
#print()
#test_solution1 = local_search(10, 100)
#print("--------------------------------------------------")
#print()

#print("For a csp with 10 variables and 500 constraints:")
#print()
#test_solution1 = local_search(10, 500)
#print("--------------------------------------------------")
#print()

print("For a csp with 10 variables and 125 constraints:")
print()
test_solution1 = local_search_hundred_times(10, 125)
print("--------------------------------------------------")
print()

print("For a csp with 20 variables and 60 constraints:")
print()
test_solution1 = local_search_hundred_times(20, 60)
print("--------------------------------------------------")
print()

#print("For a csp with 3 variables and 27 constraints:")
#print()
#test_solution1 = local_search(3, 27)
#print("--------------------------------------------------")
#print()

#print("For a csp with 4 variables and 80 constraints:")
#print()
#test_solution1 = local_search(4, 80)
#print("--------------------------------------------------")
#print()

#print("For a csp with 100 variables and 2 constraints:")
#print()
#test_solution1 = local_search(100, 2)
#print("--------------------------------------------------")
#print()
            
