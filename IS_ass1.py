import numpy as np
import pygad
import random
import json
import itertools
def print_solution(solution_matrix):
    solution_matrix = solution.reshape((num_reviewers, num_papers))

    header = "Reviewers\\Papers||| " + " | ".join([f"Paper {i+1}" for i in range(num_papers)])
    print(header)
    print("-" * len(header))

    # Print each reviewer's assignments
    for reviewer_idx in range(num_reviewers):
        row = f"Reviewer {reviewer_idx+1:<6} |||   " + "  |   ".join(f"{value:<4}" for value in solution_matrix[reviewer_idx])
        print(row)


def solution_is_valid(solution,get_sum=True,print_values=False, return_matrix = False):
    solution = solution.flatten()
    penalty_authorship = 0
    penalty_min_reviews = 0
    penalty_max_reviews = 0
    penalty_reviewer_capacity = 0
    penalty_friends = 0
    penalty_authorship = np.sum(solution * A.flatten())
    reshaped_sol = solution.reshape((num_reviewers, num_papers))
    per_reviewer = np.sum(reshaped_sol, axis=1)
    per_paper = np.sum(reshaped_sol, axis=0)
    penalty_min_reviews = np.sum(per_paper < min_reviews_per_paper)
    penalty_max_reviews = np.sum(per_paper > max_reviews_per_paper)
    penalty_reviewer_capacity = np.sum(per_reviewer > reviewer_capacity)
    co_review_matrix = np.dot(reshaped_sol, reshaped_sol.T)
    friend_review_counts = F * co_review_matrix
    penalty_friends = np.sum(friend_review_counts) // 2
    authored_papers_by_friends = np.dot(F, A)
    penalty_matrix = reshaped_sol * authored_papers_by_friends
   # penalty_friends += np.sum(penalty_matrix)

    if print_values:
        print("penalty_authorship: ", penalty_authorship)
        print("penalty_min_reviews: ", penalty_min_reviews)
        print("penalty_max_reviews: ", penalty_max_reviews)
        print("penalty_reviewer_capacity: ", penalty_reviewer_capacity)
        print("penalty_friends: ", penalty_friends)
        print("Sum: ", penalty_authorship + penalty_min_reviews + penalty_max_reviews + penalty_reviewer_capacity + penalty_friends)

    if return_matrix:
        sol = (penalty_authorship, penalty_min_reviews, penalty_max_reviews, penalty_reviewer_capacity, penalty_friends)
        return np.array(sol)
    if (not get_sum):
        return (penalty_authorship, penalty_min_reviews, penalty_max_reviews, penalty_reviewer_capacity, penalty_friends)
    else:
        return (penalty_authorship + penalty_min_reviews + penalty_max_reviews + penalty_reviewer_capacity + penalty_friends)
    

def brute_force_find_valid_solution(num_reviewers, num_papers):
    # Generate the initial solution matrix with all 0s
    solution_matrix = np.zeros((num_reviewers, num_papers), dtype=int)

    # Total number of entries in the matrix
    total_entries = num_reviewers * num_papers

    # Iterate through all possible combinations of 0s and 1s
    for i in range(2**total_entries):
        # Convert the current number to binary and pad with zeros
        binary_representation = bin(i)[2:].zfill(total_entries)
        
        # Fill the solution matrix with the current combination of 0s and 1s
        for idx, bit in enumerate(binary_representation):
            row = idx // num_papers
            col = idx % num_papers
            solution_matrix[row, col] = int(bit)
        
        # Check if the current solution is valid
        if (solution_is_valid(solution_matrix,get_sum=True) == 0):
            return solution_matrix

    # Return None if no valid solution is found
    return None

def create_fitness_function(fitnes_penalty):
    def fitness_func(ga_instance, solution, solution_idx):
        preference_score = np.sum(solution * P.flatten())
        #print(preference_score)

        penalty = 0
        penalty += np.sum(solution * A.flatten()) * fitnes_penalty["authorship"]
        print(penalty)
        
        reshaped_sol = solution.reshape((num_reviewers, num_papers))
        #print(reshaped_sol)

        #penalty for exceeding reviewer capacity, not meeting min reviews per paper, exceeding max reviews per paper
        per_reviewer = np.sum(reshaped_sol, axis=1)
        per_paper = np.sum(reshaped_sol, axis=0)

        #to do bolj kot je napacno vecji je penalty
        #penalty += np.sum(per_paper < min_reviews_per_paper) * fitnes_penalty["min_reviews"]
        #print(penalty)
        #penalty += np.sum(per_paper > max_reviews_per_paper) * fitnes_penalty["max_reviews"]
        #print(penalty)
        #penalty += np.sum(per_reviewer > reviewer_capacity) * fitnes_penalty["reviewer_capacity"]
        #print(penalty)
       # #print("autorship", (per_reviewer * A))
       # print("per_reviewer", per_reviewer) 
        #print(reshaped_sol)
        penalty += (np.maximum(0, per_paper - max_reviews_per_paper) *fitnes_penalty["max_reviews"]).sum()
        penalty += (np.maximum(0, min_reviews_per_paper - per_paper) *fitnes_penalty["min_reviews"]).sum() **4
        penalty += (np.maximum(0, per_reviewer - reviewer_capacity) * fitnes_penalty["reviewer_capacity"]).sum() **2
        
        #var = np.maximum(0, per_reviewer - reviewer_capacity).sum() **2
        
        #print("var", var)

        # how many papers have reviewers that are friends
        co_review_matrix = np.dot(reshaped_sol, reshaped_sol.T)
        friend_review_counts = F * co_review_matrix 
        penalty += np.sum(friend_review_counts) // 2 * fitnes_penalty["friends"]
        #print(penalty)

        #how many friends reviewed papers that their friends authored
        #authored_papers_by_friends = np.dot(F, A)
        #penalty_matrix = reshaped_sol * authored_papers_by_friends
        #penalty += np.sum(penalty_matrix) * fitnes_penalty["friends"]
        #print(penalty)
        #return  preference_score - penalty
        return  - penalty
    return fitness_func

def custom_mutation(offspring, number_mutatuios=3):
    initial_vector_constrains = solution_is_valid(offspring, get_sum=False,print_values=False, return_matrix=True)
    #print(initial_vector_constrains)
    constraint_vector = np.array([1 if val == 0 else 0 for val in initial_vector_constrains])
    #print(constraint_vector)
    #print( offspring)
    while number_mutatuios > 0:
        idx = 0
        gene_idx = np.random.randint(0, offspring.shape[1])
        # Perform mutation
       # print("gene_idx", gene_idx)
        offspring[idx, gene_idx] = 1 - offspring[idx, gene_idx]
        
        # Check if the mutated solution meets the constraints
        new_constraint_vector = solution_is_valid(offspring, get_sum=False, print_values=False, return_matrix=True)
        #print("new_constraint_vector", new_constraint_vector)
        if np.all(new_constraint_vector[constraint_vector == 1] == 0):
            number_mutatuios -= 1
        else:
            # If not, revert the mutation
            offspring[idx, gene_idx] = 1 - offspring[idx, gene_idx]
        #print("new_constraint_vector", new_constraint_vector)
    print("offsprin", offspring)
    return offspring


def initial_population(num_reviewers, num_papers, population_size,num=5):
    best_population = None
    best_count = None
    first = True
    for i in range(num):
        population = np.random.randint(0, 2, size=(population_size, num_reviewers * num_papers))
        count = 0
        for j in range(population_size):
            count += solution_is_valid(population[j], get_sum=True,print_values=False)
        count /= population_size
        if first or count > best_count:
            first = False
            best_count = count
            best_population = population

    print("Best count: ", best_count)
    return population


with open('datasets/easy_dataset_1.json', 'r') as file:
    data = json.load(file)
num_papers = data['num_papers']
num_reviewers = data['num_reviewers']
reviewer_capacity = data['reviewer_capacity']
min_reviews_per_paper = data['min_reviews_per_paper']
max_reviews_per_paper = data['max_reviews_per_paper']
#print(num_papers, num_reviewers, reviewer_capacity, min_reviews_per_paper, max_reviews_per_paper)

num_generations = 100
population_size = 50
num_parents_mating = 10

P = np.array(data['preferences'])
F = np.array(data['friendships'])
A = np.array(data['authorship'])
print(P)
fitnes_penalty = {
    "friends": 10,
    "authorship": 30,
    "min_reviews": 10,
    "max_reviews": 3,
    "reviewer_capacity": 25
}
initial_pop = initial_population(num_reviewers, num_papers, population_size)
# print(initial_pop)

# for i in range(population_size):
#     rez = fitness_func(initial_pop[i])
#     print(rez)

ga_instance = pygad.GA(
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    fitness_func=create_fitness_function(fitnes_penalty),
    sol_per_pop=population_size,
    num_genes=num_reviewers * num_papers,
    initial_population=initial_pop,
    crossover_type="scattered",
    mutation_type=custom_mutation,
    mutation_percent_genes=5,
    gene_type=int,
    gene_space=[0, 1],
    stop_criteria= "saturate_50",
    parent_selection_type="tournament"
)


#ga_instance.plot_fitness()
#ga_instance.run()
#solution, solution_fitness, solution_idx = ga_instance.best_solution()
#print("Best solution:", solution.reshape((num_reviewers, num_papers)))
#print("Best solution fitness:", solution_fitness)

# Generate a random solution
#
solution = initial_population(num_reviewers, num_papers, 1)
print("solution", solution) 
#print_solution(solution)
#print("Solution is valid: ", solution_is_valid(solution,get_sum=True,print_values=True))
mutated = custom_mutation(solution)
print_solution(mutated)
#print(brute_force_find_valid_solution(5, 5))