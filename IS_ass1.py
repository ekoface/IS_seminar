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
    

def fitness_func_prefence(ga_instance, solution, solution_idx):
    preference_score = np.sum(solution * P.flatten())
    return preference_score


def create_fitness_function_basic(fitness_penalty):
    def fitness_func_basic(ga_instance, solution, solution_idx):
        preference_score = np.sum(solution * P.flatten())
        penalty = 0
        penalty += np.sum(solution * A.flatten()) * fitness_penalty["authorship"]
        reshaped_sol = solution.reshape((num_reviewers, num_papers))
        per_reviewer = np.sum(reshaped_sol, axis=1)
        per_paper = np.sum(reshaped_sol, axis=0)
        penalty += np.sum(per_paper < min_reviews_per_paper) * fitness_penalty["min_reviews"]
        penalty += np.sum(per_paper > max_reviews_per_paper) * fitness_penalty["max_reviews"]
        penalty += np.sum(per_reviewer > reviewer_capacity) * fitness_penalty["reviewer_capacity"]
        co_review_matrix = np.dot(reshaped_sol, reshaped_sol.T)
        friend_review_counts = F * co_review_matrix
        penalty += np.sum(friend_review_counts) // 2 * fitness_penalty["friends"]

        return preference_score - penalty
    return fitness_func_basic
    

#added constraints for reviewer who are friends with authors
def create_fitness_function(fitnes_penalty):
    def fitness_func(ga_instance, solution, solution_idx):
        preference_score = np.sum(solution * P.flatten())

        penalty = 0
        penalty += np.sum(solution * A.flatten()) * fitnes_penalty["authorship"]

        reshaped_sol = solution.reshape((num_reviewers, num_papers))
        #penalty for exceeding reviewer capacity, not meeting min reviews per paper, exceeding max reviews per paper
        per_reviewer = np.sum(reshaped_sol, axis=1)
        per_paper = np.sum(reshaped_sol, axis=0)

        penalty += (np.maximum(0, per_paper - max_reviews_per_paper) *fitnes_penalty["max_reviews"]).sum()
        penalty += (np.maximum(0, min_reviews_per_paper - per_paper) *fitnes_penalty["min_reviews"]).sum() 
        penalty += (np.maximum(0, per_reviewer - reviewer_capacity) * fitnes_penalty["reviewer_capacity"]).sum() 

        # how many papers have reviewers that are friends
        co_review_matrix = np.dot(reshaped_sol, reshaped_sol.T)
        friend_review_counts = F * co_review_matrix 
        penalty += np.sum(friend_review_counts) // 2 * fitnes_penalty["friends"]

        #how many friends reviewed papers that their friends authored
        authored_papers_by_friends = np.dot(F, A)
        penalty_matrix = reshaped_sol * authored_papers_by_friends
        penalty += np.sum(penalty_matrix) * fitnes_penalty["friends"]
        

        return  preference_score - penalty
    return fitness_func

def custom_crossover(parents, offspring_size, ga_instance):
    offspring = np.zeros(offspring_size)
    for k in range(offspring_size[0]):
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k + 1) % parents.shape[0]
        parent1 = parents[parent1_idx]
        parent2 = parents[parent2_idx]
        parent1 = parent1.flatten()
        parent2 = parent2.flatten()
        offspring[k] = parent1
        crossover_point = np.random.randint(0, parent1.shape[0])
        offspring[k, crossover_point:] = parent2[crossover_point:]
    return offspring
def custom_mutation(offspring, ga_instance):
    number_tries = 7
    check_valid = random.randint(0, 2) == 1
    for idx in range(offspring.shape[0]):
        initial_vector_constrains = solution_is_valid(offspring[idx], get_sum=False, print_values=False, return_matrix=True)
        constraint_vector = np.array([1 if val == 0 else 0 for val in initial_vector_constrains])
        
        number_mutations = offspring.shape[1] 
        while number_mutations > 0 and number_tries > 0:
            gene_idx = np.random.randint(0, offspring.shape[1])
            # Perform mutation
            offspring[idx, gene_idx] = 1 - offspring[idx, gene_idx]
            
            # Check if the mutated solution meets the constraints
            new_constraint_vector = solution_is_valid(offspring[idx], get_sum=False, print_values=False, return_matrix=True)
            
            # Ensure the mutation respects the constraint vector
            if np.all(new_constraint_vector[constraint_vector == 1] == 0):
                number_tries -= 1
            else:
                # If not, revert the mutation
                offspring[idx, gene_idx] = 1 - offspring[idx, gene_idx]
                number_mutations -= 1
    if(number_tries != 0):
        print("Mutation failed",number_mutations," times",number_tries)
    return offspring

def generate_valid_matrix(num_reviewers, num_papers, min_reviews_per_paper, max_reviews_per_paper, reviewer_capacity):
    matrix = np.zeros((num_reviewers, num_papers), dtype=int)
    
    for paper in range(num_papers):
        num_reviews = np.random.randint(min_reviews_per_paper, max_reviews_per_paper)
        reviewers = []
        
        # Ensure that each paper gets the required number of reviews
        while len(reviewers) < num_reviews:
            potential_reviewer = np.random.choice(num_reviewers)
            if np.sum(matrix[potential_reviewer, :]) < reviewer_capacity and potential_reviewer not in reviewers:
                reviewers.append(potential_reviewer)
            else:
                for previous_paper in range(paper):
                    if matrix[potential_reviewer, previous_paper] == 1:
                        if np.sum(matrix[:, previous_paper]) > min_reviews_per_paper:
                            matrix[potential_reviewer, paper] = 1
                            matrix[potential_reviewer, previous_paper] = 0
                            reviewers.append(potential_reviewer)
                            break
        
        for reviewer in reviewers:
            matrix[reviewer, paper] = 1
        #print("done",paper)
        #print(matrix)
    return matrix

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

num_generations = 1000
population_size = 500
num_parents_mating = 5

P = np.array(data['preferences'])
F = np.array(data['friendships'])
A = np.array(data['authorship'])
#print(P)
fitnes_penalty = {
    "friends": 2,
    "authorship": 8,
    "min_reviews": 12,
    "max_reviews": 12,
    "reviewer_capacity": 9
}
initial_pop = initial_population(num_reviewers, num_papers, population_size)

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
    parent_selection_type="rank",
)


#ga_instance.plot_fitness()
ga_instance.run()
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Best solution:", solution.reshape((num_reviewers, num_papers)))
print("Best solution fitness:", solution_fitness)
is_valid = solution_is_valid(solution, get_sum=True, print_values=True)
# Generate a random solution
#
#solution = initial_population(num_reviewers, num_papers, 1)
#print("solution", solution) 
#print_solution(solution)
#print("Solution is valid: ", solution_is_valid(solution,get_sum=True,print_values=True))
#mutated = custom_mutation(solution)
#print_solution(mutated)
#print(brute_force_find_valid_solution(5, 5))