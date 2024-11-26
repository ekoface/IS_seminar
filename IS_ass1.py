import numpy as np
import pygad
import random
import json

def fitness_func(solution):
    preference_score = np.sum(solution * P.flatten())

    penalty = 0
    penalty += np.sum(solution * A.flatten())

    reshaped_sol = solution.reshape((num_reviewers, num_papers))

    #penalty for exceeding reviewer capacity, not meeting min reviews per paper, exceeding max reviews per paper
    per_reviewer = np.sum(reshaped_sol, axis=1)
    per_paper = np.sum(reshaped_sol, axis=0)
    penalty += np.sum(per_paper < min_reviews_per_paper)
    penalty += np.sum(per_paper > max_reviews_per_paper)
    penalty += np.sum(per_reviewer > reviewer_capacity)

    # how many papers have reviewers that are friends
    co_review_matrix = np.dot(reshaped_sol, reshaped_sol.T)
    friend_review_counts = F * co_review_matrix
    penalty = np.sum(friend_review_counts) // 2
    
    #how many friends reviewed papers that their friends authored
    authored_papers_by_friends = np.dot(F, A)
    penalty_matrix = reshaped_sol * authored_papers_by_friends
    penalty += np.sum(penalty_matrix)


    return preference_score - penalty


def initial_population(num_reviewers, num_papers, population_size):
    population = np.random.Generator(0, 2, size=(population_size, num_reviewers * num_papers))
    return population

# def crossover_func(parents, offspring_size, ga_instance):
#     #to do
#     return offspring

# def mutation_func(offspring, ga_instance):
#     #to do
#     return offspring


with open('datasets/easy_dataset_1.json', 'r') as file:
    data = json.load(file)

num_generations = 100
num_parents_mating = 4
population_size = 10

num_papers = data['num_papers']
num_reviewers = data['num_reviewers']
reviewer_capacity = data['reviewer_capacity']
min_reviews_per_paper = data['min_reviews_per_paper']
max_reviews_per_paper = data['max_reviews_per_paper']

P = np.array(data['preferences'])
F = np.array(data['friendships'])
A = np.array(data['authorship'])

initial_pop = initial_population(num_reviewers, num_papers, population_size)

rez = fitness_func(initial_pop[0])
print(rez)

# ga_instance = pygad.GA(num_generations=num_generations,
#                        num_parents_mating=num_parents_mating,
#                        fitness_func=fitness_func,
#                        sol_per_pop=population_size,
#                        num_genes=num_reviewers * num_papers,
#                        initial_population=initial_pop,
#                        crossover_type=crossover_func,
#                        mutation_type=mutation_func,
#                        mutation_percent_genes=5)

# ga_instance.run()

# solution, solution_fitness = ga_instance.best_solution()
# print("Best solution:", solution.reshape((num_reviewers, num_papers)))
# print("Best solution fitness:", solution_fitness)

