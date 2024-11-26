import numpy as np
import pygad
import random
import json


# Define the fitness function
def fitness_func(solution, solution_idx):
    total_score = 0
    num_reviewers, num_papers = conference_data.preference_matrix.shape

    for reviewer in range(num_reviewers):
        for paper in range(num_papers):
            if solution[reviewer][paper] == 1:
                total_score += conference_data.preference_matrix[reviewer][paper]
                if conference_data.authorship_constraints[reviewer][paper] == 1:
                    total_score -= 10  # Penalize authorship constraint violation
                for friend in range(num_reviewers):
                    if conference_data.friendship_matrix[reviewer][friend] == 1 and solution[friend][paper] == 1:
                        total_score -= 5  # Penalize friendship constraint violation

    return total_score


def initial_population(num_reviewers, num_papers, population_size):
    population = []
    for _ in range(population_size):
        individual = np.zeros((num_reviewers, num_papers), dtype=int)
        for paper in range(num_papers):
            reviewers = random.sample(range(num_reviewers), 2)  # Assign 2 reviewers per paper
            for reviewer in reviewers:
                individual[reviewer][paper] = 1
        population.append(individual.flatten())
    return population

# Define the crossover function
def crossover_func(parents, offspring_size, ga_instance):
    offspring = []
    for k in range(offspring_size[0]):
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k + 1) % parents.shape[0]
        crossover_point = np.random.randint(0, parents.shape[1])
        offspring.append(np.concatenate((parents[parent1_idx, :crossover_point], parents[parent2_idx, crossover_point:])))
    return np.array(offspring)

# Define the mutation function
def mutation_func(offspring, ga_instance):
    for idx in range(offspring.shape[0]):
        mutation_point = np.random.randint(0, offspring.shape[1])
        offspring[idx, mutation_point] = 1 - offspring[idx, mutation_point]
    return offspring


with open('easy_dataset_1.json', 'r') as file:
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

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_func,
                       sol_per_pop=population_size,
                       num_genes=num_reviewers * num_papers,
                       initial_population=initial_pop,
                       crossover_type=crossover_func,
                       mutation_type=mutation_func,
                       mutation_percent_genes=5)

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Best solution:", solution.reshape((num_reviewers, num_papers)))
print("Best solution fitness:", solution_fitness)