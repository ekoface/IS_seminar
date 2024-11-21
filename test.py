import numpy as np
import json
import random


class Reviewer:
    def __init__(self, id, capacity, friendships, authorship, preferences):
        self.id = id
        self.capacity = capacity
        self.current_load = 0
        self.friendships = friendships
        self.authorship = authorship
        self.preferences = preferences

    def can_review(self):
        return self.current_load < self.capacity

    def assign_paper(self):
        if self.can_review():
            self.current_load += 1
            return True
        return False

    def is_author(self, paper_id):
        return self.authorship[paper_id] == 1
    def is_friend(self, reviewer_id):
        return self.friendships[reviewer_id] == 1
    def print(self):
        print(f"Reviewer {self.id}:")
        print(f"  Capacity: {self.capacity}")
        print(f"  Current Load: {self.current_load}")
        print(f"  Authorship: {self.authorship}")
        print(f"  Friendships: {self.friendships}")
        print(f"  Preferences: {self.preferences}")
        print()


class Paper:
    def __init__(self, id, min_reviews, max_reviews, assigned_reviewers):
        self.id = id
        self.min_reviews = min_reviews
        self.max_reviews = max_reviews
        self.current_reviews = np.sum(assigned_reviewers)
        self.assigned_reviewers = assigned_reviewers

    def is_fully_reviewed(self):
        return len(self.assigned_reviewers) >= self.min_reviews

    def print(self):
        print(f"Paper {self.id}:")
        print(f"  Min Reviews: {self.min_reviews}")
        print(f"  Max Reviews: {self.max_reviews}")
        print(f"  Assigned Reviewers: {self.assigned_reviewers}")
        print(f"  Current Reviews: {self.current_reviews}")
        print()
    
    def add_reviewer(self, reviewer):
        if len(self.assigned_reviewers) < self.max_reviews:
            self.assigned_reviewers.append(reviewer)
            current_reviews += 1
            return True
        return False
    def remove_reviewer(self, reviewer):
        if reviewer in self.assigned_reviewers:
            self.assigned_reviewers.remove(reviewer)
            current_reviews -= 1
            return True
        return False

class Example:
    def __init__(self, reviewers,papers):
        self.reviewers = reviewers
        self.papers = papers

    def print(self):
        for reviewer in self.reviewers:
            reviewer.print()
        for paper in self.papers:
            paper.print()
    def mutatuin(self):
        pass
class Assignment:
    def __init__(self, data):
        self.examples = []
        self.data = data
        self.reviewers = data['num_reviewers']
        self.papers = data['num_papers']
        self.population = [[]]

    def intialize_reviewers_papers(self,population):
        current_reviewer_list = []
        current_paper_list = []
        for idx, example in enumerate(population):
            for idx, reviewer in enumerate(example):
                current_reviewer = Reviewer(id=idx, capacity=data['reviewer_capacity'], friendships=np.array(data['friendships'])[:,idx], authorship=np.array(data['authorship'])[:,idx], preferences=np.array(data['preferences'])[:,idx])
                current_reviewer_list.append(current_reviewer)
            for idx in range(example.shape[1]):
                paper = example[:,idx]
                current_paper = Paper(id=idx, min_reviews=3, max_reviews=5 ,assigned_reviewers=paper)
                current_paper_list.append(current_paper)
            self.examples.append(Example(current_reviewer_list, current_paper_list))
        
    
    def print(self):
        for example in self.examples:
            example.print()
        print(self.population)

    def mutatuion(self, assignment, mutation_rate=0.1):
        num_reviewers, num_papers = assignment.shape
        number = mutation_rate * (num_reviewers * num_papers)
        for i in range(number):
            while True:
                reviewer_idx = random.randint(0, num_reviewers - 1)
                paper_idx = random.randint(0, num_papers - 1)


    def evaluate_assignment(self, assignment_matrix):
        penalty = 0

        for reviewer_id, paper_id in zip(*np.where(assignment_matrix == 1)):
            reviewer = self.reviewers[reviewer_id]
            paper = self.papers[paper_id]

            if not reviewer.can_review() or paper.is_fully_reviewed():
                penalty += 1
                continue

            if reviewer.is_author(paper_id):
                penalty += 1

            for other_reviewer in paper.assigned_reviewers:
                if reviewer.friendships[other_reviewer.id] == 1:
                    penalty += 1

            paper.add_reviewer(reviewer)

        for paper in self.papers:
            if not paper.is_fully_reviewed():
                penalty += 1

        return np.sum(self.preferences * assignment_matrix) - penalty

    def genetic_algorithm(self, iterations=10000, population_size=500):
        population = [
            np.random.randint(0, 2, ((self.reviewers), (self.papers)))
            for _ in range(population_size)
        ]
        self.population = population

        self.intialize_reviewers_papers(population)
        return (0,1)
        for _ in range(iterations):
            fitness_scores = [self.evaluate_assignment(ind.copy()) for ind in population]

            # Handle non-positive fitness scores by shifting them
            min_fitness = min(fitness_scores)
            if min_fitness <= 0:
                fitness_scores = [score - min_fitness + 1 for score in fitness_scores]

            # Selection
            selected = random.choices(population, weights=fitness_scores, k=population_size // 2)

            # Crossover
            offspring = []
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    parent1, parent2 = selected[i], selected[i + 1]
                    crossover_point = np.random.randint(1, len(self.papers))
                    child = np.hstack((parent1[:, :crossover_point], parent2[:, crossover_point:]))
                    offspring.append(child)

            # Update population
            population = selected + offspring

        best_solution = max(population, key=lambda ind: self.evaluate_assignment(ind.copy()))
        best_fitness = self.evaluate_assignment(best_solution.copy())
        return best_solution, best_fitness


# Load data
with open('./datasets/easy_dataset_1.json', 'r') as file:
    data = json.load(file)

assignment_system = Assignment(data)
best_assignment, best_fitness = assignment_system.genetic_algorithm(iterations=10, population_size=1)
assignment_system.print()


'''
reviewers = [
    Reviewer(
        id=i,
        capacity=data['reviewer_capacity'],
        friendships=np.array(data['friendships'])[i, :],
        authorship=np.array(data['authorship'])[i, :],
        preferences=np.array(data['preferences'])[i, :],
    )
    for i in range(data['num_reviewers'])
]

papers = [
    Paper(
        id=j,
        min_reviews=data['min_reviews_per_paper'],
        max_reviews=data['max_reviews_per_paper'],
    )
    for j in range(data['num_papers'])
]

for reviewer in reviewers:
    reviewer.print()

for paper in papers:
    paper.print()
num_reviewers = np.array(data['num_reviewers'])
num_papers = np.array(data['num_papers'])
test = np.random.randint(0, 2, (num_reviewers, num_reviewers))

print("friendships",np.array(data['friendships']))
reviewers = []
papers = []
for idx,reviewer in enumerate(test):
    current_reviewer = Reviewer(id=idx, capacity=2, friendships=np.array(data['friendships'])[:,idx], authorship=np.array(data['authorship'])[:,idx], preferences=np.array(data['preferences'])[:,idx])
    reviewers.append(current_reviewer)
print(test)
for idx in (range(test.shape[1])):
    paper = test[:,idx]
    current_paper = Paper(id=idx, min_reviews=3, max_reviews=5 ,assigned_reviewers=paper)

    papers.append(current_paper)

for idx, paper in enumerate(range(test.shape[1])):
    current_paper = Paper(id=idx, min_reviews=3, max_reviews=5)
    current_paper.initialize_reviewers(paper)
    papers.append(current_paper)
    print("paper",paper)
    print(current_paper.print())
    break
    
'''


'''
assignment_system = Assignment(reviewers, papers)
best_assignment, best_fitness = assignment_system.genetic_algorithm(iterations=10, population_size=20)

print("Best Assignment:\n", best_assignment)
print("Best Fitness Score:", best_fitness)
'''