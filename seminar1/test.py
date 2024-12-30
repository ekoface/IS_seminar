import numpy as np
import json
import random


class Reviewer:
    """
    A class to represent a reviewer in a peer review system.

    Attributes:
    ----------
    id : int
        Unique identifier for the reviewer.
    capacity : int
        The maximum number of papers the reviewer can handle.
    current_load : int
        The current number of papers assigned to the reviewer.
    friendships : list[int]
        A binary list indicating friendships with other reviewers (1 = friend, 0 = not a friend).
    authorship : list[int]
        A binary list indicating authorship of papers (1 = author, 0 = not an author).
    preferences : list[int]
        A list representing the reviewer's preferences for papers (e.g., scores or priorities).
    """

    def __init__(self, id, capacity, friendships, authorship, preferences,current_load):
        """
        Initializes a Reviewer instance.

        Parameters:
        ----------
        id : int
            Unique identifier for the reviewer.
        capacity : int
            Maximum number of papers the reviewer can handle.
        friendships : list[int]
            Binary list of friendships with other reviewers.
        authorship : list[int]
            Binary list of authorship for papers.
        preferences : list[int]
            List of preferences for papers.
        """
        self.id = id
        self.capacity = capacity
        self.current_load = current_load # Start with no assigned papers
        self.friendships = friendships
        self.authorship = authorship
        self.preferences = preferences

    def can_review(self):
        """
        Checks if the reviewer can accept more papers.

        Returns:
        -------
        bool
            True if the current load is less than the capacity, False otherwise.
        """
        return self.current_load < self.capacity

    def assign_paper(self):
        """
        Assigns a paper to the reviewer if they can review more papers.

        Returns:
        -------
        bool
            True if the paper was successfully assigned, False otherwise.
        """
        print("ok")
        if self.can_review():
            self.current_load += 1
            print("changed")
            return True
        return False
    def remove_paper(self):
        self.current_load -= 1
    def is_author(self, paper_id):
        """
        Checks if the reviewer is an author of a given paper.

        Parameters:
        ----------
        paper_id : int
            The ID of the paper to check.

        Returns:
        -------
        bool
            True if the reviewer is the author of the paper, False otherwise.
        """
        return self.authorship[paper_id] == 1

    def is_friend(self, reviewer_id):
        """
        Checks if the reviewer is friends with another reviewer.

        Parameters:
        ----------
        reviewer_id : int
            The ID of the reviewer to check.

        Returns:
        -------
        bool
            True if the reviewer is friends with the other reviewer, False otherwise.
        """
        return self.friendships[reviewer_id] == 1

    def print(self):
        """
        Prints the details of the reviewer.
        """
        print(f"Reviewer {self.id}:")
        print(f"  Capacity: {self.capacity}")
        print(f"  Current Load: {self.current_load}")
        print(f"  Authorship: {self.authorship}")
        print(f"  Friendships: {self.friendships}")
        print(f"  Preferences: {self.preferences}")
        print()

class Paper:
    """
    A class to represent a paper in a peer review system.

    Attributes:
    ----------
    id : int
        Unique identifier for the paper.
    min_reviews : int
        Minimum number of reviews required for the paper.
    max_reviews : int
        Maximum number of reviews allowed for the paper.
    current_reviews : int
        Current number of reviews assigned to the paper.
    assigned_reviewers : list[int]
        List of reviewers currently assigned to this paper.
    """

    def __init__(self, id, min_reviews, max_reviews, assigned_reviewers):
        """
        Initializes a Paper instance.

        Parameters:
        ----------
        id : int
            Unique identifier for the paper.
        min_reviews : int
            Minimum number of reviews required for the paper.
        max_reviews : int
            Maximum number of reviews allowed for the paper.
        assigned_reviewers : list[int]
            List of reviewers initially assigned to this paper.
        """
        self.id = id
        self.min_reviews = min_reviews
        self.max_reviews = max_reviews
        self.current_reviews = np.sum(assigned_reviewers)  # Sum of reviewers currently assigned
        self.assigned_reviewers = assigned_reviewers

    def is_fully_reviewed(self):
        """
        Checks if the paper has the minimum required number of reviews.

        Returns:
        -------
        bool
            True if the number of assigned reviewers meets or exceeds the minimum reviews, False otherwise.
        """
        return len(self.assigned_reviewers) >= self.min_reviews

    def add_reviewer(self, idx):
        """
        Assigns a new reviewer to the paper if the maximum review limit has not been reached.

        Parameters:
        ----------
        reviewer : int
            ID of the reviewer to add.

        Returns:
        -------
        bool
            True if the reviewer was successfully assigned, False otherwise.
        """
        self.assigned_reviewers[idx] = 1
        self.current_reviews += 1

    def remove_reviewer(self, idx):
        """
        Removes an assigned reviewer from the paper if they are currently assigned.

        Parameters:
        ----------
        reviewer : int
            ID of the reviewer to remove.

        Returns:
        -------
        bool
            True if the reviewer was successfully removed, False otherwise.
        """
        self.assigned_reviewers[idx] = 0
        self.current_reviews -= 1
    def is_friend(self, reviewer_id):
        pass
    def print(self):
        """
        Prints the details of the paper.
        """
        print(f"Paper {self.id}:")
        print(f"  Min Reviews: {self.min_reviews}")
        print(f"  Max Reviews: {self.max_reviews}")
        print(f"  Assigned Reviewers: {self.assigned_reviewers}")
        print(f"  Current Reviews: {self.current_reviews}")
        print()

class Individual:
    def __init__(self, table,reviewers,papers,):
        self.table = table
        self.reviewers = []
        self.papers = []

    def print(self):
        for reviewer in self.reviewers:
            reviewer.print()
        for paper in self.papers:
            paper.print()

    def initialize_reviewers_papers(self,example, data): 
        
        for idx, reviewer in enumerate(example):
                current_reviewer = Reviewer(id=idx, capacity=data['reviewer_capacity'], friendships=np.array(data['friendships'])[:,idx], authorship=np.array(data['authorship'])[:,idx], preferences=np.array(data['preferences'])[:,idx],current_load=np.sum(np.array(example)[idx]))
                self.reviewers.append(current_reviewer)
        for idx in range(example.shape[1]):
            paper = example[:,idx]
            current_paper = Paper(id=idx, min_reviews=3, max_reviews=5 ,assigned_reviewers=paper)
            self.papers.append(current_paper)

    def valid_mutations(self,table,position,value,data):
        #have to check if the mutation is valid if it has changed in T[0][0] then only change if the reviewer one has capacity and the paper has not reached max reviews else, return false Else true
        # have to check the reviewer and the position.
        reviewer,paper = position
        #self.print()
        if value == 1:
            if self.reviewers[reviewer].can_review() and self.papers[paper].current_reviews < self.papers[paper].max_reviews:
                current_reviewers = self.papers[paper].assigned_reviewers
                    # Check if the new reviewer is friends with any of the current reviewers
                for current_reviewer in current_reviewers:
                    if data['friendships'][reviewer][current_reviewer] == 1:
                        print("are friends")
                        print(data['friendships'])
                        return False

                self.reviewers[reviewer].assign_paper()
                self.papers[paper].add_reviewer(reviewer)

                return True
            else:
                return False
        else:
            if self.papers[paper].current_reviews - 1 >= self.papers[paper].min_reviews:
                self.reviewers[reviewer].remove_paper()
                self.papers[paper].remove_reviewer(reviewer)
                return True
            else: 
                return False

    def fitness_function(self,data):
        # get friends: 
        fitnness = 0
        for paper in self.papers:
            unique_friendships = np.sum((paper.assigned_reviewers[:, None] * paper.assigned_reviewers[None, :]) * np.triu(data['friendships']))

            reviewer_friends_count = unique_friendships
            #print("data['friendships']",data['friendships'])
           # print("paper.assigned_reviewers",paper.assigned_reviewers)
           # print("reviewer friends count",reviewer_friends_count)
            fitnness -= reviewer_friends_count *20
            
            autorship_score = np.sum(paper.assigned_reviewers* np.array(data['authorship'])[:,paper.id])
           # print("data['authorship']",data['authorship'])
           # print("np.array(data['authorship'])[:,paper.id])",np.array(data['authorship'])[:,paper.id])
           # print(autorship_score)
            fitnness -= autorship_score *20
            preference_score = np.sum(paper.assigned_reviewers* np.array(data['preferences'])[:,paper.id])

           # print("data['preferences']",data['preferences'])

           # print("preference_score",preference_score, "paper_id",paper.id) 
            fitnness += preference_score
            if(paper.current_reviews < paper.min_reviews):
                fitnness -= 100
            if(paper.current_reviews > paper.max_reviews):
                fitnness -= 100
        for reviewer in self.reviewers:
            if reviewer.current_load > reviewer.capacity:
                fitnness -= 100
       # print("fitness",fitnness) 
        return fitnness
    

class Genetic_Algorithm:
    def __init__(self, data):
        self.individuals = []
        self.data = data
        self.reviewers = data['num_reviewers']
        self.papers = data['num_papers']
        self.population = []

    def intialize_reviewers_papers(self,population):
        current_reviewer_list = []
        current_paper_list = []
        for idx, example in enumerate(population):
            new_individual = Individual(table=example, reviewers=current_reviewer_list, papers=current_paper_list)
            new_individual.initialize_reviewers_papers(example, self.data)
            self.individuals.append(new_individual)
            
    
    def print(self):
        for individual in self.individuals:
            individual.print()

    def mutatuion_step(self,individual,position,value,data):
        #individual.print()
        can_do = individual.valid_mutations(individual.table,position,value,data)
        return can_do

    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parents to generate a child.

        Parameters:
        ----------
        parent1 : np.ndarray
            Assignment matrix for the first parent.
        parent2 : np.ndarray
            Assignment matrix for the second parent.

        Returns:
        -------
        np.ndarray
            Assignment matrix for the child.
        """
        crossover_point = np.random.randint(1, parent1.shape[1])  # Randomly select a crossover column
        child = np.hstack((parent1[:, :crossover_point], parent2[:, crossover_point:]))
        return child

    def mutation(self, assignment, mutation_rate=0.1):
        """
        Perform mutation on an assignment matrix.

        Parameters:
        ----------
        assignment : np.ndarray
            Assignment matrix to mutate.
        mutation_rate : float
            Probability of flipping each bit in the assignment matrix.
git
        Returns:
        -------
        np.ndarray
            Mutated assignment matrix.
        """
        num_reviewers, num_papers = assignment.shape
        for reviewer_idx in range(num_reviewers):
            for paper_idx in range(num_papers):
                if random.random() < mutation_rate:
                    assignment[reviewer_idx, paper_idx] = 1 - assignment[reviewer_idx, paper_idx]  # Flip 0 to 1 or 1 to 0
        return assignment

    def genetic_algorithm(self, iterations=10000, population_size=500, mutation_rate=0.1):
        """
        Run the genetic algorithm to optimize paper-reviewer assignments.

        Parameters:
        ----------
        iterations : int
            Number of iterations to run the genetic algorithm.
        population_size : int
            Size of the population.
        mutation_rate : float
            Rate of mutation in the population.

        Returns:
        -------
        Individual
            The best individual (assignment) found during optimization.
        """
        # Initialize the population
        population = [
            np.random.randint(0, 2, (self.reviewers, self.papers))
            for _ in range(population_size)
        ]
        self.population = population
        self.intialize_reviewers_papers(population)

        for _ in range(iterations):
            # Evaluate fitness
            fitness_scores = [individual.fitness_function(self.data) for individual in self.individuals]

            # Handle non-positive fitness scores by shifting them
            min_fitness = min(fitness_scores)
            if min_fitness <= 0:
                fitness_scores = [score - min_fitness + 1 for score in fitness_scores]

            # Selection
            selected = random.choices(self.individuals, weights=fitness_scores, k=population_size // 2)

            # Generate offspring through crossover
            offspring = []
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    parent1, parent2 = selected[i].table, selected[i + 1].table
                    child_table = self.crossover(parent1, parent2)
                    child_table = self.mutation(child_table, mutation_rate)
                    child = Individual(child_table, reviewers=[], papers=[])
                    child.initialize_reviewers_papers(child_table, self.data)
                    offspring.append(child)

            # Replace the old population with offspring
            self.individuals = selected + offspring

        # Return the best individual
        best_individual = max(self.individuals, key=lambda x: x.fitness_function(self.data))
        best_fitness_score = best_individual.fitness_function(self.data)
        return best_individual, best_fitness_score

# Load data
with open('./datasets/easy_dataset_1.json', 'r') as file:
    data = json.load(file)

assignment_system = Genetic_Algorithm(data)
population = [
            np.random.randint(0, 2, (assignment_system.reviewers, assignment_system.papers))
            for _ in range(1)
        ]
print(population)
assignment_system.intialize_reviewers_papers(population)
#assignment_system.print()
value = assignment_system.individuals[0].table[2][0] 
if value == 1: value = 0
else: value = 1
print("value",value)
print("valid_mutations", assignment_system.mutatuion_step(assignment_system.individuals[0],(2,0),value,data))
assignment_system.individuals[0].print()
#assignment_system.print()

#best_assignment,best_fitness_score= assignment_system.genetic_algorithm(iterations=100, population_size=100)

#print("Best Assignment:\n", best_assignment.table)
#print("Best Fitness Score:", best_fitness_score)


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