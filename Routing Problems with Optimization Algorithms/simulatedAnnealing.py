import sys
import random
import copy
import numpy as np
import pandas as pd
import time

# This class represents a state
class State:
    def __init__(self, route=[], distance=0):
        self.route = route
        self.distance = distance

    def __eq__(self, other):
        for i in range(len(self.route)):
            if self.route[i] != other.route[i]:
                return False
        return True

    def __lt__(self, other):
        return self.distance < other.distance

    def __repr__(self):
        return ('({0},{1})\n'.format(self.route, self.distance))

    def copy(self):
        return State(self.route, self.distance)

    def deepcopy(self):
        return State(copy.deepcopy(self.route), copy.deepcopy(self.distance))

    def update_distance(self, matrix, home):
        self.distance = 0
        from_index = home
        for i in range(len(self.route)):
            self.distance += matrix[from_index][self.route[i]]
            from_index = self.route[i]
        self.distance += matrix[from_index][home]

# This class represents a city
class City:
    def __init__(self, index, distance):
        self.index = index
        self.distance = distance

    def __lt__(self, other):
        return self.distance < other.distance

# Return true with probability p
def probability(p):
    return p > random.uniform(0.0, 1.0)

# Schedule function for simulated annealing
def exp_schedule(k=20, lam=0.005, limit=1000):
    return lambda t: (k * np.exp(-lam * t) if t < limit else 0)

def get_best_solution_by_distance(matrix, home):
    route = []
    from_index = home
    length = len(matrix) - 1

    while len(route) < length:
        row = matrix[from_index]
        cities = {}
        for i in range(len(row)):
            cities[i] = City(i, row[i])

        del cities[home]
        for i in route:
            del cities[i]

        sorted_cities = list(cities.values())
        sorted_cities.sort()
        from_index = sorted_cities[0].index
        route.append(from_index)

    state = State(route)
    state.update_distance(matrix, home)
    return state

def mutate(matrix, home, state, mutation_rate=0.01):
    mutated_state = state.deepcopy()

    for i in range(len(mutated_state.route)):
        if random.random() < mutation_rate:
            j = int(random.random() * len(state.route))
            city_1 = mutated_state.route[i]
            city_2 = mutated_state.route[j]
            mutated_state.route[i] = city_2
            mutated_state.route[j] = city_1

    mutated_state.update_distance(matrix, home)
    return mutated_state

def simulated_annealing(matrix, home, initial_state, mutation_rate=0.01, schedule=exp_schedule()):
    best_state = initial_state

    for t in range(sys.maxsize):
        T = schedule(t)

        if T == 0:
            return best_state

        neighbor = mutate(matrix, home, best_state, mutation_rate)
        delta_e = best_state.distance - neighbor.distance

        if delta_e > 0 or probability(np.exp(delta_e / T)):
            best_state = neighbor

def main():
    cities = ["Aksaray", "Ankara", "Çankırı", "Eskişehir", "Kayseri", "Kırıkkale", "Kırşehir", "Konya", "Nevşehir", "Niğde", "Sivas", "Yozgat"]
    city_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    home = 11

    matrix = [[0, 238, 312, 396, 155, 209, 96, 151, 73, 116, 353, 212],
              [238, 0, 132, 236, 346, 79, 222, 271, 286, 326, 440, 216],
              [313, 133, 0, 376, 347, 107, 215, 356, 307, 391, 399, 171],
              [396, 236, 376, 0, 512, 319, 388, 330, 452, 492, 704, 456],
              [155, 346, 347, 512, 0, 243, 135, 303, 81, 131, 197, 176],
              [209, 79, 107, 319, 243, 0, 110, 252, 203, 287, 364, 140],
              [96, 222, 215, 388, 135, 110, 0, 289, 95, 174, 327, 115],
              [151, 271, 356, 330, 303, 252, 289, 0, 222, 239, 502, 368],
              [73, 286, 307, 452, 81, 203, 95, 222, 0, 86, 279, 157],
              [116, 326, 391, 492, 131, 287, 174, 239, 86, 0, 329, 280],
              [353, 440, 399, 705, 197, 364, 327, 502, 279, 329, 0, 224],
              [212, 216, 171, 456, 176, 140, 115, 368, 157, 280, 224, 0]]

    num_iterations = int(input("Enter the number of iterations: "))
    start_time = time.time()
    results_df = pd.DataFrame(columns=["Iteration", "Route", "Total Distance"])

    best_state = None
    best_distance = sys.maxsize

    for iteration in range(num_iterations):
        state = get_best_solution_by_distance(matrix, home)
        state = simulated_annealing(matrix, home, state, 0.1)

        results_df = results_df._append({
            "Iteration": iteration + 1,
            "Route": f"{cities[home]} -> {' -> '.join(cities[i] for i in state.route)} -> {cities[home]}",
            "Total Distance": state.distance
        }, ignore_index=True)

        print(f'-- Simulated annealing solution - Iteration {iteration + 1} --')
        print(cities[home], end='')
        for i in range(0, len(state.route)):
            print(' -> ' + cities[state.route[i]], end='')
        print(' -> ' + cities[home], end='')
        print('\nTotal distance: {0} miles'.format(state.distance))
        print()

        if state.distance < best_distance:
            best_state = state.copy()
            best_distance = state.distance

    best_row_index = results_df[results_df["Total Distance"] == best_distance].index[0]
    results_df.at[best_row_index, "Best"] = "Yes"

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Total execution time: {execution_time} seconds")

    results_df.to_excel("D:\\Computer Engineering\\GÜZ\\Intro Heuristic Algorithims\\Project\\VehicleRoutingOptimization\\Results\\simulatedAnnealing_Yozgat.xlsx", index=False)

if __name__ == "__main__":
    main()

