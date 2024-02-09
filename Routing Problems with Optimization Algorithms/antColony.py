import sys
import random
import copy
import numpy as np
import pandas as pd
import time

class Ant:
    def __init__(self, home, num_cities):
        self.home = home
        self.num_cities = num_cities
        self.tabu_list = [home]
        self.distance_travelled = 0

    def choose_next_city(self, pheromone_matrix, distance_matrix, alpha=1.0, beta=2.0):
        current_city = self.tabu_list[-1]
        available_cities = [city for city in range(self.num_cities) if city not in self.tabu_list]

        if not available_cities:
            return current_city  # If all cities visited, return to the home city

        pheromone_values = [pheromone_matrix[current_city][next_city] for next_city in available_cities]
        distances = [distance_matrix[current_city][next_city] for next_city in available_cities]

        probabilities = [
            (pheromone * alpha) * ((1.0 / distance) * beta) for pheromone, distance in zip(pheromone_values, distances)
        ]

        probabilities /= sum(probabilities)
        chosen_city = np.random.choice(available_cities, p=probabilities)

        return chosen_city

    def move_to_city(self, next_city, distance_matrix):
        current_city = self.tabu_list[-1]
        self.distance_travelled += distance_matrix[current_city][next_city]
        self.tabu_list.append(next_city)

    def return_to_home(self, distance_matrix):
        home_distance = distance_matrix[self.tabu_list[-1]][self.home]
        self.distance_travelled += home_distance
        self.tabu_list.append(self.home)


def update_pheromones(pheromone_matrix, ants, evaporation_rate=0.1, Q=1.0):
    for ant in ants:
        for i in range(len(ant.tabu_list) - 1):
            current_city, next_city = ant.tabu_list[i], ant.tabu_list[i + 1]
            pheromone_matrix[current_city][next_city] += Q / ant.distance_travelled
            pheromone_matrix[next_city][current_city] += Q / ant.distance_travelled

    pheromone_matrix *= (1.0 - evaporation_rate)


def ant_colony_optimization(matrix, home, num_ants, num_iterations):
    num_cities = len(matrix)
    pheromone_matrix = np.ones((num_cities, num_cities))
    best_distance = sys.maxsize
    best_route = []

    for iteration in range(num_iterations):
        ants = [Ant(home, num_cities) for _ in range(num_ants)]

        for ant in ants:
            while len(ant.tabu_list) < num_cities:
                next_city = ant.choose_next_city(pheromone_matrix, matrix)
                ant.move_to_city(next_city, matrix)

            ant.return_to_home(matrix)

        update_pheromones(pheromone_matrix, ants)

        # Find the best ant and update the best route if necessary
        best_ant = min(ants, key=lambda x: x.distance_travelled)
        if best_ant.distance_travelled < best_distance:
            best_distance = best_ant.distance_travelled
            best_route = best_ant.tabu_list

    return best_route, best_distance


def main():
    cities = ["Aksaray", "Ankara", "Çankırı", "Eskişehir", "Kayseri", "Kırıkkale", "Kırşehir", "Konya", "Nevşehir", "Niğde", "Sivas", "Yozgat"]
    city_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    home = 0 

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

    num_ants = 10
    num_iterations = int(input("Enter the number of iterations: "))
    start_time = time.time()

    # Create an empty DataFrame to store the results
    results_df = pd.DataFrame(columns=["Iteration", "Route", "Total Distance"])

    best_distance = sys.maxsize
    best_route = []
    best_row_index = None

    for iteration in range(num_iterations):
        best_route_iter, best_distance_iter = ant_colony_optimization(matrix, home, num_ants, 1)

        results_df = results_df._append({
            "Iteration": iteration + 1,
            "Route": ' -> '.join(cities[city_index] for city_index in best_route_iter),
            "Total Distance": best_distance_iter
        }, ignore_index=True)

        if best_distance_iter < best_distance:
            best_distance = best_distance_iter
            best_route = best_route_iter
            best_row_index = iteration

        print(f'-- Ant Colony Optimization solution - Iteration {iteration + 1} --')
        print(' -> '.join(cities[city_index] for city_index in best_route_iter))
        print('Total distance: {0} miles'.format(best_distance_iter))
        print()

    results_df.at[best_row_index, "Best"] = "Yes"

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Total execution time: {execution_time} seconds")

    results_df.to_excel("AntColonyOptimizationResults.xlsx", index=False)

if __name__ == "__main__":
    main()