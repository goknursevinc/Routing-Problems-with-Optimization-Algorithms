import sys
import random
import numpy as np
import pandas as pd
import time
class TabuSearch:
    def __init__(self, matrix, home, tabu_size=10, max_iterations=100):
        self.matrix = matrix
        self.num_cities = len(matrix)
        self.home = home
        self.tabu_size = tabu_size
        self.max_iterations = max_iterations
    
    def calculate_distance(self, route):
        distance = 0
        for i in range(len(route) - 1):
            distance += self.matrix[route[i]][route[i + 1]]
        distance += self.matrix[route[-1]][self.home]
        return distance

    def generate_initial_solution(self):
        solution = list(range(self.num_cities))
        random.shuffle(solution)
        return solution

    def apply_tabu(self, tabu_list):
        if len(tabu_list) > self.tabu_size:
            tabu_list.pop(0)

    def tabu_search(self):
        current_solution = self.generate_initial_solution()
        best_solution = current_solution.copy()
        tabu_list = []
        best_distance = self.calculate_distance(best_solution)

        for iteration in range(self.max_iterations):
            neighbors = self.get_neighbors(current_solution)
            feasible_neighbors = [neighbor for neighbor in neighbors if neighbor not in tabu_list]

            if not feasible_neighbors:
                break  # Stuck in a local minimum

            current_solution = min(feasible_neighbors, key=lambda x: self.calculate_distance(x))
            current_distance = self.calculate_distance(current_solution)

            if current_distance < best_distance:
                best_solution = current_solution.copy()
                best_distance = current_distance

            tabu_list.append(current_solution)
            self.apply_tabu(tabu_list)

        return best_solution, best_distance

    def get_neighbors(self, solution):
        neighbors = []
        for i in range(self.num_cities - 1):
            for j in range(i + 1, self.num_cities):
                neighbor = solution.copy()
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                neighbors.append(neighbor)
        return neighbors


def main():
    cities = ["Aksaray", "Ankara", "Çankırı", "Eskişehir", "Kayseri", "Kırıkkale", "Kırşehir", "Konya", "Nevşehir", "Niğde", "Sivas", "Yozgat"]
    city_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    home = 6

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
    start_time=time.time()
    # Create an empty DataFrame to store the results
    results_df = pd.DataFrame(columns=["Iteration", "Route", "Total Distance"])

    for iteration in range(num_iterations):
        tabu_search_instance = TabuSearch(matrix, home)
        best_solution, best_distance = tabu_search_instance.tabu_search()

        # En iyi çözümü DataFrame'e ekle
        route = [cities[city_index] for city_index in [home] + best_solution]
        results_df = results_df._append({
            "Iteration": iteration + 1,
            "Route": ' -> '.join(route),
            "Total Distance": best_distance
        }, ignore_index=True)

        print(f'-- Tabu Search solution - Iteration {iteration + 1} --')
        print(' -> '.join(route))
        print(f'Total distance: {best_distance} miles\n')
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Total execution time: {execution_time} seconds")
    
    # Excel dosyasını oluştur ve yaz
    best_row_index = results_df[results_df["Total Distance"] == results_df["Total Distance"].min()].index[0]
    results_df.at[best_row_index, "Best"] = "Yes"
    results_df.to_excel("tabuSearchKırşehir.xlsx", index=False)

if __name__ == "__main__":
    main()