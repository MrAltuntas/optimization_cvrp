import numpy as np

class GreedyCVRP:
    def __init__(self, cvrp):
        """
        Initialize the Greedy CVRP solver.

        Args:
            cvrp: CVRP problem instance
        """
        self.problem = cvrp

    def solve(self):
        """
        Solve the CVRP using a greedy approach.

        The greedy algorithm works as follows:
        1. Start at the depot
        2. Go to the nearest unvisited city that doesn't exceed the vehicle capacity
        3. If no such city exists, return to the depot and start a new route
        4. Repeat until all cities are visited

        Returns:
            routes: List of routes, each route is a list of city indices
        """
        # Get problem data
        n = self.problem.num_cities
        depot = self.problem.depot
        capacity = self.problem.capacity
        demands = self.problem.demands
        dist_matrix = self.problem.distance_matrix

        # Initialize
        unvisited = set(range(n))
        unvisited.remove(depot)  # Remove depot from cities to visit
        routes = []
        current_route = []
        current_capacity = capacity
        current_city = depot

        # Main loop
        while unvisited:
            # Find the nearest unvisited city that doesn't exceed the capacity
            best_distance = float('inf')
            best_city = None

            for city in unvisited:
                # Check if adding this city would exceed the capacity
                if demands[city] <= current_capacity:
                    distance = dist_matrix[current_city, city]
                    if distance < best_distance:
                        best_distance = distance
                        best_city = city

            # If a city is found, add it to the current route
            if best_city is not None:
                current_route.append(best_city)
                unvisited.remove(best_city)
                current_capacity -= demands[best_city]
                current_city = best_city
            # Otherwise, return to the depot and start a new route
            else:
                if current_route:  # Only add non-empty routes
                    routes.append(current_route)
                current_route = []
                current_capacity = capacity
                current_city = depot

        # Add the last route if not empty
        if current_route:
            routes.append(current_route)

        return routes