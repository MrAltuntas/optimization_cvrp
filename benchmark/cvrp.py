import math
import numpy as np

class CVRP:
    def __init__(self, instance_file):
        """
        Initialize CVRP instance from a file.

        Args:
            instance_file: Path to the instance file in TSPLIB format
        """
        self.cities = []         # List of (x, y) coordinates
        self.demands = []        # Demand for each city
        self.capacity = 0        # Vehicle capacity
        self.depot = 0           # Depot index (usually 0)
        self.distance_matrix = None  # Matrix of distances between cities

        self.load_instance(instance_file)
        self.compute_distance_matrix()

    def load_instance(self, instance_file):
        """Load CVRP instance from a TSPLIB format file."""
        with open(instance_file, 'r') as f:
            lines = f.readlines()

        # Parse the header information
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('DIMENSION'):
                self.num_cities = int(line.split(':')[1].strip())
            elif line.startswith('CAPACITY'):
                self.capacity = int(line.split(':')[1].strip())
            elif line.startswith('NODE_COORD_SECTION'):
                i += 1
                # Read city coordinates
                for j in range(self.num_cities):
                    coords = lines[i + j].strip().split()
                    # Coordinates are 1-indexed in the file
                    city_id = int(coords[0])
                    x = float(coords[1])
                    y = float(coords[2])
                    self.cities.append((x, y))
                i += self.num_cities - 1  # Adjust counter to skip the coordinates
            elif line.startswith('DEMAND_SECTION'):
                i += 1
                # Read demands
                for j in range(self.num_cities):
                    demand_info = lines[i + j].strip().split()
                    city_id = int(demand_info[0])
                    demand = int(demand_info[1])
                    self.demands.append(demand)
                i += self.num_cities - 1  # Adjust counter to skip the demands
            elif line.startswith('DEPOT_SECTION'):
                i += 1
                self.depot = int(lines[i].strip()) - 1  # Convert to 0-indexed
            i += 1

    def compute_distance_matrix(self):
        """Compute the distance matrix between all city pairs."""
        n = len(self.cities)
        self.distance_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    # Euclidean distance
                    dx = self.cities[i][0] - self.cities[j][0]
                    dy = self.cities[i][1] - self.cities[j][1]
                    self.distance_matrix[i, j] = round(math.sqrt(dx*dx + dy*dy))

    def get_total_distance(self, routes):
        """
        Calculate the total distance for a solution.

        Args:
            routes: List of routes, where each route is a list of city indices

        Returns:
            Total distance
        """
        total_distance = 0

        for route in routes:
            if not route:
                continue

            # Add distance from depot to first city
            total_distance += self.distance_matrix[self.depot, route[0]]

            # Add distances between consecutive cities
            for i in range(len(route) - 1):
                total_distance += self.distance_matrix[route[i], route[i + 1]]

            # Add distance from last city to depot
            total_distance += self.distance_matrix[route[-1], self.depot]

        return total_distance

    def is_valid_solution(self, routes):
        """
        Check if a solution is valid.

        A solution is valid if:
        1. All cities are visited exactly once
        2. Each route respects the capacity constraint

        Args:
            routes: List of routes, where each route is a list of city indices

        Returns:
            (valid, message) tuple, where valid is a boolean and message explains why the solution is invalid
        """
        # Check if all cities are visited
        visited = set()
        for route in routes:
            for city in route:
                if city in visited:
                    return False, f"City {city} is visited more than once"
                visited.add(city)

        all_cities = set(range(self.num_cities))
        all_cities.remove(self.depot)  # Remove depot
        if visited != all_cities:
            missing = all_cities - visited
            return False, f"Cities {missing} are not visited"

        # Check capacity constraints
        for i, route in enumerate(routes):
            route_demand = sum(self.demands[city] for city in route)
            if route_demand > self.capacity:
                return False, f"Route {i} exceeds capacity: {route_demand} > {self.capacity}"

        return True, "Valid solution"

    def __str__(self):
        """Return a string representation of the problem."""
        return f"CVRP instance with {self.num_cities} cities, depot at {self.depot}, capacity {self.capacity}"

    def format_solution(self, routes, total_distance=None):
        """
        Format a solution as required by CVRPLIB.

        Args:
            routes: List of routes, where each route is a list of city indices
            total_distance: Total distance (if None, it will be calculated)

        Returns:
            Formatted solution string
        """
        if total_distance is None:
            total_distance = self.get_total_distance(routes)

        solution = ""
        for i, route in enumerate(routes):
            # Convert to 1-indexed for output
            route_str = " ".join(str(city + 1) for city in route)
            solution += f"Route #{i+1}: {self.depot + 1} {route_str} {self.depot + 1}\n"

        solution += f"Cost {int(total_distance)}\n"
        return solution