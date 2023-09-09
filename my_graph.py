import networkx as nx
import matplotlib.pyplot as plt
import math
import random 
import numpy as np
import pandas as pd
imp
class Graph:
    def __init__(self, V, E, W=None):
        """
        Initializes a graph object.

        Args:
            V (list): List of vertices.
            E (list): List of edges.
            W (dict, optional): Dictionary of edge weights. Defaults to None.
        """
        self.E = [(u, v) for u, v in E]
        self.E.extend([(v, u) for u, v in E])
        self.V = list(set(V))
        self.W = W
        self.adj_list = {}
        self.weights = {}
        self.adjecency_matrx= self.adj_matrix()
        self.incidence_matrix = self.incid_matrix()
        for v in V:
            self.add_vertex(v)
        if W is None:
            for u, v in self.E:
                self.adj_list[v].add(u)
                self.adj_list[u].add(v)
                self.weights[(u, v)] = 0  # Set default weight to 0 if not provided
                self.weights[(v, u)] = 0
        else:
            for u, v in self.E:
                self.adj_list[v].add(u)
                self.adj_list[u].add(v)
                self.weights[(u, v)] = W.get((u, v), W.get((v, u)))  # Set default weight to 0 if not provided
                self.weights[(v, u)] = W.get((v, u), W.get((u, v)))  # Set default weight to 0 if not provided

    def degree(self, v):
        """
        Returns the degree of a given vertex.

        Args:
            v: The vertex.

        Returns:
            int: The degree of the vertex.
        """
        return len(self.adj_list[v])

    def neighbours(self, v):
        """
        Returns the list of neighboring vertices of a given vertex.

        Args:
            v: The vertex.

        Returns:
            list: List of neighboring vertices.
        """
        return self.adj_list[v]

    def add_vertex(self, v):
        """
        Adds a vertex to the graph.

        Args:
            v: The vertex to be added.
        """
        if v not in self.adj_list:
            self.adj_list[v] = set()
        if v not in self.V:
            self.V.append(v)

    def add_edge(self, u, v, weight=0):
        """
        Adds an edge between two vertices in the graph.

        Args:
            u: The first vertex.
            v: The second vertex.
            weight (optional): The weight of the edge. Defaults to 0.
        """
        self.add_vertex(u)
        self.add_vertex(v)
        self.adj_list[u].add(v)
        self.adj_list[v].add(u)
        self.E.extend([(u, v), (v, u)])
        self.weights[(u, v)] = weight
        self.weights[(v, u)] = weight
        self.adjecency_matrx = self.adj_matrix()
        self.incid_matrix = self.incid_matrix()

    @property
    def nv(self):
        """
        Returns the number of vertices in the graph.

        Returns:
            int: Number of vertices.
        """
        return len(self.adj_list)

    @property
    def nm(self):
        """
        Returns the number of edges in the graph.

        Returns:
            int: Number of edges.
        """
        return len(self.E)

    def remove_edge(self, u, v):
        """
        Removes an edge between two vertices in the graph.

        Args:
            u: The first vertex.
            v: The second vertex.
        """
        self.E.remove((u, v))
        self.E.remove((v, u))
        self.adj_list[u].remove(v)
        self.adj_list[v].remove(u)
        del self.weights[(u, v)]
        del self.weights[(v, u)]

    def remove_vertex(self, v):
        """
        Removes a vertex from the graph.

        Args:
            v: The vertex to be removed.
        """
        deleting = list(self.adj_list[v])
        for u in deleting:
            self.remove_edge(v, u)
        del self.adj_list[v]
        self.V.remove(v)

    def show_graph_with_edges(self):
        """
        Displays the graph with edges using a spring layout.
        """
        G = nx.Graph()
        G.add_nodes_from(self.V)
        G.add_edges_from(self.E)

        isolated_nodes = [node for node in G.nodes() if G.degree(node) == 0]

        fig, ax = plt.subplots(figsize=(5, 5))
        pos = nx.spring_layout(G)
        nx.draw_networkx(G, pos=pos, ax=ax, with_labels=True, node_color="red")
        nx.draw_networkx_nodes(G, pos=pos, ax=ax, nodelist=isolated_nodes, node_color="red")
        plt.show()

    def show_graph_with_adj(self):
        """
        Displays the graph with adjacency list representation.
        """
        G = nx.Graph()
        G.add_nodes_from(self.adj_list.keys())
        for node, neighbors in self.adj_list.items():
            G.add_edges_from((node, neighbor) for neighbor in neighbors)
        fig, ax = plt.subplots(figsize=(5, 5))
        nx.draw_networkx(G, with_labels=True, node_color="royalblue")
        plt.show()

    def bfs(self, start=None):
        """
        Performs breadth-first search (BFS) starting from a given vertex.

        Args:
            start: The vertex to start the BFS from. If not provided, it starts from a random vertex.

        Returns:
            list: List of connected components.
        """
        visited = []
        components = []

        if start is not None:
            queue = [start]
            component = []
            while queue:
                current = queue.pop(0)
                if current not in visited:
                    visited.append(current)
                    component.append(current)
                    for n in self.neighbours(current):
                        if n not in visited:
                            queue.append(n)
            components.append(component)
        else:
            for vertex in self.V:
                if vertex not in visited:
                    queue = [vertex]
                    component = []

                    while queue:
                        current = queue.pop(0)
                        if current not in visited:
                            visited.append(current)
                            component.append(current)

                            for n in self.neighbours(current):
                                if n not in visited:
                                    queue.append(n)

                    components.append(component)

        for i in components:
            print(" --> ".join([str(j) for j in i]), end="|")
        return components

    def dfs_recursive(self, start=None, visited=None):
        """
        Performs depth-first search (DFS) recursively starting from a given vertex.

        Args:
            start: The vertex to start the DFS from. If not provided, it starts from a random vertex.
            visited: List of visited vertices. Used for recursion. If not provided, an empty list is used.

        Returns:
            list: List of visited vertices in the order they were visited.
        """
        if visited is None:
            visited = []
        if start is None:
            start = self.V[random.randint(0, len(self.V) - 1)]
        visited.append(start)
        for n in self.neighbours(start):
            if n not in visited:
                self.dfs_recursive(n, visited)
        return visited

    def dfs_stack(self, start=None):
        """
        Performs depth-first search (DFS) using a stack starting from a given vertex.

        Args:
            start: The vertex to start the DFS from. If not provided, it starts from a random vertex.

        Returns:
            list: List of visited vertices in the order they were visited.
        """
        visited = []
        if start is None:
            start = self.V[random.randint(0, len(self.V) - 1)]
        stack = [start]
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.append(current)
                for n in self.neighbours(current):
                    if n not in visited:
                        stack.append(n)
        print(" --> ".join([str(i) for i in visited]), end="|")
        return visited

    def shortest_path(self, start, end):
        """
        Finds the shortest path between two vertices using Dijkstra's algorithm.

        Args:
            start: The starting vertex.
            end: The ending vertex.

        Returns:
            tuple: Tuple containing the path as a list and the total cost as a string.
                   Returns None if there is no path between the vertices.
        """
        distances = {v: math.inf for v in self.V}
        distances[start] = 0
        visited = set()
        vertices = [(0, start)]

        while vertices:
            vertices = sorted(vertices, key=lambda x: x[0])
            cur_dist, cur_v = vertices.pop(0)
            if cur_v == end:  # Terminate when reaching the end vertex
                break
            if cur_v in visited:
                continue
            visited.add(cur_v)
            for n in self.neighbours(cur_v):
                weighted_sum = cur_dist + self.weights.get((cur_v, n))
                if weighted_sum < distances[n] or distances[n] is None:
                    distances[n] = weighted_sum
                    vertices.append((weighted_sum, n))

        if distances[end] == math.inf:  # Check if end vertex is unreachable
            return None

        path = [end]
        while path[-1] != start:
            for n in self.neighbours(path[-1]):
                if distances[path[-1]] == distances[n] + self.weights.get((path[-1], n)):
                    path.append(n)
                    break
        path.reverse()

        return path, f"cost: {distances[end]}"
        
    def is_connected(self):
        """ a function that tells if a graph is connected or not 
        returns True or False """
        return len(self.bfs()[0]) == len(self.V)
    
    def mst(self):
        """ a method that returns minimumm spanning tree of a Graph,
        if the Graph is not connected it reurns mst for every connected component"""

        num_vertices = self.nv
        visited = set()
        key = {v: float('inf') for v in self.V}
        parent = {v: None for v in self.V}
        components = []

        while len(visited) != num_vertices:
            # Find an unvisited vertex to start a new component
            start_vertex = next((v for v in self.V if v not in visited), None)
            if start_vertex is None:
                break

            key[start_vertex] = 0
            component = []

            while True:
                # Find the vertex with the minimum key value among unvisited vertices
                min_key = float('inf')
                min_vertex = None

                for v in self.V:
                    if v not in visited and key[v] < min_key:
                        min_key = key[v]
                        min_vertex = v

                if min_vertex is None:
                    break

                visited.add(min_vertex)
                component.append(min_vertex)

                # Update the key values and parent for adjacent vertices
                for neighbor in self.neighbours(min_vertex):
                    if neighbor not in visited and self.weights[(min_vertex, neighbor)] < key[neighbor]:
                        key[neighbor] = self.weights[(min_vertex, neighbor)]
                        parent[neighbor] = min_vertex

            if component:
                components.append(component)

        minimum_spanning_tree = []
        for component in components:
            for v in component:
                if parent[v] is not None:
                    minimum_spanning_tree.append((parent[v], v, key[v]))

        return minimum_spanning_tree, sum([i[2] for i in minimum_spanning_tree])
    

    def assign_numbers(self):
        """in order for some methods to work, we need the vertecies to be 
        numbers, there fore this function is created """

        assigned_numbers = []
        assigned_tuples = []
        string_to_number = {}
        
        # Assign numbers to strings and create string to number mapping
        for i, string in enumerate(self.V):
            assigned_numbers.append(i)
            string_to_number[string] = i
        
        # Convert tuples of strings to tuples of numbers
        for tuple_strings in self.E:
            tuple_numbers = tuple(string_to_number[string] for string in tuple_strings)
            assigned_tuples.append(tuple_numbers)
        g = Graph(assigned_numbers, assigned_tuples)
        return g, string_to_number

    def is_bridge(self, u, v):
        """return True or False if a vertex is a brige"""
        self.remove_edge(u, v)  # Temporarily remove the edge
        is_bridge = not self.is_connected()
        self.add_edge(u, v)  # Add the edge back to the graph
        return is_bridge

    def copy(self):
        """for creating aa copy of the graph 
        returns the copy"""
        g = Graph(self.V, self.E, self.W)
        return g 

    def check_eulerian(self):
        """check for eulerian tour or cycle 
        if found, returns it 
        else it prints not found and returns None"""
        def dfs(u, graph, visited_edge, path = []):
            path.append(u)
            for v in graph.adj_list[u]:
                if visited_edge[u][v] == False:
                    visited_edge[u][v], visited_edge[v][u] = True, True
                    path = dfs(v, graph, visited_edge, path)
            return path
        def dfs(u, graph, visited_edge, path = []):
            path.append(u)
            for v in graph.adj_list[u]:
                if visited_edge[u][v] == False:
                    visited_edge[u][v], visited_edge[v][u] = True, True
                    path = dfs(v, graph, visited_edge, path)
            return path
    # for checking if the graph has an Euler path or circuit
        def check_circuit_or_path(graph, max_node):
            odd_degree_nodes = []
            for i in range(max_node):
                if i not in graph.adj_list.keys():
                    continue
                if len(graph.adj_list[graph.V[i]]) % 2 == 1:
                    odd_degree_nodes.append(i)
            if len(odd_degree_nodes) == 0:
                return 1, None
            if len(odd_degree_nodes) == 2:
                return 2, odd_degree_nodes[0]
            return 3, None

        
        max_node = len(self.V)
        assigned_numbers = None
        max_node = len(self.V)
        if isinstance(self.V[0], str):
            self, assigned_numbers = self.assign_numbers()

        visited_edge = [[False for _ in range(max_node + 1)] for _ in range(max_node + 1)]
        check, odd_node = check_circuit_or_path(self, max_node)
        if check == 3:
            print("Graph is not Eulerian")
            print("No path")
            return
        degree = 0
        start_node = None

        if check == 2:
            start_node = odd_node
            print("Graph has an Euler path")
        if check == 1:
            print("Graph has an Euler cycle")
            for v in self.V:
                if self.degree(v) > degree:
                    start_node = v

        path = dfs(start_node, self, visited_edge, list())

        if assigned_numbers:
            reversed_dict = {value: key for key, value in assigned_numbers.items()}
            path = [reversed_dict[node] for node in path]

        print(" --> ".join([str(i) for i in path]))
        return path

    def incid_matrix(self):
        """creating incid_matrix of the graph"""
        num_vertices = len(self.V)
        num_edges = len(self.E) // 2
        matrix = np.zeros((num_edges, num_vertices), dtype=int)
        if self.V:
            if isinstance(self.V[0], str):
                graph, d = self.assign_numbers()
                vertexes,edges = graph.V, graph.E
            else:
                edges = self.E
                vertexes = self.V
                
            for k, vertex in enumerate(vertexes):
                for i, edge in enumerate(edges):
                    if vertex in edge:
                        if i < num_edges and k < num_vertices:
                            matrix[i, k] = 1

            matrix_df = pd.DataFrame(matrix, columns=self.V, index=self.E[:num_edges])
            matrix_df = matrix_df.astype(int)

            return matrix_df
        else:
            return    
    def adj_matrix(self):
        """creating adj matrix"""
        matrix = np.zeros((len(self.V), len(self.V)))
        for t,i in enumerate(self.V):
            for k,j in enumerate(self.V):
                if (i,j) in self.E:
                    matrix[t,k] = 1
        matrix = pd.DataFrame(matrix, columns= self.V, index= (self.V))
        matrix =matrix.astype(int)
        return matrix
    
    def check_hammiltonian(self):
        """just like the check uelerian method"""
        for v in self.V:
            n = len(self.V)
            path = [None] * (n + 1)
            visited = set()

            def is_valid(vertex, pos):
                if pos == 0:
                    return True

                if self.adj_list[path[pos - 1]].__contains__(vertex) is False:
                    return False

                if vertex in visited:
                    return False

                return True

            def hamiltonian_cycle_or_path_util(pos, is_cycle):
                if pos == n:
                    if is_cycle:
                        if self.adj_list[path[pos - 1]].__contains__(path[0]):
                            return True
                    elif len(self.adj_list[path[pos - 1]]) != 0:
                        return True

                    return False

                for vertex in self.adj_list[path[pos - 1]]:
                    if is_valid(vertex, pos):
                        path[pos] = vertex
                        visited.add(vertex)

                        if hamiltonian_cycle_or_path_util(pos + 1, is_cycle):
                            return True

                        path[pos] = None
                        visited.remove(vertex)

                return False

            path[0] = v
            visited.add(v)
            if hamiltonian_cycle_or_path_util(1, True):
                path[-1] = path[0]
                print("the graph has Hamiltonian Cycle, therefore hammiltonian")
                print(" --> ".join([str(i) for i in path]))
                return path

            if hamiltonian_cycle_or_path_util(1, False):
                path.remove(None)
                print("the graph has Hamiltonian Path, therefore hammiltonian")
                print(" --> ".join([str(i) for i in path]))
                return path

        print("No Hamiltonian Cycle or Path exists for any vertex in the given graph.")
        return None, None

    def chromatic_number(self):
        """returns the chromatric number of a graph"""
        def is_valid(vertex, color, coloring):
            for neighbor in graph.neighbours(vertex):
                if neighbor in coloring and coloring[neighbor] == color:
                    return False
            return True

        def backtrack(vertex, coloring):
            if vertex == graph.nv:
                return True

            for color in range(v[1], graph.nv + 1):
                if is_valid(vertex, color, coloring):
                    coloring[vertex] = color
                    if backtrack(vertex + 1, coloring):
                        return True
                    coloring[vertex] = 0

            return False

        coloring = {}
        if isinstance(self.V[0], str):
            graph,d = self.assign_numbers()
            v = graph.V
        else:
            graph = self
            v = graph.V
        if backtrack(v[0], coloring):  # Start with vertex 1
            chromatic_number = max(coloring.values())
            return chromatic_number
        else:
            return None

    def is_bipartie(self):
        if self.chromatic_number() == 2 :
            return True
        else:
            return False
        

    def create_line_graph(self):
        """creating line graph 
        and the dictionary that shows what edge is which vertex"""
        new_e,  key = [],{}
        for nv,e in enumerate(self.E[:len(self.E)//2]):
            for y in self.E[:len(self.E)//2]:
                if set(e).intersection(y) and e != y:
                    new_e.append((e,y))
                    new_e.append((y,e))
            key[e] = nv
        for i in range(len(new_e)):
            m = list(list(new_e[i][0]))
            n = list(list(new_e[i][1]))
            m= key[tuple(m)]
            n= key[tuple(n)]
            new_e[i] = (m,n)
        new_v = key.values()
        LG = Graph(new_v,new_e)
        return LG, key


    def get_complementary_graph(self):
        complementary_graph = Graph([], [])
        for v in self.V:
            for u in self.V:
                if ((u,v) not in self.E or (v,u) not in self.E) and v!= u:
                    complementary_graph.add_edge(v,u)
        return complementary_graph
    
    def chromatic_number_edge(self):
        LG , d= self.create_line_graph()
        return LG.chromatic_number()
    
if __name__ == "__main__":
    drive_list = [
    Graph(set(["paris", "new", "tehran", "london"]), E=[("paris", "tehran"), ("new", "london"), ("london", "paris")], W={("paris", "tehran"): 5, ("new", "london"): 3, ("london", "paris"): 2}),# Example graph
    Graph( ['A', 'B', 'C', 'D', 'E'], [('A', 'B'), ('A', 'C'), ('B', 'C'), ('B', 'D'), ('C', 'D'), ('C', 'E'), ('D', 'E')], {('A', 'B'): 0.2, ('A', 'C'): 2, ('B', 'C'): 3, ('B', 'D'): 1, ('C', 'D'): 5, ('C', 'E'): 6, ('D', 'E'): 3}),
    Graph(set(['A', 'B', 'C', 'D', 'E']), E = [('A', 'B'),('B', 'C'),('C', 'A'),('D', 'E')]),
    Graph(["A", "B", "C", "D", "E", "F"], [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "F"), ("A", "C"), ("B", "D"), ("C", "E"), ("D", "F"), ("E", "A"), ("F", "B")]),
    Graph(V= (range(6)), E = ([(3,0),(3,4),(0,1),(0,2), (1,2),(0,4), (2,4),(2,5),(4,5)])),
    Graph(V = [0, 1, 2, 3, 4, 5, 6],E = [(0, 1),(0, 2),(1, 2),(2, 3),(2, 4),(3, 4),(3, 5),(4, 5),(5, 0) ,(3,6),(4,6),(5,6),(0,6)]),
    Graph(set(range(0, 5)), [(0,1), (0,3), (0,4),(0,2), (1,2), (3,4)]),
    Graph([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [(0, 1), (0, 4), (0, 5),(1, 2), (1, 6),(2, 3), (2, 7),(3, 4), (3, 8),(4, 9),(5, 7), (5, 8),(6, 8), (6, 9),(7, 9)]),
    Graph(V=list(range(7)), E= [(0,1), (1,2), (2,3),(3,4),(4,5),(5,6), (6,0)])]
        
    for g in drive_list :
        g.show_graph_with_adj()
    