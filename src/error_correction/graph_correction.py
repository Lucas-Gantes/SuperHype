import networkx as nx
import numpy as np


class GraphCorrection:
    def __init__(self, hyperedge_node_count, adjacency_matrix, corrections_to_do, optimization_algorithm, simple_method=True):
        self.hyperedge_node_count = hyperedge_node_count
        self.corrections_to_do = corrections_to_do
        
        self.simple_method = simple_method

        self.critical_edges_context = {}

        self.G = nx.from_numpy_array(adjacency_matrix)

        self.max_cliques = nx.algorithms.find_cliques(self.G)

        self.initial_hyperedge_distribution = self.get_initial_hyperedge_distribution()

        self.sample_critical_edges()


        self.initial_distance, self.distance_contribution = self.compute_distance()

        self.critical_edges = list(self.critical_edges_context.keys())
        self.num_critical_edges = len(self.critical_edges)
        self.critical_edges_impact = np.zeros((self.num_critical_edges, self.hyperedge_node_count.shape[0], 
                                               self.hyperedge_node_count.shape[0]), dtype=int)  # The index corresponding to a maximal clique of size n is n-2
        

        self.compute_marginal_gain()
        self.global_gain = self.critical_edge_gain()


    def get_initial_hyperedge_distribution(self):
        """
        Get the initial number of maximal cliques connected to every node of the graph
        """
        initial_he_distr = np.zeros_like(self.hyperedge_node_count, dtype=int)
        for clique in self.max_cliques:
            clique_len = len(clique)
            for node in clique:
                initial_he_distr[node, clique_len-2] += 1
        
        return initial_he_distr
    

    def sample_critical_edges(self):
        """
        Sample critical edges corresponding to the corrections to do
        """
        for n in self.corrections_to_do["cliques simple completion"]:
            self.simple_completion(n)

        for n in self.corrections_to_do["clique separation"]:
            self.clique_separation(n)


    def simple_completion(self, n):
        """
        Find critical edges likely to lead to the fusion of two n-cliques
        """
        self.critical_edges_context["simple completion"][n] = {}
        n_cliques = [clique for clique in self.max_cliques if len(clique)==n]
        for i in range(len(n_cliques)):
            for j in range(i+1, len(n_cliques)):
                clique1 = sorted(n_cliques)
                clique2 = sorted(n_cliques)
                common_nodes = list(set(clique1) & set(clique2))
                
                if len(common_nodes)==n-1:
                    # Find the non-common nodes
                    assert len(list(set(clique1) ^ set(clique2))) == 2
                    critical_edge = tuple(sorted(list(set(clique1) ^ set(clique2))))

                    # If it has not been done, create an entry for the simple fusion of n-cliques in the self.critical_edges_context dictionnary
                    if "simple completion" not in self.critical_edges_context[critical_edge]:
                        self.critical_edges_context[critical_edge]["simple completion"] = {}
                    if n not in self.critical_edges_context[critical_edge]["simple completion"]:
                        self.critical_edges_context[critical_edge]["simple completion"][n] = []

                    # Add clique1 and clique2 to the context of the critical edge
                    self.critical_edges_context[critical_edge]["simple completion"][n].append([clique1, clique2])
    

    def clique_separation(self, n):
        """
        Add every edges in n-cliques. Those edges are likely to separate a n-clique if they are removed.
        """
        n_cliques = [clique for clique in self.max_cliques if len(clique)==n]

        for clique in n_cliques:
            for i in range(n):
                for j in range(i+1, n):
                    critical_edge = tuple(sorted([clique[i], clique[j]]))

                    # If it has not been done, create an entry for the separation of n-cliques in the self.critical_edges_context dictionnary
                    if "clique separation" not in self.critical_edges_context[critical_edge]:
                        self.critical_edges_context[critical_edge]["clique separation"] = {}
                    if n not in self.critical_edges_context[critical_edge]["clique separation"]:
                        self.critical_edges_context[critical_edge]["clique separation"][n] = []
                    
                    # Add the clique to the context of the critical edge
                    self.critical_edges_context[critical_edge]["clique separation"][n].append(clique)


    def compute_marginal_gain(self):
        """
        Find the marginal gain of the modification of the critical edges
        """
        # First, get the impact of the modification of every critical edge on the hyperedge distribution
        for edge_id in range(self.num_critical_edges):
            if "simple completion" in self.critical_edges_context[self.critical_edges[edge_id]]:
                for n in self.critical_edges_context[self.critical_edges[edge_id]]["simple completion"].keys():
                    self.critical_edges_impact[edge_id, n-2] -= 2  # The two maximal n-cliques are merged, so they are no longer maximal
                    self.critical_edges_impact[edge_id, n-1] += 1  # The fusion of the two maximal n-cliques creates a maximal n+1-clique
                    
                    for completion_configuration in self.critical_edges_context[self.critical_edges[edge_id]]["simple completion"][n]:
                        assert len(completion_configuration) == 2
                        common_nodes = list(set(completion_configuration[0]) & set(completion_configuration[1]))
                        diff_nodes = list(set(completion_configuration[0]) ^ set(completion_configuration[1]))
                        assert len(diff_nodes) == 2
                        
                        for node in common_nodes:
                            self.critical_edges_impact[edge_id, node, n-2] -= 2  # The two maximal n-cliques are merged, so they are no longer maximal
                            self.critical_edges_impact[edge_id, node, n-1] += 1  # The fusion of the two maximal n-cliques creates a maximal n+1-clique
                        for node in diff_nodes:
                            self.critical_edges_impact[edge_id, node, n-2] -= 1  # The node is in only one of the former maximal n-cliques
                            self.critical_edges_impact[edge_id, node, n-1] += 1  # The fusion of the two maximal n-cliques creates a maximal n+1-clique
    
            if "clique separation" in self.critical_edges_context[self.critical_edges[edge_id]]:
                for n in self.critical_edges_context[self.critical_edges[edge_id]]["clique separation"].keys():
                    for separation_configuration in self.critical_edges_context[self.critical_edges[edge_id]]["simple completion"][n]:
                        critical_edge = list(self.critical_edges[edge_id])
                        assert set(critical_edge) <= set(separation_configuration)
                        
                        for node in list(set(separation_configuration) - set(critical_edge)):
                            self.critical_edges_impact[edge_id, node, n-2] -= 1  # The separation destroys a maximal clique of size n
                            self.critical_edges_impact[edge_id, node, n-3] += 2  # The separationn creates two maximal cliques of size n-1, and this node is in both of them
                        
                        for node in critical_edge:
                            self.critical_edges_impact[edge_id, node, n-2] -= 1  # The separation destroys a maximal clique of size n
                            self.critical_edges_impact[edge_id, node, n-3] += 1  # The separationn creates two maximal cliques of size n-1, but this node is only in one of them

    def critical_edge_gain(self):
        """
        Compute the gain of the modification of a critical edge
        """
        edge_gain = np.zeros(self.num_critical_edges)
        for edge_id in range(self.num_critical_edges):
            edge_gain[edge_id] = np.sum((self.critical_edges_impact[edge_id] + self.initial_hyperedge_distribution - self.hyperedge_node_count)**2)
        return edge_gain
    

    def update_maximal_cliques(self, critical_edge=None):
        """
        Update the maximal cliques of the graph after a modification
        If ciritical_edge is None, recompute the maximal cliques from scratch.
        If critical_edge is not None, update the maximal cliques based on the modification of this edge.
        """
        if critical_edge is None:
            self.max_cliques = nx.algorithms.find_cliques(self.G)
        
        else:
            raise NotImplementedError("Updating maximal cliques with a critical edge is not implemented yet.")
    

    def update_critical_edges(self, critical_edge=None):
        """
        Update the critical edges context after a modification of a critical edge
        """
        if critical_edge is None:
            self.critical_edges_context = {}
            self.sample_critical_edges()
            self.compute_marginal_gain()
        else:
            raise NotImplementedError("Updating critical edges context with a critical edge is not implemented yet.")
        

    def compute_distance(self):
        """
        Compute a distance between a hyperedge distribution and the predicted one
        """
        dist_contribution = np.sum((self.initial_hyperedge_distribution-self.hyperedge_node_count)**2, axis=1)
        return np.sum(dist_contribution), dist_contribution


    def compute_distance_diff(self, critical_edge):
        """
        Compute the distance variation corresponding to the modification of a given critical edge
        """
        pass

    def modify_graph(self, critical_edge):
        u,v = critical_edge
        if not self.G.has_edge(u, v):
            self.G.add_edge(u, v)
        else:
            self.G.remove_edge(u, v)

        if self.simple_method:
            # Simple method: maximal clique distribution and critical edges are recalculated from scratch
            self.max_cliques = nx.algorithms.find_cliques(self.G)
            self.critical_edges_context = {}
            self.initial_hyperedge_distribution = self.get_initial_hyperedge_distribution()
            self.initial_hyperedge_distribution = self.get_initial_hyperedge_distribution()

            self.sample_critical_edges()

            self.initial_distance, self.distance_contribution = self.compute_distance()

            self.critical_edges = list(self.critical_edges_context.keys())
            self.num_critical_edges = len(self.critical_edges)
            self.critical_edges_impact = np.zeros((self.num_critical_edges, self.hyperedge_node_count.shape[0], 
                                                self.hyperedge_node_count.shape[0]), dtype=int)  # The index corresponding to a maximal clique of size n is n-2
            
            self.compute_marginal_gain()

        else:
            raise NotADirectoryError("Optimized method not implemented yet.")


    def optimization_process(self):
        """
        Run the optimization process to find the best critical edge to modify
        """
        if self.optimization_algorithm == "greedy":
            self.greedy_optimization()
        else:
            raise NotImplementedError(self.optimization_algorithm)


    def greedy_optimization(self):
        stop = False
        modified_edges = []
        while not stop:
            # Compute the gain of every critical edge
            self.global_gain = self.critical_edge_gain()

            # Find the critical edge with the highest gain
            best_edge_id = np.argmax(self.global_gain)
            best_edge = self.critical_edges[best_edge_id]

            # If the gain is negative, stop the optimization process
            if self.global_gain[best_edge_id] < 0:
                stop = True
                continue

            # Modify the graph based on the best critical edge
            self.modify_graph(best_edge)

            


