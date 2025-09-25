import torch
from src import utils
import numpy as np
from time import time 
import opt_einsum as oe

class DummyExtraFeatures:
    def __init__(self):
        """ This class does not compute anything, just returns empty tensors."""

    def __call__(self, noisy_data):
        X = noisy_data['X_t']
        e = noisy_data['E_t']
        Y = noisy_data['y_t']
        nb_layers = e.shape[-1]
        empty_X = X.new_empty((*X.shape[:-1], 0))
        empty_x = X.new_empty((*X.shape[:-1], nb_layers, 0))
        empty_e = e.new_empty((*e.shape, 0))
        empty_Y = Y.new_empty((Y.shape[0], 0))
        empty_y = Y.new_empty((Y.shape[0], nb_layers, 0))

        return utils.PlaceHolderMultilayer(X=empty_X,
                                 x=empty_x, 
                                 e=empty_e, 
                                 Y=empty_Y,
                                 y=empty_y)


class ExtraFeatures:
    def __init__(self, extra_features_type, dataset_info):
        self.max_n_nodes = dataset_info.max_n_nodes
        self.ncycles = NodeCycleFeatures()
        self.features_type = extra_features_type
        if extra_features_type in ['eigenvalues', 'all']:
            self.eigenfeatures = EigenFeatures(mode=extra_features_type)

    def __call__(self, noisy_data):
        bs, n, _, nb_layers = noisy_data['E_t'].shape
        layer_graphs = noisy_data['E_t'].permute(0, 3, 1, 2)  # (bs, n, n, layer) -> (bs, layer, n, n)
        layer_graphs = layer_graphs.reshape(-1, layer_graphs.size(2), layer_graphs.size(3))  # (bs*layer, n, n)
        node_mask_repeat = noisy_data['node_mask'].unsqueeze(1).expand(-1, nb_layers, -1)  # (bs, n) -> (bs, nb_layer, n)
        node_mask_repeat = node_mask_repeat.reshape(-1, n)  # (bs*layer, n)
        new_noisy_data = {'E_t': layer_graphs, 'node_mask': node_mask_repeat}
        
        n = new_noisy_data['node_mask'].sum(dim=1).unsqueeze(1) / self.max_n_nodes
        x_cycles, y_cycles = self.ncycles(new_noisy_data)       # (bs, n_cycles)

        if self.features_type == 'cycles':
            E = new_noisy_data['E_t']
            extra_edge_attr = torch.zeros((*E.shape[:-1], 0)).type_as(E)
            extra_features = utils.PlaceHolder(X=x_cycles, E=extra_edge_attr, y=torch.hstack((n, y_cycles)))

        elif self.features_type == 'eigenvalues':
            eigenfeatures = self.eigenfeatures(new_noisy_data)
            E = new_noisy_data['E_t']
            extra_edge_attr = torch.zeros((*E.shape[:-1], 0)).type_as(E)
            n_components, batched_eigenvalues = eigenfeatures   # (bs, 1), (bs, 10)
            extra_features = utils.PlaceHolder(X=x_cycles, E=extra_edge_attr, y=torch.hstack((n, y_cycles, n_components,
                                                                                    batched_eigenvalues)))
        elif self.features_type == 'all':
            eigenfeatures = self.eigenfeatures(new_noisy_data)
            E = new_noisy_data['E_t']
            extra_edge_attr = torch.zeros((*E.shape[:-1], 0)).type_as(E)
            n_components, batched_eigenvalues, nonlcc_indicator, k_lowest_eigvec = eigenfeatures   # (bs, 1), (bs, 10),
                                                                                                # (bs, n, 1), (bs, n, 2)

            extra_features = utils.PlaceHolder(X=torch.cat((x_cycles, nonlcc_indicator, k_lowest_eigvec), dim=-1),
                                     E=extra_edge_attr,
                                     y=torch.hstack((n, y_cycles, n_components, batched_eigenvalues)))
        else:
            raise ValueError(f"Features type {self.features_type} not implemented")

        bs, n, _, nb_layers = noisy_data['E_t'].shape
        dX = extra_features.X.shape[-1]
        dE = extra_features.E.shape[-1]
        dy = extra_features.y.shape[-1]
        extra_features_x = extra_features.X.reshape((bs, nb_layers, n, dX)).permute(0, 2, 1, 3)  # (bs*layer, n, d) -> (bs, n, layer, d)
        extra_features_X = extra_features_x.reshape(bs, n, -1)  # (bs, n, layer, d) -> (bs, n, d*layer)
        extra_features_e = extra_features.E.reshape((bs, nb_layers, n, n, dE)).permute(0, 2, 3, 1, 4)  # (bs*layer, n, n, d) -> (bs, n, n, layer, d)
        extra_features_y = extra_features.y.reshape((bs, nb_layers, dy))  # (bs*layer, d) -> (bs, layer, d))
        extra_features_Y = extra_features_y.reshape(bs, -1)  # (bs, d, layer) -> (bs, d*layer)

        return utils.PlaceHolderMultilayer(X=extra_features_X,
                                            x=extra_features_x,
                                            e=extra_features_e,
                                            Y=extra_features_Y,
                                            y=extra_features_y)

class NodeCycleFeatures:
    def __init__(self):
        self.kcycles = KNodeCycles()

    def __call__(self, noisy_data):
        adj_matrix = noisy_data['E_t'].float()  # (bs*l, n, n)

        x_cycles, y_cycles = self.kcycles.k_cycles(adj_matrix=adj_matrix)   # (bs, n_cycles)

        x_cycles = x_cycles.type_as(adj_matrix) * noisy_data['node_mask'].unsqueeze(-1)
        # Avoid large values when the graph is dense
        x_cycles = x_cycles / 10
        y_cycles = y_cycles / 10
        x_cycles[x_cycles > 1] = 1
        y_cycles[y_cycles > 1] = 1
        return x_cycles, y_cycles


class EigenFeatures:
    """
    Code taken from : https://github.com/Saro00/DGN/blob/master/models/pytorch/eigen_agg.py
    """
    def __init__(self, mode):
        """ mode: 'eigenvalues' or 'all' """
        self.mode = mode

    def __call__(self, noisy_data):
        E_t = noisy_data['E_t']
        mask = noisy_data['node_mask']
        A = E_t.float() * mask.unsqueeze(1) * mask.unsqueeze(2)
        L = compute_laplacian(A, normalize=False)
        mask_diag = 2 * L.shape[-1] * torch.eye(A.shape[-1]).type_as(L).unsqueeze(0)
        mask_diag = mask_diag * (~mask.unsqueeze(1)) * (~mask.unsqueeze(2))
        L = L * mask.unsqueeze(1) * mask.unsqueeze(2) + mask_diag

        if self.mode == 'eigenvalues':
            eigvals = torch.linalg.eigvalsh(L)        # bs, n
            eigvals = eigvals.type_as(A) / torch.sum(mask, dim=1, keepdim=True)

            n_connected_comp, batch_eigenvalues = get_eigenvalues_features(eigenvalues=eigvals)
            return n_connected_comp.type_as(A), batch_eigenvalues.type_as(A)

        elif self.mode == 'all':
            eigvals, eigvectors = torch.linalg.eigh(L)
            eigvals = eigvals.type_as(A) / torch.sum(mask, dim=1, keepdim=True)
            eigvectors = eigvectors * mask.unsqueeze(2) * mask.unsqueeze(1)
            # Retrieve eigenvalues features
            n_connected_comp, batch_eigenvalues = get_eigenvalues_features(eigenvalues=eigvals)

            # Retrieve eigenvectors features
            nonlcc_indicator, k_lowest_eigenvector = get_eigenvectors_features(vectors=eigvectors,
                                                                               node_mask=noisy_data['node_mask'],
                                                                               n_connected=n_connected_comp)
            return n_connected_comp, batch_eigenvalues, nonlcc_indicator, k_lowest_eigenvector
        else:
            raise NotImplementedError(f"Mode {self.mode} is not implemented")


def compute_laplacian(adjacency, normalize: bool):
    """
    adjacency : batched adjacency matrix (bs, n, n)
    normalize: can be None, 'sym' or 'rw' for the combinatorial, symmetric normalized or random walk Laplacians
    Return:
        L (n x n ndarray): combinatorial or symmetric normalized Laplacian.
    """
    diag = torch.sum(adjacency, dim=-1)     # (bs, n)
    n = diag.shape[-1]
    D = torch.diag_embed(diag)      # Degree matrix      # (bs, n, n)
    combinatorial = D - adjacency                        # (bs, n, n)

    if not normalize:
        return (combinatorial + combinatorial.transpose(1, 2)) / 2

    diag0 = diag.clone()
    diag[diag == 0] = 1e-12

    diag_norm = 1 / torch.sqrt(diag)            # (bs, n)
    D_norm = torch.diag_embed(diag_norm)        # (bs, n, n)
    L = torch.eye(n).unsqueeze(0) - D_norm @ adjacency @ D_norm
    L[diag0 == 0] = 0
    return (L + L.transpose(1, 2)) / 2


def get_eigenvalues_features(eigenvalues, k=5):
    """
    values : eigenvalues -- (bs, n)
    node_mask: (bs, n)
    k: num of non zero eigenvalues to keep
    """
    ev = eigenvalues
    bs, n = ev.shape
    n_connected_components = (ev < 1e-5).sum(dim=-1)
    assert (n_connected_components > 0).all(), (n_connected_components, ev)

    to_extend = max(n_connected_components) + k - n
    if to_extend > 0:
        eigenvalues = torch.hstack((eigenvalues, 2 * torch.ones(bs, to_extend).type_as(eigenvalues)))
    indices = torch.arange(k).type_as(eigenvalues).long().unsqueeze(0) + n_connected_components.unsqueeze(1)
    first_k_ev = torch.gather(eigenvalues, dim=1, index=indices)
    return n_connected_components.unsqueeze(-1), first_k_ev


def get_eigenvectors_features(vectors, node_mask, n_connected, k=2):
    """
    vectors (bs, n, n) : eigenvectors of Laplacian IN COLUMNS
    returns:
        not_lcc_indicator : indicator vectors of largest connected component (lcc) for each graph  -- (bs, n, 1)
        k_lowest_eigvec : k first eigenvectors for the largest connected component   -- (bs, n, k)
    """
    bs, n = vectors.size(0), vectors.size(1)

    # Create an indicator for the nodes outside the largest connected components
    first_ev = torch.round(vectors[:, :, 0], decimals=3) * node_mask                        # bs, n
    # Add random value to the mask to prevent 0 from becoming the mode
    random = torch.randn(bs, n, device=node_mask.device) * (~node_mask)                                   # bs, n
    first_ev = first_ev + random
    most_common = torch.mode(first_ev, dim=1).values                                    # values: bs -- indices: bs
    mask = ~ (first_ev == most_common.unsqueeze(1))
    not_lcc_indicator = (mask * node_mask).unsqueeze(-1).float()

    # Get the eigenvectors corresponding to the first nonzero eigenvalues
    to_extend = max(n_connected) + k - n
    if to_extend > 0:
        vectors = torch.cat((vectors, torch.zeros(bs, n, to_extend).type_as(vectors)), dim=2)   # bs, n , n + to_extend
    indices = torch.arange(k).type_as(vectors).long().unsqueeze(0).unsqueeze(0) + n_connected.unsqueeze(2)    # bs, 1, k
    indices = indices.expand(-1, n, -1)                                               # bs, n, k
    first_k_ev = torch.gather(vectors, dim=2, index=indices)       # bs, n, k
    first_k_ev = first_k_ev * node_mask.unsqueeze(2)

    return not_lcc_indicator, first_k_ev

def batch_trace(X):
    """
    Expect a matrix of shape B N N, returns the trace in shape B
    :param X:
    :return:
    """
    diag = torch.diagonal(X, dim1=-2, dim2=-1)
    trace = diag.sum(dim=-1)
    return trace


def batch_diagonal(X):
    """
    Extracts the diagonal from the last two dims of a tensor
    :param X:
    :return:
    """
    return torch.diagonal(X, dim1=-2, dim2=-1)


class KNodeCycles:
    """ Builds cycle counts for each node in a graph.
    """

    def __init__(self):
        super().__init__()

    def calculate_kpowers(self):
        self.k1_matrix = self.adj_matrix.float()
        self.d = self.adj_matrix.sum(dim=-1)
        self.k2_matrix = self.k1_matrix @ self.adj_matrix.float()
        self.k3_matrix = self.k2_matrix @ self.adj_matrix.float()
        self.k4_matrix = self.k3_matrix @ self.adj_matrix.float()
        self.k5_matrix = self.k4_matrix @ self.adj_matrix.float()
        self.k6_matrix = self.k5_matrix @ self.adj_matrix.float()

    def k3_cycle(self):
        """ tr(A ** 3). """
        c3 = batch_diagonal(self.k3_matrix)
        return (c3 / 2).unsqueeze(-1).float(), (torch.sum(c3, dim=-1) / 6).unsqueeze(-1).float()

    def k4_cycle(self):
        diag_a4 = batch_diagonal(self.k4_matrix)
        c4 = diag_a4 - self.d * (self.d - 1) - (self.adj_matrix @ self.d.unsqueeze(-1)).sum(dim=-1)
        return (c4 / 2).unsqueeze(-1).float(), (torch.sum(c4, dim=-1) / 8).unsqueeze(-1).float()

    def k5_cycle(self):
        diag_a5 = batch_diagonal(self.k5_matrix)
        triangles = batch_diagonal(self.k3_matrix)
        c5 = diag_a5 - 2 * triangles * self.d - (self.adj_matrix @ triangles.unsqueeze(-1)).sum(dim=-1) + triangles
        return (c5 / 2).unsqueeze(-1).float(), (c5.sum(dim=-1) / 10).unsqueeze(-1).float()

    def k6_cycle(self):
        term_1_t = batch_trace(self.k6_matrix)
        term_2_t = batch_trace(self.k3_matrix ** 2)
        term3_t = torch.sum(self.adj_matrix * self.k2_matrix.pow(2), dim=[-2, -1])
        d_t4 = batch_diagonal(self.k2_matrix)
        a_4_t = batch_diagonal(self.k4_matrix)
        term_4_t = (d_t4 * a_4_t).sum(dim=-1)
        term_5_t = batch_trace(self.k4_matrix)
        term_6_t = batch_trace(self.k3_matrix)
        term_7_t = batch_diagonal(self.k2_matrix).pow(3).sum(-1)
        term8_t = torch.sum(self.k3_matrix, dim=[-2, -1])
        term9_t = batch_diagonal(self.k2_matrix).pow(2).sum(-1)
        term10_t = batch_trace(self.k2_matrix)

        c6_t = (term_1_t - 3 * term_2_t + 9 * term3_t - 6 * term_4_t + 6 * term_5_t - 4 * term_6_t + 4 * term_7_t +
                3 * term8_t - 12 * term9_t + 4 * term10_t)
        return None, (c6_t / 12).unsqueeze(-1).float()

    def k_cycles(self, adj_matrix, verbose=False):
        self.adj_matrix = adj_matrix
        self.calculate_kpowers()

        k3x, k3y = self.k3_cycle()
        assert (k3x >= -0.1).all()

        k4x, k4y = self.k4_cycle()
        assert (k4x >= -0.1).all()

        k5x, k5y = self.k5_cycle()
        assert (k5x >= -0.1).all(), k5x

        _, k6y = self.k6_cycle()
        # assert (k6y >= -0.1).all()
        k6y = torch.clamp(k6y, min=0.0)

        kcyclesx = torch.cat([k3x, k4x, k5x], dim=-1)
        kcyclesy = torch.cat([k3y, k4y, k5y, k6y], dim=-1)
        return kcyclesx, kcyclesy
    

class CliqueComputation:
    """
    Extract the 3-cliques and 4-cliques from the graphs
    """
    def __init__(self, dataset_info=None, clique_sizes=[3, 4], algorithm='sequential', clip_values={'edges': 100, 'nodes': 1000, 'layers': 10000}):
        self.clique_sizes = clique_sizes
        self.max_n_nodes = dataset_info.max_n_nodes if dataset_info else None
        self.algorithm = algorithm
        self.clip_values = clip_values

    def __call__(self, noisy_data):
        X = noisy_data['X_t']
        e = noisy_data['E_t']
        Y = noisy_data['y_t']

        x_clique_features, e_clique_features, y_clique_features = self.get_cliques(noisy_data)
        # empty_edge_attr = e.new_zeros((*e.shape[:-1], 0))
        return utils.PlaceHolderMultilayer(X=X, 
                                           x=x_clique_features, 
                                           e=e_clique_features, 
                                           Y=Y,
                                           y=y_clique_features)

    def get_cliques(self, noisy_data):
        x_clique_features = []
        e_clique_features = []
        y_clique_features = []
        # begin = time()
        if self.algorithm == 'parallel':
            for clique_size in self.clique_sizes:
                if clique_size == 3:
                    tri_pred_x, tri_pred_e, tri_pred_y = self.get_3_cliques(noisy_data)
                    x_clique_features.append(tri_pred_x)
                    e_clique_features.append(tri_pred_e)
                    y_clique_features.append(tri_pred_y)
                elif clique_size == 4:
                    quad_pred_x, quad_pred_e, quad_pred_y = self.get_4_cliques(noisy_data)
                    x_clique_features.append(quad_pred_x)
                    e_clique_features.append(quad_pred_e)
                    y_clique_features.append(quad_pred_y)
                else:
                    raise NotImplementedError(f"Unsupported clique size: {clique_size}")
            if len(self.clique_sizes) == 0:
                bs, n, _, l = noisy_data['E_t'].shape
                x_clique_features.append(torch.zeros((bs, n, l), dtype=torch.int64).to(noisy_data['E_t'].device))
                e_clique_features.append(torch.zeros((bs, n, n, l), dtype=torch.int64).to(noisy_data['E_t'].device))
                y_clique_features.append(torch.zeros((bs, l), dtype=torch.int64).to(noisy_data['E_t'].device))
        elif self.algorithm == 'sequential4':
            tri_pred_x, tri_pred_e, tri_pred_y, quad_pred_e, quad_pred_x, quad_pred_y = self.get_cliques_sequential4(noisy_data)
            x_clique_features.append(tri_pred_x)
            e_clique_features.append(tri_pred_e)
            y_clique_features.append(tri_pred_y)
            x_clique_features.append(quad_pred_x)
            e_clique_features.append(quad_pred_e)
            y_clique_features.append(quad_pred_y)
        elif self.algorithm == 'sequential3':
            tri_pred_x, tri_pred_e, tri_pred_y = self.get_cliques_sequential3(noisy_data)
            x_clique_features.append(tri_pred_x)
            e_clique_features.append(tri_pred_e)
            y_clique_features.append(tri_pred_y)
        else:
            raise NotImplementedError(f"Unsupported algorithm for clique sampling: {self.algorithm}")
        
        # print(f"Time to compute cliques: {time() - begin:.2f} seconds")
        
        x_clique_features = torch.stack(x_clique_features, dim=-1).to(noisy_data['E_t'].device)
        e_clique_features = torch.stack(e_clique_features, dim=-1).to(noisy_data['E_t'].device)
        y_clique_features = torch.stack(y_clique_features, dim=-1).to(noisy_data['E_t'].device)
        
        return x_clique_features, e_clique_features, y_clique_features

    def get_cliques_sequential3(self, noisy_data):
        E = noisy_data['E_t'].float()
        bs, n, _, l = E.shape
        tri_pred_e = np.zeros((bs, n, n, l), dtype=np.int64)
        for b in range(bs):
            print("Computing cliques for graph n°", b)
            for l in range(l):
                for i in range(n):
                    for j in range(i + 1, n):
                        if E[b,i,j,l]:
                            for k in range(j + 1, n):
                                if E[b, j, k, l] and E[b, k, i, 0]:
                                    tri_pred_e[b, i, j, l] += 1
                                    tri_pred_e[b, j, k, l] += 1
                                    tri_pred_e[b, k, i, l] += 1
                                    tri_pred_e[b, j, i, l] += 1
                                    tri_pred_e[b, k, j, l] += 1
                                    tri_pred_e[b, i, k, l] += 1
                                            
        tri_pred_e = torch.tensor(tri_pred_e, dtype=torch.int64)  
        tri_pred_x = torch.einsum('bijd->bid', tri_pred_e).long() / 2.0
        tri_pred_y = torch.einsum('bid->bd', tri_pred_x) / 3.0

        return tri_pred_x, tri_pred_e, tri_pred_y
    
    def get_cliques_sequential4(self, noisy_data):
        E = noisy_data['E_t'].float()
        bs, n, _, l = E.shape
        tri_pred_e = np.zeros((bs, n, n, l), dtype=np.int64)
        quad_pred_e = np.zeros((bs, n, n, l), dtype=np.int64)
        for b in range(bs):
            print("Computing cliques for graph n°", b)
            for l in range(l):
                for i in range(n):
                    for j in range(i + 1, n):
                        if E[b,i,j,l]:
                            for k in range(j + 1, n):
                                if E[b, j, k, l] and E[b, k, i, 0]:
                                    tri_pred_e[b, i, j, l] += 1
                                    tri_pred_e[b, j, k, l] += 1
                                    tri_pred_e[b, k, i, l] += 1
                                    tri_pred_e[b, j, i, l] += 1
                                    tri_pred_e[b, k, j, l] += 1
                                    tri_pred_e[b, i, k, l] += 1
                                    for m in range(k+1, n):
                                        if E[b, i, m, l] and E[b, j, m, l] and E[b, k, m, l]:
                                            quad_pred_e[b, i, j, l] += 1
                                            quad_pred_e[b, i, k, l] += 1
                                            quad_pred_e[b, i, m, l] += 1
                                            quad_pred_e[b, j, i, l] += 1
                                            quad_pred_e[b, k, i, l] += 1
                                            quad_pred_e[b, m, i, l] += 1

                                            quad_pred_e[b, j, k, l] += 1
                                            quad_pred_e[b, j, m, l] += 1
                                            quad_pred_e[b, k, j, l] += 1
                                            quad_pred_e[b, m, j, l] += 1

                                            quad_pred_e[b, k, m, l] += 1
                                            quad_pred_e[b, m, k, l] += 1
                                            
        tri_pred_e = torch.tensor(tri_pred_e, dtype=torch.int64)
        quad_pred_e = torch.tensor(quad_pred_e, dtype=torch.int64)    
        tri_pred_x = torch.einsum('bijd->bid', tri_pred_e).long() / 2.0
        tri_pred_y = torch.einsum('bid->bd', tri_pred_x) / 3.0
        quad_pred_x = torch.einsum('bijd->bid', quad_pred_e).long() / 3.0
        quad_pred_y = torch.einsum('bid->bd', quad_pred_x) / 4.0

        return tri_pred_x, tri_pred_e, tri_pred_y, quad_pred_e, quad_pred_x, quad_pred_y
    

    def get_3_cliques(self, noisy_data):
        E = noisy_data['E_t'].float()
        tri_pred_e = oe.contract('bijd,bjkd,bikd->bijd', E, E, E, optimize='auto', memory_limit='max_input').long()
        tri_pred_x = torch.einsum('bijd->bid', tri_pred_e).long() / 2.0
        tri_pred_y = torch.einsum('bid->bd', tri_pred_x) / 3.0

        tri_pred_e[tri_pred_e > self.clip_values['edges']] = -1
        tri_pred_x[tri_pred_x > self.clip_values['nodes']] = -1
        tri_pred_y[tri_pred_y > self.clip_values['layers']] = -1
        return tri_pred_x, tri_pred_e, tri_pred_y


    def get_4_cliques(self, noisy_data):
        E = noisy_data['E_t'].float()
        bs, n, _, l = E.shape
        quad_pred_e = torch.zeros((bs, n, n, l), dtype=torch.float16).to(E.device)
        E_casted = E.type(torch.float16)
        for d in range(bs):
            for layer in range(l):
                E_reduced = E_casted[d, :, :, layer]
                quad_pred_e[d, :, :, layer] = oe.contract('ij,ik,il,jk,jl,kl->ij', E_reduced, E_reduced, E_reduced, E_reduced, E_reduced, E_reduced, optimize='auto', memory_limit='max_input').long() / 2.0
        
        quad_pred_e = quad_pred_e.type(torch.float64)
        quad_pred_x = torch.einsum('bijd->bid', quad_pred_e).long() / 3.0
        quad_pred_y = torch.einsum('bid->bd', quad_pred_x) / 4.0

        quad_pred_e[quad_pred_e > self.clip_values['edges']] = -1
        quad_pred_x[quad_pred_x > self.clip_values['nodes']] = -1
        quad_pred_y[quad_pred_y > self.clip_values['layers']] = -1
        return quad_pred_x.type(E.dtype), quad_pred_e.type(E.dtype), quad_pred_y.type(E.dtype)