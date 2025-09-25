import math

import torch
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn import functional as F
from torch import Tensor

from src import utils
from src.diffusion import diffusion_utils
from src.models.layers import Ltoy, Xtoy, Etoy, masked_sigmoid, masked_softmax


class GlobalLocalAttention(nn.Module):
    """
    A layer that performs attention and FiLM between global and local features.
    """
    def __init__(self, dG, dL, n_head, **kw):
        super().__init__()
        assert dL % n_head == 0, f"dx: {dL} -- nhead: {n_head}"
        self.dl = dL
        self.dg = dG
        self.df = int(dL / n_head)
        self.n_head = n_head

        # Attention
        self.q = Linear(dL, dL, **kw)
        self.k = Linear(dL, dL, **kw)
        self.v = Linear(dL, dL, **kw)

        # FiLM G to L (former y and X)
        self.g_l_mul = Linear(dG, dL, **kw)
        self.g_l_add = Linear(dG, dL, **kw)

        # Process G
        self.g_g = Linear(dG, dG, **kw)
        self.l_g = Ltoy(dL, dG)

        # Output layers
        self.l_out = Linear(dL, dL, **kw)
        self.g_out = nn.Sequential(nn.Linear(dG, dG), nn.ReLU(), nn.Linear(dG, dG))

    def forward(self, G, L, node_mask=None):
        """
        :param G: ..., dG        global features
        :param L: ..., nb_locals, dL  local features
        :param node_mask: G.shape[:-1]   (optional: if None, no masking is applied)
        The mask is only between multi-layer node embeddings and multi-layer graph labels
        :return: newG and newL with the same shape.
        """
        if node_mask is not None:
            nb_nodes = node_mask.shape[-1]
            assert L.shape[-2] == nb_nodes, f"Expected L shape to have {nb_nodes} nodes, got {L.shape[-2]}"
            L_mask = node_mask.unsqueeze(-1)  # (..., nb_locals, 1)
        else:
            L_mask = torch.ones(L.shape[:-1], device=L.device, dtype=L.dtype).unsqueeze(-1)  # (..., nb_locals, 1)

        Ymask1 = L_mask.unsqueeze(-2)  # (..., nb_locals, 1, 1)
        Ymask2 = L_mask.unsqueeze(-3)  # (..., 1, nb_locals, 1)

        # 1. Map L to keys and queries
        Q = self.q(L) * L_mask  # (..., nb_locals, dL)
        K = self.k(L) * L_mask  # (..., nb_locals, dL)
        diffusion_utils.assert_correctly_masked(Q, L_mask)

        # 2. Reshape to (..., nb_locals, n_head, df) with dx = n_head * df

        Q = Q.reshape(Q.shape[:-1] + (self.n_head, self.df,))
        K = K.reshape(K.shape[:-1] + (self.n_head, self.df,))

        # Reshape Q and K to prepare for attention computation

        Q = Q.unsqueeze(-4)                              # (..., 1, nb_locals, n_head, df)
        K = K.unsqueeze(-3)                              # (..., nb_locals, 1, n head, df)

        # Compute unnormalized attentions. Y is (..., nb_locals, nb_locals, n_head, df)
        Y = Q * K
        Y = Y / math.sqrt(Y.size(-1))
        diffusion_utils.assert_correctly_masked(Y, (Ymask1 * Ymask2).unsqueeze(-1))

        # Compute attentions
        softmax_mask = L_mask.unsqueeze(-3).unsqueeze(-1)  # ..., 1, nb_locals, 1, 1
        softmax_mask = softmax_mask.expand_as(Y)
        attn = masked_softmax(Y, softmax_mask, dim=-3)  #..., nb_locals, nb_locals, n_head, df

        V = self.v(L) * L_mask  # ..., nb_locals, dL
        V = V.reshape(V.shape[:-1] + (self.n_head, self.df,))  # (..., nb_locals, n_head, df)
        V = V.unsqueeze(-4)  # (..., 1, nb_locals, n_head, df)

        # Compute weighted values
        weighted_V = attn * V  # (..., nb_locals, nb_locals, n_head, df)
        weighted_V = weighted_V.sum(dim=-3)  # (..., nb_locals, n_head, df)

        # Send output to input dim
        weighted_V = weighted_V.flatten(start_dim=-2)  # (..., nb_locals, dL)

        # Incorporate G to L
        gl1 = self.g_l_add(G).unsqueeze(-2)  # (..., 1, dL)
        gl2 = self.g_l_mul(G).unsqueeze(-2)  # (..., 1, dL)
        newL = gl1 + (gl2 + 1) * weighted_V

        # Output L
        newL = self.l_out(newL) * L_mask  # (..., nb_locals, dL)
        diffusion_utils.assert_correctly_masked(newL, L_mask)

        # Process G based on L
        g_repr = self.g_g(G)
        l_repr = self.l_g(L)
        newG = g_repr + l_repr
        newG = self.g_out(newG)  # ..., dG

        return newG, newL

class GlobalLocalCrossAttention(nn.Module):
    """
    A layer that performs self-attention and cross-attention between global and local features.
    """
    def __init__(self, dG, dL, n_head, hidden_dG, hidden_dL, dropout : float = 0.1, layer_norm_eps : float = 1e-5, **kw):
        super().__init__()
        assert dL % n_head == 0, f"dx: {dL} -- nhead: {n_head}"
        self.dl = dL
        self.dg = dG
        self.df = int(dL / n_head)
        self.n_head = n_head

        # Attention
        self.selfAttentionLocal = nn.MultiheadAttention(
            embed_dim=dL, 
            num_heads=n_head, 
            batch_first=True, 
            dropout=dropout
        )
        
        self.crossAttentionLocalToGlobal = nn.MultiheadAttention(
            embed_dim=dG,
            num_heads=n_head,
            batch_first=True,
            kdim=dL,
            vdim=dL,
            dropout=dropout
        )

        self.crossAttentionGlobalToLocal = nn.MultiheadAttention(
            embed_dim=dL,
            num_heads=n_head,
            batch_first=True,
            kdim=dG,
            vdim=dG,
            dropout=dropout
        )

        self.dropout_SelfAtt = Dropout(dropout)
        self.dropout_CrossAttGL = Dropout(dropout)
        self.dropout_CrossAttLG = Dropout(dropout)

        self.norm_SelfAtt = LayerNorm(dL, eps=layer_norm_eps, **kw)
        self.norm_CrossAttL = LayerNorm(dL, eps=layer_norm_eps, **kw)
        self.norm_CrossAttG = LayerNorm(dG, eps=layer_norm_eps, **kw)
        self.norm_L = LayerNorm(dL, eps=layer_norm_eps, **kw)
        self.norm_G = LayerNorm(dG, eps=layer_norm_eps, **kw)

        self.lin_L1 = Linear(2*dL, hidden_dL, **kw)
        self.dropout_L1 = Dropout(dropout)
        self.lin_L2 = Linear(hidden_dL, dL, **kw)
        self.dropout_L2 = Dropout(dropout)
        self.lin_G1 = Linear(dG, hidden_dG, **kw)
        self.dropout_G1 = Dropout(dropout)
        self.lin_G2 = Linear(hidden_dG, dG, **kw)
        self.dropout_G2 = Dropout(dropout)

        self.activation = F.relu

    def forward(self, G, L, L_mask, G_mask):
        """
        :param G: ..., dG        global features
        :param L: ..., nb_locals, dL  local features
        :param L_mask: bs, nb_locals -> a mask for attention between X and Y used to mask multi-layer node embeddings of padded nodes
        :param G_mask: bs -> a mask for attention batween X and x used to mask elements of the batch
        :return: newG and newL with the same shape.
        """
        if L_mask is not None:
            bs, nb_nodes = L_mask.size()
            S = 1
            nh = self.n_head

            key_mask_L = (L_mask == 0)

        else:
            attn_mask_L = None
            key_mask_L = None


        # Self-attention and cross attention between local and global features
        self_att_L, _ = self.selfAttentionLocal(L, L, L, key_padding_mask=key_mask_L)
        cross_att_GtoL, _ = self.crossAttentionGlobalToLocal(L, G, G)
        cross_att_LtoG, _ = self.crossAttentionLocalToGlobal(G, L, L, key_padding_mask=key_mask_L)
    
        if L_mask is not None:
            self_att_L = self_att_L.masked_fill(key_mask_L.unsqueeze(-1), 0.0)
            cross_att_GtoL = cross_att_GtoL.masked_fill((L_mask == 0).unsqueeze(-1), 0.0)


        # Dropout
        self_att_L = self.dropout_SelfAtt(self_att_L)
        cross_att_GtoL = self.dropout_CrossAttGL(cross_att_GtoL)
        cross_att_LtoG = self.dropout_CrossAttLG(cross_att_LtoG)

        # Apply layer norm n°1
        self_att_L = self.norm_SelfAtt(self_att_L + L)
        cross_att_GtoL = self.norm_CrossAttL(cross_att_GtoL + L)
        cross_att_LtoG = self.norm_CrossAttG(cross_att_LtoG + G)

        # Concatenation of self-attention and cross-attention for local features
        newL = torch.cat((self_att_L, cross_att_GtoL), dim=-1)

        # Apply MLP
        newL = self.dropout_L2(self.lin_L2(self.dropout_L1(self.activation(self.lin_L1(newL)))))
        newG = self.dropout_G2(self.lin_G2(self.dropout_G1(self.activation(self.lin_G1(cross_att_LtoG)))))

        # Apply layer norm n°2
        newL = self.norm_L(newL + self_att_L + cross_att_GtoL)
        newG = self.norm_G(newG + cross_att_LtoG)

        if L_mask is not None:
            newL = newL.masked_fill(key_mask_L.unsqueeze(-1), 0.0)
        
        if G_mask is not None:
            G_mask_reshaped = (G_mask.unsqueeze(-1).unsqueeze(-1) == 0)  # bs, 1, 1
            newL = newL.masked_fill(G_mask_reshaped, 0.0)
            newG = newG.masked_fill(G_mask_reshaped, 0.0)

        return newG, newL


class GlobalLocalTransformer(nn.Module):
    """
    A class to add skipped connexions, layer normalization and dropout after the attention
    between global and local features.
    """
    def __init__(self, dG: int, dL: int, n_head: int, dim_ffG: int,
                 dim_ffL: int, dropout: float = 0.1, layer_norm_eps: float = 1e-5, 
                 device=None, dtype=None) -> None:
        kw = {'device': device, 'dtype': dtype}
        super().__init__()

        self.self_attn = GlobalLocalAttention(dG, dL, n_head, **kw)

        self.linG1 = Linear(dG, dim_ffG, **kw)
        self.linG2 = Linear(dim_ffG, dG, **kw)
        self.normG1 = LayerNorm(dG, eps=layer_norm_eps, **kw)
        self.normG2 = LayerNorm(dG, eps=layer_norm_eps, **kw)
        self.dropoutG1 = Dropout(dropout)
        self.dropoutG2 = Dropout(dropout)
        self.dropoutG3 = Dropout(dropout)

        self.linL1 = Linear(dL, dim_ffL, **kw)
        self.linL2 = Linear(dim_ffL, dL, **kw)
        self.normL1 = LayerNorm(dL, eps=layer_norm_eps, **kw)
        self.normL2 = LayerNorm(dL, eps=layer_norm_eps, **kw)
        self.dropoutL1 = Dropout(dropout)
        self.dropoutL2 = Dropout(dropout)
        self.dropoutL3 = Dropout(dropout)

        self.activation = F.relu

    def forward(self, G, L, node_mask=None):
        """ 
        Pass the input through the encoder layer.
        :param G: ..., dG        global features
        :param L: ..., nb_locals, dL  local features
        :param node_mask: G.shape[:-1]   (optional: if None, no masking is applied)
        The mask is only between multi-layer node embeddings and multi-layer graph labels
        :return: newG and newL with the same shape.
        """

        newG, newL = self.self_attn(G, L, node_mask=node_mask)

        newG_d = self.dropoutG1(newG)
        G = self.normG1(G + newG_d)

        newL_d = self.dropoutL1(newL)
        L = self.normL1(L + newL_d)

        ff_outputG = self.linG2(self.dropoutG2(self.activation(self.linG1(G))))
        ff_outputG = self.dropoutG3(ff_outputG)
        G = self.normG2(G + ff_outputG)

        ff_outputL = self.linL2(self.dropoutL2(self.activation(self.linL1(L))))
        ff_outputL = self.dropoutL3(ff_outputL)
        L = self.normL2(L + ff_outputL)

        return G, L


class GraphSuperpositionTransformer(nn.Module):
    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict, hidden_dims: dict,
                 output_dims: dict, act_fn_in, act_fn_out, cross_attention, triplet_interactions, parallel=True, single_layer=False):
        super().__init__()
        self.n_layers = n_layers

        self.out_dim_x = output_dims['x']
        self.out_dim_y = output_dims['y']
        self.out_dim_e = output_dims['e']
        self.out_dim_X = output_dims['X']
        self.out_dim_Y = output_dims['Y']

        self.mlp_in_x = nn.Sequential(nn.Linear(input_dims['x'], hidden_mlp_dims['x']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['x'], hidden_dims['dx']), act_fn_in)
        
        self.mlp_in_y = nn.Sequential(nn.Linear(input_dims['y'], hidden_mlp_dims['y']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['y'], hidden_dims['dy']), act_fn_in)

        self.mlp_in_e = nn.Sequential(nn.Linear(input_dims['e'], hidden_mlp_dims['e']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['e'], hidden_dims['de']), act_fn_in)
    
        self.mlp_in_X = nn.Sequential(nn.Linear(input_dims['X'], hidden_mlp_dims['X']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['X'], hidden_dims['dX']), act_fn_in)
        
        self.mlp_in_Y = nn.Sequential(nn.Linear(input_dims['Y'], hidden_mlp_dims['Y']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['Y'], hidden_dims['dY']), act_fn_in)
        
        self.xeyTrLayers = [MultiXEyTransformerLayer(dx=hidden_dims['dx'],
                                                de=hidden_dims['de'], 
                                                dy=hidden_dims['dy'],
                                                n_head=hidden_dims['n_head_xey'],
                                                dim_ffX=hidden_dims['dim_ffx_xey'],
                                                dim_ffE=hidden_dims['dim_ffe_xey'],
                                                dim_ffy=hidden_dims['dim_ffy_xey'],
                                                triplet_interactions=triplet_interactions) for _ in range(n_layers+1)]
        
        self.xeyTrLayers = nn.ModuleList(self.xeyTrLayers)

        self.cross_attention = cross_attention

        self.single_layer = single_layer
        self.parallel = parallel
        if not single_layer:
            print("The network is in multi-layer mode")
            if cross_attention:
                self.XxTrLayers = [GlobalLocalCrossAttention(dG=hidden_dims['dX'],
                                                        dL=hidden_dims['dx'],
                                                        n_head=hidden_dims['n_head_Xx'],
                                                        hidden_dG=hidden_dims['dim_ffX_Xx'],
                                                        hidden_dL=hidden_dims['dim_ffx_Xx']) for _ in range(n_layers)]
                
                self.XxTrLayers = nn.ModuleList(self.XxTrLayers)

                self.YyTrLayers = [GlobalLocalCrossAttention(dG=hidden_dims['dY'],
                                                        dL=hidden_dims['dy'],
                                                        n_head=hidden_dims['n_head_Yy'],
                                                        hidden_dG=hidden_dims['dim_ffY_Yy'],
                                                        hidden_dL=hidden_dims['dim_ffy_Yy']) for _ in range(n_layers)]
                
                self.YyTrLayers = nn.ModuleList(self.YyTrLayers)

                self.YXTrLayers = [GlobalLocalCrossAttention(dG=hidden_dims['dY'],
                                                        dL=hidden_dims['dX'],
                                                        n_head=hidden_dims['n_head_YX'],
                                                        hidden_dG=hidden_dims['dim_ffY_YX'],
                                                        hidden_dL=hidden_dims['dim_ffX_YX']) for _ in range(n_layers)]
                
                self.YXTrLayers = nn.ModuleList(self.YXTrLayers)
            else:
                self.XxTrLayers = [GlobalLocalTransformer(dG=hidden_dims['dX'],
                                                        dL=hidden_dims['dx'],
                                                        n_head=hidden_dims['n_head_Xx'],
                                                        dim_ffG=hidden_dims['dim_ffX_Xx'],
                                                        dim_ffL=hidden_dims['dim_ffx_Xx']) for _ in range(n_layers)]
                
                self.XxTrLayers = nn.ModuleList(self.XxTrLayers)

                self.YyTrLayers = [GlobalLocalTransformer(dG=hidden_dims['dY'],
                                                        dL=hidden_dims['dy'],
                                                        n_head=hidden_dims['n_head_Yy'],
                                                        dim_ffG=hidden_dims['dim_ffY_Yy'],
                                                        dim_ffL=hidden_dims['dim_ffy_Yy']) for _ in range(n_layers)]
                
                self.YyTrLayers = nn.ModuleList(self.YyTrLayers)

                self.YXTrLayers = [GlobalLocalTransformer(dG=hidden_dims['dY'],
                                                        dL=hidden_dims['dX'],
                                                        n_head=hidden_dims['n_head_YX'],
                                                        dim_ffG=hidden_dims['dim_ffY_YX'],
                                                        dim_ffL=hidden_dims['dim_ffX_YX']) for _ in range(n_layers)]
                
                self.YXTrLayers = nn.ModuleList(self.YXTrLayers)
            
            if parallel:
                print("The network is in parallel mode")
                act_fn_inter = nn.ReLU()
                
                layer_norm_eps = 1e-5
                self.concat_X = [nn.Sequential(nn.Linear(2*hidden_dims['dX'], hidden_mlp_dims['X']), act_fn_inter,
                                        nn.Linear(hidden_mlp_dims['X'], hidden_dims['dX'])) for _ in range(n_layers)]
                self.concat_X = nn.ModuleList(self.concat_X)
                self.norm_X = [LayerNorm(hidden_dims['dX'], eps=layer_norm_eps) for _ in range(n_layers)]
                self.norm_X = nn.ModuleList(self.norm_X)

                self.concat_x = [nn.Sequential(nn.Linear(2*hidden_dims['dx'], hidden_mlp_dims['x']), act_fn_inter,
                                        nn.Linear(hidden_mlp_dims['x'], hidden_dims['dx'])) for _ in range(n_layers)]
                self.concat_x = nn.ModuleList(self.concat_x)
                self.norm_x = [LayerNorm(hidden_dims['dx'], eps=layer_norm_eps) for _ in range(n_layers)]
                self.norm_x = nn.ModuleList(self.norm_x)

                self.concat_Y = [nn.Sequential(nn.Linear(2*hidden_dims['dY'], hidden_mlp_dims['Y']), act_fn_inter,
                                        nn.Linear(hidden_mlp_dims['Y'], hidden_dims['dY'])) for _ in range(n_layers)]
                self.concat_Y = nn.ModuleList(self.concat_Y)
                self.norm_Y = [LayerNorm(hidden_dims['dY'], eps=layer_norm_eps) for _ in range(n_layers)]
                self.norm_Y = nn.ModuleList(self.norm_Y)

                self.concat_y = [nn.Sequential(nn.Linear(2*hidden_dims['dy'], hidden_mlp_dims['y']), act_fn_inter,
                                        nn.Linear(hidden_mlp_dims['y'], hidden_dims['dy'])) for _ in range(n_layers)]
                self.concat_y = nn.ModuleList(self.concat_y)
                self.norm_y = [LayerNorm(hidden_dims['dy'], eps=layer_norm_eps) for _ in range(n_layers)]
                self.norm_y = nn.ModuleList(self.norm_y)
            
            else:
                print("The network is in sequential mode")
        else:
            print("The network is in single-layer mode")

            

        self.mlp_out_X = nn.Sequential(nn.Linear(hidden_dims['dX'], hidden_mlp_dims['X']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['X'], output_dims['X']))
        
        self.mlp_out_Y = nn.Sequential(nn.Linear(hidden_dims['dY'], hidden_mlp_dims['Y']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['Y'], output_dims['Y']))
        
        self.mlp_out_e = nn.Sequential(nn.Linear(hidden_dims['de'], hidden_mlp_dims['e']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['e'], output_dims['e']))
        
        self.mlp_out_x = nn.Sequential(nn.Linear(hidden_dims['dx'], hidden_mlp_dims['x']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['x'], output_dims['x']))
        
        self.mlp_out_y = nn.Sequential(nn.Linear(hidden_dims['dy'], hidden_mlp_dims['y']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['y'], output_dims['y']))



    def forward(self, X, Y, e, x, y, node_mask, kernel=None):
        """
        X: multi-layer node embeddings (bs, n, dX)
        Y: multi-layer graph embeddings (bs, dY)
        e: edge embeddings (bs, n, n, l, de)
        x: layer-specific node embeddings (bs, n, l, dx)
        y: layer-specific graph embedding (bs, l, dy)
        node_mask: (bs, n)
        kernel: (bs, n, n, l) (if None, no kernel is applied for single-layer features update)
        """
        # For debug only
        input_dims = {'X': X.shape, 'x': x.shape, 'Y': Y.shape, 'y': y.shape, 'e': e.shape}

        # Checking if the batch size is consistent
        assert X.shape[0] == Y.shape[0]
        assert Y.shape[0] == e.shape[0]
        assert e.shape[0] == x.shape[0]
        assert x.shape[0] == y.shape[0]
        assert y.shape[0] == node_mask.shape[0]
        if kernel is not None:
            assert node_mask.shape[0] == kernel.shape[0]

        # Checking if the number of nodes is consistent
        assert X.shape[1] == x.shape[1]
        assert x.shape[1] == e.shape[1]
        assert e.shape[1] == e.shape[2]
        assert e.shape[2] == node_mask.shape[1]
        if kernel is not None:
            assert node_mask.shape[1] == kernel.shape[1]
            assert kernel.shape[1] == kernel.shape[2]

        # Checking if the number of layers is consistent
        assert e.shape[3] == x.shape[2]
        assert x.shape[2] == y.shape[1]
        if kernel is not None:
            assert y.shape[1] == kernel.shape[3]

        # Step 1: calculation of the masks
        bs, n, l, _ = x.shape

        diag_mask = torch.eye(n)
        diag_mask = ~diag_mask.type_as(e).bool()
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(bs, -1, -1, -1, -1)  # (bs, n, n, 1, 1)
        
        node_mask_layers = node_mask.unsqueeze(1)  # (bs, n) -> (bs, 1, n)
        node_mask_layers = node_mask_layers.expand(-1, l, -1)  # (bs, 1, n) -> (bs, l, n)
        node_mask_layers = node_mask_layers.reshape(-1, n)  # (bs, l, n) -> (bs*l, n)

        node_mask_Xx = node_mask.reshape(-1)  # (bs, n) -> (bs*n)

        # Step 2: application of the input MLPs
        X_to_out = X[..., :self.out_dim_X]
        Y_to_out = Y[..., :self.out_dim_Y]
        e_to_out = e[..., :self.out_dim_e]
        x_to_out = x[..., :self.out_dim_x]
        y_to_out = y[..., :self.out_dim_y]

        new_e = self.mlp_in_e(e)
        new_e = (new_e + new_e.transpose(1, 2)) / 2
        assert torch.allclose(new_e, new_e.transpose(1, 2)), "new_e is not symmetric after symetrization"
        
        after_in = utils.PlaceHolderMultilayer(X=self.mlp_in_X(X), 
                                                Y=self.mlp_in_Y(Y), 
                                                e=new_e,
                                                x=self.mlp_in_x(x),
                                                y=self.mlp_in_y(y)).mask(node_mask)
        
        X, Y, e, x, y = after_in.X, after_in.Y, after_in.e, after_in.x, after_in.y
        
        # Step 3: a loop to apply the transformer layers
        for layer_id in range(self.n_layers):
            if self.parallel:
                # Message passing between layer-specific features
                dx = x.shape[-1]
                x_perm_xey = x.permute(0, 2, 1, 3)  # (bs, n, l, dx) -> (bs, l, n, dx)
                x_in_xey = x_perm_xey.reshape(-1, n, dx)  # (bs*l, n, dx)

                de = e.shape[-1]
                e_perm_xey = e.permute(0, 3, 1, 2, 4)  # (bs, n, n, l, de) -> (bs, l, n, n, de)
                e_in_xey = e_perm_xey.reshape(-1, n, n, de)  # (bs, l, n, n, de) -> (bs*l, n, n, de)

                dy = y.shape[-1]
                y_in_xey = y.reshape(-1, dy)  # (bs, l, dy) -> (bs*l, dy)
                
                if kernel is not None:
                    reshaped_kernel = kernel.permute(0, 3, 1, 2)  # (bs, n, n, l) -> (bs, l, n, n)
                    reshaped_kernel = kernel.reshape(-1, n, n)  # (bs, l, n, n) -> (bs*l, n, n)
                else:
                    reshaped_kernel = None

                x_xey, e_xey, y_xey = self.xeyTrLayers[layer_id](x_in_xey, e_in_xey, y_in_xey, node_mask_layers, reshaped_kernel)

                x_xey = x_xey.reshape(bs, l, n, dx)  # (bs*l, n, dx) -> (bs, l, n, dx)
                x_xey = x_xey.permute(0, 2, 1, 3)  # (bs, l, n, dx) -> (bs, n, l, dx)

                e_xey = e_xey.reshape(bs, l, n, n, de)  # (bs*l, n, n, de) -> (bs, l, n, n, de)
                e_xey = e_xey.permute(0, 2, 3, 1, 4)  # (bs, n, n, l, de)

                y_xey = y_xey.reshape(bs, l, dy)  # (bs*l, dy) -> (bs, l, dy)


                # Message passing between multi-layer features
                if self.cross_attention:
                    Y_reshaped_XY = Y.unsqueeze(-2) # (bs, dY) -> (bs, 1, dY)
                    Y_reshaped_XY, X_XY = self.YXTrLayers[layer_id](Y_reshaped_XY, X, L_mask=node_mask, G_mask=None)
                    Y_XY = Y_reshaped_XY.squeeze(-2)
                else:
                    Y_XY, X_XY = self.YXTrLayers[layer_id](Y, X, node_mask=node_mask)

                # Message passing between multi-layer features and layer-specific features
                dx = x.shape[-1]
                x_in_Xx = x.reshape(-1, l, dx)  # (bs, n, l, dx) -> (bs*n, l, dx)
                dX = X.shape[-1]
                X_in_Xx = X.reshape(-1, dX)  # (bs, n, dX) -> (bs*n, dX)
                if self.cross_attention:
                    X_in_reshaped_Xx = X_in_Xx.unsqueeze(-2)  # (bs*n, dX) -> (bs*n, 1, dX)
                    X_reshaped_Xx, x_Xx = self.XxTrLayers[layer_id](X_in_reshaped_Xx, x_in_Xx, L_mask=None, G_mask=node_mask_Xx)
                    X_Xx = X_reshaped_Xx.squeeze(-2)
                else:
                    X_Xx, x_Xx = self.XxTrLayers[layer_id](X_in_Xx, x_in_Xx)
                x_Xx = x_Xx.reshape(bs, n, l, dx)  # (bs*n, l, dx) -> (bs, n, l, dx)
                X_Xx = X_Xx.reshape(bs, n, dX)  # (bs*n, dX) -> (bs, n, dX)

                if self.cross_attention:
                    Y_reshaped_Yy = Y.unsqueeze(-2) # (bs, dY) -> (bs, 1, dY)
                    Y_reshaped_Yy, y_Yy = self.YyTrLayers[layer_id](Y_reshaped_Yy, y, L_mask=None, G_mask=None)
                    Y_Yy = Y_reshaped_Yy.squeeze(-2)
                else:
                    Y_Yy, y_Yy = self.YyTrLayers[layer_id](Y, y)
                
                X_concat = torch.cat([X_Xx, X_XY], dim=-1)
                Y_concat = torch.cat([Y_Yy, Y_XY], dim=-1)
                x_concat = torch.cat([x_Xx, x_xey], dim=-1)
                y_concat = torch.cat([y_Yy, y_xey], dim=-1)

                X = self.concat_X[layer_id](X_concat)
                x = self.concat_x[layer_id](x_concat)
                Y = self.concat_Y[layer_id](Y_concat)
                y = self.concat_y[layer_id](y_concat)
                e = e_xey

            else:
                # Message passing between layer-specific features
                dx = x.shape[-1]
                x_perm = x.permute(0, 2, 1, 3)  # (bs, n, l, dx) -> (bs, l, n, dx)
                x_in = x_perm.reshape(-1, n, dx)  # (bs*l, n, dx)

                de = e.shape[-1]
                e_perm = e.permute(0, 3, 1, 2, 4)  # (bs, n, n, l, de) -> (bs, l, n, n, de)
                e_in = e_perm.reshape(-1, n, n, de)  # (bs, l, n, n, de) -> (bs*l, n, n, de)

                dy = y.shape[-1]
                y_in = y.reshape(-1, dy)  # (bs, l, dy) -> (bs*l, dy)
                
                if kernel is not None:
                    reshaped_kernel = kernel.permute(0, 3, 1, 2)  # (bs, n, n, l) -> (bs, l, n, n)
                    reshaped_kernel = kernel.reshape(-1, n, n)  # (bs, l, n, n) -> (bs*l, n, n)
                else:
                    reshaped_kernel = None

                x, e, y = self.xeyTrLayers[layer_id](x_in, e_in, y_in, node_mask_layers, reshaped_kernel)

                x = x.reshape(bs, l, n, dx)  # (bs*l, n, dx) -> (bs, l, n, dx)
                x = x.permute(0, 2, 1, 3)  # (bs, l, n, dx) -> (bs, n, l, dx)

                e = e.reshape(bs, l, n, n, de)  # (bs*l, n, n, de) -> (bs, l, n, n, de)
                e = e.permute(0, 2, 3, 1, 4)  # (bs, n, n, l, de)

                y = y.reshape(bs, l, dy)  # (bs*l, dy) -> (bs, l, dy)

                if not self.single_layer:
                    # Message passing between multi-layer features
                    if self.cross_attention:
                        Y_reshaped = Y.unsqueeze(-2) # (bs, dY) -> (bs, 1, dY)
                        Y_reshaped, X = self.YXTrLayers[layer_id](Y_reshaped, X, L_mask=node_mask, G_mask=None)
                        Y = Y_reshaped.squeeze(-2)
                    else:
                        Y, X = self.YXTrLayers[layer_id](Y, X, node_mask=node_mask)

                    # Message passing between multi-layer features and layer-specific features
                    dx = x.shape[-1]
                    x_in = x.reshape(-1, l, dx)  # (bs, n, l, dx) -> (bs*n, l, dx)
                    dX = X.shape[-1]
                    X_in = X.reshape(-1, dX)  # (bs, n, dX) -> (bs*n, dX)
                    if self.cross_attention:
                        X_in_reshaped = X_in.unsqueeze(-2)  # (bs*n, dX) -> (bs*n, 1, dX)
                        X_reshaped, x = self.XxTrLayers[layer_id](X_in_reshaped, x_in, L_mask=None, G_mask=node_mask_Xx)
                        X = X_reshaped.squeeze(-2)
                    else:
                        X, x = self.XxTrLayers[layer_id](X_in, x_in)
                    x = x.reshape(bs, n, l, dx)  # (bs*n, l, dx) -> (bs, n, l, dx)
                    X = X.reshape(bs, n, dX)  # (bs*n, dX) -> (bs, n, dX)

                    if self.cross_attention:
                        Y_reshaped = Y.unsqueeze(-2) # (bs, dY) -> (bs, 1, dY)
                        Y_reshaped, y = self.YyTrLayers[layer_id](Y_reshaped, y, L_mask=None, G_mask=None)
                        Y = Y_reshaped.squeeze(-2)
                    else:
                        Y, y = self.YyTrLayers[layer_id](Y, y)

        if self.parallel == False:
            # Final message passing between layer-specific features
            dx = x.shape[-1]
            x_perm = x.permute(0, 2, 1, 3)  # (bs, n, l, dx) -> (bs, l, n, dx)
            x_in = x_perm.reshape(-1, n, dx)  # (bs*l, n, dx)

            de = e.shape[-1]
            e_perm = e.permute(0, 3, 1, 2, 4)  # (bs, n, n, l, de) -> (bs, l, n, n, de)
            e_in = e_perm.reshape(-1, n, n, de)  # (bs, l, n, n, de) -> (bs*l, n, n, de)

            dy = y.shape[-1]
            y_in = y.reshape(-1, dy)  # (bs, l, dy) -> (bs*l, dy)

            x, e, y = self.xeyTrLayers[layer_id](x_in, e_in, y_in, node_mask_layers)

            x = x.reshape(bs, l, n, dx)  # (bs*l, n, dx) -> (bs, l, n, dx)
            x = x.permute(0, 2, 1, 3)  # (bs, l, n, dx) -> (bs, n, l, dx)

            e = e.reshape(bs, l, n, n, de)  # (bs*l, n, n, de) -> (bs, l, n, n, de)
            e = e.permute(0, 2, 3, 1, 4)  # (bs, n, n, l, de)

            y = y.reshape(bs, l, dy)  # (bs*l, dy) -> (bs, l, dy)


        # Step 4: application of the output MLPs
        X = self.mlp_out_X(X) + X_to_out
        Y = self.mlp_out_Y(Y) + Y_to_out
        e = (self.mlp_out_e(e) + e_to_out) * diag_mask
        x = self.mlp_out_x(x) + x_to_out
        y = self.mlp_out_y(y) + y_to_out

        e = (e + torch.transpose(e, 1, 2)) / 2

        return utils.PlaceHolderMultilayer(X=X, x=x, e=e, Y=Y, y=y).mask(node_mask=node_mask)



class MultiLayerAttention(nn.Module):
    """  
    Attention between multiple layers.
    """
    def __init__(self, dx: int, dy: int, n_head: int, dim_ffX: int = 2048,
                 dim_ffy: int = 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, device=None, dtype=None) -> None:
        kw = {'device': device, 'dtype': dtype}
        super().__init__()
        self.node_attention = nn.MultiheadAttention(embed_dim=dx, num_heads=n_head, batch_first=True, dropout=dropout)
        self.global_attention = nn.MultiheadAttention(embed_dim=dy, num_heads=n_head, batch_first=True, dropout=dropout)

        self.dropout_x1 = Dropout(dropout)
        self.dropout_x2 = Dropout(dropout)
        self.dropout_x3 = Dropout(dropout)
        self.dropout_y1 = Dropout(dropout)
        self.dropout_y2 = Dropout(dropout)
        self.dropout_y3 = Dropout(dropout)

        self.lin_x1 = Linear(dx, dim_ffX, **kw)
        self.lin_x2 = Linear(dim_ffX, dx, **kw)
        self.lin_y1 = Linear(dy, dim_ffy, **kw)
        self.lin_y2 = Linear(dim_ffy, dy, **kw)

        self.norm_x1 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.norm_x2 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.norm_y1 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.norm_y2 = LayerNorm(dy, eps=layer_norm_eps, **kw)

        self.activation = F.relu
    
    def forward(self, x, y):
        """
        x: Node features of shape (batch_size, num_nodes, num_layers, feature_dim)
        y: Global features of shape (batch_size, num_layers, feature_dim)
        """
        batch_size, num_nodes, num_layers, feature_dim = x.shape

        # Concatenate node features across layers
        x_concat = x.reshape(batch_size*num_nodes, num_layers, feature_dim) # (batch_size*num_nodes, num_layers, feature_dim)
        # Apply node attention
        x_attn, _ = self.node_attention(x_concat, x_concat, x_concat)
        x_attn = self.dropout_x1(x_attn)
        x = self.norm_x1(x_concat + x_attn)
        # Reshape back to original shape
        x = x.reshape(batch_size, num_nodes, num_layers, feature_dim) # (batch_size, num_nodes, num_layers, feature_dim)

        x2 = self.dropout_x2(x)
        ff_output_x = self.lin_x2(self.dropout_x2(self.activation(self.lin_x1(x2))))
        ff_output_x = self.dropout_x3(ff_output_x)
        x = self.norm_x2(x + ff_output_x)

        # Apply self-attention on global features
        y_attn, _ = self.global_attention(y, y, y)
        y_attn = self.dropout_y1(y_attn)
        y = self.norm_y1(y + y_attn)
        
        ff_output_y = self.lin_y2(self.dropout_y2(self.activation(self.lin_y1(y))))
        ff_output_y = self.dropout_y3(ff_output_y)
        y = self.norm_y2(y + ff_output_y)

        return x, y


class MultiGraphLayer(nn.Module):
    def __init__(self, dx: int, de: int, dy: int, n_head: int, dim_ffX: int = 2048,
                 dim_ffE: int = 128, dim_ffy: int = 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, device=None, dtype=None) -> None:
        super().__init__()
        self.transformer_layer = MultiXEyTransformerLayer(dx, de, dy, n_head, dim_ffX, dim_ffE, dim_ffy,
                                                          dropout, layer_norm_eps, device, dtype)
        self.attention_layer = MultiLayerAttention(dx, dy, n_head, dim_ffX=dim_ffX,
                                                   dim_ffy=dim_ffy, dropout=dropout,
                                                   layer_norm_eps=layer_norm_eps, device=device, dtype=dtype)
    
    def forward(self, x, e, y, node_mask):
        """
        X: Node features of shape (batch_size, num_nodes, feature_dim)
        E: Edge features of shape (batch_size, num_nodes, num_nodes, feature_dim)
        y: Global features of shape (batch_size, feature_dim)
        node_mask: Mask for the src keys per batch (optional)
        """
        # Apply the attention layer
        x, y = self.attention_layer(x, y)

        # Reshape x, e and y to separate layers
        batch_size, num_nodes, num_layers, x_feature_dim = x.shape
        _, _, _, _, e_feature_dim = e.shape
        _, _, y_feature_dim = y.shape
        x_perm = x.permute(0, 2, 1, 3)  # (bs, num_nodes, num_layers, feature_dim) -> (bs, num_layers, num_nodes, feature_dim)
        e_perm = e.permute(0, 3, 1, 2, 4)  # (bs, num_nodes, num_nodes, num_layers, e_feature_dim) -> (bs, num_layers, num_nodes, num_nodes, e_feature_dim)
        x_reshaped = x_perm.reshape(-1, num_nodes, x_feature_dim)  # (bs * num_layers, num_nodes, feature_dim)
        e_reshaped = e_perm.reshape(-1, num_nodes, num_nodes, e_feature_dim)  # (bs * num_layers, num_nodes, num_nodes, num_layers, e_feature_dim)
        y_reshaped = y.reshape(-1, y_feature_dim)  # (bs * num_layers, feature_dim)

        node_mask_reshaped = node_mask.unsqueeze(1).expand(-1, num_layers, -1)  # (bs, num_layers, num_nodes)
        node_mask_reshaped = node_mask_reshaped.reshape(-1, num_nodes)  # (bs * num_layers, num_nodes)
        # Ensure the node_mask is of the correct shape
        assert node_mask_reshaped.shape == (batch_size * num_layers, num_nodes), \
            f"Expected node_mask shape {(batch_size * num_layers, num_nodes)}, got {node_mask_reshaped.shape}"

        # Apply the transformer layer
        x, e, y = self.transformer_layer(x_reshaped, e_reshaped, y_reshaped, node_mask_reshaped)

        # Reshape back to original shape
        x = x.reshape(batch_size, num_layers, num_nodes, x_feature_dim).permute(0, 2, 1, 3)
        e = e.reshape(batch_size, num_layers, num_nodes, num_nodes, e_feature_dim).permute(0, 2, 3, 1, 4)
        y = y.reshape(batch_size, num_layers, y_feature_dim)

        return x, e, y

class MultiXEyTransformerLayer(nn.Module):
    """ Transformer that updates node, edge and global features
        d_x: node features
        d_e: edge features
        dy : global features
        n_head: the number of heads in the multi_head_attention
        dim_feedforward: the dimension of the feedforward network model after self-attention
        dropout: dropout probablility. 0 to disable
        layer_norm_eps: eps value in layer normalizations.
    """
    def __init__(self, dx: int, de: int, dy: int, n_head: int, dim_ffX: int = 2048,
                 dim_ffE: int = 128, dim_ffy: int = 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, triplet_interactions = False,
                 device=None, dtype=None) -> None:
        kw = {'device': device, 'dtype': dtype}
        super().__init__()

        self.self_attn = MultiLayerNodeEdgeBlock(dx, de, dy, n_head, **kw)

        self.linX1 = Linear(dx, dim_ffX, **kw)
        self.linX2 = Linear(dim_ffX, dx, **kw)
        self.normX1 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.normX2 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.dropoutX1 = Dropout(dropout)
        self.dropoutX2 = Dropout(dropout)
        self.dropoutX3 = Dropout(dropout)

        self.linE1 = Linear(de, dim_ffE, **kw)
        self.linE2 = Linear(dim_ffE, de, **kw)
        self.normE1 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.normE2 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.dropoutE1 = Dropout(dropout)
        self.dropoutE2 = Dropout(dropout)
        self.dropoutE3 = Dropout(dropout)

        self.lin_y1 = Linear(dy, dim_ffy, **kw)
        self.lin_y2 = Linear(dim_ffy, dy, **kw)
        self.norm_y1 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.norm_y2 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.dropout_y1 = Dropout(dropout)
        self.dropout_y2 = Dropout(dropout)
        self.dropout_y3 = Dropout(dropout)

        self.triplet_interactions = triplet_interactions

        if triplet_interactions is not None:
            self.dropout_E_triplets = Dropout(dropout)
            self.norm_E_triplets = LayerNorm(de, eps=layer_norm_eps, **kw)
            
            if triplet_interactions == "attention":
                self.triplet_block = TripletAttention(de=de, n_head=n_head)

            elif triplet_interactions == "aggregation":
                self.triplet_block = TripletAggregation(de=de, d_hidden=de)

        self.activation = F.relu

    def forward(self, X: Tensor, E: Tensor, y, node_mask: Tensor, kernel: Tensor = None):
        """ Pass the input through the encoder layer.
            X: (bs, n, d)
            E: (bs, n, n, d)
            y: (bs, dy)
            node_mask: (bs, n) Mask for the src keys per batch (optional)
            kernel: (bs, n, n) (if None, no kernel is applied)
            Output: newX, newE, new_y with the same shape.
        """

        newX, newE, new_y = self.self_attn(X, E, y, node_mask=node_mask, kernel=kernel)

        newX_d = self.dropoutX1(newX)
        X = self.normX1(X + newX_d)

        newE_d = self.dropoutE1(newE)
        E = self.normE1(E + newE_d)

        new_y_d = self.dropout_y1(new_y)
        y = self.norm_y1(y + new_y_d)

        if self.triplet_interactions:
            newE_triplets = self.triplet_block(E, node_mask)
            newE_triplets_d = self.dropout_E_triplets(newE_triplets)
            E = self.norm_E_triplets(newE_triplets_d + E)

        ff_outputX = self.linX2(self.dropoutX2(self.activation(self.linX1(X))))
        ff_outputX = self.dropoutX3(ff_outputX)
        X = self.normX2(X + ff_outputX)

        ff_outputE = self.linE2(self.dropoutE2(self.activation(self.linE1(E))))
        ff_outputE = self.dropoutE3(ff_outputE)
        E = self.normE2(E + ff_outputE)

        ff_output_y = self.lin_y2(self.dropout_y2(self.activation(self.lin_y1(y))))
        ff_output_y = self.dropout_y3(ff_output_y)
        y = self.norm_y2(y + ff_output_y)

        return X, E, y


class MultiLayerNodeEdgeBlock(nn.Module):
    """ Self attention layer that also updates the representations on the edges. """
    def __init__(self, dx, de, dy, n_head, **kwargs):
        super().__init__()
        assert dx % n_head == 0, f"dx: {dx} -- nhead: {n_head}"
        self.dx = dx
        self.de = de
        self.dy = dy
        self.df = int(dx / n_head)
        self.n_head = n_head

        # Attention
        self.q = Linear(dx, dx)
        self.k = Linear(dx, dx)
        self.v = Linear(dx, dx)

        # FiLM E to X
        self.e_add = Linear(de, dx)
        self.e_mul = Linear(de, dx)

        # FiLM y to E
        self.y_e_mul = Linear(dy, dx)           # Warning: here it's dx and not de
        self.y_e_add = Linear(dy, dx)

        # FiLM y to X
        self.y_x_mul = Linear(dy, dx)
        self.y_x_add = Linear(dy, dx)

        # Process y
        self.y_y = Linear(dy, dy)
        self.x_y = Xtoy(dx, dy)
        self.e_y = Etoy(de, dy)

        # Output layers
        self.x_out = Linear(dx, dx)
        self.e_out = Linear(dx, de)
        self.y_out = nn.Sequential(nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy))

    def forward(self, X, E, y, node_mask, kernel = None):
        """
        :param X: bs, n, d        node features
        :param E: bs, n, n, d     edge features
        :param y: bs, dz           global features
        :param node_mask: bs, n
        :param kernel: bs, n, n
        :return: newX, newE, new_y with the same shape.
        """
        # assert torch.isfinite(X).all(), "X contains NaN values"
        # assert torch.isfinite(E).all(), "E contains NaN values"
        # assert torch.isfinite(y).all(), "y contains NaN values"

        bs, n, _ = X.shape
        x_mask = node_mask.unsqueeze(-1)        # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)           # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)           # bs, 1, n, 1

        # 1. Map X to keys and queries
        Q = self.q(X) * x_mask           # (bs, n, dx)
        K = self.k(X) * x_mask           # (bs, n, dx)
        diffusion_utils.assert_correctly_masked(Q, x_mask)
        # 2. Reshape to (bs, n, n_head, df) with dx = n_head * df

        Q = Q.reshape((Q.size(0), Q.size(1), self.n_head, self.df))
        K = K.reshape((K.size(0), K.size(1), self.n_head, self.df))

        Q = Q.unsqueeze(2)                              # (bs, 1, n, n_head, df)
        K = K.unsqueeze(1)                              # (bs, n, 1, n head, df)

        # Compute unnormalized attentions. Y is (bs, n, n, n_head, df)
        Y = Q * K
        Y = Y / math.sqrt(Y.size(-1))
        # assert torch.isfinite(Y).all(), "Y contains NaN values"
        diffusion_utils.assert_correctly_masked(Y, (e_mask1 * e_mask2).unsqueeze(-1))

        E1 = self.e_mul(E) * e_mask1 * e_mask2                        # bs, n, n, dx
        E1 = E1.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))

        E2 = self.e_add(E) * e_mask1 * e_mask2                        # bs, n, n, dx
        E2 = E2.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))

        # Incorporate edge features to the self attention scores.
        Y = Y * (E1 + 1) + E2                  # (bs, n, n, n_head, df)

        # assert torch.isfinite(Y).all(), "Y contains NaN values after adding E"

        # Incorporate y to E
        newE = Y.flatten(start_dim=3)                      # bs, n, n, dx
        # assert torch.isfinite(newE).all(), "newE contains NaN values before FiLM"
        ye1 = self.y_e_add(y).unsqueeze(1).unsqueeze(1)  # bs, 1, 1, de
        ye2 = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
        # assert torch.isfinite(ye1).all(), "ye1 contains NaN values"
        # assert torch.isfinite(ye2).all(), "ye2 contains NaN values"
        newE = ye1 + (ye2 + 1) * newE

        # print("newE: "+str(newE))

        # assert torch.isfinite(e_mask1).all(), "e_mask1 contains NaN values"
        # assert torch.isfinite(e_mask2).all(), "e_mask2 contains NaN values"
        # assert torch.isfinite(self.e_out(newE)).all(), "e_out(newE) contains NaN values"
        # assert torch.isfinite(e_mask1 * e_mask2).all(), "e_mask1 * e_mask2 contains NaN values"

        # Output E
        newE = self.e_out(newE) * e_mask1 * e_mask2      # bs, n, n, de
        # assert torch.isfinite(newE).all(), "newE contains NaN values"
        diffusion_utils.assert_correctly_masked(newE, e_mask1 * e_mask2)

        # Multiply Y with the kernel
        if kernel is not None:
            reshaped_kernel = kernel.unsqueeze(-1).unsqueeze(-1)  # bs, n, n, 1, 1
            Y = Y - reshaped_kernel

        # Compute attentions. attn is still (bs, n, n, n_head, df)
        softmax_mask = e_mask2.expand(-1, n, -1, self.n_head)    # bs, 1, n, 1
        attn = masked_softmax(Y, softmax_mask, dim=2)  # bs, n, n, n_head

        V = self.v(X) * x_mask                        # bs, n, dx
        V = V.reshape((V.size(0), V.size(1), self.n_head, self.df))
        V = V.unsqueeze(1)                                     # (bs, 1, n, n_head, df)

        # Compute weighted values
        weighted_V = attn * V
        weighted_V = weighted_V.sum(dim=2)

        # Send output to input dim
        weighted_V = weighted_V.flatten(start_dim=2)            # bs, n, dx

        # Incorporate y to X
        yx1 = self.y_x_add(y).unsqueeze(1)
        yx2 = self.y_x_mul(y).unsqueeze(1)
        newX = yx1 + (yx2 + 1) * weighted_V

        # Output X
        newX = self.x_out(newX) * x_mask
        diffusion_utils.assert_correctly_masked(newX, x_mask)

        # Process y based on X axnd E
        y = self.y_y(y)
        e_y = self.e_y(E)
        x_y = self.x_y(X)
        new_y = y + x_y + e_y
        new_y = self.y_out(new_y)               # bs, dy

        return newX, newE, new_y


class MultiLayerGraphTransformer(nn.Module):
    """
    n_layers : int -- number of layers
    dims : dict -- contains dimensions for each feature type
    """
    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict, hidden_dims: dict,
                 output_dims: dict, act_fn_in, act_fn_out, nb_labels=None):
        super().__init__()
        self.n_layers = n_layers

        self.out_dim_x = output_dims['x']
        self.out_dim_e = output_dims['e']
        self.out_dim_y = output_dims['y']
        self.nb_labels = nb_labels

        self.mlp_in_X = nn.Sequential(nn.Linear(input_dims['x'], hidden_mlp_dims['X']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['X'], hidden_dims['dx']), act_fn_in)

        self.mlp_in_E = nn.Sequential(nn.Linear(input_dims['e'], hidden_mlp_dims['E']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['E'], hidden_dims['de']), act_fn_in)

        self.mlp_in_y = nn.Sequential(nn.Linear(input_dims['y'], hidden_mlp_dims['y']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['y'], hidden_dims['dy']), act_fn_in)
        
        self.tf_layers = [MultiGraphLayer(dx=hidden_dims['dx'],
                                          de=hidden_dims['de'], dy=hidden_dims['dy'],
                                          n_head=hidden_dims['n_head'],
                                          dim_ffX=hidden_dims['dim_ffX'],
                                          dim_ffE=hidden_dims['dim_ffE'],
                                          dim_ffy=hidden_dims['dim_ffy']) for _ in range(n_layers)]
        self.tf_layers = nn.ModuleList(self.tf_layers)

        self.mlp_out_X = nn.Sequential(nn.Linear(hidden_dims['dx'], hidden_mlp_dims['X']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['X'], output_dims['x']))

        self.mlp_out_E = nn.Sequential(nn.Linear(hidden_dims['de'], hidden_mlp_dims['E']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['E'], output_dims['e']))

        self.mlp_out_y = nn.Sequential(nn.Linear(hidden_dims['dy'], hidden_mlp_dims['y']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['y'], output_dims['y']))
        
        if nb_labels is not None:
            self.act_fn_labels = nn.ReLU()

    def forward(self, x, e, y, node_mask):
        bs, n = x.shape[0], x.shape[1]

        diag_mask = torch.eye(n)
        diag_mask = ~diag_mask.type_as(e).bool()
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(bs, -1, -1, -1, -1)
        
        X_to_out = x[..., :self.out_dim_x]
        E_to_out = e[..., :self.out_dim_e]
        y_to_out = y[..., :self.out_dim_y]

        new_e = self.mlp_in_E(e)

        new_e = (new_e + new_e.transpose(1, 2)) / 2
        assert torch.allclose(new_e, new_e.transpose(1, 2)), "new_e is not symmetric after symmetrization"

        after_in = utils.PlaceHolder(X=self.mlp_in_X(x), E=new_e, y=self.mlp_in_y(y)).mask(node_mask, layer_labels=True)
        x, e, y = after_in.X, after_in.E, after_in.y

        for layer in self.tf_layers:
            x, e, y = layer(x, e, y, node_mask)

        X = self.mlp_out_X(x)
        E = self.mlp_out_E(e)
        y = self.mlp_out_y(y)

        X = (X + X_to_out)
        E = (E + E_to_out) * diag_mask
        y = y + y_to_out

        E = 1/2 * (E + torch.transpose(E, 1, 2))

        if self.nb_labels is not None:
            # Assuming the last dimensions of X contains node labels
            X[..., -self.nb_labels:] = self.act_fn_labels(X[..., -self.nb_labels:])

        output = utils.PlaceHolder(X=X, E=E, y=y).mask(node_mask, layer_labels=True)

        assert output.E.shape[-1] == 1
        assert output.X.shape[-1] == 1

        output.E = output.E.squeeze(-1)
        output.X = output.X.mean(-2)  # Average on the layers

        return output


class TripletAttention(nn.Module):
    def __init__(self, de, n_head, dropout=0.1):
        super().__init__()
        self.df = de // n_head
        self.n_head = n_head

        self.q_proj = Linear(de, de, bias=False)
        self.p_proj = Linear(de, de, bias=False)
        self.v_proj = Linear(de, de, bias=False)
               
        self.bias_proj = Linear(de, n_head, bias=False)
        self.gate_proj = Linear(de, n_head, bias=False)

        self.triplet_dropout = Dropout(dropout)

        self.mix_in_out = Linear(2*de, de, bias=False)
    
    def compute_attention(self, q, p, v, bias, gate, mask_tri):
        """
        q: queries for (i,j) -> (bs, n_head, n, n, df)
        p: keys for (_,k) -> (bs, n_head, n, n, df)
        v: values for (_,k) -> (bs, n_head, n, n, df)
        bias, gate for (i,k) -> (bs, n_head, n, 1, n)
        mask_tri -> (bs, n, n, n)
        """
        y = ((q @ p.transpose(-1, -2)).squeeze(-1)) / math.sqrt(self.df) + bias  # (bs, n_head, n, n, n)
        y_masked = y.masked_fill(mask_tri.unsqueeze(1) == False, -1e-9)
        y_masked = self.triplet_dropout(y_masked)
        attn = torch.softmax(y_masked, dim=-1) * torch.sigmoid(gate)

        output = ((attn.unsqueeze(-1))*(v.unsqueeze(2))).sum(dim=-2)  # sum over k, (bs, n_head, n, n, df)
        
        return output


    def forward(self, e, node_mask=None):
        """
        e: edge embeddings -> (bs, n, n, de)
        node_mask -> (bs, n) or None
        """
        assert e.shape[1] == e.shape[2]

        # Calculation of mask_tri
        if node_mask is not None:
            assert node_mask.shape[0] == e.shape[0]
            assert node_mask.shape[1] == e.shape[1]
            node_mask1 = node_mask.unsqueeze(-1).unsqueeze(-1)  # (bs, n, 1, 1)
            node_mask2 = node_mask.unsqueeze(1).unsqueeze(-1)  # (bs, 1, n, 1)
            node_mask3 = node_mask.unsqueeze(1).unsqueeze(1)  # (bs, 1, 1, n)
            tri_mask = node_mask1 & node_mask2 & node_mask3  # (bs, n, n, n)
        else:
            tri_mask = torch.ones(e.shape[0], e.shape[1], e.shape[1], e.shape[1], dtype=torch.bool)

        bs, n, _, de = e.shape

        # Calculation of queries, keys, values, biases and gates with linear layers
        q = self.q_proj(e)  # (bs, n, n, de)
        p = self.p_proj(e)  # (bs, n, n, de)
        v = self.v_proj(e)  # (bs, n, n, de)

        bias = self.bias_proj(e)  # (bs, n, n, n_head)
        gate = self.gate_proj(e)  # (bs, n, n, n_head)

        # Projections are splitted in separate heads and dimensions are permuted
        q = q.view(bs, n, n, self.n_head, self.df).permute(0, 3, 1, 2, 4)  # (bs, n_head, n, n, df) -> (i,j)
        p = p.view(bs, n, n, self.n_head, self.df).permute(0, 3, 2, 1, 4)  # (bs, n_head, n, n, df) -> (j,k)
        v = v.view(bs, n, n, self.n_head, self.df).permute(0, 3, 2, 1, 4)  # (bs, n_head, n, n, df) -> (j,k)

        bias = bias.view(bs, n, n, self.n_head).permute(0, 3, 1, 2)  # (bs, n_head, n, n)
        gate = gate.view(bs, n, n, self.n_head).permute(0, 3, 1, 2)  # (bs, n_head, n, n)

        # Also reshape bias and gate
        bias = bias.unsqueeze(-2)  # (bs, n_head, n, 1, n) -> (i, (j,) k)
        gate = gate.unsqueeze(-2)  # (bs, n_head, n, 1, n) -> (i, (j,) k)

        # Triplet attention inward
        att_in = self.compute_attention(q, p, v, bias, gate, tri_mask)  # (bs, n_head, n, n, df)

        # Triplet attention outward
        att_out = self.compute_attention(q,
                                         p.transpose(2, 3),  # exchange j and k
                                         v.transpose(2, 3),
                                         bias.transpose(-1, -2),
                                         gate.transpose(-1, -2),
                                         mask_tri=tri_mask
                                         )

        concat_att = torch.cat([att_in, att_out], dim=-1)  # (bs, n_head, n, n, 2*df)
        concat_att = concat_att.permute(0, 2, 3, 1, 4).contiguous()  # (bs, n, n, n_head, 2*df)
        concat_att = concat_att.view(bs, n, n, -1)  # (bs, n, n, 2*n_head*df)

        return self.mix_in_out(concat_att)
    

class TripletAggregation(nn.Module):
    def __init__(self, de, d_hidden, dropout = 0.1):
        super().__init__()
        self.v_proj = Linear(de, d_hidden, bias=False)

        self.bias_proj = Linear(de, 1, bias=False)
        self.gate_proj = Linear(de, 1, bias=False)

        self.dropout = Dropout(dropout)

        self.mix_in_out = Linear(2*d_hidden, de, bias=False)

    def forward(self, e, node_mask=None):
        """
        e: edge embeddings -> (bs, n, n, de)
        node_mask -> (bs, n) or None
        """
        assert e.shape[1] == e.shape[2]

        bs, n, _, de = e.shape

        # Calculation of values, biases and gates with linear layers
        v = self.v_proj(e)  # (bs, n, n, d_hidden)

        bias = self.bias_proj(e).squeeze(-1)  # (bs, n, n)
        gate = torch.sigmoid(self.gate_proj(e)).squeeze(-1)  # (bs, n, n)

        if node_mask is not None:
            bias = bias.masked_fill(node_mask.unsqueeze(1) == False, -1e-9)
            gate = gate * (node_mask.unsqueeze(1))
            v = v * (node_mask.unsqueeze(1).unsqueeze(-1))

        s_in = torch.softmax(bias, dim=-1)  # (bs, n, n)
        s_in_do = self.dropout(s_in)  # (bs, n, n)
        aggr_in = s_in_do * gate  # (bs, n, n)
        aggr_in = torch.einsum('bik,bjkd->bijd', aggr_in, v) # (bs, n, n, d_hidden)

        s_out = torch.softmax(bias.transpose(1, 2), dim=-1)  # (bs, n, n)
        s_out_do = self.dropout(s_out)  # (bs, n, n)
        aggr_out = s_out_do * (gate.transpose(1, 2))  # (bs, n, n)
        aggr_out = torch.einsum('bki,bkjd->bijd', aggr_out, v.transpose(1,2)) # (bs, n, n, d_hidden)

        concat_aggr = torch.cat([aggr_in, aggr_out], dim=-1)  # (bs, n, n, 2*d_hidden)

        return self.mix_in_out(concat_aggr)