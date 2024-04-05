import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

from src.utils import ccorr

class Aggregator(nn.Module):
    def __init__(self, h_dim, num_ents, num_edges, num_bases, n_hidden_layers, encoder_name,
                 self_loop=False, dropout=0.0, use_cuda=False):
        super(Aggregator, self).__init__()
        self.h_dim = h_dim
        self.self_loop = self_loop
        self.num_ents = num_ents
        self.num_edges = num_edges
        self.use_cuda = use_cuda
        self.layers = torch.nn.ModuleList()
        self.encoder_name = encoder_name
        self.n_hidden_layers = n_hidden_layers
        print('using encoder_name: ' + encoder_name)
        if encoder_name == "rgcn":
            for n in range(self.n_hidden_layers):
                self.layers.append(RGCNLayer(self.h_dim, self.h_dim, self.num_edges, num_bases,
                                    activation=F.rrelu, self_loop=self.self_loop, dropout=dropout))
        elif encoder_name == "compgcn":
            for n in range(self.n_hidden_layers):
                self.layers.append(CompGCNLayer(self.h_dim, self.h_dim, self.num_edges, num_bases,
                                    activation=F.rrelu, self_loop=self.self_loop, dropout=dropout))
        else:
            raise NotImplementedError

    def forward(self, g, ent_embeds, rel_embeds):

            g.ndata['h'] = ent_embeds[g.ndata['id']].view(-1, ent_embeds.shape[1])
            g.edata['h'] = rel_embeds[g.edata['type']].view(-1, rel_embeds.shape[1])

            for layer in self.layers:
                layer(g)

            node_repr = g.ndata.pop('h')  # [num_ents, h_dim]

            return node_repr

class AggregateLayer(nn.Module):
    def __init__(self, in_feat, out_feat, bias=None, activation=None,
                 self_loop=False, dropout=0.0):
        super(AggregateLayer, self).__init__()
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        if self.bias == True:
            self.bias = self.get_param(out_feat)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = self.get_param([in_feat, out_feat])

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    # define how propagation is done in subclass
    def propagate(self, g, reverse):
        raise NotImplementedError

    def get_param(self, shape):
        param = nn.Parameter(torch.Tensor(*shape))
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))
        return param


class RGCNLayer(AggregateLayer):
    def __init__(self, in_feat, out_feat, num_edges, num_bases, bias=None,
                 activation=None, self_loop=False, dropout=0.0):
        super(RGCNLayer, self).__init__(in_feat, out_feat, bias,
                                        self_loop=self_loop, dropout=dropout)
        self.num_edges = num_edges
        self.num_bases = num_bases
        assert self.num_bases > 0

        self.out_feat = out_feat

        self.submat_in = in_feat // self.num_bases
        self.submat_out = out_feat // self.num_bases

        self.weight = self.get_param([self.num_edges, self.num_bases * self.submat_in * self.submat_out])

    def msg_func(self, edges):
        weight = self.weight.index_select(0, edges.data['type']).view(-1, self.submat_in, self.submat_out)
        node = edges.src['h'].view(-1, 1, self.submat_in)
        msg = torch.bmm(node, weight).view(-1, self.out_feat)
        return {'msg': msg}

    def propagate(self, g):
        # h = sum(msg_i, i from all neighbors)
        g.update_all(lambda x: self.msg_func(x), fn.sum(msg='msg', out='h'), self.apply_func)

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}

    def forward(self, g):
        if self.self_loop:
            loop_message = torch.mm(g.ndata['h'], self.loop_weight)
            if self.dropout is not None:
                loop_message = self.dropout(loop_message)

        self.propagate(g)

        # apply bias and self loop
        node_repr = g.ndata['h']
        if self.bias:
            node_repr = node_repr + self.bias
        if self.self_loop:
            node_repr = node_repr + loop_message
        if self.activation:
            node_repr = self.activation(node_repr)
        if self.dropout is not None:
            node_repr = self.dropout(node_repr)

        g.ndata['h'] = node_repr
        return g


class CompGCNLayer(AggregateLayer):
    def __init__(self, in_feat, out_feat, num_edges, num_bases, bias=None,
                 activation=None, self_loop=False, dropout=0.0, combine_fn="corr"):
        super(CompGCNLayer, self).__init__(in_feat, out_feat, bias,
                                           self_loop=self_loop,
                                           dropout=dropout)
        assert combine_fn in ["corr", "sub", "mult"]
        if combine_fn == "sub":
            self.combine_fn = torch.sub
        elif combine_fn == "mult":
            self.combine_fn = torch.mul
        else:
            self.combine_fn = ccorr
        self.num_edges = num_edges
        self.num_bases = num_bases
        assert self.num_bases > 0

        self.out_feat = out_feat

        self.submat_in = in_feat // self.num_bases
        self.submat_out = out_feat // self.num_bases

        self.rel = None

        self.weight = self.get_param([self.num_edges, self.num_bases * self.submat_in * self.submat_out])
        self.w_rel = self.get_param([in_feat, out_feat])

    def msg_func(self, edges):

        weight = self.weight.index_select(0, edges.data['type']).view(-1, self.submat_in, self.submat_out)
        node = edges.src['h'].view(-1, 1, self.submat_in)
        edge = edges.data['h'].view(-1, 1, self.submat_in)
        combined = self.combine_fn(node, edge)
        msg = torch.bmm(combined, weight).view(-1, self.out_feat)
        return {'msg': msg}

    def propagate(self, g):
        g.update_all(lambda x: self.msg_func(x), fn.sum(msg='msg', out='h'), self.apply_func)

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}

    def forward(self, g):
        if self.self_loop:
            loop_message = torch.mm(g.ndata['h'], self.loop_weight)
            if self.dropout is not None:
                loop_message = self.dropout(loop_message)

        self.propagate(g)

        # apply bias and self loop
        node_repr = g.ndata['h']
        rel_repr = g.edata['h']
        self.rel = rel_repr

        if self.bias:
            node_repr = node_repr + self.bias
        if self.self_loop:
            node_repr = node_repr + loop_message
        if self.activation:
            node_repr = self.activation(node_repr)
        if self.dropout is not None:
            node_repr = self.dropout(node_repr)

        rel_repr = torch.matmul(self.rel, self.w_rel)

        g.edata['h'] = rel_repr
        g.ndata['h'] = node_repr
        return g
