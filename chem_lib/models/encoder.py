import torch
from collections import OrderedDict
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros

num_atom_type = 120  # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6  # including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3


class HyperSetEncoder2(nn.Module):
    def __init__(self, inp_dim,hidden_dim,out_dim, n_layer1=3, n_layer2=3,batch_norm=False, dropout=0.0,g_layer=5,gpu=0):
        super(HyperSetEncoder2, self).__init__()
        
        self.n_layer1=n_layer1
        self.n_layer2=n_layer2
        self.g_layer=g_layer
        self.nl=n_layer1+n_layer2
        self.dropout=dropout
        self.batch_norm=batch_norm
        self.gpu_id=gpu

        num_dims_listo = [hidden_dim] * (n_layer1+n_layer2-1)+[out_dim]
        num_dims_listi=num_dims_listo[:]
        num_dims_listi[n_layer1-1]=hidden_dim*2
        self.layer_list = OrderedDict()
        for l in range(n_layer1+n_layer2):
            if l==0:
                self.layer_list['fc{}'.format(l)] = nn.Linear(inp_dim, num_dims_listo[l])
            else:
                self.layer_list['fc{}'.format(l)] = nn.Linear(num_dims_listi[l-1], num_dims_listo[l])
            if l < n_layer1+n_layer2 - 1:
                if batch_norm:
                    self.layer_list['norm{}'.format(l)] = nn.BatchNorm1d(num_features=num_dims_listo[l])
                self.layer_list['relu{}'.format(l)] = nn.LeakyReLU()
                if dropout > 0:
                    self.layer_list['drop{}'.format(l)] = nn.Dropout(p=dropout)
        self.layer_list=nn.ModuleDict(self.layer_list)


    def cat_label(self,all_emb):
        n,d=all_emb.size()
        k=int((n)/2)
        l0=torch.zeros((k,1))
        l1=torch.ones((k,1))
        ln=torch.cat((l1,l0),-1)
        lp=torch.cat((l0,l1),-1)
        ls=torch.cat((ln,lp),0).to(all_emb.device)        
        ae=torch.cat((all_emb,ls),-1)
        return ae
    def forward(self,all_emb,pos=None):
        n,d=all_emb.size()
        x=all_emb
        if pos!=None:
            poso=torch.zeros((n,self.g_layer))
            poso[:,pos]=1
            poso=poso.to(x.device)   
            x=torch.cat((x,poso),-1)
        
        for l in range(self.nl):
            x0=x
            x=self.layer_list['fc{}'.format(l)](x)
            if l < self.nl - 1:
                if self.batch_norm:
                    x=self.layer_list['norm{}'.format(l)](x)
                x=self.layer_list['relu{}'.format(l)](x)
                if self.dropout > 0:
                    self.layer_list['drop{}'.format(l)](x)
            if l>0 and l<self.nl-1 and l!=self.n_layer1:
                x+=x0
            if l==self.n_layer1-1:
                ks=int(x.size(0)/2)
                p0=torch.mean(x[0:ks],0)
                p1=torch.mean(x[ks:],0)
                x=torch.cat((p0,p1),-1)
        
        return x
        
class Ctrl(nn.Module):

    def __init__(self, inp_dim=300, hidden_dim=300,out_dim=1,n_layer=3,g_layer=5,gpu=2,batch_norm=0,dropout=0):
        super().__init__()
        self.n_layer=n_layer
        self.g_layer=g_layer
        self.out_dim=out_dim
        self.gpu=gpu
        self.batch_norm=batch_norm
        self.dropout=dropout
        num_dims_list = [hidden_dim+g_layer] * (n_layer-1)+[out_dim]
        self.setenc=HyperSetEncoder2(inp_dim=300+2,hidden_dim=300,out_dim=inp_dim,gpu=self.gpu,n_layer2=1)
        self.layer_list = OrderedDict()
        for l in range(n_layer):
            if l==0:
                self.layer_list['fc{}'.format(l)] = nn.Linear(inp_dim+g_layer, num_dims_list[l])
            else:
                self.layer_list['fc{}'.format(l)] = nn.Linear(num_dims_list[l-1], num_dims_list[l])
            if l < n_layer- 1:
                if batch_norm:
                    self.layer_list['norm{}'.format(l)] = nn.BatchNorm1d(num_features=num_dims_list[l])
                self.layer_list['relu{}'.format(l)] = nn.LeakyReLU()
                if dropout > 0:
                    self.layer_list['drop{}'.format(l)] = nn.Dropout(p=dropout)
        self.layer_list=nn.ModuleDict(self.layer_list)

    def forward(self, node_feat,pos=0):
        nd,_=node_feat.size()
        k=int((nd)/2)
        nf=self.setenc.cat_label(node_feat)
        sup=self.setenc(nf).unsqueeze(0)#1,300
        poso=torch.zeros((1,self.g_layer))
        poso[0,pos]=1
        poso=poso.to(sup.device)   
        x=torch.cat((sup,poso),-1)

        for l in range(self.n_layer):
            x0=x
            x=self.layer_list['fc{}'.format(l)](x)
            if l < self.n_layer - 1:
                if self.batch_norm:
                    x=self.layer_list['norm{}'.format(l)](x)
                x=self.layer_list['relu{}'.format(l)](x)
                if self.dropout > 0:
                    self.layer_list['drop{}'.format(l)](x)
                x+=x0
        return x#21,1

class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not. 


    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, emb_dim, aggr="add"):
        super(GINConv, self).__init__()
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, emb_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)

class GCNConv(MessagePassing):

    def __init__(self, emb_dim, aggr="add"):
        super(GCNConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        norm = self.norm(edge_index, x.size(0), x.dtype)

        x = self.linear(x)

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings, norm=norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)

class GATConv(MessagePassing):
    def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr="add"):
        super(GATConv, self).__init__()

        self.aggr = aggr

        self.emb_dim = emb_dim
        self.heads = heads
        self.negative_slope = negative_slope

        self.weight_linear = torch.nn.Linear(emb_dim, heads * emb_dim)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))

        self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))

        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, heads * emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, heads * emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        x = self.weight_linear(x).view(-1, self.heads, self.emb_dim)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, edge_index, x_i, x_j, edge_attr):
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j += edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out = aggr_out + self.bias

        return aggr_out

class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr="mean"):
        super(GraphSAGEConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        x = self.linear(x)

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p=2, dim=-1)

class GNN(torch.nn.Module):
    """


    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """

    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0, gnn_type="gin", batch_norm=True,mod=0):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.mod=mod
        self.pool = global_mean_pool
        self.p=nn.Parameter(torch.zeros((self.num_layer)))

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if self.mod in [16,17,18,25,26]:
                
                '''gen=HyperSetEncoder2(inp_dim=emb_dim*2+num_layer+2,hidden_dim=emb_dim,out_dim=emb_dim,g_layer=num_layer)
                self.add_module('modw{}'.format(layer), gen)
                gen=HyperSetEncoder2(inp_dim=emb_dim*2+num_layer+2,hidden_dim=emb_dim,out_dim=emb_dim,g_layer=num_layer)
                self.add_module('modb{}'.format(layer), gen)'''
                gen=HyperSetEncoder2(inp_dim=emb_dim+2,hidden_dim=emb_dim,out_dim=emb_dim,g_layer=num_layer)
                self.add_module('modw{}'.format(layer), gen)
                gen=HyperSetEncoder2(inp_dim=emb_dim+2,hidden_dim=emb_dim,out_dim=emb_dim,g_layer=num_layer)
                self.add_module('modb{}'.format(layer), gen)
            elif self.mod in [20,21,22,23,24]:
                gen=HyperSetEncoder2(inp_dim=emb_dim+2,hidden_dim=emb_dim,out_dim=emb_dim,g_layer=num_layer)
                self.add_module('modw{}'.format(layer), gen)
                gen=HyperSetEncoder2(inp_dim=emb_dim+2,hidden_dim=emb_dim,out_dim=emb_dim,g_layer=num_layer)
                self.add_module('modb{}'.format(layer), gen)
                self.ctrl=Ctrl(g_layer=num_layer)
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr="add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))

        ###List of batchnorms
        self.use_batch_norm = batch_norm
        if self.use_batch_norm:
            self.batch_norms = torch.nn.ModuleList()
            for layer in range(num_layer):
                self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, *argv):
        if len(argv) == 6:
            x, edge_index, edge_attr,batch,fps,lps = argv[0], argv[1], argv[2],argv[3],argv[4],argv[5]
        elif len(argv) == 5:
            x, edge_index, edge_attr,batch,fps = argv[0], argv[1], argv[2],argv[3],argv[4]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        elif len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
            fps=None
        else:
            raise ValueError("unmatched number of arguments.")
        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        h_list = [x]
        film_list=[]
        p_list=[]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            if self.use_batch_norm:
                h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            
            if self.mod in [16,17,18,25,26]:
                if fps==None:
                    t=self.pool(h,batch)#20,300
                    '''t0=self.pool(h_list[0],batch)
                    t=torch.cat((t0,t),-1)
                    t=self._modules['modw{}'.format(0)].cat_label(t)
                    w=self._modules['modw{}'.format(0)](t,layer).unsqueeze(0)
                    b=self._modules['modb{}'.format(0)](t,layer).unsqueeze(0)#1,300'''
                    t=self._modules['modw{}'.format(layer)].cat_label(t)
                    w=self._modules['modw{}'.format(layer)](t).unsqueeze(0)
                    b=self._modules['modb{}'.format(layer)](t).unsqueeze(0)
                    if self.mod in [25]:
                        p=torch.sigmoid(self.p[layer])
                        h=h=h*(1-p+w*p)+b*p
                    elif self.mod in [26]:
                        #p=torch.sigmoid(self.p[layer])
                        p=self.p[layer]
                        w=w/torch.norm(w)
                        b=b/torch.norm(b)
                        h=h*(1+w*p)+b*p
                    else:
                        w=w/torch.norm(w)
                        b=b/torch.norm(b)
                        #w=w/torch.sum(torch.abs(w))
                        #b=b/torch.sum(torch.abs(b))
                        h=h*(1+w)+b
                    
                    film_list.append(w)
                    film_list.append(b)
                else:
                    w=fps[layer*2].unsqueeze(0)
                    b=fps[layer*2+1].unsqueeze(0)#1,300
                    if self.mod in [25]:
                        p=torch.sigmoid(self.p[layer])
                        h=h=h*(1-p+w*p)+b*p
                    elif self.mod in [26]:
                        #p=torch.sigmoid(self.p[layer])
                        p=self.p[layer]
                        w=w/torch.norm(w)
                        b=b/torch.norm(b)
                        h=h*(1+w*p)+b*p
                    else:
                        w=w/torch.norm(w)
                        b=b/torch.norm(b)
                        #w=w/torch.sum(torch.abs(w))
                        #b=b/torch.sum(torch.abs(b))
                        h=h*(1+w)+b
                    film_list.append(w)
                    film_list.append(b)
            elif self.mod in [20,21,22,23,24]:
                #print(fps,lps)
                if fps==None:
                    
                    t=self.pool(h,batch)#20,300
                    if self.mod in [22]:
                        if layer in [4]:
                            p=torch.ones((1,1)).to(t.device)        
                        else:
                            p=torch.zeros((1,1)).to(t.device) 
                    elif self.mod in [23]:
                        pt=self.ctrl(t,layer)
                        #print(pt)
                        if pt>0:
                            p=torch.ones((1,1)).to(t.device)        
                        else:
                            p=torch.zeros((1,1)).to(t.device) 
                    else:
                        p=torch.sigmoid(self.ctrl(t,layer))
                    p_list.append(p)
                    if self.mod in [24]:
                        t=self._modules['modw{}'.format(0)].cat_label(t)
                        w=self._modules['modw{}'.format(0)](t).unsqueeze(0)
                        b=self._modules['modb{}'.format(0)](t).unsqueeze(0)#1,300
                    else:
                        t=self._modules['modw{}'.format(layer)].cat_label(t)
                        w=self._modules['modw{}'.format(layer)](t).unsqueeze(0)
                        b=self._modules['modb{}'.format(layer)](t).unsqueeze(0)#1,300
                    h=h*(1-p+w*p)+b*p
                    
                    film_list.append(w)
                    film_list.append(b)
                else:
                    
                    w=fps[layer*2].unsqueeze(0)
                    b=fps[layer*2+1].unsqueeze(0)#1,300
                    p=lps[layer]
                    p_list.append(p)
                    h=h*(1-p+w*p)+b*p
                    film_list.append(w)
                    film_list.append(b)
            h_list.append(h)
            
        

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]
        if self.mod in [16,17,18,25,26]:
            #return node_representation,torch.cat(film_list,0),torch.sigmoid(self.p).detach()
            return node_representation,torch.cat(film_list,0),self.p.detach()
        elif self.mod in [20,21,22,23,24]:
            return node_representation,torch.cat(film_list,0),torch.cat(p_list,0)
        else:
            return node_representation,None,None

class GNN_Encoder(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat

    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """

    def __init__(self, num_layer, emb_dim,  JK="last", drop_ratio=0, graph_pooling="mean", gnn_type="gin",batch_norm=True,mod=0):
        super(GNN_Encoder, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_workers = 2
        self.mod=mod

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type=gnn_type,batch_norm=batch_norm,mod=self.mod)

        # Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

    def from_pretrained(self, model_file, gpu_id):
        if torch.cuda.is_available():
            self.gnn.load_state_dict(torch.load(model_file, map_location='cuda:' + str(gpu_id)),False)
        else:
            self.gnn.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')),False)

    def forward(self, *argv):
        
        if len(argv) == 5:
            x, edge_index, edge_attr, batch,fps = argv[0], argv[1], argv[2], argv[3], argv[4]
            lps=None
        elif len(argv) == 6:
            x, edge_index, edge_attr, batch,fps,lps = argv[0], argv[1], argv[2], argv[3], argv[4],argv[5]
        elif len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
            fps=None
            lps=None
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            print(len(argv))
            raise ValueError("unmatched number of arguments.")

        node_representation,fp,lp = self.gnn(x, edge_index, edge_attr,batch,fps,lps)

        graph_representation = self.pool(node_representation, batch)

        return graph_representation, node_representation,fp,lp#10,300



