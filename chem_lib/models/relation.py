
from collections import OrderedDict
from tkinter import E, N

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, inp_dim, hidden_dim, num_layers,batch_norm=False, dropout=0.):
        super(MLP, self).__init__()
        layer_list = OrderedDict()
        in_dim = inp_dim
        for l in range(num_layers):
            layer_list['fc{}'.format(l)] = nn.Linear(in_dim, hidden_dim)
            if l < num_layers - 1:
                if batch_norm:
                    layer_list['norm{}'.format(l)] = nn.BatchNorm1d(num_features=hidden_dim)
                layer_list['relu{}'.format(l)] = nn.LeakyReLU()
                if dropout > 0:
                    layer_list['drop{}'.format(l)] = nn.Dropout(p=dropout)
            in_dim = hidden_dim
        if num_layers > 0:
            self.network = nn.Sequential(layer_list)
        else:
            self.network = nn.Identity()

    def forward(self, emb):
        out = self.network(emb)
        return out

class Attention(nn.Module):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    """
    def __init__(self, dim, num_heads=1, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attention_dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return x

class ContextMLP(nn.Module):
    def __init__(self, inp_dim, hidden_dim, num_layers,pre_fc=0,batch_norm=False, dropout=0.,ctx_head=1,att=1):
        super(ContextMLP, self).__init__()
        self.pre_fc = pre_fc #0, 1
        self.att=att
        in_dim = inp_dim
        out_dim = hidden_dim
        dropout=0
        if self.att==0:
            self.mlp_proj = MLP(inp_dim=inp_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                batch_norm=batch_norm, dropout=dropout)
        else:
            if self.pre_fc:
                hidden_dim=int(hidden_dim//2)  
                self.attn_layer = Attention(hidden_dim,num_heads=ctx_head,attention_dropout=dropout)        
                self.mlp_proj = MLP(inp_dim=inp_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                    batch_norm=batch_norm, dropout=dropout)
            else:
                self.attn_layer = Attention(inp_dim)
                inp_dim=int(inp_dim*2)
                self.mlp_proj = MLP(inp_dim=inp_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                    batch_norm=batch_norm, dropout=dropout)

    def forward(self, s_emb, q_emb):
        if self.att:
            if self.pre_fc:
                s_emb = self.mlp_proj(s_emb)
                q_emb = self.mlp_proj(q_emb)
            n_support = s_emb.size(0)
            n_query = q_emb.size(0)

            s_emb_rep = s_emb.unsqueeze(0).repeat(n_query, 1, 1)
            q_emb_rep = q_emb.unsqueeze(1)
            all_emb = torch.cat((s_emb_rep, q_emb_rep), 1)######q=-1
            orig_all_emb =  all_emb

            n_shot=int(n_support//2)
            neg_proto_emb = all_emb[:,:n_shot].mean(1).unsqueeze(1).repeat(1, n_support + 1, 1)
            pos_proto_emb = all_emb[:,n_shot:2*n_shot].mean(1).unsqueeze(1).repeat(1, n_support + 1, 1)
            all_emb =torch.stack((all_emb, neg_proto_emb,pos_proto_emb), -2)
            #torch.Size([16, 21, 3, 300])
            q,s,n, d = all_emb.shape
            x=all_emb.reshape((q*s,n,d))
            attn_x =self.attn_layer(x)
            attn_x=attn_x.reshape((q,s,n, d))
            all_emb = attn_x[:,:,0,]

            all_emb = torch.cat([all_emb, orig_all_emb],dim=-1)

            if not self.pre_fc:
                all_emb = self.mlp_proj(all_emb)
            #print(all_emb.size())torch.Size([16, 21, 128])
            return all_emb, None
        else:
            n_support = s_emb.size(0)
            n_query = q_emb.size(0)

            s_emb_rep = s_emb.unsqueeze(0).repeat(n_query, 1, 1)
            q_emb_rep = q_emb.unsqueeze(1)
            all_emb = torch.cat((s_emb_rep, q_emb_rep), 1)######q=-1
            all_emb = self.mlp_proj(all_emb)
            return all_emb, None

class HyperSetEncoder(nn.Module):
    def __init__(self, inp_dim,hidden_dim,out_dim, n_layer1=4, n_layer2=4,batch_norm=False, dropout=0.0,gpu=0):
        super(HyperSetEncoder, self).__init__()
        
        self.n_layer1=n_layer1
        self.n_layer2=n_layer2
        self.nl=n_layer1+n_layer2
        self.dropout=dropout
        self.batch_norm=batch_norm
        self.gpu_id=gpu

        num_dims_list = [hidden_dim] * (n_layer1+n_layer2-1)+[out_dim]
        self.layer_list = OrderedDict()
        for l in range(n_layer1+n_layer2):
            if l==0:
                self.layer_list['fc{}'.format(l)] = nn.Linear(inp_dim, num_dims_list[l])
            else:
                self.layer_list['fc{}'.format(l)] = nn.Linear(num_dims_list[l-1], num_dims_list[l])
            if l < n_layer1+n_layer2 - 1:
                if batch_norm:
                    self.layer_list['norm{}'.format(l)] = nn.BatchNorm1d(num_features=num_dims_list[l])
                self.layer_list['relu{}'.format(l)] = nn.LeakyReLU()
                if dropout > 0:
                    self.layer_list['drop{}'.format(l)] = nn.Dropout(p=dropout)
        self.layer_list=nn.ModuleDict(self.layer_list)


    def cat_label(self,all_emb):
        bz,n,d=all_emb.size()
        k=int((n-1)/2)
        l0=torch.zeros((bz,k,1))
        l1=torch.ones((bz,k,1))
        ln=torch.cat((l1,l0),-1)
        lp=torch.cat((l0,l1),-1)
        ls=torch.cat((ln,lp),1).to(self.gpu_id)
        ae=torch.cat((all_emb[:,:-1,:],ls),-1)
        return ae
    def forward(self,all_emb):
        x=all_emb
        for l in range(self.nl):
            x0=x
            x=self.layer_list['fc{}'.format(l)](x)
            if l < self.nl - 1:
                if self.batch_norm:
                    x=self.layer_list['norm{}'.format(l)](x)
                x=self.layer_list['relu{}'.format(l)](x)
                if self.dropout > 0:
                    self.layer_list['drop{}'.format(l)](x)
            if l>0 and l<self.nl-1:
                x+=x0
            if l==self.n_layer1-1:
                x=torch.mean(x,1)
        return x

class HyperSetEncoder2(nn.Module):
    def __init__(self, inp_dim,hidden_dim,out_dim, n_layer1=3, n_layer2=3,batch_norm=False, dropout=0.0,gpu=0):
        super(HyperSetEncoder2, self).__init__()
        
        self.n_layer1=n_layer1
        self.n_layer2=n_layer2
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
        bz,n,d=all_emb.size()
        k=int((n-1)/2)
        l0=torch.zeros((bz,k,1))
        l1=torch.ones((bz,k,1))
        ln=torch.cat((l1,l0),-1)
        lp=torch.cat((l0,l1),-1)
        ls=torch.cat((ln,lp),1).to(self.gpu_id)
        ae=torch.cat((all_emb[:,:-1,:],ls),-1)
        return ae
    def forward(self,all_emb):
        x=all_emb
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
                ks=int(x.size(1)/2)
                p0=torch.mean(x[:,0:ks],1)
                p1=torch.mean(x[:,ks:],1)
                x=torch.cat((p0,p1),-1)
        return x

class ADC(nn.Module):
    def __init__(self, inp_dim,hidden_dim, n_layer=4,batch_norm=False, dropout=0.0,gpu=0):
        super(ADC, self).__init__()
        
        
        self.nl=n_layer
        self.dropout=dropout
        self.batch_norm=batch_norm
        self.gpu_id=gpu

        num_dims_list = [hidden_dim] * (n_layer-1)+[inp_dim]
        self.layer_list_w = OrderedDict()
        for l in range(self.nl):
            if l==0:
                self.layer_list_w['fc{}'.format(l)] = nn.Linear(inp_dim, num_dims_list[l])
            else:
                self.layer_list_w['fc{}'.format(l)] = nn.Linear(num_dims_list[l-1], num_dims_list[l])
            if l < self.nl - 1:
                if batch_norm:
                    self.layer_list_w['norm{}'.format(l)] = nn.BatchNorm1d(num_features=num_dims_list[l])
                self.layer_list_w['relu{}'.format(l)] = nn.LeakyReLU()
                if dropout > 0:
                    self.layer_list_w['drop{}'.format(l)] = nn.Dropout(p=dropout)
        self.layer_list_w=nn.ModuleDict(self.layer_list_w)

        num_dims_list = [hidden_dim] * (self.nl-1)+[1]
        self.layer_list_b = OrderedDict()
        for l in range(self.nl):
            if l==0:
                self.layer_list_b['fc{}'.format(l)] = nn.Linear(inp_dim, num_dims_list[l])
            else:
                self.layer_list_b['fc{}'.format(l)] = nn.Linear(num_dims_list[l-1], num_dims_list[l])
            if l < self.nl - 1:
                if batch_norm:
                    self.layer_list_b['norm{}'.format(l)] = nn.BatchNorm1d(num_features=num_dims_list[l])
                self.layer_list_b['relu{}'.format(l)] = nn.LeakyReLU()
                if dropout > 0:
                    self.layer_list_b['drop{}'.format(l)] = nn.Dropout(p=dropout)
        self.layer_list_b=nn.ModuleDict(self.layer_list_b)


    def forward(self,all_emb):
        ks=int((all_emb.size(1)-1)/2)
        x_in=torch.stack((all_emb[:,:ks,:].mean(1),all_emb[:,ks:-1,:].mean(1)),1)
        w=x_in
        for l in range(self.nl):
            w0=w
            w=self.layer_list_w['fc{}'.format(l)](w)
            if l < self.nl - 1:
                if self.batch_norm:
                    w=self.layer_list_w['norm{}'.format(l)](w)
                w=self.layer_list_w['relu{}'.format(l)](w)
                if self.dropout > 0:
                    self.layer_list_w['drop{}'.format(l)](w)
            if l>0 and l<self.nl-1:
                w+=w0#16,2,128
        w=w.transpose(-1,-2)
        b=x_in
        for l in range(self.nl):
            b0=b
            b=self.layer_list_b['fc{}'.format(l)](b)
            if l < self.nl - 1:
                if self.batch_norm:
                    b=self.layer_list_b['norm{}'.format(l)](b)
                b=self.layer_list_b['relu{}'.format(l)](b)
                if self.dropout > 0:
                    self.layer_list_b['drop{}'.format(l)](b)
            if l>0 and l<self.nl-1:
                b+=b0#16,2,1
        b=b.transpose(-1,-2)
        #print(all_emb.size(),w.size(),b.size())
        x=torch.bmm(all_emb,w)+b
        
        return x
            
class Atten(nn.Module):
        #self.W_q=nn.Parameter(torch.Tensor(in_features,hidden_features))
        #self.W_k=nn.Parameter(torch.Tensor(in_features,hidden_features))
        #self.W_v=nn.Parameter(torch.Tensor(in_features,hidden_features))

    def __init__(self, dim, num_heads=1,lm=1,k_shot=10,attention_dropout=0., projection_dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attention_dropout)

    def forward(self, node_feat,ra=0):
        nq,nd,_=node_feat.size()
        x=node_feat[:]
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        #[16, 1, 21, 128]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        #16,h,21,21
        if ra==1:
            return attn
        else:
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            return x

class Atten_full(nn.Module):
        #self.W_q=nn.Parameter(torch.Tensor(in_features,hidden_features))
        #self.W_k=nn.Parameter(torch.Tensor(in_features,hidden_features))
        #self.W_v=nn.Parameter(torch.Tensor(in_features,hidden_features))

    def __init__(self, dim, num_heads=1,lm=1,k_shot=10,attention_dropout=0., projection_dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        #self.qkv = nn.Linear(dim, dim * 3, bias=False)
        #self.attn_drop = nn.Dropout(attention_dropout)

    def forward(self, node_feat,mat):
        nq,nd,_=node_feat.size()
        x=node_feat[:]
        B, N, C = x.shape
        qkv = torch.matmul(node_feat,mat).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return x

class Atten_ker(nn.Module):

    def __init__(self, dim, num_heads=1,lm=1,k_shot=10,attention_dropout=0., projection_dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        #self.qkv = nn.Linear(dim, dim * 3, bias=False)
        #self.attn_drop = nn.Dropout(attention_dropout)

    def forward(self, node_feat,mat):
        nq,nd,_=node_feat.size()
        x=node_feat[:]
        B, N, C = x.shape
        k = torch.matmul(node_feat,mat).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k=k[0]

        attn = (node_feat.unsqueeze(1) @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        return attn
        
class MultiCG(nn.Module):
        #self.W_q=nn.Parameter(torch.Tensor(in_features,hidden_features))
        #self.W_k=nn.Parameter(torch.Tensor(in_features,hidden_features))
        #self.W_v=nn.Parameter(torch.Tensor(in_features,hidden_features))

    def __init__(self, dim=128, c=32,k_shot=10):
        super().__init__()
        self.c = c
        self.dim=dim
        self.ws=nn.parameter.Parameter(0.1*torch.rand(c,dim,dim))

    def forward(self, node_feat,ra=0):
        nq,nd,_=node_feat.size()
        x=node_feat[:].unsqueeze(1)
        xs=torch.matmul(x,self.ws)
        return xs#16,c,128,128

class Ctrl(nn.Module):

    def __init__(self, inp_dim=256, hidden_dim=256,out_dim=1,n_layer=3,gpu=2,batch_norm=0,dropout=0):
        super().__init__()
        self.n_layer=n_layer
        self.out_dim=out_dim
        self.gpu=gpu
        self.batch_norm=batch_norm
        self.dropout=dropout
        num_dims_list = [hidden_dim] * (n_layer-1)+[out_dim]
        self.setenc=HyperSetEncoder2(inp_dim=128+2,hidden_dim=128,out_dim=128,gpu=self.gpu,n_layer2=1)
        self.layer_list = OrderedDict()
        for l in range(n_layer):
            if l==0:
                self.layer_list['fc{}'.format(l)] = nn.Linear(inp_dim, num_dims_list[l])
            else:
                self.layer_list['fc{}'.format(l)] = nn.Linear(num_dims_list[l-1], num_dims_list[l])
            if l < n_layer- 1:
                if batch_norm:
                    self.layer_list['norm{}'.format(l)] = nn.BatchNorm1d(num_features=num_dims_list[l])
                self.layer_list['relu{}'.format(l)] = nn.LeakyReLU()
                if dropout > 0:
                    self.layer_list['drop{}'.format(l)] = nn.Dropout(p=dropout)
        self.layer_list=nn.ModuleDict(self.layer_list)

    def forward(self, node_feat):
        nq,nd,_=node_feat.size()
        k=int((nd-1)/2)
        nf=self.setenc.cat_label(node_feat)
        sup=self.setenc(nf)#16,128
        x=torch.cat((sup,node_feat[:,-1,:]),-1)
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



class NodeUpdateNetwork(nn.Module):
    def __init__(self, inp_dim, out_dim, n_layer=2, batch_norm=False, dropout=0.0):
        super(NodeUpdateNetwork, self).__init__()
        # set size
        num_dims_list = [out_dim] * n_layer  # [num_features * r for r in ratio]
        if n_layer > 1:
            num_dims_list[0] =  out_dim*2

        # layers
        layer_list = OrderedDict()
        for l in range(len(num_dims_list)):
            layer_list['conv{}'.format(l)] = nn.Conv2d(
                in_channels=num_dims_list[l - 1] if l > 0 else (1 +1) * inp_dim,
                out_channels=num_dims_list[l],
                kernel_size=1,
                bias=False)
            if batch_norm:
                layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=num_dims_list[l])
            layer_list['relu{}'.format(l)] = nn.LeakyReLU()

            if dropout > 0 and l == (len(num_dims_list) - 1):
                layer_list['drop{}'.format(l)] = nn.Dropout2d(p=dropout)

        self.network = nn.Sequential(layer_list)

    def forward(self, node_feat, edge_feat):
        # get size
        if len(node_feat.size())==4:#bz,c,n,e     bz,c,n,n
            sz=node_feat.size()
            diag_mask = 1.0 - torch.eye(node_feat.size(2)).unsqueeze(0).unsqueeze(0).repeat(node_feat.size(0), sz[1], 1, 1).to(
            node_feat.device)
            # set diagonal as zero and normalize
            edge_feat = F.normalize(edge_feat * diag_mask, p=1, dim=-1)
            edge_feat=edge_feat.reshape(-1,sz[2],sz[2])
            node_feat=node_feat.reshape(-1,sz[2],sz[3])
            

            aggr_feat=torch.bmm(edge_feat,node_feat)
            node_feat=torch.cat((node_feat,aggr_feat),-1).transpose(1,2)
            node_feat = self.network(node_feat.unsqueeze(-1))#bz*c,128,n,1
            node_feat=node_feat.transpose(1,2).squeeze(-1).reshape(sz[0],sz[1],sz[2],sz[3])
            '''node_feat=node_feat.transpose(1,2).unsqueeze(-1)
            node_feat = self.network(node_feat).squeeze(-1).transpose(1,2)#bz*c,n,128
            node_feat=torch.bmm(edge_feat,node_feat)
            node_feat=node_feat.transpose(1,2).squeeze(-1).reshape(sz[0],sz[1],sz[2],sz[3])'''
        else:
            num_tasks = node_feat.size(0)
            num_data = node_feat.size(1)

            # get eye matrix (batch_size x 2 x node_size x node_size)
            diag_mask = 1.0 - torch.eye(num_data).unsqueeze(0).unsqueeze(0).repeat(num_tasks, 1, 1, 1).to(node_feat.device)

            # set diagonal as zero and normalize
            edge_feat = F.normalize(edge_feat * diag_mask, p=1, dim=-1)
            
            # compute attention and aggregate
            aggr_feat = torch.bmm(torch.cat(torch.split(edge_feat, 1, 1), 1).squeeze(1), node_feat)

            node_feat = torch.cat([node_feat, torch.cat(aggr_feat.split(num_data, 1), -1)], -1).transpose(1, 2)
            # non-linear transform
            node_feat = self.network(node_feat.unsqueeze(-1)).transpose(1, 2).squeeze(-1)
        return node_feat


class EdgeUpdateNetwork(nn.Module):
    def __init__(self, in_features, hidden_features, n_layer=3, top_k=-1,
                 batch_norm=False, dropout=0.0, adj_type='dist', activation='softmax'):
        super(EdgeUpdateNetwork, self).__init__()
        self.top_k = top_k
        self.adj_type = adj_type
        self.activation = activation

        num_dims_list = [hidden_features] * n_layer  # [num_features * r for r in ratio]
        if n_layer > 1:
            num_dims_list[0] = 2 * hidden_features
        if n_layer > 3:
            num_dims_list[1] = 2 * hidden_features
        # layers
        layer_list = OrderedDict()
        for l in range(len(num_dims_list)):
            # set layer
            layer_list['conv{}'.format(l)] = nn.Conv2d(in_channels=num_dims_list[l - 1] if l > 0 else in_features,
                                                       out_channels=num_dims_list[l],
                                                       kernel_size=1,
                                                       bias=False)
            if batch_norm:
                layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=num_dims_list[l], )
            layer_list['relu{}'.format(l)] = nn.LeakyReLU()

            if dropout > 0:
                layer_list['drop{}'.format(l)] = nn.Dropout2d(p=dropout)

        layer_list['conv_out'] = nn.Conv2d(in_channels=num_dims_list[-1],
                                           out_channels=1,
                                           kernel_size=1)
        self.sim_network = nn.Sequential(layer_list)

    def softmax_with_mask(self, adj, mask=None):
        if mask is not None:
            adj_new = adj - (1 - mask.expand_as(adj)) * 1e8
        else:
            adj_new = adj
        n_q, n_edge, n1, n2 = adj_new.size()
        adj_new = adj_new.reshape(n_q * n_edge * n1, n2)
        adj_new = F.softmax(adj_new, dim=-1)
        adj_new = adj_new.reshape((n_q, n_edge, n1, n2))
        return adj_new

    def forward(self, node_feat, edge_feat=None):  # x: bs*N*num_feat
        # compute abs(x_i, x_j)
        if len(node_feat.size())==4:#bz,c,n,e
            x_i = node_feat.unsqueeze(2)#bz,c,1,n,e
            x_j = torch.transpose(x_i, 2, 3)
            x_ij = torch.abs(x_i - x_j)#bz,c,n,n,e
            x_ij = torch.transpose(x_ij, 2, 4) #bz,c,e,n,n
            if self.adj_type == 'sim':
                x_ij = torch.exp(-x_ij)
            
            sz=x_ij.size()
            x_ij=x_ij.reshape(-1,x_ij.size(2),x_ij.size(3),x_ij.size(4))
            sim_val = self.sim_network(x_ij)
            sim_val=sim_val.reshape(sz[0],sz[1],sz[3],sz[4])#16,c,n,n
            diag_mask = 1.0 - torch.eye(node_feat.size(2)).unsqueeze(0).unsqueeze(0).repeat(node_feat.size(0), sz[1], 1, 1).to(
            node_feat.device)
            #16,c,n,n
        else:
            x_i = node_feat.unsqueeze(2)
            x_j = torch.transpose(x_i, 1, 2)
            x_ij = torch.abs(x_i - x_j)
            x_ij = torch.transpose(x_ij, 1, 3)  # size: bs x fs X N x N  (2,128,11,11)
            if self.adj_type == 'sim':
                x_ij = torch.exp(-x_ij)
            
            sim_val = self.sim_network(x_ij)#16,1,n,n
            # compute similarity/dissimilarity (batch_size x feat_size x num_samples x num_samples)
            diag_mask = 1.0 - torch.eye(node_feat.size(1)).unsqueeze(0).unsqueeze(0).repeat(node_feat.size(0), 1, 1, 1).to(
                node_feat.device)
        if self.activation == 'softmax':
            sim_val = self.softmax_with_mask(sim_val, diag_mask)
        elif self.activation == 'sigmoid':
            sim_val = torch.sigmoid(sim_val) * diag_mask
        else:
            sim_val = sim_val * diag_mask

        adj_val = sim_val

        if self.top_k > 0:
            n_q, n_edge, n1, n2 = adj_val.size()
            k=min(self.top_k,n1)
            adj_temp = adj_val.reshape(n_q*n_edge*n1,n2)
            topk, indices = torch.topk(adj_temp, k)
            mask = torch.zeros_like(adj_temp)
            mask = mask.scatter(1, indices, 1)
            mask = mask.reshape((n_q, n_edge, n1, n2))
            if self.activation == 'softmax':
                adj_val = self.softmax_with_mask(adj_val, mask)
            else:
                adj_val = adj_val * mask

        return adj_val, edge_feat
class LinearClassifier(nn.Module):
    def __init__(self,gnn_inp_dim,inp_dim,num_class,drop):
        super(LinearClassifier, self).__init__()
        self.pre_dropout=drop
        self.fc1 = nn.Sequential(nn.Linear(gnn_inp_dim, inp_dim), nn.LeakyReLU())
        if self.pre_dropout>0:
            self.predrop2 = nn.Dropout(p=self.pre_dropout)
        self.fc2 = nn.Linear(inp_dim, num_class)

    def forward(self,node_feat): 
        if self.pre_dropout>0:
            node_feat=self.predrop2(node_feat)
        node_feat = self.fc1(node_feat)
        #node_feat = self.res_alpha * all_emb +  node_feat

        s_feat = node_feat[:, :-1, :]
        q_feat = node_feat[:, -1, :]

        s_logits = self.fc2(s_feat)
        q_logits = self.fc2(q_feat)
        return s_logits,q_logits

class TaskAwareRelation(nn.Module):
    def __init__(self, inp_dim, hidden_dim, num_layers, edge_n_layer, num_class=2,
                res_alpha=0., top_k=-1, node_concat=True, batch_norm=False, dropout=0.0,
                 adj_type='sim', activation='softmax',pre_dropout=0.0,mod=1,ar=1,gpu=0,adc=0,k_shot=10,lm=0,c=32):
        super(TaskAwareRelation, self).__init__()
        self.c=c
        self.gpu=gpu
        self.inp_dim = inp_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.node_concat = node_concat
        self.res_alpha = res_alpha
        self.dropout_rate = dropout
        self.pre_dropout = pre_dropout
        self.adj_type=adj_type
        self.mod=mod
        self.ar=ar
        self.adc=adc
        self.k_shot=k_shot
        self.lm=lm
        node_n_layer = max(1, min(int(edge_n_layer // 2), 2))
        gnn_inp_dim = self.inp_dim
        if self.pre_dropout>0:
            self.predrop1 = nn.Dropout(p=self.pre_dropout)
        for i in range(self.num_layers):
            if self.mod==5:
                if self.lm:
                    lmt=nn.LayerNorm((k_shot,gnn_inp_dim))
                    self.add_module('lm{}'.format(i), lmt)
                gen=HyperSetEncoder(inp_dim=gnn_inp_dim*2+2,hidden_dim=gnn_inp_dim*gnn_inp_dim,out_dim=gnn_inp_dim*gnn_inp_dim,gpu=gpu,n_layer1=4,n_layer2=4)
                self.add_module('gen{}'.format(i), gen)
                atl=Atten_ker(gnn_inp_dim,1,lm=0,k_shot=self.k_shot)
                self.add_module('atl{}'.format(i), atl)
                module_l = NodeUpdateNetwork(inp_dim=gnn_inp_dim, out_dim=hidden_dim, n_layer=node_n_layer,
                                            batch_norm=batch_norm,
                                            dropout=dropout if i < self.num_layers - 1 else 0.0)
                self.add_module('node_layer{}'.format(i), module_l)
            elif self.mod==6:
                if self.lm:
                    lmt=nn.LayerNorm((k_shot,gnn_inp_dim))
                    self.add_module('lm{}'.format(i), lmt)
                atl=Atten(gnn_inp_dim,1,lm=0,k_shot=self.k_shot)
                modulator_w=HyperSetEncoder(inp_dim=gnn_inp_dim*2+2,hidden_dim=gnn_inp_dim*2,out_dim=gnn_inp_dim,gpu=gpu)
                self.add_module('atl{}'.format(i), atl)
                self.add_module('modw{}'.format(i), modulator_w)
                module_l = NodeUpdateNetwork(inp_dim=gnn_inp_dim, out_dim=hidden_dim, n_layer=node_n_layer,
                                            batch_norm=batch_norm,
                                            dropout=dropout if i < self.num_layers - 1 else 0.0)
                self.add_module('node_layer{}'.format(i), module_l)
            elif self.mod==7:
                if self.lm:
                    lmt=nn.LayerNorm((k_shot,gnn_inp_dim))
                    self.add_module('lm{}'.format(i), lmt)
                gen=HyperSetEncoder(inp_dim=gnn_inp_dim*2+2,hidden_dim=gnn_inp_dim*2,out_dim=gnn_inp_dim,gpu=gpu)
                self.em=nn.parameter.Parameter(torch.rand(gnn_inp_dim,gnn_inp_dim))
                self.add_module('gen{}'.format(i), gen)
                atl=Atten_ker(gnn_inp_dim,1,lm=0,k_shot=self.k_shot)
                self.add_module('atl{}'.format(i), atl)
                module_l = NodeUpdateNetwork(inp_dim=gnn_inp_dim, out_dim=hidden_dim, n_layer=node_n_layer,
                                            batch_norm=batch_norm,
                                            dropout=dropout if i < self.num_layers - 1 else 0.0)
                self.add_module('node_layer{}'.format(i), module_l)
            elif self.mod==8:
                #gen=HyperSetEncoder(inp_dim=gnn_inp_dim*2+2,hidden_dim=gnn_inp_dim*2,out_dim=self.c,gpu=gpu)
                gen=HyperSetEncoder2(inp_dim=gnn_inp_dim*2+2,hidden_dim=gnn_inp_dim,out_dim=self.c,gpu=gpu)
                self.add_module('gen{}'.format(i), gen)
                mcg=MultiCG(k_shot=self.k_shot,dim=gnn_inp_dim,c=self.c)
                self.add_module('mcg{}'.format(i), mcg)
                module_w = EdgeUpdateNetwork(in_features=gnn_inp_dim, hidden_features=hidden_dim, n_layer=edge_n_layer,
                                            top_k=top_k, batch_norm=batch_norm, adj_type=adj_type,
                                            activation=activation, dropout=dropout if i < self.num_layers - 1 else 0.0)
                module_l = NodeUpdateNetwork(inp_dim=gnn_inp_dim, out_dim=hidden_dim, n_layer=node_n_layer,
                                            batch_norm=batch_norm,
                                            dropout=dropout if i < self.num_layers - 1 else 0.0)
                self.add_module('edge_layer{}'.format(i), module_w)
                self.add_module('node_layer{}'.format(i), module_l)
            elif self.mod==9:
                #gen=HyperSetEncoder(inp_dim=gnn_inp_dim*2+2,hidden_dim=gnn_inp_dim*2,out_dim=self.c,gpu=gpu)
                gen=HyperSetEncoder2(inp_dim=gnn_inp_dim+2,hidden_dim=gnn_inp_dim,out_dim=gnn_inp_dim,gpu=gpu)
                self.add_module('modw{}'.format(0), gen)
                gen1=HyperSetEncoder2(inp_dim=gnn_inp_dim+2,hidden_dim=gnn_inp_dim,out_dim=gnn_inp_dim,gpu=gpu)
                self.add_module('modb{}'.format(0), gen1)
                module_w = EdgeUpdateNetwork(in_features=gnn_inp_dim, hidden_features=hidden_dim, n_layer=edge_n_layer,
                                            top_k=top_k, batch_norm=batch_norm, adj_type=adj_type,
                                            activation=activation, dropout=dropout if i < self.num_layers - 1 else 0.0)
                module_l = NodeUpdateNetwork(inp_dim=gnn_inp_dim, out_dim=hidden_dim, n_layer=node_n_layer,
                                            batch_norm=batch_norm,
                                            dropout=dropout if i < self.num_layers - 1 else 0.0)
                self.add_module('edge_layer{}'.format(i), module_w)
                self.add_module('node_layer{}'.format(i), module_l)
            elif self.mod==10:
                #gen=HyperSetEncoder(inp_dim=gnn_inp_dim*2+2,hidden_dim=gnn_inp_dim*2,out_dim=self.c,gpu=gpu)
                gen=HyperSetEncoder2(inp_dim=gnn_inp_dim+2,hidden_dim=gnn_inp_dim,out_dim=gnn_inp_dim,gpu=gpu)
                self.add_module('modw{}'.format(0), gen)
                gen1=HyperSetEncoder2(inp_dim=gnn_inp_dim+2,hidden_dim=gnn_inp_dim,out_dim=gnn_inp_dim,gpu=gpu)
                self.add_module('modb{}'.format(0), gen1)
                gen2=HyperSetEncoder2(inp_dim=gnn_inp_dim*2+2,hidden_dim=gnn_inp_dim,out_dim=self.c,gpu=gpu)
                self.add_module('gen{}'.format(i), gen2)
                mcg=MultiCG(k_shot=self.k_shot,dim=gnn_inp_dim,c=self.c)
                self.add_module('mcg{}'.format(i), mcg)
                module_w = EdgeUpdateNetwork(in_features=gnn_inp_dim, hidden_features=hidden_dim, n_layer=edge_n_layer,
                                            top_k=top_k, batch_norm=batch_norm, adj_type=adj_type,
                                            activation=activation, dropout=dropout if i < self.num_layers - 1 else 0.0)
                module_l = NodeUpdateNetwork(inp_dim=gnn_inp_dim, out_dim=hidden_dim, n_layer=node_n_layer,
                                            batch_norm=batch_norm,
                                            dropout=dropout if i < self.num_layers - 1 else 0.0)
                self.add_module('edge_layer{}'.format(i), module_w)
                self.add_module('node_layer{}'.format(i), module_l)
            elif self.mod==11:#share ctrl
                gen=HyperSetEncoder2(inp_dim=gnn_inp_dim+2,hidden_dim=gnn_inp_dim,out_dim=gnn_inp_dim,gpu=gpu)
                self.add_module('modw{}'.format(0), gen)
                gen1=HyperSetEncoder2(inp_dim=gnn_inp_dim+2,hidden_dim=gnn_inp_dim,out_dim=gnn_inp_dim,gpu=gpu)
                self.add_module('modb{}'.format(0), gen1)
                ctrl=Ctrl(gpu=gpu)
                self.add_module('ctrl{}'.format(0), ctrl)
                lc=LinearClassifier(gnn_inp_dim,inp_dim,num_class,self.pre_dropout)
                self.add_module('lc{}'.format(i), lc)
                module_w = EdgeUpdateNetwork(in_features=gnn_inp_dim, hidden_features=hidden_dim, n_layer=edge_n_layer,
                                            top_k=top_k, batch_norm=batch_norm, adj_type=adj_type,
                                            activation=activation, dropout=dropout if i < self.num_layers - 1 else 0.0)
                module_l = NodeUpdateNetwork(inp_dim=gnn_inp_dim, out_dim=hidden_dim, n_layer=node_n_layer,
                                            batch_norm=batch_norm,
                                            dropout=dropout if i < self.num_layers - 1 else 0.0)
                self.add_module('edge_layer{}'.format(i), module_w)
                self.add_module('node_layer{}'.format(i), module_l)
            elif self.mod==12:#individual ctrl
                gen=HyperSetEncoder2(inp_dim=gnn_inp_dim+2,hidden_dim=gnn_inp_dim,out_dim=gnn_inp_dim,gpu=gpu)
                self.add_module('modw{}'.format(0), gen)
                gen1=HyperSetEncoder2(inp_dim=gnn_inp_dim+2,hidden_dim=gnn_inp_dim,out_dim=gnn_inp_dim,gpu=gpu)
                self.add_module('modb{}'.format(0), gen1)
                ctrl=Ctrl(gpu=gpu)
                self.add_module('ctrl{}'.format(i), ctrl)
                lc=LinearClassifier(gnn_inp_dim,inp_dim,num_class,self.pre_dropout)
                self.add_module('lc{}'.format(i), lc)
                module_w = EdgeUpdateNetwork(in_features=gnn_inp_dim, hidden_features=hidden_dim, n_layer=edge_n_layer,
                                            top_k=top_k, batch_norm=batch_norm, adj_type=adj_type,
                                            activation=activation, dropout=dropout if i < self.num_layers - 1 else 0.0)
                module_l = NodeUpdateNetwork(inp_dim=gnn_inp_dim, out_dim=hidden_dim, n_layer=node_n_layer,
                                            batch_norm=batch_norm,
                                            dropout=dropout if i < self.num_layers - 1 else 0.0)
                self.add_module('edge_layer{}'.format(i), module_w)
                self.add_module('node_layer{}'.format(i), module_l)
            elif self.mod in [13,16,17,21,22,23,24,25,26]:#
                gen=HyperSetEncoder2(inp_dim=gnn_inp_dim+2,hidden_dim=gnn_inp_dim,out_dim=gnn_inp_dim,gpu=gpu)
                self.add_module('modw{}'.format(0), gen)
                gen1=HyperSetEncoder2(inp_dim=gnn_inp_dim+2,hidden_dim=gnn_inp_dim,out_dim=gnn_inp_dim,gpu=gpu)
                self.add_module('modb{}'.format(0), gen1)
                ctrl=Ctrl(gpu=gpu)
                self.add_module('ctrl{}'.format(0), ctrl)
                adc=ADC(inp_dim=gnn_inp_dim,hidden_dim=gnn_inp_dim)
                self.add_module('adc{}'.format(0), adc)
                module_w = EdgeUpdateNetwork(in_features=gnn_inp_dim, hidden_features=hidden_dim, n_layer=edge_n_layer,
                                            top_k=top_k, batch_norm=batch_norm, adj_type=adj_type,
                                            activation=activation, dropout=dropout if i < self.num_layers - 1 else 0.0)
                module_l = NodeUpdateNetwork(inp_dim=gnn_inp_dim, out_dim=hidden_dim, n_layer=node_n_layer,
                                            batch_norm=batch_norm,
                                            dropout=dropout if i < self.num_layers - 1 else 0.0)
                self.add_module('edge_layer{}'.format(i), module_w)
                self.add_module('node_layer{}'.format(i), module_l)
            elif self.mod==14:#individual ctrl
                gen=HyperSetEncoder2(inp_dim=gnn_inp_dim+2,hidden_dim=gnn_inp_dim,out_dim=gnn_inp_dim,gpu=gpu)
                self.add_module('modw{}'.format(0), gen)
                gen1=HyperSetEncoder2(inp_dim=gnn_inp_dim+2,hidden_dim=gnn_inp_dim,out_dim=gnn_inp_dim,gpu=gpu)
                self.add_module('modb{}'.format(0), gen1)
                ctrl=Ctrl(gpu=gpu)
                self.add_module('ctrl{}'.format(i), ctrl)
                adc=ADC(inp_dim=gnn_inp_dim,hidden_dim=gnn_inp_dim)
                self.add_module('adc{}'.format(0), adc)
                module_w = EdgeUpdateNetwork(in_features=gnn_inp_dim, hidden_features=hidden_dim, n_layer=edge_n_layer,
                                            top_k=top_k, batch_norm=batch_norm, adj_type=adj_type,
                                            activation=activation, dropout=dropout if i < self.num_layers - 1 else 0.0)
                module_l = NodeUpdateNetwork(inp_dim=gnn_inp_dim, out_dim=hidden_dim, n_layer=node_n_layer,
                                            batch_norm=batch_norm,
                                            dropout=dropout if i < self.num_layers - 1 else 0.0)
                self.add_module('edge_layer{}'.format(i), module_w)
                self.add_module('node_layer{}'.format(i), module_l)
            elif self.mod==15:#individual ctrl
                #gen=HyperSetEncoder2(inp_dim=gnn_inp_dim+2,hidden_dim=gnn_inp_dim,out_dim=gnn_inp_dim,gpu=gpu)
                #self.add_module('modw{}'.format(0), gen)
                #gen1=HyperSetEncoder2(inp_dim=gnn_inp_dim+2,hidden_dim=gnn_inp_dim,out_dim=gnn_inp_dim,gpu=gpu)
                #self.add_module('modb{}'.format(0), gen1)
                ctrl=Ctrl(gpu=gpu)
                self.add_module('ctrl{}'.format(0), ctrl)
                adc=ADC(inp_dim=gnn_inp_dim,hidden_dim=gnn_inp_dim)
                self.add_module('adc{}'.format(0), adc)
                module_w = EdgeUpdateNetwork(in_features=gnn_inp_dim, hidden_features=hidden_dim, n_layer=edge_n_layer,
                                            top_k=top_k, batch_norm=batch_norm, adj_type=adj_type,
                                            activation=activation, dropout=dropout if i < self.num_layers - 1 else 0.0)
                module_l = NodeUpdateNetwork(inp_dim=gnn_inp_dim, out_dim=hidden_dim, n_layer=node_n_layer,
                                            batch_norm=batch_norm,
                                            dropout=dropout if i < self.num_layers - 1 else 0.0)
                self.add_module('edge_layer{}'.format(i), module_w)
                self.add_module('node_layer{}'.format(i), module_l)
            elif self.mod==4:
                if self.lm:
                    lmt=nn.LayerNorm((k_shot,gnn_inp_dim))
                    self.add_module('lm{}'.format(i), lmt)
                atl=Atten(gnn_inp_dim,1,lm=0,k_shot=self.k_shot)
                self.add_module('atl{}'.format(i), atl)
                module_l = NodeUpdateNetwork(inp_dim=gnn_inp_dim, out_dim=hidden_dim, n_layer=node_n_layer,
                                            batch_norm=batch_norm,
                                            dropout=dropout if i < self.num_layers - 1 else 0.0)
                self.add_module('node_layer{}'.format(i), module_l)
            elif self.mod in [2,3]:
                if self.mod==3:
                    modulator_w=HyperSetEncoder(inp_dim=gnn_inp_dim*2+2,hidden_dim=gnn_inp_dim*2,out_dim=gnn_inp_dim,gpu=gpu)
                    modulator_b=HyperSetEncoder(inp_dim=gnn_inp_dim*2+2,hidden_dim=gnn_inp_dim*2,out_dim=gnn_inp_dim,gpu=gpu)
                    self.add_module('modw{}'.format(i), modulator_w)
                    self.add_module('modb{}'.format(i), modulator_b)
                if self.lm:
                    lmt=nn.LayerNorm((k_shot,gnn_inp_dim))
                    self.add_module('lm{}'.format(i), lmt)
                atl=Atten(gnn_inp_dim,1,lm=0,k_shot=self.k_shot)
                self.add_module('atl{}'.format(i), atl)
            elif self.mod in [0,1,18,20]:
                if self.mod==1:
                    if self.ar==1:
                        modulator_w=HyperSetEncoder(inp_dim=gnn_inp_dim*2+2,hidden_dim=gnn_inp_dim*2,out_dim=gnn_inp_dim,gpu=gpu)
                        modulator_b=HyperSetEncoder(inp_dim=gnn_inp_dim*2+2,hidden_dim=gnn_inp_dim*2,out_dim=gnn_inp_dim,gpu=gpu)
                    else:
                        modulator_w=HyperSetEncoder(inp_dim=gnn_inp_dim+2,hidden_dim=gnn_inp_dim*2,out_dim=gnn_inp_dim,gpu=gpu)
                        modulator_b=HyperSetEncoder(inp_dim=gnn_inp_dim+2,hidden_dim=gnn_inp_dim*2,out_dim=gnn_inp_dim,gpu=gpu)
                module_w = EdgeUpdateNetwork(in_features=gnn_inp_dim, hidden_features=hidden_dim, n_layer=edge_n_layer,
                                            top_k=top_k, batch_norm=batch_norm, adj_type=adj_type,
                                            activation=activation, dropout=dropout if i < self.num_layers - 1 else 0.0)
                module_l = NodeUpdateNetwork(inp_dim=gnn_inp_dim, out_dim=hidden_dim, n_layer=node_n_layer,
                                            batch_norm=batch_norm,
                                            dropout=dropout if i < self.num_layers - 1 else 0.0)
                self.add_module('edge_layer{}'.format(i), module_w)
                self.add_module('node_layer{}'.format(i), module_l)
                if self.mod==1:
                    self.add_module('modw{}'.format(i), modulator_w)
                    self.add_module('modb{}'.format(i), modulator_b)

            if self.node_concat:
                gnn_inp_dim = gnn_inp_dim + hidden_dim
            else:
                gnn_inp_dim = hidden_dim
        if self.adc==0:
            '''self.fc1 = nn.Sequential(nn.Linear(gnn_inp_dim, inp_dim), nn.LeakyReLU())
            if self.pre_dropout>0:
                self.predrop2 = nn.Dropout(p=self.pre_dropout)
            self.fc2 = nn.Linear(inp_dim, num_class)'''
            self.lc=LinearClassifier(gnn_inp_dim,inp_dim,num_class,self.pre_dropout)
        elif self.adc==1:
            pass
        elif self.adc==2:
            if self.ar==1:
                self.lastmod=HyperSetEncoder(inp_dim=gnn_inp_dim*2+2,hidden_dim=gnn_inp_dim*2,out_dim=gnn_inp_dim,gpu=gpu)
            else:
                self.lastmod=HyperSetEncoder(inp_dim=gnn_inp_dim+2,hidden_dim=gnn_inp_dim*2,out_dim=gnn_inp_dim,gpu=gpu)
        elif self.adc==4:
            self.adaptive_classifier=ADC(inp_dim=gnn_inp_dim,hidden_dim=gnn_inp_dim)
        assert 0 <= res_alpha <= 1

    def forward(self, all_emb, q_emb=None, return_adj=False, return_emb=False):
        b,n,d=all_emb.size()
        b=int(b)
        node_feat=all_emb
        nf0=all_emb[:]
        if self.pre_dropout>0:
            node_feat=self.predrop1(node_feat)
            
        edge_feat_list = []
        if return_adj:
            x_i = node_feat.unsqueeze(2)
            x_j = torch.transpose(x_i, 1, 2)
            init_adj = torch.abs(x_i - x_j)
            init_adj = torch.transpose(init_adj, 1, 3)  # size: bs x fs X N x N  (2,128,11,11)
            if self.adj_type == 'sim':
                init_adj = torch.exp(-init_adj)
            diag_mask = 1.0 - torch.eye(node_feat.size(1)).unsqueeze(0).unsqueeze(0).repeat(node_feat.size(0), 1, 1, 1).to(
                node_feat.device)
            init_adj = init_adj*diag_mask
            edge_feat_list.append(init_adj)
        if self.mod in [9,10,11,12,13,14]:
                nf=self._modules['modw{}'.format(0)].cat_label(node_feat)
                w=self._modules['modw{}'.format(0)](nf).unsqueeze(1)
                b=self._modules['modb{}'.format(0)](nf).unsqueeze(1)
                node_feat=node_feat*(1+w)+b
        if self.mod in [11,12,13,14,15,16,17,21,22,23,24,25,26]:
            self.plist=[]
            self.qlist=[]
        for i in range(self.num_layers):
            if self.mod==5:
                node_feat_new=node_feat[:]
                if self.lm:
                    node_feat_new=self.self._modules['lm{}'.format(i)](node_feat_new)
                nf=self._modules['gen{}'.format(i)].cat_label(torch.cat((nf0,node_feat_new),-1))
                m=self._modules['gen{}'.format(i)](nf).reshape(-1,node_feat.size(-1),node_feat.size(-1))
                adj = self._modules['atl{}'.format(i)](node_feat_new,m)  
                node_feat_new = self._modules['node_layer{}'.format(i)](node_feat_new, adj)
            elif self.mod==6:
                node_feat_new=node_feat[:]
                if self.lm:
                    node_feat_new=self.self._modules['lm{}'.format(i)](node_feat_new)
                nf=self._modules['modw{}'.format(i)].cat_label(torch.cat((nf0,node_feat_new),-1))
                w=self._modules['modw{}'.format(i)](nf).unsqueeze(1)
                node_feat_new=node_feat_new*w
                adj = self._modules['atl{}'.format(i)](node_feat_new,1)  
                node_feat_new = self._modules['node_layer{}'.format(i)](node_feat_new, adj)  
            elif self.mod==7:
                node_feat_new=node_feat[:]
                if self.lm:
                    node_feat_new=self.self._modules['lm{}'.format(i)](node_feat_new)
                nf=self._modules['gen{}'.format(i)].cat_label(torch.cat((nf0,node_feat_new),-1))
                em=self.em.repeat(b,1,1)
                iem=torch.inverse(self.em).repeat(b,1,1)
                ev=self._modules['gen{}'.format(i)](nf).reshape(-1,node_feat.size(-1),1)
                m=torch.bmm(iem,ev*em)
                adj = self._modules['atl{}'.format(i)](node_feat_new,m)  
                node_feat_new = self._modules['node_layer{}'.format(i)](node_feat_new, adj)
            elif self.mod==8 or self.mod==10:
                node_feat_new=node_feat[:]
                nf=self._modules['gen{}'.format(i)].cat_label(torch.cat((nf0,node_feat_new),-1))
                node_feat_new=self._modules['mcg{}'.format(i)](node_feat_new) 
                wei=self._modules['gen{}'.format(i)](nf)
                wei=torch.softmax(wei,-1)#bz,c
                adj, _ = self._modules['edge_layer{}'.format(i)](node_feat_new)   
                node_feat_new = self._modules['node_layer{}'.format(i)](node_feat_new, adj)#bz,c,n,e
                node_feat_new=node_feat_new.transpose(1,2)#bz,n,c,e
                wei=wei.unsqueeze(1).unsqueeze(-1).repeat(1,node_feat_new.size(1),1,1)
                node_feat_new*=wei
                node_feat_new=torch.sum(node_feat_new,2)
            elif self.mod==11:
                node_feat_new=node_feat[:]
                adj, _ = self._modules['edge_layer{}'.format(i)](node_feat_new)   
                node_feat_new = self._modules['node_layer{}'.format(i)](node_feat_new, adj)#bz,n,e
                sp=self._modules['ctrl{}'.format(0)](node_feat_new)   
                self.plist.append(sp)
                _,tq=self._modules['lc{}'.format(i)](node_feat_new)
                tq=tq.unsqueeze(1)#16,1,2
                if i==0:
                    self.qlist=tq
                else:
                    self.qlist=torch.cat((self.qlist,tq),1)
                if i==self.num_layers-1:
                    
                    plist=torch.cat(self.plist,-1)#16,nl
                    #print(self.qlist.size())16,nl,2
                    return plist,self.qlist,0
            elif self.mod==12:
                node_feat_new=node_feat[:]
                adj, _ = self._modules['edge_layer{}'.format(i)](node_feat_new)   
                node_feat_new = self._modules['node_layer{}'.format(i)](node_feat_new, adj)#bz,n,e
                sp=self._modules['ctrl{}'.format(i)](node_feat_new)   
                self.plist.append(sp)
                _,tq=self._modules['lc{}'.format(i)](node_feat_new)
                tq=tq.unsqueeze(1)#16,1,2
                if i==0:
                    self.qlist=tq
                else:
                    self.qlist=torch.cat((self.qlist,tq),1)
                if i==self.num_layers-1:
                    
                    plist=torch.cat(self.plist,-1)#16,nl
                    #print(self.qlist.size())16,nl,2
                    return plist,self.qlist,0
            elif self.mod in [13,16,17,21,22,23,24,25,26]:
                node_feat_new=node_feat[:]
                adj, _ = self._modules['edge_layer{}'.format(i)](node_feat_new)   
                node_feat_new = self._modules['node_layer{}'.format(i)](node_feat_new, adj)#bz,n,e
                sp=self._modules['ctrl{}'.format(0)](node_feat_new)   
                self.plist.append(sp)
                logits=self._modules['adc{}'.format(0)](node_feat_new)
                s_logits = logits[:,:-1,:]
                tq = logits[:,-1,:]
                tq=tq.unsqueeze(1)#16,1,2
                if i==0:
                    self.qlist=tq
                else:
                    self.qlist=torch.cat((self.qlist,tq),1)
                if i==3:
                    s_feat = torch.mean(node_feat_new[:, :-1, :],0)
                    q_feat = node_feat_new[:, -1, :]
                if i==self.num_layers-1:
                    
                    plist=torch.cat(self.plist,-1)#16,nl
                    #print(self.qlist.size())16,nl,2
                    if return_emb:
                        return plist,self.qlist,0,s_feat, q_feat
                    else:
                        return plist,self.qlist,0
            elif self.mod==14:
                node_feat_new=node_feat[:]
                adj, _ = self._modules['edge_layer{}'.format(i)](node_feat_new)   
                node_feat_new = self._modules['node_layer{}'.format(i)](node_feat_new, adj)#bz,n,e
                sp=self._modules['ctrl{}'.format(i)](node_feat_new)   
                self.plist.append(sp)
                logits=self._modules['adc{}'.format(0)](node_feat_new)
                s_logits = logits[:,:-1,:]
                tq = logits[:,-1,:]
                tq=tq.unsqueeze(1)#16,1,2
                if i==0:
                    self.qlist=tq
                else:
                    self.qlist=torch.cat((self.qlist,tq),1)
                if i==self.num_layers-1:
                    
                    plist=torch.cat(self.plist,-1)#16,nl
                    #print(self.qlist.size())16,nl,2
                    return plist,self.qlist,0
            elif self.mod==15:
                node_feat_new=node_feat[:]
                adj, _ = self._modules['edge_layer{}'.format(i)](node_feat_new)   
                node_feat_new = self._modules['node_layer{}'.format(i)](node_feat_new, adj)#bz,n,e
                sp=self._modules['ctrl{}'.format(0)](node_feat_new)   
                self.plist.append(sp)
                logits=self._modules['adc{}'.format(0)](node_feat_new)
                s_logits = logits[:,:-1,:]
                tq = logits[:,-1,:]
                tq=tq.unsqueeze(1)#16,1,2
                if i==0:
                    self.qlist=tq
                else:
                    self.qlist=torch.cat((self.qlist,tq),1)
                if i==self.num_layers-1:
                    
                    plist=torch.cat(self.plist,-1)#16,nl
                    #print(self.qlist.size())16,nl,2

                    if return_emb:
                        return plist,self.qlist,0,s_feat, q_feat
                    else:
                        return plist,self.qlist,0
            elif self.mod==4:
                node_feat_new=node_feat[:]
                if self.lm:
                    node_feat_new=self.self._modules['lm{}'.format(i)](node_feat_new)
                adj = self._modules['atl{}'.format(i)](node_feat_new,1)  
                node_feat_new = self._modules['node_layer{}'.format(i)](node_feat_new, adj)    
            elif self.mod in [2,3]:
                node_feat_new=node_feat[:]
                if self.lm:
                    node_feat_new=self.self._modules['lm{}'.format(i)](node_feat_new)
                if self.mod==3:
                    nf=self._modules['modw{}'.format(i)].cat_label(torch.cat((nf0,node_feat_new),-1))
                    w=self._modules['modw{}'.format(i)](nf).unsqueeze(1)
                    b=self._modules['modb{}'.format(i)](nf).unsqueeze(1)
                    node_feat_new=node_feat_new*w+b
                node_feat_new = self._modules['atl{}'.format(i)](node_feat_new)
                adj=1
            else:
                if self.mod==1:
                    if self.ar==1:
                        nf=self._modules['modw{}'.format(i)].cat_label(torch.cat((nf0,node_feat),-1))
                    else:
                        nf=self._modules['modw{}'.format(i)].cat_label(node_feat)
                    w=self._modules['modw{}'.format(i)](nf).unsqueeze(1)
                    b=self._modules['modb{}'.format(i)](nf).unsqueeze(1)
                    node_feat=node_feat*w+b

                adj, _ = self._modules['edge_layer{}'.format(i)](node_feat)   
                node_feat_new = self._modules['node_layer{}'.format(i)](node_feat, adj)
            if self.node_concat:
                node_feat = torch.cat([node_feat, node_feat_new], 2)
            else:
                node_feat = node_feat_new
            
            edge_feat_list.append(adj)
        
        if self.adc==0:
            '''if self.pre_dropout>0:
                node_feat=self.predrop2(node_feat)
            node_feat = self.fc1(node_feat)
            node_feat = self.res_alpha * all_emb +  node_feat

            s_feat = node_feat[:, :-1, :]
            q_feat = node_feat[:, -1, :]

            s_logits = self.fc2(s_feat)
            q_logits = self.fc2(q_feat)'''
            s_logits,q_logits=self.lc(node_feat)

        elif self.adc==1:

            ks=int((node_feat.size(1)-1)/2)
            proto0=torch.mean(node_feat[:,:ks,:],1).unsqueeze(1)#21,1,128
            proto1=torch.mean(node_feat[:,ks:-1,:],1).unsqueeze(1)
            '''s0= torch.sum(torch.abs(node_feat[:, :-1, :]-proto0.unsqueeze(1)),-1).unsqueeze(-1)
            s1= torch.sum(torch.abs(node_feat[:, :-1, :]-proto1.unsqueeze(1)),-1).unsqueeze(-1)'''
            fx=node_feat[:, -1, :].unsqueeze(1)#21,1,128
            q0=2*torch.sum(fx*proto0,-1)-torch.sum(proto0*proto0,-1)
            q1=2*torch.sum(fx*proto1,-1)-torch.sum(proto1*proto1,-1)
            q_logits=torch.cat((q0,q1),-1)
            s_logits=0
            #s_logits=-torch.cat((s0,s1),-1)
            #q_logits=-torch.cat((q0,q1),-1)
            

        elif self.adc==2:
            if self.ar==1:
                nf=self.lastmod.cat_label(torch.cat((nf0,node_feat),-1))
            else:
                nf=self.lastmod.cat_label(node_feat)
            w=self.lastmod(nf).unsqueeze(1)
            node_feat=node_feat*w
            ks=int((node_feat.size(1)-1)/2)
            proto0=torch.mean(node_feat[:,:ks,:],1)
            proto1=torch.mean(node_feat[:,ks:-1,:],1)
            s0= torch.sum(torch.square(node_feat[:, :-1, :]-proto0.unsqueeze(1)),-1).unsqueeze(-1)
            s1= torch.sum(torch.square(node_feat[:, :-1, :]-proto1.unsqueeze(1)),-1).unsqueeze(-1)
            q0= torch.sum(torch.square(node_feat[:, -1, :]-proto0),-1).unsqueeze(-1)
            q1= torch.sum(torch.square(node_feat[:, -1, :]-proto1),-1).unsqueeze(-1)
            s_logits=-torch.cat((s0,s1),-1)
            q_logits=-torch.cat((q0,q1),-1)
        elif self.adc==3:
            ks=int((node_feat.size(1)-1)/2)
            u=torch.mean(torch.mean(node_feat[:,:-1,:],1),0)#128
            u0=torch.mean(torch.mean(node_feat[:,:ks,:],1),0)
            u1=torch.mean(torch.mean(node_feat[:,ks:-1,:],1),0)

            t=node_feat[0,:-1,:].unsqueeze(-1)#20,128,1
            t=t-u.unsqueeze(0).unsqueeze(-1)
            sigma=torch.sum(torch.bmm(t,t.transpose(1,2)),0)/(2*ks-1)

            t=node_feat[0,:ks,:].unsqueeze(-1)#10,128,1
            t=t-u0.unsqueeze(0).unsqueeze(-1)
            sigma0=torch.sum(torch.bmm(t,t.transpose(1,2)),0)

            t=node_feat[0,ks:-1,:].unsqueeze(-1)#10,128,1
            t=t-u1.unsqueeze(0).unsqueeze(-1)
            sigma1=torch.sum(torch.bmm(t,t.transpose(1,2)),0)

            if ks>1:
                a=ks/(ks+1)
                sigma0=a*sigma0/(ks-1)+(1-a)*sigma#+torch.eye(128).to(self.gpu)
                sigma1=a*sigma1/(ks-1)+(1-a)*sigma#+torch.eye(128).to(self.gpu)
            else:
                sigma0=sigma#+torch.eye(128).to(self.gpu)
                sigma1=sigma#+torch.eye(128).to(self.gpu)
            sigma0=torch.inverse(sigma0).unsqueeze(0).repeat(node_feat.size(0),1,1)#16,128,128
            sigma1=torch.inverse(sigma1).unsqueeze(0).repeat(node_feat.size(0),1,1)
            xi=node_feat[:,-1,:].unsqueeze(1)#16,1,128
            u0=u0.unsqueeze(0).unsqueeze(0).repeat(xi.size(0),1,1)
            t=xi-u0
            q0=0.5*torch.matmul(torch.bmm(t,sigma0),t.transpose(1,2)).squeeze(-1)
            t=xi-u1
            q1=0.5*torch.matmul(torch.bmm(t,sigma1),t.transpose(1,2)).squeeze(-1)
    
            s_logits=0
            q_logits=-torch.cat((q0,q1),-1)
        elif self.adc==4:
            
            node_feat = self.res_alpha * all_emb +  node_feat

            logits=self.adaptive_classifier(node_feat)

            s_logits = logits[:,:-1,:]
            q_logits = logits[:,-1,:]
        print(return_emb)
        if return_emb:
            return s_logits, q_logits, edge_feat_list, s_feat, q_feat
        else:
            return s_logits, q_logits, edge_feat_list
