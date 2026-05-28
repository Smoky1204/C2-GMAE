import torch
import torch.nn as nn
import torch.nn.functional as F

from pos_enc import PEG
from utils import get_activation
from conv import GATConv


class Model(torch.nn.Module):
    def __init__(self, encoder, out_dim, args=None):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(args.embed_dim, out_dim)
    
    def forward(self, x, edge_index, eigvals=None, eigvecs=None, mask_tokens=None, PE=None, pe_bias=None):
        h, pe = self.encoder.embed(x, edge_index, PE, pe_bias=pe_bias)
        pred = self.classifier(h)
        return pred, pe


class GraphEncoder(torch.nn.Module):
    def __init__(self, out_dim, args=None):
        super().__init__()
        self.gnn_dropout = args.gnn_dropout

        self.num_layer = args.enc_gnn_layer
        emb_dim = args.embed_dim
        hid_dim = emb_dim // args.heads

        self.gnns = nn.ModuleList()
        if self.num_layer == 1:
            self.gnns.append(GATConv(in_channels=args.feat_dim, out_channels=out_dim, heads=args.heads, args=args))
        else:
            self.gnns.append(GATConv(in_channels=args.feat_dim, out_channels=hid_dim, heads=args.heads, args=args))
            self.activations = nn.ModuleList()
            for layer in range(self.num_layer - 1):
                self.gnns.append(GATConv(in_channels=emb_dim, out_channels=hid_dim, heads=args.heads, args=args))        
                self.activations.append(get_activation(args.gnn_activation))

        self.pe_enc = PEG(args=args)
        

    def forward(self, x, x_masked, edge_index, PE=None, PE_noise=None, pe_bias=None):

        pe = self.pe_enc(PE)  
        pe_noise = self.pe_enc(PE_noise)
        if pe_bias is not None:
            pe = pe + pe_bias.unsqueeze(1)
            pe_noise = pe_noise + pe_bias.unsqueeze(1)

        h_list = [x]
        h_masked_list = [x_masked]
        pe_list = [pe]
        pe_noise_list = [pe_noise]
        for layer in range(self.num_layer):
            h = F.dropout(h_list[layer], p=self.gnn_dropout, training=self.training)
            h, pe_noise = self.gnns[layer](h, pe_noise_list[layer], edge_index)

            h_masked = F.dropout(h_masked_list[layer], p=self.gnn_dropout, training=self.training)
            h_masked, pe = self.gnns[layer](h_masked, pe_list[layer], edge_index)
            if layer != self.num_layer - 1:
                h = self.activations[layer](h)
                h_masked = self.activations[layer](h_masked)
            
            h_masked_list.append(h_masked)
            h_list.append(h)
            pe_noise_list.append(pe_noise)
            pe_list.append(pe)
        
        return h_masked_list[-1], pe_noise_list[-1]

    def embed(self, x, edge_index, PE=None, pe_bias=None):
        pe = self.pe_enc(PE)
        if pe_bias is not None:
            pe = pe + pe_bias.unsqueeze(1)

        h_list = [x]
        pe_list = [pe]
        for layer in range(self.num_layer):
            h = F.dropout(h_list[layer], p=self.gnn_dropout, training=self.training)
            h, pe = self.gnns[layer](h, pe_list[layer], edge_index)

            if layer != self.num_layer - 1:
                h = self.activations[layer](h)
            
            h_list.append(h)
            pe_list.append(pe)
        
        return h_list[-1], pe_list[-1]
