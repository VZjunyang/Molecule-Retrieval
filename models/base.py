from torch import nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, model_name, num_node_features, nout, nhid, graph_hidden_channels):
        self.graph_encoder = None
        self.text_encoder = None
        
    def forward(self, graph_batch, input_ids, attention_mask):
        return None, None
    
    def get_text_encoder(self):
        return self.text_encoder
    
    def get_graph_encoder(self):
        return self.graph_encoder