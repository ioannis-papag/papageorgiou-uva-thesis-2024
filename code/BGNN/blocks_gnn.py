import numpy as np
import torch
from torch import nn
from object_encoder import ObjectEncoder
from matplotlib import pyplot as plt

#from BlocksCNN import BlocksCNN


def unsorted_segment_sum(tensor, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, tensor.size(1))
    result = tensor.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, tensor.size(1))
    result.scatter_add_(0, segment_ids, tensor)
    return result


class EncoderMLP(nn.Module):
    """MLP encoder, maps observation to latent state."""
    
    def __init__(self, input_dim, output_dim, hidden_dim, num_objects):
        super(EncoderMLP, self).__init__()

        self.num_objects = num_objects
        self.input_dim = input_dim
        self.act_fn = nn.ReLU()

        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, ins):
        #h_flat = ins.permute(0, 3, 1, 2).view(-1, self.num_objects, self.input_dim)
        h_flat = ins.view(-1, self.num_objects, self.input_dim)
        #h_flat = ins.permute(0, 3, 1, 2).reshape(-1, self.num_objects, self.input_dim)
        h = self.act_fn(self.fc1(h_flat))
        h = self.fc2(h)
        h = self.act_fn(self.ln(h))
        h = self.fc3(h)

        return h

class BlocksGNN(torch.nn.Module):
    """GNN-based transition function."""
    def __init__(self, input_dim=512, hidden_dim=512, num_objects=7):
        super(BlocksGNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_objects = num_objects


        self.edge_mlp = nn.Sequential(
            nn.Linear(input_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))

        node_input_dim = input_dim + hidden_dim

        self.node_mlp = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))
        
        self.final_node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 3))
        
        self.final_edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )

        self.edge_list = None
        self.batch_size = 0

    def _edge_model(self, source, target):

        out = torch.cat([source, target], dim=1)
        return self.edge_mlp(out)

    def _node_model(self, node_attr, edge_index, edge_attr):
        if edge_attr is not None:
            row, _ = edge_index
            agg = unsorted_segment_sum(edge_attr, row, num_segments=node_attr.size(0))
            out = torch.cat([node_attr, agg], dim=1)
        else:
            out = node_attr

        return self.node_mlp(out)
    

    def _get_edge_list_fully_connected(self, batch_size, num_objects):
        # Only re-evaluate if necessary (e.g. if batch size changed).
        if self.edge_list is None or self.batch_size != batch_size:
            self.batch_size = batch_size

            # Create fully-connected adjacency matrix for single sample.
            adj_full = torch.ones(num_objects, num_objects)
            #adj_full -= torch.eye(num_objects)
            self.edge_list = adj_full.nonzero()

            # Remove diagonal.

            # Copy `batch_size` times and add offset.
            self.edge_list = self.edge_list.repeat(batch_size, 1)
            offset = torch.arange(0, batch_size * num_objects, num_objects).unsqueeze(-1)
            offset = offset.expand(batch_size, num_objects * (num_objects-1+1))
            offset = offset.contiguous().view(-1)
            self.edge_list += offset.unsqueeze(-1)

            # Transpose to COO format -> Shape: [2, num_edges].
            self.edge_list = self.edge_list.transpose(0, 1)
            self.edge_list = self.edge_list.cuda()


        return self.edge_list

    def forward(self, states):

        batch_size = states.size(0)
        num_nodes = states.size(1)

        # states: [batch_size (B), num_objects, embedding_dim]
        # node_attr: Flatten states tensor to [B * num_objects, embedding_dim]
        node_attr = states.view(-1, self.input_dim)

        edge_attr = None
        edge_index = None

        if num_nodes > 1:
            # edge_index: [B * (num_objects*[num_objects-1]), 2] edge list
            edge_index = self._get_edge_list_fully_connected(batch_size, num_nodes)

            row, col = edge_index
            edge_attr = self._edge_model(node_attr[row], node_attr[col])


        node_attr = self._node_model(node_attr, edge_index, edge_attr)

        edge_attr = self._edge_model(node_attr[row], node_attr[col])

        #--Second Round of Message passing
        #node_attr = self._node_model(node_attr, edge_index, edge_attr)
        #edge_attr = self._edge_model(node_attr[row], node_attr[col])
        #--Second Round of Message passing


        node_output = self.final_node_mlp(node_attr)
        edge_output = self.final_edge_mlp(edge_attr)

        node_output = node_output.view(batch_size, num_nodes, 3).view(batch_size, -1)
        edge_output = edge_output.view(batch_size, num_nodes, num_nodes).view(batch_size, -1)
        # [batch_size, num_nodes, hidden_dim]
        #x = node_attr.view(batch_size, num_nodes, -1)
    
        #x = x.view(x.size(0), -1)

        #result = self.final_mlp(x)

        #return result
        #return node_attr.view(batch_size, num_nodes, -1).view(batch_size, -1)
        return torch.cat((node_output, edge_output), dim=1)
    
class BlocksCNN(nn.Module):
    """CNN encoder, maps observation to obj-specific feature maps."""

    # def __init__(self, input_dim, hidden_dim, num_objects):
    #     super(BlocksCNN, self).__init__()
    #     self.cnn1 = nn.Conv2d(
    #         input_dim, hidden_dim, (10, 10), stride=10)
    #     self.cnn2 = nn.Conv2d(hidden_dim, num_objects, (1, 1), stride=1)
    #     self.ln1 = nn.BatchNorm2d(hidden_dim)
    #     self.act1 = nn.ReLU()
    #     self.act2 = nn.Sigmoid()

    # def forward(self, obs):
    #     h = self.act1(self.ln1(self.cnn1(obs)))
    #     h = self.act2(self.cnn2(h))
    #     x = h.cpu()
    #     # for i in range(7):
    #     #     plt.imshow(x[0, i, :, :].numpy(), interpolation='nearest')
    #     #     plt.show()
    #     return h

    # def __init__(self, input_dim, hidden_dim, num_objects):
    #     super(BlocksCNN, self).__init__()

    #     self.cnn1 = nn.Conv2d(input_dim, hidden_dim, (9, 9), padding=4)
    #     self.act1 = nn.LeakyReLU()
    #     self.ln1 = nn.BatchNorm2d(hidden_dim)

    #     self.cnn2 = nn.Conv2d(hidden_dim, num_objects, (5, 5), stride=5)
    #     self.act2 = nn.Sigmoid()

    # def forward(self, obs):
    #     h = self.act1(self.ln1(self.cnn1(obs)))
    #     h = self.act2(self.cnn2(h))
    #     return h
    
    
    def __init__(self, input_dim, hidden_dim, num_objects):
        super(BlocksCNN, self).__init__()

        self.cnn1 = nn.Conv2d(input_dim, hidden_dim, (3, 3), padding=1)
        self.act1 = nn.ReLU()
        self.ln1 = nn.BatchNorm2d(hidden_dim)

        self.cnn2 = nn.Conv2d(hidden_dim, hidden_dim, (3, 3), padding=1)
        self.act2 = nn.ReLU()
        self.ln2 = nn.BatchNorm2d(hidden_dim)

        # self.cnn3 = nn.Conv2d(hidden_dim, hidden_dim, (3, 3), padding=1)
        # self.act3 = nn.ReLU()
        # self.ln3 = nn.BatchNorm2d(hidden_dim)

        self.cnn4 = nn.Conv2d(hidden_dim, num_objects, (3, 3), padding=1)
        self.act4 = nn.Sigmoid()

    def forward(self, obs):
        h = self.act1(self.ln1(self.cnn1(obs)))
        h = self.act2(self.ln2(self.cnn2(h)))
        # h = self.act3(self.ln3(self.cnn3(h)))
        h = self.act4(self.cnn4(h))
        #h = (h >= 0.5).to(dtype=torch.float)
        # for i in range(7):
        #     plt.imshow(h[0, i, :, :].numpy(), interpolation='nearest')
        #     plt.show()
        
        return h

class FullPipelineModel(torch.nn.Module):
    """Main module for a Contrastively-trained Structured World Model (C-SWM).

    Args:
        embedding_dim: Dimensionality of abstract state space.
        input_dims: Shape of input observation.
        hidden_dim: Number of hidden units in encoder and transition model.
        action_dim: Dimensionality of action space.
        num_objects: Number of object slots.
    """
    def __init__(self, embedding_dim, input_dims, hidden_dim, num_objects):
        super(FullPipelineModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_objects = num_objects
        self.input_dims = input_dims

        self.num_channels = input_dims[0]
        width_height = input_dims[1:]

        self.obj_detector = BlocksCNN(input_dim = self.num_channels, hidden_dim = self.hidden_dim//16, num_objects=self.num_objects)
        #self.obj_detector = BlocksCNN()

        width_height = np.array(width_height)
        #width_height = 48*48

        #self.obj_encoder = EncoderMLP(input_dim = np.prod(width_height), output_dim=512, hidden_dim=512, num_objects=7)
        self.obj_encoder = EncoderMLP(input_dim = 480*480, output_dim=512, hidden_dim=512, num_objects=7)
        self.gnn = BlocksGNN()



    def forward(self, obs):
        return self.gnn(self.obj_encoder(self.obj_detector(obs)))
    