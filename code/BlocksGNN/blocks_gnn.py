import torch
import torch.nn as nn
import torch.nn.functional as F

class MessagePassingGNN(nn.Module):
    def __init__(self):
        super(MessagePassingGNN, self).__init__()
        self.node_update_mlp = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

        self.node_task_mlp = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        self.edge_task_mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    
    def forward(self, node_features):
        batch_size, num_nodes, _ = node_features.shape
        
        aggregated_messages = torch.mean(node_features, dim=1, keepdim=True).expand(-1, num_nodes, -1)

        updated_node_features = self.node_update_mlp(aggregated_messages.reshape(-1, 512)).reshape(batch_size, num_nodes, -1)

        updated_node_features = node_features + updated_node_features
        node_task_out = self.node_task_mlp(updated_node_features.reshape(-1, 512))
        node_task_out = torch.sigmoid(node_task_out).reshape(batch_size, num_nodes * 3) 

        edge_task_out = []
        for b in range(batch_size):
            for i in range(num_nodes):
                for j in range(num_nodes):
                    edge_feature = torch.cat((updated_node_features[b, i], updated_node_features[b, j]), dim=0)
                    edge_task_out.append(self.edge_task_mlp(edge_feature))
        
        edge_task_out = torch.sigmoid(torch.stack(edge_task_out)).reshape(batch_size, num_nodes * num_nodes)
        
        return node_task_out, edge_task_out


'''model = MessagePassingGNN()
batch_node_features = torch.randn(10, 7, 512)  # Simulated batched input
node_outputs, edge_outputs = model(batch_node_features)

print("Node Outputs Shape:", node_outputs.shape)  # Should be [batch_size, 21]
print("Edge Outputs Shape:", edge_outputs.shape)  # Should be [batch_size, 49]'''