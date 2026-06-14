
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchinfo import summary

class MLP(nn.Module):
    """
    Multi-Layer Perceptron: Simple 3-layer network with same depth as others
    Architecture: input -> 1024 -> 512 -> 256 -> output
    Features: Basic ReLU activation, moderate dropout, NO batch norm, NO residual
    """
    def __init__(self, input_dim, hidden_layer_list=None, num_class=2):
        super(MLP, self).__init__()
        # Allow small/light configs via hidden_layer_list (len 3). Default small: [512, 64, 16].
        hl = hidden_layer_list or [512, 64, 16]
        hl = (hl + [512, 64, 16])[:3]
        self.fc1 = nn.Linear(input_dim, hl[0])
        self.fc2 = nn.Linear(hl[0], hl[1])
        self.fc3 = nn.Linear(hl[1], hl[2])
        self.fc4 = nn.Linear(hl[2], num_class)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)  # Lower dropout - simplest regularization
        self.num_class = num_class

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.fc4(x)
        return x  # Raw logits only
    

class RNN(nn.Module):
    def __init__(self, input_dim, num_class, hidden_layer_list=None, sequence_length=None):
        super(RNN, self).__init__()
        hidden_layer_list = hidden_layer_list or []
        # Small defaults: [rnn_hidden, fc_hidden]
        self.hidden_size = hidden_layer_list[0] if len(hidden_layer_list) >= 1 else 64
        self.num_layers = 2
        self.rnn = nn.RNN(
            input_dim,
            self.hidden_size,
            self.num_layers,
            batch_first=True,
            dropout=0.3 if self.num_layers > 1 else 0.0
        )
        fc_hidden = hidden_layer_list[1] if len(hidden_layer_list) >= 2 else 16
        self.fc1 = nn.Linear(self.hidden_size, fc_hidden)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(fc_hidden, num_class)
        self.num_class = num_class

    def forward(self, x):
        # Handle both 2D and 3D input
        if x.dim() == 2:
            # Convert 2D to 3D by adding sequence dimension
            x = x.unsqueeze(1)  # [batch, 1, features]
        elif x.dim() == 3 and x.size(2) != self.rnn.input_size:
            x = x.transpose(1, 2)
        
        rnn_out, _ = self.rnn(x)
        last_output = rnn_out[:, -1, :]
        x = F.relu(self.fc1(last_output))
        x = self.dropout(x)
        x = self.fc2(x)
        return x  # Raw logits only

class LSTM(nn.Module):
    def __init__(self, input_dim, num_class, hidden_layer_list=None, sequence_length=None):
        super(LSTM, self).__init__()
        hidden_layer_list = hidden_layer_list or []
        # Small defaults: [lstm_hidden, fc_hidden]
        self.hidden_size = hidden_layer_list[0] if len(hidden_layer_list) >= 1 else 64
        self.num_layers = 2
        self.lstm = nn.LSTM(
            input_dim,
            self.hidden_size,
            self.num_layers,
            batch_first=True,
            dropout=0.3 if self.num_layers > 1 else 0.0
        )
        fc_hidden = hidden_layer_list[1] if len(hidden_layer_list) >= 2 else 16
        self.fc1 = nn.Linear(self.hidden_size, fc_hidden)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(fc_hidden, num_class)
        self.num_class = num_class

    def forward(self, x):
        # Handle both 2D and 3D input
        if x.dim() == 2:
            # Convert 2D to 3D by adding sequence dimension
            x = x.unsqueeze(1)  # [batch, 1, features]
        elif x.dim() == 3 and x.size(2) != self.lstm.input_size:
            x = x.transpose(1, 2)
            
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        x = F.relu(self.fc1(last_output))
        x = self.dropout(x)
        x = self.fc2(x)
        return x  # Raw logits only

class GRU(nn.Module):
    def __init__(self, input_dim, num_class, hidden_layer_list=None, sequence_length=None):
        super(GRU, self).__init__()
        hidden_layer_list = hidden_layer_list or []
        # Small defaults: [gru_hidden, fc_hidden]
        self.hidden_size = hidden_layer_list[0] if len(hidden_layer_list) >= 1 else 64
        self.num_layers = 2
        self.gru = nn.GRU(
            input_dim,
            self.hidden_size,
            self.num_layers,
            batch_first=True,
            dropout=0.3 if self.num_layers > 1 else 0.0
        )
        fc_hidden = hidden_layer_list[1] if len(hidden_layer_list) >= 2 else 16
        self.fc1 = nn.Linear(self.hidden_size, fc_hidden)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(fc_hidden, num_class)
        self.num_class = num_class

    def forward(self, x):
        # Handle both 2D and 3D input
        if x.dim() == 2:
            # Convert 2D to 3D by adding sequence dimension
            x = x.unsqueeze(1)  # [batch, 1, features]
        elif x.dim() == 3 and x.size(2) != self.gru.input_size:
            x = x.transpose(1, 2)
            
        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]
        x = F.relu(self.fc1(last_output))
        x = self.dropout(x)
        x = self.fc2(x)
        return x  # Raw logits only

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
