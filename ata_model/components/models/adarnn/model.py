"""
AdaRNN: Adaptive Learning and Forecasting for Time Series
Adapted for IoT Attack Detection (Classification)

Original paper: https://arxiv.org/abs/2108.04443 (CIKM 2021)

Key innovations:
1. TDC (Temporal Distribution Characterization) - discovers domains in time series
2. TDM (Temporal Distribution Matching) - aligns distributions across time periods
3. Temporal importance weighting - learns which timesteps matter most

This implementation adapts AdaRNN for:
- Classification (attack detection) instead of regression
- IoT network traffic data
- Binary and multi-class classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from .loss_transfer import TransferLoss


class AdaRNN(nn.Module):
    """
    AdaRNN: Adaptive RNN with Temporal Distribution Matching
    
    This model learns domain-invariant representations by:
    1. Using GRU/LSTM to encode temporal features
    2. Learning importance weights for each timestep
    3. Aligning hidden state distributions across time periods
    
    Args:
        n_input: Input feature dimension
        n_hiddens: List of hidden sizes for each RNN layer
        n_output: Output dimension (number of classes)
        dropout: Dropout rate
        len_seq: Sequence length
        model_type: 'AdaRNN' or 'Boosting'
        trans_loss: Transfer loss type ('mmd', 'coral', 'adv', 'cosine')
        use_bottleneck: Whether to use bottleneck layer
        bottleneck_width: Bottleneck layer width
        rnn_type: 'gru' or 'lstm'
    """
    
    def __init__(
        self,
        n_input: int = 27,
        n_hiddens: List[int] = [64, 64],
        n_output: int = 2,
        dropout: float = 0.0,
        len_seq: int = 1,
        model_type: str = 'AdaRNN',
        trans_loss: str = 'mmd',
        use_bottleneck: bool = True,
        bottleneck_width: int = 64,
        rnn_type: str = 'gru'
    ):
        super(AdaRNN, self).__init__()
        
        self.n_input = n_input
        self.num_layers = len(n_hiddens)
        self.hiddens = n_hiddens
        self.n_output = n_output
        self.model_type = model_type
        self.trans_loss = trans_loss
        self.len_seq = len_seq
        self.use_bottleneck = use_bottleneck
        self.rnn_type = rnn_type
        
        # Build RNN layers
        in_size = self.n_input
        features = nn.ModuleList()
        
        for hidden in n_hiddens:
            if rnn_type == 'gru':
                rnn = nn.GRU(
                    input_size=in_size,
                    num_layers=1,
                    hidden_size=hidden,
                    batch_first=True,
                    dropout=dropout
                )
            else:  # LSTM
                rnn = nn.LSTM(
                    input_size=in_size,
                    num_layers=1,
                    hidden_size=hidden,
                    batch_first=True,
                    dropout=dropout
                )
            features.append(rnn)
            in_size = hidden
        
        self.features = nn.Sequential(*features)
        
        # Bottleneck and output layers
        if use_bottleneck:
            self.bottleneck = nn.Sequential(
                nn.Linear(n_hiddens[-1], bottleneck_width),
                nn.BatchNorm1d(bottleneck_width),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(bottleneck_width, bottleneck_width),
                nn.BatchNorm1d(bottleneck_width),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self._init_bottleneck()
            self.fc = nn.Linear(bottleneck_width, n_output)
        else:
            self.fc = nn.Linear(n_hiddens[-1], n_output)
        
        nn.init.xavier_normal_(self.fc.weight)
        
        # AdaRNN specific: Temporal importance gates
        if self.model_type == 'AdaRNN':
            self._init_gates()
    
    def _init_bottleneck(self):
        """Initialize bottleneck weights"""
        for m in self.bottleneck.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)
    
    def _init_gates(self):
        """Initialize temporal importance gates for AdaRNN"""
        # Gate network: learns importance weight for each timestep
        gate = nn.ModuleList()
        bnlst = nn.ModuleList()
        
        for i in range(self.num_layers):
            # Gate input: concatenated source and target hidden states
            gate_input_dim = self.len_seq * self.hiddens[i] * 2
            gate_weight = nn.Linear(gate_input_dim, self.len_seq)
            gate.append(gate_weight)
            bnlst.append(nn.BatchNorm1d(self.len_seq))
        
        self.gate = gate
        self.bn_lst = bnlst
        self.softmax = nn.Softmax(dim=0)
        
        # Initialize gate weights
        for i in range(len(self.hiddens)):
            nn.init.xavier_normal_(self.gate[i].weight)
            nn.init.constant_(self.gate[i].bias, 0.0)
    
    def gru_features(self, x: torch.Tensor, predict: bool = False):
        """
        Extract features through GRU layers
        
        Returns:
            out: Final hidden states
            out_list: Hidden states from each layer
            out_weight_list: Temporal importance weights (for AdaRNN)
        """
        x_input = x
        out_list = []
        out_weight_list = [] if self.model_type == 'AdaRNN' and not predict else None
        
        for i in range(self.num_layers):
            out, _ = self.features[i](x_input.float())
            x_input = out
            out_list.append(out)
            
            # Compute temporal importance weights
            if self.model_type == 'AdaRNN' and not predict:
                out_gate = self._process_gate_weight(x_input, i)
                out_weight_list.append(out_gate)
        
        return out, out_list, out_weight_list
    
    def _process_gate_weight(self, out: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Compute temporal importance weights using gate network
        
        Key insight: Different timesteps have different importance
        for domain alignment. This learns which timesteps to focus on.
        """
        batch_size = out.shape[0]
        half = batch_size // 2
        
        # Split source and target
        x_s = out[:half]
        x_t = out[half:]
        
        # Concatenate along feature dimension
        x_all = torch.cat((x_s, x_t), dim=2)
        x_all = x_all.reshape(x_all.shape[0], -1)
        
        # Compute importance weights
        weight = torch.sigmoid(self.bn_lst[layer_idx](
            self.gate[layer_idx](x_all.float())))
        weight = torch.mean(weight, dim=0)
        weight = self.softmax(weight).squeeze()
        
        return weight
    
    def _get_features_split(self, output_list: List[torch.Tensor]):
        """Split output features into source and target"""
        fea_list_src, fea_list_tar = [], []
        
        for fea in output_list:
            half = fea.size(0) // 2
            fea_list_src.append(fea[:half])
            fea_list_tar.append(fea[half:])
        
        return fea_list_src, fea_list_tar
    
    def forward_pre_train(
        self,
        x: torch.Tensor,
        len_win: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, List]:
        """
        Forward pass for pre-training phase
        Learns temporal importance weights
        
        Args:
            x: Concatenated source and target features (batch*2, seq_len, n_input)
            len_win: Window size for temporal alignment
            
        Returns:
            predictions: Model predictions
            loss_transfer: Transfer loss
            out_weight_list: Temporal importance weights
        """
        if x.size(1) != self.len_seq:
            raise ValueError(f"Input seq length ({x.size(1)}) must match configured len_seq ({self.len_seq}).")

        out, out_list, out_weight_list = self.gru_features(x)
        
        # Get final representation
        fea = out[:, -1, :]  # Take last timestep
        
        if self.use_bottleneck:
            fea = self.bottleneck(fea)
        
        predictions = self.fc(fea)
        
        # Compute transfer loss with temporal alignment
        out_list_s, out_list_t = self._get_features_split(out_list)
        
        loss_transfer = torch.zeros(1, device=x.device)
        
        for i in range(len(out_list_s)):
            criterion = TransferLoss(
                loss_type=self.trans_loss,
                input_dim=out_list_s[i].shape[2]
            )
            
            # Align each timestep with importance weighting
            for j in range(self.len_seq):
                # Define alignment window
                i_start = max(0, j - len_win)
                i_end = min(self.len_seq - 1, j + len_win)
                
                for k in range(i_start, i_end + 1):
                    if self.model_type == 'AdaRNN' and out_weight_list is not None:
                        if out_weight_list[i].dim() == 0:
                            weight = out_weight_list[i]
                        elif j < out_weight_list[i].shape[0]:
                            weight = out_weight_list[i][j]
                        else:
                            weight = x.new_tensor(1.0 / self.len_seq)
                    else:
                        weight = x.new_tensor(1.0 / self.len_seq)
                    
                    loss_transfer = loss_transfer + weight * criterion.compute(
                        out_list_s[i][:, j, :],
                        out_list_t[i][:, k, :]
                    )
        
        return predictions, loss_transfer, out_weight_list
    
    def forward_Boosting(
        self,
        x: torch.Tensor,
        weight_mat: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for boosting phase
        Uses learned weights with boosting updates
        """
        if x.size(1) != self.len_seq:
            raise ValueError(f"Input seq length ({x.size(1)}) must match configured len_seq ({self.len_seq}).")

        out, out_list, _ = self.gru_features(x)
        
        fea = out[:, -1, :]
        
        if self.use_bottleneck:
            fea = self.bottleneck(fea)
        
        predictions = self.fc(fea)
        
        # Split features
        out_list_s, out_list_t = self._get_features_split(out_list)
        
        # Initialize weights if not provided
        if weight_mat is None:
            weight_mat = (1.0 / self.len_seq) * torch.ones(
                self.num_layers, self.len_seq, device=x.device
            )
        
        loss_transfer = torch.zeros(1, device=x.device)
        dist_mat = torch.zeros(self.num_layers, self.len_seq, device=x.device)
        
        for i in range(len(out_list_s)):
            criterion = TransferLoss(
                loss_type=self.trans_loss,
                input_dim=out_list_s[i].shape[2]
            )
            
            for j in range(self.len_seq):
                loss_trans = criterion.compute(
                    out_list_s[i][:, j, :],
                    out_list_t[i][:, j, :]
                )
                loss_transfer = loss_transfer + weight_mat[i, j] * loss_trans
                dist_mat[i, j] = loss_trans
        
        return predictions, loss_transfer, dist_mat, weight_mat
    
    def update_weight_Boosting(
        self,
        weight_mat: torch.Tensor,
        dist_old: torch.Tensor,
        dist_new: torch.Tensor
    ) -> torch.Tensor:
        """
        Update boosting weights based on distribution distance change
        
        Increase weight where distance increased (harder to align)
        """
        epsilon = 1e-12
        dist_old = dist_old.detach()
        dist_new = dist_new.detach()
        
        # Increase weights where distance increased
        mask = dist_new > dist_old + epsilon
        weight_mat[mask] = weight_mat[mask] * (1 + torch.sigmoid(dist_new[mask] - dist_old[mask]))
        
        # Normalize
        weight_norm = torch.norm(weight_mat, dim=1, p=1)
        weight_mat = weight_mat / weight_norm.unsqueeze(1).repeat(1, self.len_seq)
        
        return weight_mat
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference: predict without domain alignment
        """
        out, _, _ = self.gru_features(x, predict=True)
        fea = out[:, -1, :]
        
        if self.use_bottleneck:
            fea = self.bottleneck(fea)
        
        return self.fc(fea)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass for inference"""
        return self.predict(x)


class AdaRNNClassifier(nn.Module):
    """
    AdaRNN wrapper for classification tasks
    
    Simplified interface for IoT attack detection:
    - Handles both binary and multi-class classification
    - Automatic class weighting for imbalanced data
    - Easy integration with existing training pipelines
    """
    
    def __init__(
        self,
        input_dim: int = 27,
        hidden_dims: List[int] = [64, 64],
        num_classes: int = 2,
        seq_len: int = 1,
        dropout: float = 0.2,
        trans_loss: str = 'mmd',
        use_bottleneck: bool = True,
        bottleneck_dim: int = 64
    ):
        super(AdaRNNClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.input_dim = input_dim
        
        self.adarnn = AdaRNN(
            n_input=input_dim,
            n_hiddens=hidden_dims,
            n_output=num_classes,
            dropout=dropout,
            len_seq=seq_len,
            model_type='AdaRNN',
            trans_loss=trans_loss,
            use_bottleneck=use_bottleneck,
            bottleneck_width=bottleneck_dim
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass"""
        # Ensure correct input shape: (batch, seq_len, features)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        return self.adarnn.predict(x)
    
    def forward_with_transfer(
        self,
        x_source: torch.Tensor,
        x_target: torch.Tensor,
        len_win: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List]:
        """
        Forward pass with transfer loss computation
        
        Args:
            x_source: Source domain samples
            x_target: Target domain samples
            len_win: Alignment window
            
        Returns:
            pred_source: Source predictions
            pred_target: Target predictions  
            loss_transfer: Transfer loss
            weights: Temporal importance weights
        """
        # Ensure correct shape
        if x_source.dim() == 2:
            x_source = x_source.unsqueeze(1)
        if x_target.dim() == 2:
            x_target = x_target.unsqueeze(1)
        
        # Concatenate for joint processing
        x_all = torch.cat([x_source, x_target], dim=0)
        
        # Forward with transfer
        pred_all, loss_transfer, weights = self.adarnn.forward_pre_train(x_all, len_win)
        
        # Split predictions
        n_source = x_source.size(0)
        pred_source = pred_all[:n_source]
        pred_target = pred_all[n_source:]
        
        return pred_source, pred_target, loss_transfer, weights
    
    def forward_boosting(
        self,
        x_source: torch.Tensor,
        x_target: torch.Tensor,
        weight_mat: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with boosting-based transfer
        """
        if x_source.dim() == 2:
            x_source = x_source.unsqueeze(1)
        if x_target.dim() == 2:
            x_target = x_target.unsqueeze(1)
        
        x_all = torch.cat([x_source, x_target], dim=0)
        
        pred_all, loss_transfer, dist_mat, weight_mat = self.adarnn.forward_Boosting(
            x_all, weight_mat
        )
        
        n_source = x_source.size(0)
        pred_source = pred_all[:n_source]
        pred_target = pred_all[n_source:]
        
        return pred_source, pred_target, loss_transfer, dist_mat, weight_mat


class AdaRNNLightClassifier(nn.Module):
    """
    Lightweight AdaRNN variant with two specialist heads and a gate.

    Design goals:
    - keep the original AdaRNN backbone and transfer loss
    - add a simple normal head + attack head + gate head
    - enable selected-backprop style expert losses without a full NEC+ pipeline
    """

    def __init__(
        self,
        input_dim: int = 27,
        hidden_dims: List[int] = [64, 64],
        num_classes: int = 2,
        seq_len: int = 1,
        dropout: float = 0.2,
        trans_loss: str = 'mmd',
        use_bottleneck: bool = True,
        bottleneck_dim: int = 64
    ):
        super(AdaRNNLightClassifier, self).__init__()

        self.num_classes = num_classes
        self.input_dim = input_dim

        self.adarnn = AdaRNN(
            n_input=input_dim,
            n_hiddens=hidden_dims,
            n_output=num_classes,
            dropout=dropout,
            len_seq=seq_len,
            model_type='AdaRNN',
            trans_loss=trans_loss,
            use_bottleneck=use_bottleneck,
            bottleneck_width=bottleneck_dim
        )

        head_dim = bottleneck_dim if use_bottleneck else hidden_dims[-1]
        self.normal_head = nn.Linear(head_dim, num_classes)
        self.attack_head = nn.Linear(head_dim, num_classes)
        self.gate_head = nn.Linear(head_dim, 1)

        nn.init.xavier_normal_(self.normal_head.weight)
        nn.init.xavier_normal_(self.attack_head.weight)
        nn.init.xavier_normal_(self.gate_head.weight)
        nn.init.constant_(self.normal_head.bias, 0.0)
        nn.init.constant_(self.attack_head.bias, 0.0)
        nn.init.constant_(self.gate_head.bias, 0.0)

    def _encode(self, x: torch.Tensor):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        out, out_list, out_weight_list = self.adarnn.gru_features(x, predict=True)
        fea = out[:, -1, :]

        if self.adarnn.use_bottleneck:
            fea = self.adarnn.bottleneck(fea)

        return fea, out_list, out_weight_list

    def _combine_logits(self, features: torch.Tensor):
        normal_logits = self.normal_head(features)
        attack_logits = self.attack_head(features)
        gate_prob = torch.sigmoid(self.gate_head(features))
        combined_logits = (1.0 - gate_prob) * normal_logits + gate_prob * attack_logits
        return combined_logits, normal_logits, attack_logits, gate_prob.squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features, _, _ = self._encode(x)
        combined_logits, _, _, _ = self._combine_logits(features)
        return combined_logits

    def forward_with_details(self, x: torch.Tensor):
        features, _, _ = self._encode(x)
        return self._combine_logits(features)

    def _compute_transfer_loss(
        self,
        x_all: torch.Tensor,
        len_win: int = 0,
        weight_mat: Optional[torch.Tensor] = None,
        boosting: bool = False
    ):
        if boosting:
            _, loss_transfer, dist_mat, weight_mat = self.adarnn.forward_Boosting(x_all, weight_mat)
            return loss_transfer, dist_mat, weight_mat

        _, loss_transfer, out_weight_list = self.adarnn.forward_pre_train(x_all, len_win)
        return loss_transfer, out_weight_list

    def forward_with_transfer(
        self,
        x_source: torch.Tensor,
        x_target: torch.Tensor,
        len_win: int = 0
    ):
        if x_source.dim() == 2:
            x_source = x_source.unsqueeze(1)
        if x_target.dim() == 2:
            x_target = x_target.unsqueeze(1)

        x_all = torch.cat([x_source, x_target], dim=0)
        loss_transfer, out_weight_list = self._compute_transfer_loss(x_all, len_win=len_win, boosting=False)

        features, _, _ = self._encode(x_all)
        combined_logits, normal_logits, attack_logits, gate_prob = self._combine_logits(features)

        n_source = x_source.size(0)
        return (
            combined_logits[:n_source],
            combined_logits[n_source:],
            loss_transfer,
            out_weight_list,
            normal_logits[:n_source],
            normal_logits[n_source:],
            attack_logits[:n_source],
            attack_logits[n_source:],
            gate_prob[:n_source],
            gate_prob[n_source:]
        )

    def forward_boosting(
        self,
        x_source: torch.Tensor,
        x_target: torch.Tensor,
        weight_mat: Optional[torch.Tensor] = None
    ):
        if x_source.dim() == 2:
            x_source = x_source.unsqueeze(1)
        if x_target.dim() == 2:
            x_target = x_target.unsqueeze(1)

        x_all = torch.cat([x_source, x_target], dim=0)
        loss_transfer, dist_mat, weight_mat = self._compute_transfer_loss(
            x_all,
            weight_mat=weight_mat,
            boosting=True
        )

        features, _, _ = self._encode(x_all)
        combined_logits, normal_logits, attack_logits, gate_prob = self._combine_logits(features)

        n_source = x_source.size(0)
        return (
            combined_logits[:n_source],
            combined_logits[n_source:],
            loss_transfer,
            dist_mat,
            weight_mat,
            normal_logits[:n_source],
            normal_logits[n_source:],
            attack_logits[:n_source],
            attack_logits[n_source:],
            gate_prob[:n_source],
            gate_prob[n_source:]
        )


# Simple MLP-based AdaRNN for tabular data (no sequence)
class AdaRNNMLP(nn.Module):
    """
    AdaRNN-style domain adaptation for tabular (non-sequential) data
    
    Uses MLP instead of RNN, but retains:
    - Distribution matching between domains
    - Importance weighting for features
    """
    
    def __init__(
        self,
        input_dim: int = 27,
        hidden_dims: List[int] = [128, 64],
        num_classes: int = 2,
        dropout: float = 0.2,
        trans_loss: str = 'mmd'
    ):
        super(AdaRNNMLP, self).__init__()
        
        self.num_classes = num_classes
        self.trans_loss = trans_loss
        self.num_layers = len(hidden_dims)
        
        # Build encoder layers
        layers = []
        in_dim = input_dim
        for hidden in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden
        
        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)
        
        # Feature importance weights
        self.importance_weight = nn.Parameter(torch.ones(input_dim))
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward"""
        # Apply importance weighting
        x = x * torch.softmax(self.importance_weight, dim=0)
        
        features = self.encoder(x)
        return self.classifier(features)
    
    def forward_with_transfer(
        self,
        x_source: torch.Tensor,
        x_target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward with transfer loss"""
        # Apply importance weighting
        weights = torch.softmax(self.importance_weight, dim=0)
        x_source = x_source * weights
        x_target = x_target * weights
        
        # Get features
        feat_source = self.encoder(x_source)
        feat_target = self.encoder(x_target)
        
        # Predictions
        pred_source = self.classifier(feat_source)
        pred_target = self.classifier(feat_target)
        
        # Transfer loss
        criterion = TransferLoss(loss_type=self.trans_loss, input_dim=feat_source.shape[1])
        loss_transfer = criterion.compute(feat_source, feat_target)
        
        return pred_source, pred_target, loss_transfer


if __name__ == "__main__":
    # Test the models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test AdaRNN
    print("Testing AdaRNN...")
    model = AdaRNN(
        n_input=27,
        n_hiddens=[64, 64],
        n_output=2,
        len_seq=1,
        trans_loss='mmd'
    ).to(device)
    
    x = torch.randn(32, 1, 27).to(device)
    out = model.predict(x)
    print(f"  Input: {x.shape} -> Output: {out.shape}")
    
    # Test AdaRNNClassifier
    print("\nTesting AdaRNNClassifier...")
    classifier = AdaRNNClassifier(
        input_dim=27,
        hidden_dims=[64, 64],
        num_classes=2,
        seq_len=1
    ).to(device)
    
    x_s = torch.randn(16, 27).to(device)
    x_t = torch.randn(16, 27).to(device)
    
    pred_s, pred_t, loss_trans, _ = classifier.forward_with_transfer(x_s, x_t)
    print(f"  Source pred: {pred_s.shape}, Target pred: {pred_t.shape}")
    print(f"  Transfer loss: {loss_trans.item():.6f}")
    
    # Test AdaRNNMLP (for tabular data)
    print("\nTesting AdaRNNMLP...")
    mlp_model = AdaRNNMLP(
        input_dim=27,
        hidden_dims=[128, 64],
        num_classes=2
    ).to(device)
    
    pred_s, pred_t, loss_trans = mlp_model.forward_with_transfer(x_s, x_t)
    print(f"  Source pred: {pred_s.shape}, Transfer loss: {loss_trans.item():.6f}")
