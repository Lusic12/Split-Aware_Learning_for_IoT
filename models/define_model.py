import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_btf(x, input_dim):
    if x.dim() == 2:
        if x.size(-1) != input_dim:
            raise ValueError(f"Expected input_dim={input_dim}, got {x.size(-1)}")
        return x.unsqueeze(1)
    if x.dim() == 3:
        if x.size(-1) == input_dim:
            return x
        if x.size(1) == input_dim:
            return x.transpose(1, 2)
        raise ValueError(
            f"Expected feature dim {input_dim} on last/second axis, got {tuple(x.shape)}"
        )
    raise ValueError(f"Unsupported input shape: {tuple(x.shape)}")


class Time2Vec(nn.Module):
    def __init__(self, dim, activation="sin"):
        super().__init__()
        if dim < 1:
            raise ValueError("Time2Vec dim must be >= 1")
        self.dim = int(dim)
        self.linear = nn.Linear(1, 1)
        if self.dim > 1:
            self.periodic = nn.Linear(1, self.dim - 1)
        else:
            self.periodic = None
        if activation == "cos":
            self.activation = torch.cos
        else:
            self.activation = torch.sin

    def forward(self, t):
        # t: [B, L, 1]
        linear = self.linear(t)
        if self.periodic is None:
            return linear
        periodic = self.activation(self.periodic(t))
        return torch.cat([linear, periodic], dim=-1)


class TimeEmbedding(nn.Module):
    def __init__(self, max_len, dim, emb_type="learned", activation="sin", normalize=True):
        super().__init__()
        self.max_len = int(max_len)
        self.dim = int(dim)
        self.emb_type = emb_type
        self.normalize = bool(normalize)
        if self.emb_type == "time2vec":
            self.time2vec = Time2Vec(self.dim, activation=activation)
            self.embedding = None
        else:
            self.embedding = nn.Embedding(self.max_len, self.dim)
            self.time2vec = None

    def forward(self, seq_len, batch_size, device=None, dtype=None):
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.max_len}")
        if device is None:
            device = self.embedding.weight.device if self.embedding is not None else None
        if dtype is None:
            dtype = (
                self.embedding.weight.dtype
                if self.embedding is not None
                else torch.float32
            )

        positions = torch.arange(seq_len, device=device)
        if self.embedding is not None:
            emb = self.embedding(positions).to(dtype=dtype)
            emb = emb.unsqueeze(0)
        else:
            pos = positions.to(dtype=dtype)
            if self.normalize:
                denom = max(seq_len - 1, 1)
                pos = pos / denom
            pos = pos.view(1, seq_len, 1)
            emb = self.time2vec(pos)
        if batch_size is not None:
            emb = emb.expand(batch_size, -1, -1)
        return emb


def build_time_embedding(time_embedding, sequence_length):
    cfg = time_embedding or {}
    if not cfg or not cfg.get("enabled", False):
        return None, 0
    if sequence_length is None and cfg.get("max_len") is None:
        raise ValueError("time_embedding requires sequence_length or max_len")
    max_len = cfg.get("max_len", sequence_length)
    dim = int(cfg.get("dim", 8))
    emb_type = cfg.get("type", "learned")
    activation = cfg.get("activation", "sin")
    normalize = cfg.get("normalize", True)
    return TimeEmbedding(max_len, dim, emb_type=emb_type, activation=activation, normalize=normalize), dim


class CNN(nn.Module):
    """
    Improved CNN: Multiple conv layers + enhanced FC layers
    Conv Architecture: input → 256 → 128 → 64 channels
    FC Architecture: flatten → 256 → 128 → 64 → output
    Features: Multiple conv layers, batch norm, dropout, better feature extraction
    """
    def __init__(
        self,
        input_dim,
        num_class,
        hidden_layer_list=None,
        sequence_length=20,
        strict_input=False,
        time_embedding=None,
    ):
        super(CNN, self).__init__()
        self.input_dim = int(input_dim)
        # FC sizes can be overridden via hidden_layer_list (expects 4 values)
        fc_defaults = [512, 256, 128, 64]
        fc_sizes = (hidden_layer_list or fc_defaults)
        fc_sizes = (fc_sizes + fc_defaults)[:4]  # pad/fallback to 4 layers
        
        # Ensure minimum sequence length
        self.sequence_length = max(sequence_length, 8)
        self.time_embed, self.time_emb_dim = build_time_embedding(time_embedding, sequence_length)
        total_input_dim = self.input_dim + self.time_emb_dim
        
        # Multi-layer Conv architecture: 256 → 128 → 64
        self.conv1 = nn.Conv1d(total_input_dim, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(256)
        
        self.conv2 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        
        # Pooling layers - smaller kernel for small input dimensions
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)  # More gentle pooling
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)  # More gentle pooling
        
        # Adaptive pooling for consistent output size regardless of input dimension
        self.adaptive_pool = nn.AdaptiveAvgPool1d(8)  # Increased output size
        
        # Enhanced FC layers: 512 → 256 → 128 → 64 → output
        self.fc1 = nn.Linear(64 * 8, fc_sizes[0])  # After conv3 and adaptive pool (64 channels * 8 length)
        self.fc2 = nn.Linear(fc_sizes[0], fc_sizes[1])
        self.fc3 = nn.Linear(fc_sizes[1], fc_sizes[2])
        self.fc4 = nn.Linear(fc_sizes[2], fc_sizes[3])
        self.fc5 = nn.Linear(fc_sizes[3], num_class)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.num_class = num_class
        self.strict_input = strict_input

    def forward(self, x):
        if self.strict_input and x.dim() != 3:
            raise RuntimeError(
                f"CNN expects 3D input [B, T, F], got {tuple(x.shape)}. "
                "Use SequenceDataset or the inference pipeline to build sequences."
            )
        # Handle different input dimensions
        if x.dim() == 3:
            x = _ensure_btf(x, self.input_dim)
        elif x.dim() == 2:
            x = _ensure_btf(x, self.input_dim)

        if self.time_embed is not None:
            time_emb = self.time_embed(
                seq_len=x.size(1),
                batch_size=x.size(0),
                device=x.device,
                dtype=x.dtype,
            )
            x = torch.cat([x, time_emb], dim=-1)
        # [B, T, F] → [B, F, T] for Conv1d
        x = x.transpose(1, 2)
        
        # Conv layers with progressive channel reduction: 256 → 128 → 64
        x = self.dropout(self.relu(self.bn1(self.conv1(x))))
        x = self.pool1(x)
        
        x = self.dropout(self.relu(self.bn2(self.conv2(x))))
        x = self.pool2(x)
        
        x = self.dropout(self.relu(self.bn3(self.conv3(x))))
        x = self.adaptive_pool(x)
        
        # Flatten for FC layers
        x = x.view(x.size(0), -1)
        
        # Enhanced FC layers: 512 → 256 → 128 → 64 → output
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.dropout(self.relu(self.fc4(x)))
        x = self.fc5(x)
        
        return x  # Raw logits only
class DNN(nn.Module):
    """
    Deep Neural Network: Deep architecture with batch normalization
    Architecture: input -> 1024 -> 512 -> 256 -> output
    Features: Batch normalization, moderate dropout, deep layers
    """
    def __init__(self, input_dim, hidden_layer_list=None, num_class=2):
        super(DNN, self).__init__()
        hl = hidden_layer_list or [1024, 512, 256]
        hl = (hl + [1024, 512, 256])[:3]
        # DNN: Deep architecture with batch normalization
        self.fc1 = nn.Linear(input_dim, hl[0])
        self.bn1 = nn.BatchNorm1d(hl[0])
        self.fc2 = nn.Linear(hl[0], hl[1])
        self.bn2 = nn.BatchNorm1d(hl[1])
        self.fc3 = nn.Linear(hl[1], hl[2])
        self.bn3 = nn.BatchNorm1d(hl[2])
        self.fc4 = nn.Linear(hl[2], num_class)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.num_class = num_class

    def forward(self, x):
        if x.dim() == 3:
            x = x[:, -1, :]
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        x = self.dropout(self.relu(self.bn3(self.fc3(x))))
        x = self.fc4(x)
        return x  # Raw logits only

class MLP(nn.Module):
    """
    Multi-Layer Perceptron: Simple 3-layer network with same depth as others
    Architecture: input -> 1024 -> 512 -> 256 -> output
    Features: Basic ReLU activation, moderate dropout, NO batch norm, NO residual
    """
    def __init__(self, input_dim, hidden_layer_list=None, num_class=2):
        super(MLP, self).__init__()
        hl = hidden_layer_list or [1024, 512, 256]
        hl = (hl + [1024, 512, 256])[:3]
        # MLP: Simple 3-layer architecture (same depth as DNN/RaD_FFNN for fair comparison)
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

class RaD_FFNN(nn.Module):
    """
    Mô hình FFNN tùy chỉnh: sử dụng các lớp theo thứ tự từ hidden_layer_list[3] đến hidden_layer_list[0].
    """
    def __init__(self, input_dim, hidden_layer_list, num_class=2):
        super(RaD_FFNN, self).__init__()
        # Dynamic architecture based on hidden_layer_list: [h3, h2, h1, h0]
        # Layers: input -> h3 -> h2 -> h1 -> h0 -> output
        self.fc1 = nn.Linear(input_dim, hidden_layer_list[3])
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layer_list[3], hidden_layer_list[2])
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_layer_list[2], hidden_layer_list[1])
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_layer_list[1], hidden_layer_list[0])
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(hidden_layer_list[0], num_class)
        self.num_class = num_class

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.relu4(self.fc4(x))
        x = self.fc5(x)
        return x  # Raw logits only



class TemporalConvNet(nn.Module):
    def __init__(self, input_channels, output_channels, num_layers=2, kernel_size=25, dilation_base=2, hidden_layers=(512,256,128), stride=1):
        super().__init__()
        self.tcn_layers = nn.ModuleList()
        self.paddings = []
        for i in range(num_layers):
            dilation = dilation_base ** i
            conv = nn.Conv1d(
                output_channels if i > 0 else input_channels,
                output_channels,
                kernel_size,
                stride=stride,
                dilation=dilation
            )
            self.tcn_layers.append(nn.Sequential(conv, nn.ReLU()))
            self.paddings.append((0, dilation * (kernel_size - 1)))
        self.conv1d_layers = nn.ModuleList()
        prev_channels = input_channels
        for hidden_dim in hidden_layers:
            self.conv1d_layers.append(nn.Conv1d(prev_channels, hidden_dim, kernel_size=kernel_size, stride=stride))
            prev_channels = hidden_dim
        self.conv1d_layers.append(nn.Conv1d(prev_channels, output_channels, kernel_size=kernel_size, stride=stride))

    def forward(self, x):
        conv1d_output = x
        for conv in self.conv1d_layers:
            conv1d_output = F.pad(conv1d_output, (0, self.tcn_layers[0][0].kernel_size[0] - 1))
            conv1d_output = conv(conv1d_output)
            conv1d_output = F.relu(conv1d_output)
        tcn_output = x
        for i, layer in enumerate(self.tcn_layers):
            tcn_output = F.pad(tcn_output, self.paddings[i])
            tcn_output = layer(tcn_output)
        out = tcn_output + conv1d_output
        return out

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, model_dim=128, dim_feedforward=256, num_heads=4, num_layers=4, dropout=0.3):
        super().__init__()
        self.conv1d = nn.Conv1d(input_dim, model_dim, kernel_size=1)
        self.layer_norm = nn.LayerNorm(model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=self.layer_norm)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = x.permute(2, 0, 1)
        x = self.transformer_encoder(x)
        x = self.layer_norm(x)
        x = x.mean(dim=0)
        return x

class Transformer_TemporalCNN(nn.Module):
    def __init__(self, config, task_type):
        super().__init__()
        input_channels = config.get("tcn_input_channels", len(config["features"]))
        tcn_cfg = config["tcn_config"]
        trans_cfg = config["transformer_config"]
        fusion_cfg = config.get("fusion_config", {})
        hidden_size = config.get("hidden_size", [128, 64, 32])
        # Sử dụng trực tiếp các giá trị từ config
        output_channels = tcn_cfg.get("output_channels", 128)
        self.feature_temporal = TemporalConvNet(
            input_channels=input_channels,
            output_channels=output_channels,
            num_layers=tcn_cfg.get("num_layers", 2),
            kernel_size=tcn_cfg.get("kernel_size", 25),
            dilation_base=tcn_cfg.get("dilation_base", 2),
            hidden_layers=tuple(tcn_cfg.get("hidden_layers", [128, 64])),
            stride=tcn_cfg.get("stride", 1)
        )
        self.norm_vis = nn.BatchNorm1d(output_channels)
        self.feat_fc = nn.Conv1d(output_channels, hidden_size[0], kernel_size=1)
        self.activ = nn.ReLU()
        self.dropout = nn.Dropout(fusion_cfg.get("dropout", 0.3))
        num_outputs = 2 if task_type == 'binary_class' else config["train_config"].get("num_class", 5)
        # Sử dụng trực tiếp các giá trị từ config
        actual_model_dim = trans_cfg.get("model_dim", 128)
        self.out_head = nn.Sequential(
            nn.Linear(actual_model_dim, hidden_size[2]),
            nn.BatchNorm1d(hidden_size[2]),
            nn.Linear(hidden_size[2], num_outputs)
        )
        self.transformer = TransformerEncoder(
            input_dim=hidden_size[0],
            model_dim=trans_cfg.get("model_dim", 128),
            dim_feedforward=trans_cfg.get("dim_feedforward", 256),
            num_heads=trans_cfg.get("num_heads", 4),
            num_layers=trans_cfg.get("num_layers", 4),
            dropout=trans_cfg.get("dropout", 0.3)
        )
        self.config = config
        self.task_type = task_type

    def forward(self, sample):
        if not isinstance(sample, dict):
            sample = {"sequence": sample}
        feature = sample["sequence"]
        feature = feature.transpose(1, 2)
        feature = self.feature_temporal(feature)
        feature = self.norm_vis(feature)
        feature = feature.transpose(1, 2)
        feat = feature.transpose(1, 2)
        feat = self.feat_fc(feat)
        feat = self.activ(feat)
        feat = self.dropout(feat)
        feat = feat.transpose(2, 1)
        out = self.transformer(feat)  # [B, model_dim]
        out = self.out_head(out)
        return out

class Binary_classification(Transformer_TemporalCNN):
    def __init__(self, config):
        super(Binary_classification, self).__init__(config, task_type='binary_class')

class Multi_classification(Transformer_TemporalCNN):
    def __init__(self, config):
        super(Multi_classification, self).__init__(config, task_type='multi_class')

# ----------------------------
# Các model khác (CNN, DNN, MLP, RaD_FFNN) được định nghĩa dưới dạng ví dụ đơn giản
# Bạn có thể thay đổi cài đặt tùy ý.

class RNN(nn.Module):
    def __init__(
        self,
        input_dim,
        num_class,
        hidden_layer_list=None,
        sequence_length=None,
        time_embedding=None,
    ):
        super(RNN, self).__init__()
        self.input_dim = int(input_dim)
        hidden_layer_list = hidden_layer_list or []
        # Allow smaller configs: [rnn_hidden, fc_hidden]
        self.hidden_size = hidden_layer_list[0] if len(hidden_layer_list) >= 1 else 512
        self.num_layers = 2
        self.time_embed, self.time_emb_dim = build_time_embedding(time_embedding, sequence_length)
        rnn_input_dim = self.input_dim + self.time_emb_dim
        self.rnn = nn.RNN(
            rnn_input_dim,
            self.hidden_size,
            self.num_layers,
            batch_first=True,
            dropout=0.3 if self.num_layers > 1 else 0.0
        )
        # Additional FC layers for better parameter count
        fc_hidden = hidden_layer_list[1] if len(hidden_layer_list) >= 2 else 256
        self.fc1 = nn.Linear(self.hidden_size, fc_hidden)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(fc_hidden, num_class)
        self.num_class = num_class

    def forward(self, x):
        x = _ensure_btf(x, self.input_dim)
        if self.time_embed is not None:
            time_emb = self.time_embed(
                seq_len=x.size(1),
                batch_size=x.size(0),
                device=x.device,
                dtype=x.dtype,
            )
            x = torch.cat([x, time_emb], dim=-1)

        rnn_out, _ = self.rnn(x)
        last_output = rnn_out[:, -1, :]
        x = F.relu(self.fc1(last_output))
        x = self.dropout(x)
        x = self.fc2(x)
        return x  # Raw logits only

class LSTM(nn.Module):
    def __init__(
        self,
        input_dim,
        num_class,
        hidden_layer_list=None,
        sequence_length=None,
        time_embedding=None,
    ):
        super(LSTM, self).__init__()
        self.input_dim = int(input_dim)
        hidden_layer_list = hidden_layer_list or []
        # Allow smaller configs: [lstm_hidden, fc_hidden]
        self.hidden_size = hidden_layer_list[0] if len(hidden_layer_list) >= 1 else 512
        self.num_layers = 2
        self.time_embed, self.time_emb_dim = build_time_embedding(time_embedding, sequence_length)
        lstm_input_dim = self.input_dim + self.time_emb_dim
        self.lstm = nn.LSTM(
            lstm_input_dim,
            self.hidden_size,
            self.num_layers,
            batch_first=True,
            dropout=0.3 if self.num_layers > 1 else 0.0
        )
        # Additional FC layers for better parameter count
        fc_hidden = hidden_layer_list[1] if len(hidden_layer_list) >= 2 else 256
        self.fc1 = nn.Linear(self.hidden_size, fc_hidden)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(fc_hidden, num_class)
        self.num_class = num_class

    def forward(self, x):
        x = _ensure_btf(x, self.input_dim)
        if self.time_embed is not None:
            time_emb = self.time_embed(
                seq_len=x.size(1),
                batch_size=x.size(0),
                device=x.device,
                dtype=x.dtype,
            )
            x = torch.cat([x, time_emb], dim=-1)

        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        x = F.relu(self.fc1(last_output))
        x = self.dropout(x)
        x = self.fc2(x)
        return x  # Raw logits only

class GRU(nn.Module):
    def __init__(
        self,
        input_dim,
        num_class,
        hidden_layer_list=None,
        sequence_length=None,
        time_embedding=None,
    ):
        super(GRU, self).__init__()
        self.input_dim = int(input_dim)
        hidden_layer_list = hidden_layer_list or []
        # Allow smaller student models via hidden_layer_list: [gru_hidden, fc_hidden]
        self.hidden_size = hidden_layer_list[0] if len(hidden_layer_list) >= 1 else 512
        self.num_layers = 2
        self.time_embed, self.time_emb_dim = build_time_embedding(time_embedding, sequence_length)
        gru_input_dim = self.input_dim + self.time_emb_dim
        self.gru = nn.GRU(
            gru_input_dim,
            self.hidden_size,
            self.num_layers,
            batch_first=True,
            dropout=0.3 if self.num_layers > 1 else 0.0
        )
        fc_hidden = hidden_layer_list[1] if len(hidden_layer_list) >= 2 else 256
        self.fc1 = nn.Linear(self.hidden_size, fc_hidden)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(fc_hidden, num_class)
        self.num_class = num_class

    def forward(self, x):
        x = _ensure_btf(x, self.input_dim)
        if self.time_embed is not None:
            time_emb = self.time_embed(
                seq_len=x.size(1),
                batch_size=x.size(0),
                device=x.device,
                dtype=x.dtype,
            )
            x = torch.cat([x, time_emb], dim=-1)

        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]
        x = F.relu(self.fc1(last_output))
        x = self.dropout(x)
        x = self.fc2(x)
        return x  # Raw logits only

def create_model(
    model_type,
    input_dim,
    num_class,
    hidden_layer_list=None,
    sequence_length=None,
    config=None,
    strict_input=False,
    time_embedding=None,
):
    model_type = model_type.lower()
    sequence_models = {"cnn", "rnn", "lstm", "gru"}
    if model_type in sequence_models and sequence_length is None:
        raise ValueError(
            "sequence_length must be provided for sequence models: cnn/rnn/lstm/gru."
        )

    if model_type == "tcn_transformer":
        if config is None:
            raise ValueError("Config must be provided for tcn_transformer model.")
        required_keys = ["tcn_config", "transformer_config", "fusion_config", "hidden_size"]
        missing = [key for key in required_keys if key not in config]
        if missing:
            raise ValueError(
                f"Missing config keys for tcn_transformer: {', '.join(missing)}. "
                "Define them in the YAML."
            )
        if "features" not in config and "tcn_input_channels" not in config:
            raise ValueError(
                "Config must include either 'features' or 'tcn_input_channels' for tcn_transformer."
            )
        model = Binary_classification(config) if num_class == 2 else Multi_classification(config)
    else:
        builders = {
            "cnn": lambda: CNN(
                input_dim,
                num_class,
                hidden_layer_list=hidden_layer_list,
                sequence_length=sequence_length,
                strict_input=strict_input,
                time_embedding=time_embedding,
            ),
            "dnn": lambda: DNN(input_dim, hidden_layer_list=hidden_layer_list, num_class=num_class),
            "rnn": lambda: RNN(
                input_dim,
                num_class,
                hidden_layer_list=hidden_layer_list,
                sequence_length=sequence_length,
                time_embedding=time_embedding,
            ),
            "lstm": lambda: LSTM(
                input_dim,
                num_class,
                hidden_layer_list=hidden_layer_list,
                sequence_length=sequence_length,
                time_embedding=time_embedding,
            ),
            "gru": lambda: GRU(
                input_dim,
                num_class,
                hidden_layer_list=hidden_layer_list,
                sequence_length=sequence_length,
                time_embedding=time_embedding,
            ),
            "mlp": lambda: MLP(input_dim, hidden_layer_list=hidden_layer_list, num_class=num_class),
            "rad_ffnn": lambda: RaD_FFNN(input_dim, hidden_layer_list, num_class=num_class),
        }
        builder = builders.get(model_type)
        if builder is None:
            raise ValueError(f"Unknown model_type: {model_type}")
        model = builder()
    
    return model
