"""
Transfer Loss Functions for AdaRNN
Supports: MMD (Maximum Mean Discrepancy), CORAL, Adversarial, Cosine
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MMD_loss(nn.Module):
    """
    Maximum Mean Discrepancy loss
    Measures distance between two distributions using kernel methods
    """
    def __init__(self, kernel_type='linear', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_type = kernel_type
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
    
    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        """Gaussian kernel for RBF MMD"""
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        
        L2_distance = ((total0 - total1) ** 2).sum(2)
        
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) 
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)
    
    def linear_kernel(self, source, target):
        """Linear kernel for MMD"""
        return torch.mm(source, target.t())
    
    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd(source, target)
        elif self.kernel_type == 'rbf':
            return self.rbf_mmd(source, target)
        else:
            return self.linear_mmd(source, target)
    
    def linear_mmd(self, source, target):
        """Linear MMD distance"""
        delta = source.mean(0) - target.mean(0)
        loss = torch.dot(delta, delta)
        return loss
    
    def rbf_mmd(self, source, target):
        """RBF kernel MMD distance"""
        batch_size = int(source.size()[0])
        kernels = self.gaussian_kernel(
            source, target,
            kernel_mul=self.kernel_mul,
            kernel_num=self.kernel_num
        )
        
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        
        loss = torch.mean(XX + YY - XY - YX)
        return loss


def CORAL(source, target):
    """
    CORAL: CORrelation ALignment
    Aligns second-order statistics (covariances) of source and target
    """
    d = source.size(1)
    ns, nt = source.size(0), target.size(0)
    
    # Source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm / (ns - 1)
    
    # Target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt / (nt - 1)
    
    # Frobenius norm of covariance difference
    loss = torch.sum(torch.mul((xc - xct), (xc - xct)))
    loss = loss / (4 * d * d)
    
    return loss


class DomainDiscriminator(nn.Module):
    """Domain discriminator for adversarial domain adaptation"""
    def __init__(self, input_dim, hidden_dim=64):
        super(DomainDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 2)
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.1)
    
    def forward(self, x):
        return self.net(x)


class GradientReversal(torch.autograd.Function):
    """Gradient Reversal Layer for adversarial training"""
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


def adversarial_loss(source, target, input_dim, hidden_dim=64):
    """
    Adversarial domain adaptation loss
    Uses a domain discriminator to minimize domain discrepancy
    """
    device = source.device
    discriminator = DomainDiscriminator(input_dim, hidden_dim).to(device)
    
    # Create domain labels
    source_labels = torch.zeros(source.size(0), dtype=torch.long, device=device)
    target_labels = torch.ones(target.size(0), dtype=torch.long, device=device)
    
    # Combine features and labels
    features = torch.cat([source, target], dim=0)
    labels = torch.cat([source_labels, target_labels], dim=0)
    
    # Forward through discriminator
    predictions = discriminator(features)
    loss = F.cross_entropy(predictions, labels)
    
    return loss


def cosine_distance(source, target):
    """Cosine similarity based distance"""
    source_mean = source.mean(0)
    target_mean = target.mean(0)
    
    cos_sim = F.cosine_similarity(source_mean.unsqueeze(0), target_mean.unsqueeze(0))
    return 1 - cos_sim.squeeze()


class TransferLoss(object):
    """
    Unified interface for different transfer loss functions
    Supports: mmd, mmd_rbf, coral, cosine, adv (adversarial)
    """
    def __init__(self, loss_type='mmd', input_dim=512):
        self.loss_type = loss_type
        self.input_dim = input_dim
    
    def compute(self, source, target):
        """
        Compute transfer loss between source and target features
        
        Args:
            source: Source domain features (batch_size, feature_dim)
            target: Target domain features (batch_size, feature_dim)
        
        Returns:
            Transfer loss value
        """
        if self.loss_type == 'mmd' or self.loss_type == 'mmd_lin':
            mmd_loss = MMD_loss(kernel_type='linear')
            loss = mmd_loss(source, target)
        
        elif self.loss_type == 'mmd_rbf':
            mmd_loss = MMD_loss(kernel_type='rbf')
            loss = mmd_loss(source, target)
        
        elif self.loss_type == 'coral':
            loss = CORAL(source, target)
        
        elif self.loss_type == 'cosine' or self.loss_type == 'cos':
            loss = cosine_distance(source, target)
        
        elif self.loss_type == 'adv':
            loss = adversarial_loss(source, target, self.input_dim)
        
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
        
        return loss


if __name__ == "__main__":
    # Test the loss functions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    source = torch.randn(32, 64).to(device)
    target = torch.randn(32, 64).to(device)
    
    print("Testing Transfer Loss Functions:")
    for loss_type in ['mmd', 'mmd_rbf', 'coral', 'cosine', 'adv']:
        trans_loss = TransferLoss(loss_type=loss_type, input_dim=64)
        loss = trans_loss.compute(source, target)
        print(f"  {loss_type}: {loss.item():.6f}")
