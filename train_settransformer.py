import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import GSCIFAR10  


# ---------------------
# Set Transformer Components
# ---------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        self.scale = (self.head_dim) ** -0.5

    def forward(self, Q, K, V):
        # Q: (B, QN, D)
        # K: (B, KN, D)
        # V: (B, VN, D)
        B, QN, D = Q.size()
        _, KN, _ = K.size()
        _, VN, _ = V.size()

        q = self.query(Q).view(B, QN, self.num_heads, self.head_dim).transpose(1, 2)  # (B, h, QN, d_h)
        k = self.key(K).view(B, KN, self.num_heads, self.head_dim).transpose(1, 2)   # (B, h, KN, d_h)
        v = self.value(V).view(B, VN, self.num_heads, self.head_dim).transpose(1, 2) # (B, h, VN, d_h)

        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B, h, QN, KN)
        attn_weights = attn_weights.softmax(dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)  # (B, h, QN, d_h)
        out = out.transpose(1, 2).contiguous().view(B, QN, D)  # (B, QN, D)
        out = self.dropout(self.out(out))
        return out


class SAB(nn.Module):
    """
    Set Attention Block
    Applies MHA and FFN with residual connections
    """
    def __init__(self, dim, num_heads=4, mlp_ratio=2, dropout=0.1):
        super(SAB, self).__init__()
        self.mha = MultiHeadAttention(dim, num_heads=num_heads, dropout=dropout)
        self.ln1 = nn.LayerNorm(dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim*mlp_ratio),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim*mlp_ratio, dim),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(dim)
        
    def forward(self, X):
        # MHA block
        X = X + self.mha(X, X, X)
        X = self.ln1(X)
        
        # Feed-forward block
        X = X + self.ffn(X)
        X = self.ln2(X)
        
        return X


class PMA(nn.Module):
    def __init__(self, dim, num_heads=4, k=1, dropout=0.1):
        super(PMA, self).__init__()
        self.k = k
        self.seeds = nn.Parameter(torch.randn(k, dim))
        self.mha = MultiHeadAttention(dim, num_heads=num_heads, dropout=dropout)
        self.ln = nn.LayerNorm(dim)
        
    def forward(self, X):
        # X: (B, N, D)
        B, N, D = X.size()
        # S: (B, k, D)
        S = self.seeds.unsqueeze(0).expand(B, self.k, D)
        
        # Q=S, K=X, V=X for PMA
        out = S + self.mha(S, X, X) 
        out = self.ln(out)
        
        return out.mean(dim=1)  # (B, D)


class SetTransformerClassifier(nn.Module):
    def __init__(self, input_dim=8, dim=128, num_heads=4, num_sab_layers=2, mlp_ratio=2, dropout=0.1, num_classes=10):
        super(SetTransformerClassifier, self).__init__()
        
        # Initial linear embedding of each Gaussian
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Stack multiple SAB blocks
        sab_layers = []
        for _ in range(num_sab_layers):
            sab_layers.append(SAB(dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout))
        self.sab = nn.Sequential(*sab_layers)
        
        # Pool with PMA
        self.pma = PMA(dim, num_heads=num_heads, k=1, dropout=dropout)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes)
        )
        
    def forward(self, x):
        # x: (B, N=128, D=8)
        x = self.input_proj(x)  # (B, N, dim)
        x = self.sab(x)          # (B, N, dim)
        x = self.pma(x)          # (B, dim)
        x = self.classifier(x)   # (B, num_classes)
        return x

# ---------------------
# Training & Evaluation
# ---------------------
def train_model(dataset, test_dataset, num_classes=10, num_epochs=100, batch_size=128, lr=1e-3, device='cuda'):
    """
    Train the SetTransformer-based model on the dataset.
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    model = SetTransformerClassifier(input_dim=8, dim=192, num_heads=6, num_sab_layers=12, mlp_ratio=2, dropout=0.1, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_acc = 0.0
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for data, labels in dataloader:
            data = data.to(device)   
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * data.size(0)
            _, preds = torch.max(outputs, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += data.size(0)
        
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples * 100.0
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f} Accuracy: {accuracy:.2f}%")
        
        # Evaluate every 10 epochs or so
        if (epoch+1) % 10 == 0:
            val_loss, val_acc = evaluate_model(model, test_dataset, batch_size=batch_size, device=device)
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_model.pth')
            model.train()  # back to training mode
        
        if (epoch+1) % 50 == 0:
            # save model checkpoint
            torch.save(model.state_dict(), f'checkpoint_{epoch+1}.pth')
    
    return model


def evaluate_model(model, dataset, batch_size=128, device='cuda'):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    criterion = nn.CrossEntropyLoss()

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)
            labels = labels.to(device)

            outputs = model(data)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * data.size(0)
            _, preds = torch.max(outputs, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += data.size(0)

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples * 100.0
    print(f"Evaluation Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return average_loss, accuracy


if __name__=="__main__":
   
    train_dataset = GSCIFAR10(
        dir_path='./cifar10/train', 
        label_path='./cifar10/train', 
        orig_path='./cifar10/train_orig', 
        rendered_path='./cifar10/train_rendered',
        dynamic_loading=False, 
        raw=False, 
        normalization='mean_std', 
        xy_jitter=0.02,
        scaling_jitter=0.1,
        rotation_jitter=0.1,
        value_jitter=0.03,
        xy_flip=True,
        order_random=True
    )
    
    test_dataset = GSCIFAR10(dir_path='./cifar10/test', 
                             label_path='./cifar10/test', 
                             orig_path='./cifar10/test_orig', 
                             rendered_path='./cifar10/test_rendered',
                             dynamic_loading=False, 
                             raw=False, 
                             normalization='mean_std')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = train_model(train_dataset, test_dataset=test_dataset, 
                        num_classes=10, num_epochs=200, batch_size=256, lr=1e-4, device=device)

    # Evaluate final model
    avg_loss, acc = evaluate_model(model, test_dataset, batch_size=256, device=device)
    print("Average Loss: ", avg_loss)
    print("Accuracy: ", acc)
    
    # Save the model
    torch.save(model.state_dict(), 'settransformer_1208.pth')

