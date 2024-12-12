import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from tqdm import tqdm
from torchvision import datasets, transforms

# ----------------------
# Vision Transformer Components
# ----------------------

class PatchEmbed(nn.Module):
    """Split image into patches and embed them."""
    def __init__(self, img_size=32, patch_size=2, in_chans=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x) # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2) # (B, embed_dim, num_patches)
        x = x.transpose(1, 2) # (B, num_patches, embed_dim)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def drop_path(x, drop_prob, training):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=2,
        in_chans=3,
        num_classes=10,
        embed_dim=192,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path_rate=0.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=.02)

        # Transformer Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, drop_path=dpr[i]
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        nn.init.trunc_normal_(self.cls_token, std=.02)
        nn.init.trunc_normal_(self.head.weight, std=.02)
        nn.init.constant_(self.head.bias, 0)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.patch_embed(x) # (B, N, D)
        B, N, D = x.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, N+1, D)
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_out = x[:, 0]  # CLS token
        logits = self.head(cls_out)
        return logits

def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']

def train_one_epoch(model, dataloader, optimizer, device, scaler=None, clip_grad_norm=1.0):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total = 0

    for data, targets in tqdm(dataloader, desc="Training", leave=False):
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                logits = model(data)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            if clip_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(data)
            loss = criterion(logits, targets)
            loss.backward()
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
            optimizer.step()

        total_loss += loss.item() * data.size(0)
        _, pred = torch.max(logits, dim=1)
        correct += pred.eq(targets).sum().item()
        total += data.size(0)
    
    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc

def eval_one_epoch(model, dataloader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in tqdm(dataloader, desc="Evaluating", leave=False):
            data, targets = data.to(device), targets.to(device)
            logits = model(data)
            loss = criterion(logits, targets)

            total_loss += loss.item() * data.size(0)
            _, pred = torch.max(logits, dim=1)
            correct += pred.eq(targets).sum().item()
            total += data.size(0)
    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc

if __name__ == "__main__":
    # Hyperparameters
    batch_size = 128
    start_lr = 1e-8
    base_lr = 5e-4
    warmup_epochs = 20
    epochs = 500
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Normalization for CIFAR-10
    cifar_mean = [0.4914, 0.4822, 0.4465]
    cifar_std = [0.2023, 0.1994, 0.2010]

    # Define transforms: horizontal flip, color jitter, and normalization for train
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar_mean, std=cifar_std)
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar_mean, std=cifar_std)
    ])

    # Load CIFAR10 dataset from torchvision
    train_dataset = datasets.CIFAR10(
        root='./cifar10', 
        train=True,
        download=True,
        transform=train_transform
    )

    val_dataset = datasets.CIFAR10(
        root='./cifar10', 
        train=False,
        download=True,
        transform=val_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Define the ViT model
    model = VisionTransformer(
        img_size=32, 
        patch_size=1,
        in_chans=3, 
        num_classes=10, 
        embed_dim=192,
        depth=12, 
        num_heads=12, 
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.1,
        attn_drop=0.1,
        drop_path_rate=0.1
    ).to(device)

    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=base_lr, 
        betas=(0.9, 0.95), 
        weight_decay=0.3
    )

    # Learning rate scheduling
    start_factor = start_lr / base_lr
    end_factor = 1.0
    warmup_scheduler = LinearLR(optimizer, start_factor=start_factor, end_factor=end_factor, total_iters=warmup_epochs)
    remain_scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=epochs - warmup_epochs,
        eta_min=1e-6
    )
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, remain_scheduler], milestones=[warmup_epochs])

    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    best_val_acc = 0.0
    for epoch in range(epochs):
        print(f"Epoch [{epoch+1}/{epochs}]")
        current_lr = get_lr(optimizer)
        print(f"Current learning rate: {current_lr:.6f}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, scaler=scaler)
        val_loss, val_acc = eval_one_epoch(model, val_loader, device)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_vit_2x2_cifar10.pt")
            print("Saved best model!")
