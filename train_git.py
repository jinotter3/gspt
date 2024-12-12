import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import GSCIFAR10
from model.git_model import GaussianTransformer
from model.git_model_noembed import GaussianTransformerNoEmbed
from tqdm import tqdm
from torch.optim.lr_scheduler import SequentialLR, LinearLR, ConstantLR

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
    batch_size = 512
    start_lr = 1e-8
    base_lr = 2e-3
    warmup_epochs = 20
    epochs = 500
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Data
    train_dataset = GSCIFAR10(
        dir_path='./cifar10/train', 
        label_path='./cifar10/train', 
        orig_path='./cifar10/train_orig', 
        rendered_path='./cifar10/train_rendered',
        dynamic_loading=False, 
        raw=False, 
        normalization='mean_std', 
        xy_jitter=0.02,
        scaling_jitter=0.2,
        rotation_jitter=0.2,
        value_jitter=0.04,
        global_xy_shift=(0.1, 0.1),
        global_value_shift=0.1,
        invisibility_prob=0.2,
        xy_flip=True,
        order_random=True
    )
    
    test_dataset = GSCIFAR10(
        dir_path='./cifar10/test', 
        label_path='./cifar10/test', 
        orig_path='./cifar10/test_orig', 
        rendered_path='./cifar10/test_rendered',
        dynamic_loading=False, 
        raw=False, 
        normalization='mean_std'
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Define model
    model = GaussianTransformer(
        num_classes=10, 
        seq_len=128, 
        input_dim=8, 
        embed_dim=192,  # PiT-T like
        depth=12, 
        num_heads=12, 
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.1,
        attn_drop=0.1,
        drop_path_rate=0.1
    ).to(device)
    # model = GaussianTransformerNoEmbed(
    #     num_classes=10, 
    #     seq_len=128, 
    #     input_dim=8, 
    #     embed_dim=192,  # PiT-T like
    #     depth=12, 
    #     num_heads=12, 
    #     mlp_ratio=4.0,
    #     qkv_bias=True,
    #     drop=0.1,
    #     attn_drop=0.1,
    #     drop_path_rate=0.1
    # ).to(device)

    # Initialize optimizer with AdamW and requested betas and weight decay
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=base_lr,  # Initialize at base_lr
        betas=(0.9, 0.95), 
        weight_decay=0.3
    )

    # Compute start_factor based on base_lr
    start_factor = start_lr / base_lr  # This should be <= 1
    end_factor = 1.0

    # Linear warmup from start_factor to 1.0 over warmup_epochs
    warmup_scheduler = LinearLR(optimizer, start_factor=start_factor, end_factor=end_factor, total_iters=warmup_epochs)
    # After warmup, use cosine annealing scheduler
    remain_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=epochs - warmup_epochs,
        eta_min=1e-6 # Minimum learning rate you want to decay to
    )
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, remain_scheduler], milestones=[warmup_epochs])

    # Optionally, use mixed precision
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

        # Step the scheduler after each epoch
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_pit_model_2_f.pt")
            print("Saved best model!")



# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from dataset import GSCIFAR10
# from git_model import GaussianTransformer
# from tqdm import tqdm

# def get_lr(optimizer):
#     return optimizer.param_groups[0]['lr']

# def train_one_epoch(model, dataloader, optimizer, device):
#     model.train()
#     criterion = nn.CrossEntropyLoss()
#     total_loss = 0
#     correct = 0
#     total = 0

#     for data, targets in tqdm(dataloader, desc="Training", leave=False):
#         data, targets = data.to(device), targets.to(device)
#         # data: (B, 128, 8), targets: (B)
        
#         optimizer.zero_grad()
#         logits = model(data)
#         loss = criterion(logits, targets)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item() * data.size(0)
#         _, pred = torch.max(logits, dim=1)
#         correct += pred.eq(targets).sum().item()
#         total += data.size(0)
    
#     avg_loss = total_loss / total
#     acc = correct / total
#     return avg_loss, acc

# def eval_one_epoch(model, dataloader, device):
#     model.eval()
#     criterion = nn.CrossEntropyLoss()
#     total_loss = 0
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data, targets in tqdm(dataloader, desc="Evaluating", leave=False):
#             data, targets = data.to(device), targets.to(device)
#             logits = model(data)
#             loss = criterion(logits, targets)

#             total_loss += loss.item() * data.size(0)
#             _, pred = torch.max(logits, dim=1)
#             correct += pred.eq(targets).sum().item()
#             total += data.size(0)
#     avg_loss = total_loss / total
#     acc = correct / total
#     return avg_loss, acc

# if __name__ == "__main__":
#     # Hyperparameters
#     batch_size = 256
#     base_lr = 1e-3
#     warmup_epochs = 10
#     epochs = 120
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Load Data
#     train_dataset = GSCIFAR10(dir_path='./cifar10/train', 
#                               label_path='./cifar10/train', 
#                               orig_path='./cifar10/train_orig', 
#                               rendered_path='./cifar10/train_rendered',
#                               dynamic_loading=False, 
#                               raw=False, 
#                               normalization='mean_std', 
#                               xy_jitter=0.02,
#                               scaling_jitter=0.1,
#                               rotation_jitter=0.1,
#                               value_jitter=0.03,
#                               order_random=True)
    
#     test_dataset = GSCIFAR10(dir_path='./cifar10/test', 
#                              label_path='./cifar10/test', 
#                              orig_path='./cifar10/test_orig', 
#                              rendered_path='./cifar10/test_rendered',
#                              dynamic_loading=False, 
#                              raw=False, 
#                              normalization='mean_std')

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
#     val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

#     # Define model
#     model = GaussianTransformer(
#         num_classes=10, 
#         seq_len=128, 
#         input_dim=8, 
#         embed_dim=192,  # PiT-T like
#         depth=12, 
#         num_heads=12, 
#         mlp_ratio=4.0,
#         qkv_bias=True,
#         drop=0.1,
#         attn_drop=0.1,
#         drop_path_rate=0.1
#     ).to(device)

#     # Initialize optimizer with a very small learning rate
#     optimizer = optim.AdamW(model.parameters(), lr=1e-8, weight_decay=0.05)
    
#     # Create learning rate scheduler for linear warmup
#     def warmup_lambda(epoch):
#         if epoch < warmup_epochs:
#             return float(epoch + 1) / warmup_epochs  # Linear warmup
#         return 1.0  # Full learning rate after warmup
    
#     scheduler = torch.optim.lr_scheduler.LambdaLR(
#         optimizer, 
#         lr_lambda=lambda epoch: warmup_lambda(epoch) * base_lr / get_lr(optimizer)
#     )

#     best_val_acc = 0.0
#     for epoch in range(epochs):
#         print(f"Epoch [{epoch+1}/{epochs}]")
#         current_lr = get_lr(optimizer)
#         print(f"Current learning rate: {current_lr:.6f}")
        
#         train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
#         val_loss, val_acc = eval_one_epoch(model, val_loader, device)

#         print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
#         print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

#         # Step the scheduler after each epoch
#         scheduler.step()

#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             torch.save(model.state_dict(), "best_pit_model_2.pt")
#             print("Saved best model!")

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from dataset import GSCIFAR10
# from git_model import GaussianTransformer
# from tqdm import tqdm

# def train_one_epoch(model, dataloader, optimizer, device):
#     model.train()
#     criterion = nn.CrossEntropyLoss()
#     total_loss = 0
#     correct = 0
#     total = 0

#     for data, targets in tqdm(dataloader, desc="Training", leave=False):
#         data, targets = data.to(device), targets.to(device)
#         # data: (B, 128, 8), targets: (B)
        
#         optimizer.zero_grad()
#         logits = model(data)
#         loss = criterion(logits, targets)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item() * data.size(0)
#         _, pred = torch.max(logits, dim=1)
#         correct += pred.eq(targets).sum().item()
#         total += data.size(0)
    
#     avg_loss = total_loss / total
#     acc = correct / total
#     return avg_loss, acc

# def eval_one_epoch(model, dataloader, device):
#     model.eval()
#     criterion = nn.CrossEntropyLoss()
#     total_loss = 0
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data, targets in tqdm(dataloader, desc="Evaluating", leave=False):
#             data, targets = data.to(device), targets.to(device)
#             logits = model(data)
#             loss = criterion(logits, targets)

#             total_loss += loss.item() * data.size(0)
#             _, pred = torch.max(logits, dim=1)
#             correct += pred.eq(targets).sum().item()
#             total += data.size(0)
#     avg_loss = total_loss / total
#     acc = correct / total
#     return avg_loss, acc

# if __name__ == "__main__":
#     # Hyperparameters
#     batch_size = 256
#     lr = 3e-4
#     epochs = 100
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Load Data
#     train_dataset = GSCIFAR10(dir_path='./cifar10/train', 
#                               label_path='./cifar10/train', 
#                               orig_path='./cifar10/train_orig', 
#                               rendered_path='./cifar10/train_rendered',
#                               dynamic_loading=False, 
#                               raw=False, 
#                               normalization='mean_std', 
#                               xy_jitter=0.02,
#                               scaling_jitter=0.1,
#                               rotation_jitter=0.1,
#                               value_jitter=0.03,
#                               order_random=True)
    
#     test_dataset = GSCIFAR10(dir_path='./cifar10/test', 
#                              label_path='./cifar10/test', 
#                              orig_path='./cifar10/test_orig', 
#                              rendered_path='./cifar10/test_rendered',
#                              dynamic_loading=False, 
#                              raw=False, 
#                              normalization='mean_std')

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
#     val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

#     # Define model
#     model = GaussianTransformer(
#         num_classes=10, 
#         seq_len=128, 
#         input_dim=8, 
#         embed_dim=192,  # PiT-T like
#         depth=12, 
#         num_heads=12, 
#         mlp_ratio=4.0,
#         qkv_bias=True,
#         drop=0.1,
#         attn_drop=0.1,
#         drop_path_rate=0.1
#     ).to(device)

#     optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)

#     best_val_acc = 0.0
#     for epoch in range(epochs):
#         print(f"Epoch [{epoch+1}/{epochs}]")
#         train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
#         val_loss, val_acc = eval_one_epoch(model, val_loader, device)

#         print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
#         print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             torch.save(model.state_dict(), "best_pit_model.pt")
#             print("Saved best model!")
