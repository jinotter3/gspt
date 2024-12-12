import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from multiprocessing import Pool, Manager
from tqdm import tqdm
import time
import argparse
from torchvision import datasets, transforms
import torchvision.utils as vutils
from gsplat.project_gaussians_2d_scale_rot import project_gaussians_2d_scale_rot
from gsplat.rasterize_sum import rasterize_gaussians_sum

class GaussianBW(nn.Module):
    def __init__(self, init_num_points, H, W, channels=3):
        super().__init__()
        if channels != 1 and channels != 3:
            raise ValueError("Only 1 or 3 channels are supported.")
        self.channels = channels
        self.init_num_points = init_num_points
        self.H, self.W = H, W
        self.BLOCK_W, self.BLOCK_H = 16, 16
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.xy = nn.Parameter(torch.zeros(self.init_num_points, 2))
        self.scaling = nn.Parameter(torch.rand(self.init_num_points, 2))
        self.register_buffer('_opacity', torch.ones((self.init_num_points, 1)))
        self._rotation = nn.Parameter(torch.rand(self.init_num_points, 1))
        self.values = nn.Parameter((torch.rand(self.init_num_points, self.channels) - 0.5))
        
        self.rotation_activation = torch.sigmoid
        self.background = torch.zeros(self.channels, device=self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=2e-3)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.5)
    
    @property
    def get_scaling(self):
        return torch.abs(self.scaling) + 0.3
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation) * 2 * np.pi
    
    @property
    def get_xy(self):
        return torch.tanh(self.xy)
    
    def forward(self):
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(
            self.get_xy, self.get_scaling, self.get_rotation, self.H, self.W, self.tile_bounds
        )
        
        out_img = rasterize_gaussians_sum(
            self.xys, depths, self.radii, conics, num_tiles_hit,
            self.values, self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W,
            background=self.background, return_alpha=False
        )
        out_img = out_img.view(-1, self.H, self.W, self.channels).permute(0, 3, 1, 2)
        return out_img
    
    def init_xy_adaptive(self, target, alpha=0.5):
        grad_magnitude = self.compute_gradient_magnitude(target)
        grad_magnitude = grad_magnitude.detach()
        grad_magnitude_flat = grad_magnitude.view(-1)
        grad_magnitude_flat += 1e-8
        
        grad_probs = grad_magnitude_flat / grad_magnitude_flat.sum()
        uniform_probs = torch.full_like(grad_probs, 1 / grad_probs.numel())
        probabilities = alpha * grad_probs + (1 - alpha) * uniform_probs
        probabilities = probabilities / probabilities.sum()
        
        indices = torch.multinomial(probabilities, num_samples=self.init_num_points, replacement=True)
        H, W = grad_magnitude.shape
        y_indices = indices // W
        x_indices = indices % W
        x_coords = (x_indices.float() / (W - 1)) * 2 - 1
        y_coords = (y_indices.float() / (H - 1)) * 2 - 1
        epsilon = 1e-6
        x_coords = x_coords.clamp(min=-1 + epsilon, max=1 - epsilon)
        y_coords = y_coords.clamp(min=-1 + epsilon, max=1 - epsilon)
        xy_coords = torch.stack([x_coords, y_coords], dim=1).to(self.device)
        self.xy = nn.Parameter(torch.atanh(xy_coords))
    
    def compute_gradient_magnitude(self, image):
        sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=torch.float32, device=image.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=torch.float32, device=image.device).view(1, 1, 3, 3)
        
        grad_magnitudes = []
        for channel in range(3):
            channel_data = image[:, channel:channel+1]
            grad_x = nn.functional.conv2d(channel_data, sobel_x, padding=1)
            grad_y = nn.functional.conv2d(channel_data, sobel_y, padding=1)
            channel_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
            grad_magnitudes.append(channel_magnitude)
        
        combined_magnitude = torch.mean(torch.stack(grad_magnitudes, dim=1), dim=1)
        return combined_magnitude[0, 0]

class SingleGPUManager:
    def __init__(self, max_processes=4):
        manager = Manager()
        self.process_count = manager.Value('i', 0)
        self.max_processes = max_processes
        self.lock = manager.Lock()
    
    def acquire_gpu(self):
        while True:
            with self.lock:
                if self.process_count.value < self.max_processes:
                    self.process_count.value += 1
                    return 0
            time.sleep(0.1)
    
    def release_gpu(self, _):
        with self.lock:
            self.process_count.value -= 1

class GaussianConverter:
    def __init__(self, num_points=128, num_iterations=25000, split='train'):
        self.num_points = num_points
        self.num_iterations = num_iterations
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create directories
        self.param_dir = f'./cifar10/{split}'
        self.rendered_dir = f'./cifar10/{split}_rendered'
        self.orig_dir = f'./cifar10/{split}_orig'
        
        for directory in [self.param_dir, self.rendered_dir, self.orig_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def convert_single_sample(self, image, index):
        image_tensor = torch.FloatTensor(image).unsqueeze(0).to(self.device)
        
        model = GaussianBW(self.num_points, image_tensor.shape[-2], image_tensor.shape[-1]).to(self.device)
        model.init_xy_adaptive(image_tensor)
        
        optimizer = optim.Adam(model.parameters(), lr=2e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)
        
        for _ in range(self.num_iterations):
            optimizer.zero_grad()
            out = model()
            loss = nn.MSELoss()(out, image_tensor)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if loss < 1e-4:
                break
        
        final_psnr = 10 * torch.log10(1 / loss)
        
        with torch.no_grad():
            # Save rendered image
            final_output = model().detach().cpu()
            final_output = final_output.clamp(0, 1)
            rendered_path = os.path.join(self.rendered_dir, f'c{index:05d}.png')
            orig_path = os.path.join(self.orig_dir, f'c{index:05d}.png')
            vutils.save_image(final_output, rendered_path)
            vutils.save_image(image_tensor.cpu(), orig_path)
            
            # Save parameters
            params = {
                'xy': model.xy.detach().cpu(),
                'scaling': model.scaling.detach().cpu(),
                'rotation': model._rotation.detach().cpu(),
                'values': model.values.detach().cpu()
            }
            param_path = os.path.join(self.param_dir, f'c{index:05d}.pt')
            torch.save(params, param_path)
        
        return final_psnr.detach().cpu().item()

gpu_manager_shared = None

def init_worker(gpu_manager):
    global gpu_manager_shared
    gpu_manager_shared = gpu_manager

def process_sample(args):
    global gpu_manager_shared
    gpu_id = None
    try:
        image, label, index = args
        gpu_id = gpu_manager_shared.acquire_gpu()
        converter = GaussianConverter(split='test')
        psnr = converter.convert_single_sample(image, index)
        return index, label, psnr
    except Exception as e:
        print(f"Error processing sample {index}: {str(e)}")
        return None
    finally:
        if gpu_id is not None:
            gpu_manager_shared.release_gpu(gpu_id)

def convert_cifar10_to_gaussians(split='train', start_idx=0, num_samples=12500, num_processes=4):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    dataset = datasets.CIFAR10(root='./data', train=(split=='train'), download=True, transform=transform)
    
    end_idx = min(start_idx + num_samples, len(dataset))
    X = torch.stack([dataset[i][0] for i in range(start_idx, end_idx)]).numpy()
    y = np.array([dataset[i][1] for i in range(start_idx, end_idx)])
    
    gpu_manager_shared = SingleGPUManager(max_processes=num_processes)

    with Pool(processes=num_processes, initializer=init_worker, initargs=(gpu_manager_shared,)) as pool:
        data_args = [(X[i], y[i], start_idx+i) for i in range(len(X))]
        
        for result in tqdm(pool.imap_unordered(process_sample, data_args), total=len(data_args)):
            if result is None:
                continue
            index, label, psnr = result
            # Save label
            label_path = os.path.join(f'./cifar10/{split}', f'c{index:05d}_label.txt')
            with open(label_path, 'w') as f:
                f.write(f"{label}\n{psnr}")

if __name__ == '__main__':
    argsparser = argparse.ArgumentParser()
    argsparser.add_argument('--split', type=str, default='train', help='Dataset split to process (train or test)')
    argsparser.add_argument('--start_idx', type=int, default=0, help='Starting index for the dataset')
    argsparser.add_argument('--num_samples', type=int, default=12500, help='Number of samples to process')
    argsparser.add_argument('--num_processes', type=int, default=6, help='Number of processes to use')
    args = argsparser.parse_args()
    
    split = args.split
    num_processes = args.num_processes
    start_idx = args.start_idx
    num_samples = args.num_samples
    
    # split = 'train'  # or 'test'
    output_path = convert_cifar10_to_gaussians(
        split=split,
        start_idx=start_idx,
        num_samples=num_samples,
        num_processes=num_processes,
    )