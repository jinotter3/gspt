import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from renderer import GaussianRenderer


class GSCIFAR10(Dataset):
    def __init__(
        self,
        dir_path='./cifar10/train',
        label_path='./cifar10/train',
        orig_path='./cifar10/train_orig',
        rendered_path='./cifar10/train_rendered',
        init_images=False,
        raw=True,
        dynamic_loading=False,
        normalization=None,
        xy_jitter=0.0,
        scaling_jitter=0.0,
        rotation_jitter=0.0,
        value_jitter=0.0,
        xy_flip=False,
        order_random=False,
        global_xy_shift=(0.0, 0.0),     
        global_value_shift=0.0,           
        invisibility_prob=0.0,           
        global_rotation_angle=0.0,
        render_device='cuda'
    ):
        """
        GSCIFAR10 Dataset for loading Gaussian parameters, labels, and optional images.
        
        Args:
            dir_path (str): Directory containing .pt files with Gaussian parameters.
            label_path (str): Directory containing .txt files with labels and psnrs.
            orig_path (str): Directory containing original images.
            rendered_path (str): Directory containing rendered images.
            init_images (bool): If True and not dynamic loading, preload images into memory.
            raw (bool): If True, load parameters as raw outputs; if False, apply transformations.
            dynamic_loading (bool): If True, load .pt data on demand rather than preloading.
            normalization (None or str): If 'mean_std', apply mean-std normalization; 
                                         if 'specific', apply a custom normalization scheme.
            xy_jitter, scaling_jitter, rotation_jitter, value_jitter (float): Jitters to apply.
            order_random (bool): If True, randomly permute the order of Gaussians in each sample.
            device (str): Device to place renderer models.
        """
        self.dir_path = dir_path
        self.label_path = label_path
        self.orig_path = orig_path
        self.rendered_path = rendered_path

        self.xy_jitter = xy_jitter
        self.scaling_jitter = scaling_jitter
        self.rotation_jitter = rotation_jitter
        self.value_jitter = value_jitter
        self.xy_flip = xy_flip
        self.order_random = order_random
        self.global_xy_shift = global_xy_shift
        self.global_value_shift = global_value_shift
        self.invisibility_prob = invisibility_prob
        self.global_rotation_angle = global_rotation_angle
        self.raw = raw
        self.dynamic_loading = dynamic_loading
        self.init_images_flag = init_images and not dynamic_loading

        # Check normalization conditions
        # Normalization only makes sense when raw=False
        assert not (self.raw and normalization is not None), "Normalization only supported for raw=False."
        assert normalization in [None, 'mean_std', 'specific'], "Invalid normalization type."
        self.normalization = normalization

        # If mean/std normalization is requested, load them
        if self.normalization == 'mean_std':
            # Expecting mean/std to be dictionaries. Adjust as necessary.
            self.mean = torch.load('mean.pt', weights_only=True)
            self.std = torch.load('std.pt', weights_only=True)

        self.transform = transforms.ToTensor()

        # Load file lists and data
        self.pt_files = self.collect_files(self.dir_path, '.pt')
        self.labels, self.psnrs, self.label_files = self.load_all_labels(self.label_path)

        # Preload PT data if not dynamic
        if self.dynamic_loading:
            self.raw_data_cache = None
        else:
            # Store raw data always. We will apply transforms in __getitem__ if needed.
            self.raw_data_cache = [self.construct_data_raw(f) for f in tqdm(self.pt_files)]

        # Manage images
        self.orig_img_files = self.collect_files(self.orig_path, '.png')
        self.rendered_img_files = self.collect_files(self.rendered_path, '.png')
        
        if self.init_images_flag:
            self.orig_data = [self.load_image(f) for f in tqdm(self.orig_img_files)]
            self.rendered_data = [self.load_image(f) for f in tqdm(self.rendered_img_files)]
        else:
            self.orig_data = None
            self.rendered_data = None

        # Initialize renderers
        self.renderer_train = GaussianRenderer().to(render_device)
        self.renderer_infer = GaussianRenderer(clamp=True).to(render_device)

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        # Load raw data
        if self.dynamic_loading:
            raw_gaussians = self.construct_data_raw(self.pt_files[idx])
        else:
            raw_gaussians = self.raw_data_cache[idx]

        # Apply transforms if not raw
        if not self.raw:
            gaussians = self.apply_transforms(raw_gaussians.clone())
        else:
            gaussians = raw_gaussians

        label = self.labels[idx]
        return gaussians, label

    def get_raw_data(self, idx):
        """
        Get the raw Gaussian parameters without transformations.
        """
        if self.dynamic_loading:
            raw_gaussians = self.construct_data_raw(self.pt_files[idx])
        else:
            raw_gaussians = self.raw_data_cache[idx]
        return raw_gaussians
    
    def get_base_transformed_data(self, idx):
        """
        Get raw data and apply base transformations.
        This function assumes it is operating on raw data.
        """
        data = self.get_raw_data(idx).clone()

        # Apply base transformations:
        # 1. scaling -> ensure positive + offset: abs + 0.3
        # 2. rotation -> sigmoid to [0, 2*pi), then mod pi
        # 3. xy -> tanh
        # 4. normalization if requested
        # 5. specific normalization if requested

        # XY transform (tanh)
        data[:, :2] = torch.tanh(data[:, :2])

        # Scaling transform: abs + 0.3
        data[:, 2:4] = torch.abs(data[:, 2:4]) + 0.3

        # Rotation transform: sigmoid -> [0,2*pi) mod pi
        data[:, 4] = (torch.sigmoid(data[:, 4]) * 2 * np.pi) % np.pi
        
        return data
    
    def get_jittered_data_for_rendering(self, idx):
        """
        Get raw data and apply only jittering for visualization.
        We do NOT apply normalization or other transformations here.
        Just jitter the raw parameters so they remain in a form
        that the renderer can handle.
        """
        data = self.get_raw_data(idx).clone()

        # Apply jitter directly on raw data if desired:
        if self.xy_jitter > 0:
            data[:, :2] += torch.randn_like(data[:, :2]) * self.xy_jitter
        if self.scaling_jitter > 0:
            data[:, 2:4] += torch.randn_like(data[:, 2:4]) * self.scaling_jitter
        if self.rotation_jitter > 0:
            data[:, 4] += torch.randn_like(data[:, 4]) * self.rotation_jitter
        if self.value_jitter > 0:
            data[:, 5:] += torch.randn_like(data[:, 5:]) * self.value_jitter
        
        if self.xy_flip: 
            if torch.rand(1) > 0.5:
                data[:, 0] = -data[:, 0]
        # Ensure no ordering randomization here, or do it if you want to visualize that too:
        if self.order_random:
            data = data[torch.randperm(128)]

        return data
    
    def get_transformed_data_for_rendering(self, idx):
        """
        Get raw data and apply transformations for visualization.
        We do NOT apply normalization or other transformations here.
        Just transform the raw parameters so they remain in a form
        that the renderer can handle.
        """
        data = self.get_raw_data(idx).clone()

        # Apply jitter directly on raw data if desired:
        if self.xy_jitter > 0:
            data[:, :2] += torch.randn_like(data[:, :2]) * self.xy_jitter
        if self.scaling_jitter > 0:
            data[:, 2:4] += torch.randn_like(data[:, 2:4]) * self.scaling_jitter
        if self.rotation_jitter > 0:
            data[:, 4] += torch.randn_like(data[:, 4]) * self.rotation_jitter
        if self.value_jitter > 0:
            data[:, 5:] += torch.randn_like(data[:, 5:]) * self.value_jitter
        
        if self.xy_flip:
            # flip x
            data[:, 0] = -data[:, 0]
            # flip gaussians (rotate by pi)
            data[:, 4] = (data[:, 4] + np.pi) % (2 * np.pi)
                
        if self.global_xy_shift != (0.0, 0.0):
            shift_x, shift_y = self.global_xy_shift
            data[:, 0] += shift_x
            data[:, 1] += shift_y
            
        if self.global_value_shift != 0.0:
            data[:, 5:] += self.global_value_shift 
            
        if self.invisibility_prob > 0.0:
            mask = torch.rand(128) < self.invisibility_prob
            data[mask, 5:] = 0.0
            
        if self.global_rotation_angle != 0.0:
            angle = self.global_rotation_angle
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            # Rotate XY
            x = data[:, 0].clone()
            y = data[:, 1].clone()
            data[:, 0] = x * cos_a - y * sin_a
            data[:, 1] = x * sin_a + y * cos_a
            # Rotate angle
            data[:, 4] = (data[:, 4] + angle) % (2 * np.pi)
    
    
        # Ensure no ordering randomization here, or do it if you want to visualize that too:
        if self.order_random:
            data = data[torch.randperm(128)]

        return data

    def get_raw_data_for_rendering(self, idx):
        """
        Convenience function to get raw data specifically for rendering.
        """
        return self.get_raw_data(idx)

    def image_item(self, idx):
        """
        Return original and rendered images for the given index.
        If dynamic loading is enabled (and init_images_flag is False),
        load images on demand.
        """
        if self.dynamic_loading and not self.init_images_flag:
            # Lazy loading of images
            orig_img = self.orig_data[idx] if (self.orig_data is not None and self.orig_data[idx] is not None) else self.load_image(self.orig_img_files[idx])
            rendered_img = self.rendered_data[idx] if (self.rendered_data is not None and self.rendered_data[idx] is not None) else self.load_image(self.rendered_img_files[idx])

            # Cache them if desired
            if self.orig_data is None:
                self.orig_data = [None] * len(self.orig_img_files)
            if self.rendered_data is None:
                self.rendered_data = [None] * len(self.rendered_img_files)

            self.orig_data[idx] = orig_img
            self.rendered_data[idx] = rendered_img

            return orig_img, rendered_img
        else:
            # Already loaded or preloaded
            return self.orig_data[idx], self.rendered_data[idx]

    def info_item(self, idx):
        """
        Return label and psnr information for the given index.
        """
        return self.labels[idx], self.psnrs[idx]

    def load_all_labels(self, dir_path):
        """
        Load all labels and psnrs from label files.
        """
        label_files = self.collect_files(dir_path, '.txt')
        labels, psnrs = [], []
        for f in label_files:
            label, psnr = self.load_label(f)
            labels.append(label)
            psnrs.append(psnr)
        return labels, psnrs, label_files

    def collect_files(self, dir_path, suffix):
        """
        Collect and sort files with a given suffix from a directory.
        """
        files = os.listdir(dir_path)
        filtered = [f for f in files if f.endswith(suffix)]
        filtered.sort()
        return [os.path.join(dir_path, f) for f in filtered]

    def load_label(self, path):
        """
        Load a single label and psnr from a .txt file.
        """
        with open(path, 'r') as f:
            lines = f.read().strip().split('\n')
        label = torch.tensor(int(lines[0]), dtype=torch.int64)
        psnr = torch.tensor(float(lines[1]), dtype=torch.float32)
        return label, psnr

    def load_image(self, path):
        """
        Load and transform an image from disk.
        """
        img = Image.open(path).convert('RGB')
        return self.transform(img)

    def construct_data_raw(self, pt_file):
        """
        Load raw Gaussian parameters from a .pt file without any transformations.
        This returns the raw data as stored, suitable for renderer input.
        """
        data = torch.load(pt_file, weights_only=True)

        # Expected keys and their dimensionalities
        keys = {
            'xy': 2,
            'scaling': 2,
            'rotation': 1,
            'values': 3
        }

        return_tensor = torch.zeros((128, sum(keys.values())), dtype=torch.float32)
        start = 0
        for key, length in keys.items():
            param_data = data[key]
            return_tensor[:, start:start+length] = param_data
            start += length

        return return_tensor
    
    def apply_base_transforms(self, data):
        """
        Apply transformations, normalization, jitter, etc. to raw data.
        This function assumes it is operating on raw data.
        """
        # Data format: [ [xy], [scaling], [rotation], [values] ]
        # Indices: xy:0-1, scaling:2-3, rotation:4, values:5-7

        # Apply default transformations if raw=False
        # If not raw, we apply these steps:
        # 1. scaling -> ensure positive + offset: abs + 0.3
        # 2. rotation -> sigmoid to [0, 2*pi), then mod pi
        # 3. xy -> tanh
        # 4. normalization if requested
        # 5. specific normalization if requested

        # XY transform (tanh)
        data[:, :2] = torch.tanh(data[:, :2])

        # Scaling transform: abs + 0.3
        data[:, 2:4] = torch.abs(data[:, 2:4]) + 0.3

        # Rotation transform: sigmoid -> [0,2*pi) mod pi
        data[:, 4] = (torch.sigmoid(data[:, 4]) * 2 * np.pi) % np.pi
        
        return data

    def apply_transforms(self, data):
        """
        Apply transformations, normalization, jitter, etc. to raw data.
        This function assumes it is operating on raw data.
        """
        # Data format: [ [xy], [scaling], [rotation], [values] ]
        # Indices: xy:0-1, scaling:2-3, rotation:4, values:5-7

        # Apply default transformations if raw=False
        # If not raw, we apply these steps:
        # 1. scaling -> ensure positive + offset: abs + 0.3
        # 2. rotation -> sigmoid to [0, 2*pi), then mod pi
        # 3. xy -> tanh
        # 4. normalization if requested
        # 5. specific normalization if requested

        # XY transform (tanh)
        data[:, :2] = torch.tanh(data[:, :2])

        # Scaling transform: abs + 0.3
        data[:, 2:4] = torch.abs(data[:, 2:4]) + 0.3

        # Rotation transform: sigmoid -> [0,2*pi) mod pi
        data[:, 4] = (torch.sigmoid(data[:, 4]) * 2 * np.pi) % np.pi

        # Values are assumed to be in [0,1], no default transform except later normalization steps.

        # Apply normalization if requested
        if self.normalization == 'mean_std':
            # mean and std are dicts with keys 'xy', 'scaling', 'rotation', 'values'
            # We must apply them piecewise
            # Example:
            # data[:,0:2] = (data[:,0:2] - mean['xy']) / std['xy']
            # Make sure mean and std are tensors or broadcastable
            xy_mean, xy_std = self.mean['xy'], self.std['xy']
            scaling_mean, scaling_std = self.mean['scaling'], self.std['scaling']
            rotation_mean, rotation_std = self.mean['rotation'], self.std['rotation']
            values_mean, values_std = self.mean['values'], self.std['values']

            data[:, :2] = (data[:, :2] - xy_mean) / xy_std
            data[:, 2:4] = (data[:, 2:4] - scaling_mean) / scaling_std
            data[:, 4] = (data[:, 4] - rotation_mean) / rotation_std
            data[:, 5:] = (data[:, 5:] - values_mean) / values_std

        elif self.normalization == 'specific':
            # Specific normalization:
            # scaling -> log transform
            # rotation -> [0, pi] to [-1, 1]
            # xy -> none (already tanh)
            # values -> none ([0,1])
            data[:, 2:4] = torch.log(data[:, 2:4])
            data[:, 4] = (data[:, 4] / np.pi) * 2 - 1
            # xy and values remain as is

        # Apply jitter if requested
        if self.xy_jitter > 0 and torch.rand(1) > 0.5:
            data[:, :2] += torch.randn_like(data[:, :2]) * self.xy_jitter
        if self.scaling_jitter > 0 and torch.rand(1) > 0.5:
            data[:, 2:4] += torch.randn_like(data[:, 2:4]) * self.scaling_jitter
        if self.rotation_jitter > 0 and torch.rand(1) > 0.5:
            data[:, 4] += torch.randn_like(data[:, 4]) * self.rotation_jitter
        if self.value_jitter > 0 and torch.rand(1) > 0.5:
            data[:, 5:] += torch.randn_like(data[:, 5:]) * self.value_jitter
        if self.xy_flip and torch.rand(1) > 0.5:
                data[:, 0] = -data[:, 0]
        if self.order_random:
            data = data[torch.randperm(128)]
            
        if self.global_xy_shift != (0.0, 0.0) and torch.rand(1) > 0.5:
            shift_x, shift_y = self.global_xy_shift
            rand_x = torch.rand(1).item() * 2 - 1
            rand_y = torch.rand(1).item() * 2 - 1
            data[:, 0] += shift_x * rand_x
            data[:, 1] += shift_y * rand_y
            
        if self.global_value_shift != 0.0 and torch.rand(1) > 0.5:
            rand_val = torch.rand(1).item() * 2 - 1
            data[:, 5:] += self.global_value_shift * rand_val
            
        if self.invisibility_prob > 0.0 and torch.rand(1) > 0.5:
            mask = torch.rand(128) < self.invisibility_prob * torch.rand(1).item()
            data[mask, 5:] = 0.0  # Set RGB to 0
            
        if self.global_rotation_angle != 0.0 and torch.rand(1) > 0.5:
            angle = self.global_rotation_angle
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            # Rotate XY
            x = data[:, 0].clone()
            y = data[:, 1].clone()
            data[:, 0] = x * cos_a - y * sin_a
            data[:, 1] = x * sin_a + y * cos_a
            
            if self.normalization == 'specific':
                # rotation in [-1,1] -> convert to angle in [0, pi]
                rot_angle = (data[:, 4] + 1) / 2 * np.pi
                # add global rotation
                rot_angle = (rot_angle + angle) % np.pi
                # convert back to [-1,1]
                data[:, 4] = (rot_angle / np.pi) * 2 - 1
        return data

    def render_image(self, idx):
        """
        Render a single image in inference mode with raw data.
        """
        raw_data = self.get_raw_data(idx)
        return self.renderer_infer.forward_single(raw_data)
    
    def render_jittered_image(self, idx):
        """
        Render a single image in inference mode after applying jitter to the raw data.
        This allows you to visualize the effect of jittering.
        """
        jittered_data = self.get_jittered_data_for_rendering(idx)
        return self.renderer_infer.forward_single(jittered_data)

    def render_image_train(self, idx):
        """
        Render a single image in training mode.
        For rendering, we must pass raw data to the renderer.
        """
        raw_data = self.get_raw_data_for_rendering(idx)
        return self.renderer_train.forward_single(raw_data)

    def render_images(self, idxs):
        """
        Render multiple images in inference mode.
        For rendering, we must pass raw data to the renderer.
        """
        x = [self.get_raw_data_for_rendering(i) for i in idxs]
        x = torch.stack(x, dim=0)
        return self.renderer_infer(x)

    def render_images_train(self, idxs):
        """
        Render multiple images in training mode.
        For rendering, we must pass raw data to the renderer.
        """
        x = [self.get_raw_data_for_rendering(i) for i in idxs]
        x = torch.stack(x, dim=0)
        return self.renderer_train(x)

    def render_novel(self, data):
        """
        Render novel data in training mode.
        Here 'data' should be raw format suitable for the renderer.
        """
        return self.renderer_train(data)

    def render_novel_infer(self, data):
        """
        Render novel data in inference mode.
        Here 'data' should be raw format suitable for the renderer.
        """
        return self.renderer_infer(data)


# import os
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset
# from torchvision import transforms
# from PIL import Image
# from tqdm import tqdm

# from renderer import GaussianRenderer


# class GSCIFAR10(Dataset):
#     def __init__(
#         self,
#         dir_path='./cifar10/train',
#         label_path='./cifar10/train',
#         orig_path='./cifar10/train_orig',
#         rendered_path='./cifar10/train_rendered',
#         init_images=False,
#         raw=True,
#         dynamic_loading=False,
#         normalization=None,
#         xy_jitter=0.0,
#         scaling_jitter=0.0,
#         rotation_jitter=0.0,
#         value_jitter=0.0,
#         order_random=False,
#         device='cuda'
#     ):
#         """
#         GSCIFAR10 Dataset for loading Gaussian parameters, labels, and optional images.
        
#         Args:
#             dir_path (str): Directory containing .pt files with Gaussian parameters.
#             label_path (str): Directory containing .txt files with labels and psnrs.
#             orig_path (str): Directory containing original images.
#             rendered_path (str): Directory containing rendered images.
#             init_images (bool): If True and not dynamic loading, preload images into memory.
#             raw (bool): If True, load parameters as raw outputs; if False, apply transformations.
#             dynamic_loading (bool): If True, load .pt data on demand rather than preloading.
#             normalization (None or str): If 'mean_std', apply mean-std normalization; 
#                                          if 'specific', apply a custom normalization scheme.
#             xy_jitter, scaling_jitter, rotation_jitter, value_jitter (float): Jitters to apply.
#             order_random (bool): If True, randomly permute the order of Gaussians in each sample.
#             device (str): Device to place renderer models.
#         """
#         self.dir_path = dir_path
#         self.label_path = label_path
#         self.orig_path = orig_path
#         self.rendered_path = rendered_path

#         self.xy_jitter = xy_jitter
#         self.scaling_jitter = scaling_jitter
#         self.rotation_jitter = rotation_jitter
#         self.value_jitter = value_jitter
#         self.order_random = order_random
#         self.raw = raw
#         self.dynamic_loading = dynamic_loading
#         self.init_images_flag = init_images and not dynamic_loading

#         # Check normalization conditions
#         # Normalization only makes sense when raw=False
#         assert not (self.raw and normalization is not None), "Normalization only supported for raw=False."
#         assert normalization in [None, 'mean_std', 'specific'], "Invalid normalization type."
#         self.normalization = normalization

#         # If mean/std normalization is requested, load them
#         if self.normalization == 'mean_std':
#             self.mean = torch.load('mean.pt', weights_only=True)
#             self.std = torch.load('std.pt', weights_only=True)

#         self.transform = transforms.ToTensor()

#         # Load file lists and data
#         self.pt_files = self.collect_files(self.dir_path, '.pt')
#         self.labels, self.psnrs, self.label_files = self.load_all_labels(self.label_path)

#         # Preload PT data if not dynamic
#         if self.dynamic_loading:
#             self.data = None
#         else:
#             self.data = [self.construct_data(f) for f in tqdm(self.pt_files)]

#         # Manage images
#         self.orig_img_files = self.collect_files(self.orig_path, '.png')
#         self.rendered_img_files = self.collect_files(self.rendered_path, '.png')
        
#         if self.init_images_flag:
#             self.orig_data = [self.load_image(f) for f in tqdm(self.orig_img_files)]
#             self.rendered_data = [self.load_image(f) for f in tqdm(self.rendered_img_files)]
#         else:
#             self.orig_data = None
#             self.rendered_data = None

#         # Initialize renderers
#         self.renderer_train = GaussianRenderer().to(device)
#         self.renderer_infer = GaussianRenderer(clamp=True).to(device)

#     def __len__(self):
#         return len(self.pt_files)

#     def __getitem__(self, idx):
#         # On-demand loading if dynamic_loading is True
#         if self.dynamic_loading:
#             gaussians = self.construct_data(self.pt_files[idx])
#             gaussians = self.apply_transforms(gaussians)
#             label = self.labels[idx]
#             return gaussians, label
#         else:
#             data = self.apply_transforms(self.data[idx])
#             return data, self.labels[idx]

#     def image_item(self, idx):
#         """
#         Return original and rendered images for the given index.
#         If dynamic loading is enabled (and init_images_flag is False),
#         load images on demand.
#         """
#         if self.dynamic_loading and not self.init_images_flag:
#             # Lazy loading of images
#             orig_img = self.orig_data[idx] if (self.orig_data is not None and self.orig_data[idx] is not None) else self.load_image(self.orig_img_files[idx])
#             rendered_img = self.rendered_data[idx] if (self.rendered_data is not None and self.rendered_data[idx] is not None) else self.load_image(self.rendered_img_files[idx])

#             # Cache them if desired
#             if self.orig_data is None:
#                 self.orig_data = [None] * len(self.orig_img_files)
#             if self.rendered_data is None:
#                 self.rendered_data = [None] * len(self.rendered_img_files)

#             self.orig_data[idx] = orig_img
#             self.rendered_data[idx] = rendered_img

#             return orig_img, rendered_img
#         else:
#             # Already loaded or preloaded
#             return self.orig_data[idx], self.rendered_data[idx]

#     def info_item(self, idx):
#         """
#         Return label and psnr information for the given index.
#         """
#         return self.labels[idx], self.psnrs[idx]

#     def load_all_labels(self, dir_path):
#         """
#         Load all labels and psnrs from label files.
#         """
#         label_files = self.collect_files(dir_path, '.txt')
#         labels, psnrs = [], []
#         for f in label_files:
#             label, psnr = self.load_label(f)
#             labels.append(label)
#             psnrs.append(psnr)
#         return labels, psnrs, label_files

#     def collect_files(self, dir_path, suffix):
#         """
#         Collect and sort files with a given suffix from a directory.
#         """
#         files = os.listdir(dir_path)
#         filtered = [f for f in files if f.endswith(suffix)]
#         filtered.sort()
#         return [os.path.join(dir_path, f) for f in filtered]

#     def load_label(self, path):
#         """
#         Load a single label and psnr from a .txt file.
#         """
#         with open(path, 'r') as f:
#             lines = f.read().strip().split('\n')
#         label = torch.tensor(int(lines[0]), dtype=torch.int64)
#         psnr = torch.tensor(float(lines[1]), dtype=torch.float32)
#         return label, psnr

#     def load_image(self, path):
#         """
#         Load and transform an image from disk.
#         """
#         img = Image.open(path).convert('RGB')
#         return self.transform(img)

#     def construct_data(self, pt_file):
#         """
#         Load and process Gaussian parameters from a .pt file.
#         Applies raw or transformed loading, normalization, and jitter.
#         """
#         data = torch.load(pt_file, weights_only=True)

#         # Expected keys and their dimensionalities
#         keys = {
#             'xy': 2,
#             'scaling': 2,
#             'rotation': 1,
#             'values': 3
#         }

#         return_tensor = torch.zeros((128, sum(keys.values())), dtype=torch.float32)
#         start = 0

#         for key, length in keys.items():
#             param_data = data[key]

#             # If not raw, apply default transformations:
#             if not self.raw:
#                 if key == 'scaling':
#                     # Ensure positive scaling: abs + 0.3 offset
#                     param_data = torch.abs(param_data) + 0.3
#                 elif key == 'rotation':
#                     # Rotation: map from (0,1) via sigmoid to [0, 2*pi), then mod pi
#                     param_data = (torch.sigmoid(param_data) * 2 * np.pi) % np.pi
#                 elif key == 'xy':
#                     # XY: tanh normalization
#                     param_data = torch.tanh(param_data)

#             # Apply normalization if requested
#             if self.normalization == 'mean_std':
#                 param_data = (param_data - self.mean[key]) / self.std[key]
#             elif self.normalization == 'specific':
#                 # Specific normalization:
#                 # xy -> none (already tanh in [-1,1])
#                 # scaling -> log transform
#                 # rotation -> map [0, pi] to [-1, 1]
#                 # values -> none ([0,1])
#                 if key == 'scaling':
#                     param_data = torch.log(param_data)
#                 elif key == 'rotation':
#                     param_data = (param_data / np.pi) * 2 - 1

         

#             return_tensor[:, start:start+length] = param_data
#             start += length

       

#         return return_tensor
    
#     def apply_transforms(self, data):
#         """
#         Apply transformations to a given data tensor.
#         """
    
#         # Apply jitter if requested
#         if self.xy_jitter > 0:
#             data[:, :2] += torch.randn_like(data[:, :2]) * self.xy_jitter
#         if self.scaling_jitter > 0:
#             data[:, 2:4] += torch.randn_like(data[:, 2:4]) * self.scaling_jitter
#         if self.rotation_jitter > 0:
#             data[:, 4] += torch.randn_like(data[:, 4]) * self.rotation_jitter
#         if self.value_jitter > 0:
#             data[:, 5:] += torch.randn_like(data[:, 5:]) * self.value_jitter
#         if self.order_random:
#             data = data[torch.randperm(128)]
#         return data

#     def render_image(self, idx):
#         """
#         Render a single image in inference mode.
#         """
#         data = self[idx][0] if self.dynamic_loading else self.data[idx]
#         return self.renderer_infer.forward_single(data)

#     def render_image_train(self, idx):
#         """
#         Render a single image in training mode.
#         """
#         data = self[idx][0] if self.dynamic_loading else self.data[idx]
#         return self.renderer_train.forward_single(data)

#     def render_images(self, idxs):
#         """
#         Render multiple images in inference mode.
#         """
#         x = [self[i][0] if self.dynamic_loading else self.data[i] for i in idxs]
#         x = torch.stack(x, dim=0)
#         return self.renderer_infer(x)

#     def render_images_train(self, idxs):
#         """
#         Render multiple images in training mode.
#         """
#         x = [self[i][0] if self.dynamic_loading else self.data[i] for i in idxs]
#         x = torch.stack(x, dim=0)
#         return self.renderer_train(x)

#     def render_novel(self, data):
#         """
#         Render novel data in training mode.
#         """
#         return self.renderer_train(data)

#     def render_novel_infer(self, data):
#         """
#         Render novel data in inference mode.
#         """
#         return self.renderer_infer(data)
