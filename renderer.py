import torch.nn as nn
import torch
import numpy as np

from gsplat.project_gaussians_2d_scale_rot import project_gaussians_2d_scale_rot
from gsplat.rasterize_sum import rasterize_gaussians_sum


class GaussianRenderer(nn.Module):
    def __init__(self, H=32, W=32, num_gaussians=128, clamp=False):
        super().__init__()
        self.clamp = clamp
        self.H, self.W = H, W
        self.BLOCK_W, self.BLOCK_H = 16, 16
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        )
        self.register_buffer('scale_bound', torch.tensor([0.3, 0.3]).view(1, 2))
        self.register_buffer('_opacity', torch.ones((num_gaussians, 1)))
        self.register_buffer('_background', torch.zeros(3))
    
    def forward(self, data):
        '''
        data: (B, N, 8) tensor
        '''
        xy = data[:, :, :2]
        scale = data[:, :, 2:4]
        rotation = data[:, :, 4:5]
        color = data[:, :, 5:]
        batch_size = data.shape[0]
        output = []
        # print(xy.shape, scale.shape, rotation.shape, color.shape)
        
        for i in range(batch_size):
            xy_i = torch.tanh(xy[i])
            scaling_i = torch.abs(scale[i]) + 0.3
            rotation_i = torch.sigmoid(rotation[i])*2*np.pi
            color_i = color[i]
            
            xys, depths, radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(
                xy_i, scaling_i, rotation_i, self.H, self.W, self.tile_bounds
            )
            out_img = rasterize_gaussians_sum(
                xys, depths, radii, conics, num_tiles_hit,
                color_i, self._opacity,
                self.H, self.W, self.BLOCK_H, self.BLOCK_W,
                background=self._background, return_alpha=False
            )
            if self.clamp:
                out_img = torch.clamp(out_img, 0, 1)
            out_img = out_img.view(self.H, self.W, 3).permute(2, 0, 1).contiguous()
            output.append(out_img)
        
        return torch.stack(output)
    
    def forward_single(self, data):
        '''
        data: (N, 8) tensor
        '''
        xy = data[:, :2]
        scale = data[:, 2:4]
        rotation = data[:, 4:5]
        color = data[:, 5:]


        xy_i = torch.tanh(xy)
        scaling_i = torch.abs(scale) + 0.3
        rotation_i = torch.sigmoid(rotation)*2*np.pi
        color_i = color
        #first 10 samplesclass GaussianRenderer(nn.Module):
    def __init__(self, H=32, W=32, num_gaussians=128, clamp=False):
        super().__init__()
        self.clamp = clamp
        self.H, self.W = H, W
        self.BLOCK_W, self.BLOCK_H = 16, 16
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        )
        self.register_buffer('scale_bound', torch.tensor([0.3, 0.3]).view(1, 2))
        self.register_buffer('_opacity', torch.ones((num_gaussians, 1)))
        self.register_buffer('_background', torch.zeros(3))
    
    def forward(self, data):
        '''
        data: (B, N, 8) tensor
        '''
        xy = data[:, :, :2]
        scale = data[:, :, 2:4]
        rotation = data[:, :, 4:5]
        color = data[:, :, 5:]
        batch_size = data.shape[0]
        output = []
        # print(xy.shape, scale.shape, rotation.shape, color.shape)
        
        for i in range(batch_size):
            xy_i = torch.tanh(xy[i])
            scaling_i = torch.abs(scale[i]) + 0.3
            rotation_i = torch.sigmoid(rotation[i])*2*np.pi
            color_i = color[i]
            
            xys, depths, radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(
                xy_i, scaling_i, rotation_i, self.H, self.W, self.tile_bounds
            )
            out_img = rasterize_gaussians_sum(
                xys, depths, radii, conics, num_tiles_hit,
                color_i, self._opacity,
                self.H, self.W, self.BLOCK_H, self.BLOCK_W,
                background=self._background, return_alpha=False
            )
            if self.clamp:
                out_img = torch.clamp(out_img, 0, 1)
            out_img = out_img.view(self.H, self.W, 3).permute(2, 0, 1).contiguous()
            output.append(out_img)
        
        return torch.stack(output)
    
    def forward_single(self, data):
        '''
        data: (N, 8) tensor
        '''
        xy = data[:, :2]
        scale = data[:, 2:4]
        rotation = data[:, 4:5]
        color = data[:, 5:]


        xy_i = torch.tanh(xy)
        scaling_i = torch.abs(scale) + 0.3
        rotation_i = torch.sigmoid(rotation)*2*np.pi
        color_i = color
        #first 10 samples
        xys, depths, radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(
            xy_i, scaling_i, rotation_i, self.H, self.W, self.tile_bounds
        )
        # first 10 samples
        out_img = rasterize_gaussians_sum(
            xys, depths, radii, conics, num_tiles_hit,
            color_i, self._opacity,
            self.H, self.W, self.BLOCK_H, self.BLOCK_W,
            background=self._background, return_alpha=False
        )
        if self.clamp:
            out_img = torch.clamp(out_img, 0, 1)

        out_img = out_img.view(self.H, self.W, 3).permute(2, 0, 1).contiguous()
        xys, depths, radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(
            xy_i, scaling_i, rotation_i, self.H, self.W, self.tile_bounds
        )
        # first 10 samples
        out_img = rasterize_gaussians_sum(
            xys, depths, radii, conics, num_tiles_hit,
            color_i, self._opacity,
            self.H, self.W, self.BLOCK_H, self.BLOCK_W,
            background=self._background, return_alpha=False
        )
        if self.clamp:
            out_img = torch.clamp(out_img, 0, 1)

        out_img = out_img.view(self.H, self.W, 3).permute(2, 0, 1).contiguous()
        return out_img
# class GaussianRenderer(nn.Module):
#     def __init__(self, H=32, W=32, num_gaussians=128, clamp=False):
#         super().__init__()
#         self.H, self.W = H, W
#         self.BLOCK_W, self.BLOCK_H = 16, 16
#         self.clamp = clamp
#         self.tile_bounds = (
#             (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
#             (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
#             1,
#         )
#         # self.register_buffer('scale_bound', torch.tensor([0.3, 0.3]).view(1, 2))
#         self.register_buffer('_opacity', torch.ones((num_gaussians, 1)))
#         self.register_buffer('_background', torch.zeros(3))
    
#     def forward(self, data):
#         '''
#         data: (B, N, 8) tensor
#         '''
#         xy = data[:, :, :2]
#         scale = data[:, :, 2:4]
#         rotation = data[:, :, 4:5]
#         color = data[:, :, 5:]
#         batch_size = data.shape[0]
#         output = []
#         # print(xy.shape, scale.shape, rotation.shape, color.shape)
        
#         for i in range(batch_size):
#             xy_i = torch.tanh(xy[i])
#             scaling_i = torch.abs(scale[i]) + 0.3
#             rotation_i = torch.sigmoid(rotation[i])*2*np.pi
#             color_i = color[i]
            
#             xys, depths, radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(
#                 xy_i, scaling_i, rotation_i, self.H, self.W, self.tile_bounds
#             )
#             out_img = rasterize_gaussians_sum(
#                 xys, depths, radii, conics, num_tiles_hit,
#                 color_i, self._opacity,
#                 self.H, self.W, self.BLOCK_H, self.BLOCK_W,
#                 background=self._background, return_alpha=False
#             )
#             if self.clamp:
#                 out_img = torch.clamp(out_img, 0, 1)
#             out_img = out_img.view(self.H, self.W, 3).permute(2, 0, 1).contiguous()
#             output.append(out_img)
        
#         return torch.stack(output)
    
#     def forward_single(self, data):
#         '''
#         data: (N, 8) tensor
#         '''
#         xy = data[:, :2]
#         scale = data[:, 2:4]
#         rotation = data[:, 4:5]
#         color = data[:, 5:]


#         xy_i = torch.tanh(xy)
#         scaling_i = torch.abs(scale) + 0.3
#         rotation_i = torch.sigmoid(rotation)*2*np.pi
#         color_i = color
#         # first 10 samples
#         # print(xy_i[:10])
#         # print(scaling_i[:10])
#         # print(rotation_i[:10])
#         # print(color_i[:10])
#         xys, depths, radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(
#             xy_i, scaling_i, rotation_i, self.H, self.W, self.tile_bounds
#         )
#         # first 10 samples
#         print("=============")
#         print(xys[:10])
#         print(depths[:10])
#         print(radii[:10])
#         print(conics[:10])
#         print(num_tiles_hit[:10])
#         out_img = rasterize_gaussians_sum(
#             xys, depths, radii, conics, num_tiles_hit,
#             color_i, self._opacity,
#             self.H, self.W, self.BLOCK_H, self.BLOCK_W,
#             background=self._background, return_alpha=False
#         )
#         print(out_img.shape)
#         print(out_img)
#         if self.clamp:
#             out_img = torch.clamp(out_img, 0, 1)

#         out_img = out_img.view(self.H, self.W, 3).permute(2, 0, 1).contiguous()
#         return out_img

if __name__ == '__main__':
    # Test the renderer
    import matplotlib.pyplot as plt
    from dataset import GSCIFAR10
    from torch.utils.data import DataLoader

    dataset = train_dataset = GSCIFAR10(dir_path='./cifar10/train', dynamic_loading=True, raw=False, normalization='mean_std', order_random=True, value_jitter=0.01, scaling_jitter=0.01, rotation_jitter=0.01, xy_jitter=0.01)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    renderer = GaussianRenderer(H=32, W=32, num_gaussians=128, clamp=True).to('cuda')
    for data, labels in dataloader:
        print(data.shape)
        data = data.to('cuda')
        out_img = renderer(data)
        print(out_img.shape)
        plt.imshow(out_img[0].permute(1, 2, 0).cpu().detach().numpy())
        plt.savefig('test.png')