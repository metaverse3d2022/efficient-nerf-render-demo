from train import EfficientNeRFSystem, NerfTree_Pytorch
import torch
import time
from datasets import dataset_dict
from collections import defaultdict
from models.nerf import Embedding, NeRF
from utils import load_ckpt
import os
from torchvision import transforms
from models.sh import eval_sh
from datasets.blender import pose_spherical
from datasets.ray_utils import get_ray_directions, get_rays
import numpy as np

def config_parser():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str,
                        default='data/nerf_synthetic/lego',
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='blender',
                        help='which dataset to train/val')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[800, 800],
                        help='resolution (img_w, img_h) of the image')
    parser.add_argument('--spheric_poses', default=False, action="store_true",
                        help='whether images are taken in spheric poses ()')

    parser.add_argument('--N_samples', type=int, default=128,
                        help='number of coarse samples')
    parser.add_argument('--N_importance', type=int, default=5,
                        help='number of additional fine samples')
    parser.add_argument('--use_disp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument('--perturb', type=float, default=1.0,
                        help='factor to perturb depth sampling points')
    parser.add_argument('--noise_std', type=float, default=1.0,
                        help='std dev of noise added to regularize sigma')
        
    parser.add_argument('--loss_type', type=str, default='mse',
                        choices=['mse'],
                        help='loss to use')

    parser.add_argument('--batch_size', type=int, default=4096,
                        help='batch size')
    parser.add_argument('--chunk', type=int, default=32*1024,
                        help='chunk size to split the input to avoid OOM')
    parser.add_argument('--num_epochs', type=int, default=16,
                        help='number of training epochs')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')

    parser.add_argument('--ckpt_path', type=str, 
                        default='logs/lego_coarse128_fine5_V384/version_10/checkpoints/epoch=5-step=93750.ckpt',
                        help='pretrained checkpoint path to load')
    parser.add_argument('--prefixes_to_ignore', nargs='+', type=str, default=['loss'],
                        help='the prefixes to ignore in the checkpoint state dict')

    parser.add_argument('--optimizer', type=str, default='radam',
                        help='optimizer type',
                        choices=['sgd', 'adam', 'radam', 'ranger', 'adamw'])
    parser.add_argument('--lr', type=float, default=0.002,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='learning rate momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight decay')
    parser.add_argument('--lr_scheduler', type=str, default='poly',
                        help='scheduler type',
                        choices=['steplr', 'cosine', 'poly'])
    #### params for warmup, only applied when optimizer == 'sgd' or 'adam'
    parser.add_argument('--warmup_multiplier', type=float, default=1.0,
                        help='lr is multiplied by this factor after --warmup_epochs')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Gradually warm-up(increasing) learning rate in optimizer')
    ###########################
    #### params for steplr ####
    parser.add_argument('--decay_step', nargs='+', type=int, default=[20],
                        help='scheduler decay step')
    parser.add_argument('--decay_gamma', type=float, default=0.1,
                        help='learning rate decay amount')
    ###########################
    #### params for poly ####
    parser.add_argument('--poly_exp', type=float, default=0.9,
                        help='exponent for polynomial learning rate decay')
    ###########################

    parser.add_argument('--exp_name', type=str, default='lego_coarse128_fine5_V384',
                        help='experiment name')
    
    parser.add_argument('--coord_scope', type=float, default=3.0,
                        help='the scope of world coordnates')

    parser.add_argument('--sigma_init', type=float, default=30.0,
                        help='the init sigma')

    parser.add_argument('--sigma_default', type=float, default=-20.0,
                        help='the default sigma')

    parser.add_argument('--weight_threashold', type=float, default=1e-5,
                        help='the weight threashold')

    parser.add_argument('--uniform_ratio', type=float, default=0.01,
                        help='the percentage of uniform sampling')
    
    parser.add_argument('--beta', type=float, default=0.1,
                        help='update rate')

    parser.add_argument('--warmup_step', type=int, default=5000,
                        help='the warmup step')
    
    parser.add_argument('--weight_sparse', type=float, default=0.0,
                        help='weight of sparse loss')
    
    parser.add_argument('--weight_tv', type=float, default=0.0,
                        help='weight of tv loss')
    return parser.parse_args()


hparams = config_parser()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#system = EfficientNeRFSystem.load_from_checkpoint(checkpoint_path='logs/lego_coarse128_fine5_V384/version_0/checkpoints/epoch=15-step=250000.ckpt', map_location=torch.device('cpu'), hparams=hparams)

'''
dataset = dataset_dict["blender"]
val_dataset = dataset(split='test', root_dir=hparams.root_dir, img_wh=hparams.img_wh)

sample = val_dataset[0]

#rays, rgbs = model.decode_batch(sample)
rays = sample['rays'].to(device)
#rays = rays.squeeze()
#rgbs = rgbs.squeeze()
'''

c2w = pose_spherical(90,-30,4)
w, h = hparams.img_wh
focal = 0.5*w/np.tan(0.5*0.6911112070083618)
directions = get_ray_directions(h, w, focal)
directions = directions / torch.norm(directions, dim=-1, keepdim=True)
rays_o, rays_d = get_rays(directions, c2w)
rays = torch.cat([rays_o, rays_d], 1).to(device)

system_dict = {}

system_dict['embedding_xyz'] = Embedding(3, 10) # 10 is the default number
system_dict['embedding_dir'] = Embedding(3, 4) # 4 is the default number
system_dict['embeddings'] = [system_dict['embedding_xyz'], system_dict['embedding_dir']]

system_dict['deg'] = 2
system_dict['dim_sh'] = 3 * (system_dict['deg'] + 1)**2

system_dict['nerf_coarse'] = NeRF(D=4, W=128,
                        in_channels_xyz=63, in_channels_dir=27, 
                        skips=[2], deg=system_dict['deg'])
load_ckpt(system_dict['nerf_coarse'], hparams.ckpt_path, 'nerf_coarse')
system_dict['models'] = [system_dict['nerf_coarse']]
if hparams.N_importance > 0:
    system_dict['nerf_fine'] = NeRF(D=8, W=256,
                        in_channels_xyz=63, in_channels_dir=27, 
                        skips=[4], deg=system_dict['deg'])
    load_ckpt(system_dict['nerf_fine'], hparams.ckpt_path, 'nerf_fine')
    system_dict['models'] += [system_dict['nerf_fine']]

system_dict['sigma_init'] = hparams.sigma_init
system_dict['sigma_default'] = hparams.sigma_default

# sparse voxels
coord_scope = hparams.coord_scope
nerf_tree = NerfTree_Pytorch(xyz_min=[-coord_scope, -coord_scope, -coord_scope], 
                                    xyz_max=[coord_scope, coord_scope, coord_scope], 
                                    grid_coarse=384, 
                                    grid_fine=3,
                                    deg=system_dict['deg'], 
                                    sigma_init=system_dict['sigma_init'], 
                                    sigma_default=system_dict['sigma_default'],
                                    device='cuda')
os.makedirs(f'logs/{hparams.exp_name}/ckpts', exist_ok=True)
system_dict['nerftree_path'] = os.path.join(f'logs/{hparams.exp_name}/ckpts', 'nerftree.pt')
if hparams.ckpt_path != None and os.path.exists(system_dict['nerftree_path']):
    voxels_dict = torch.load(system_dict['nerftree_path'], map_location=torch.device('cuda'))
    nerf_tree.sigma_voxels_coarse = voxels_dict['sigma_voxels_coarse']

# fine voxels
nerf_tree.index_voxels_coarse = voxels_dict['index_voxels_coarse']
nerf_tree.voxels_fine = voxels_dict['voxels_fine']

# prepare_data

system_dict['near'] = 2.0
system_dict['far'] = 6.0
system_dict['distance'] = system_dict['far'] - system_dict['near']
near = torch.full((1,), system_dict['near'], dtype=torch.float32, device='cuda')
far = torch.full((1,), system_dict['far'], dtype=torch.float32, device='cuda')

# z_vals_coarse
system_dict['N_samples_coarse'] = hparams.N_samples
z_vals_coarse = torch.linspace(0, 1, system_dict['N_samples_coarse'], device='cuda') # (N_samples_coarse)
if not hparams.use_disp: # use linear sampling in depth space
    z_vals_coarse = near * (1-z_vals_coarse) + far * z_vals_coarse
else: # use linear sampling in disparity space
    z_vals_coarse = 1/(1/near * (1-z_vals_coarse) + 1/far * z_vals_coarse)   # (N_rays, N_samples_coarse)
system_dict['z_vals_coarse'] = z_vals_coarse.unsqueeze(0)

# z_vals_fine
system_dict['N_samples_fine'] = hparams.N_samples * hparams.N_importance
z_vals_fine = torch.linspace(0, 1, system_dict['N_samples_fine'], device='cuda') # (N_samples_coarse)
if not hparams.use_disp: # use linear sampling in depth space
    z_vals_fine = near * (1-z_vals_fine) + far * z_vals_fine
else: # use linear sampling in disparity space
    z_vals_fine = 1/(1/near * (1-z_vals_fine) + 1/far * z_vals_fine)   # (N_rays, N_samples_coarse)
system_dict['z_vals_fine'] = z_vals_fine.unsqueeze(0)



def sigma2weights(deltas, sigmas):
        # compute alpha by the formula (3)
        # if self.training:
        #noise = torch.randn(sigmas.shape, device=sigmas.device)
        #sigmas = sigmas + noise

        # alphas = 1-torch.exp(-deltas*torch.nn.ReLU()(sigmas)) # (N_rays, N_samples_)
        alphas = 1-torch.exp(-deltas*torch.nn.Softplus()(sigmas)) # (N_rays, N_samples_)
        alphas_shifted = torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas+1e-10], -1) # [1, a1, a2, ...]
        weights = alphas * torch.cumprod(alphas_shifted, -1)[:, :-1] # (N_rays, N_samples_)
        return weights, alphas


def render_rays(models,
                embeddings,
                rays,
                N_samples=64,
                use_disp=False,
                noise_std=0.0,
                N_importance=0,
                chunk=1024*32,
                white_back=False
                ):

        def inference(model, embedding_xyz, xyz_, dir_, dir_embedded, z_vals, idx_render):
            N_samples_ = xyz_.shape[1]
            # Embed directions
            xyz_ = xyz_[idx_render[:, 0], idx_render[:, 1]].view(-1, 3) # (N_rays*N_samples_, 3)
            view_dir = dir_.unsqueeze(1).expand(-1, N_samples_, -1)
            view_dir = view_dir[idx_render[:, 0], idx_render[:, 1]]
            # Perform model inference to get rgb and raw sigma
            B = xyz_.shape[0]
            '''
            out_chunks = []
            for i in range(0, B, chunk):
                out_chunks += [model(embedding_xyz(xyz_[i:i+chunk]), view_dir[i:i+chunk])]
            out = torch.cat(out_chunks, 0)
            '''
            
            out_fine = nerf_tree.query_fine(xyz_)
            sigma, sh = torch.split(out_fine, (1, system_dict['dim_sh']), dim=-1)
            #print('out: ', out_fine.shape)



            deg = 2
            rgb = eval_sh(deg=deg, sh=sh.reshape(-1, 3, (deg + 1)**2), dirs=view_dir)
            rgb = torch.sigmoid(rgb)
            out = torch.cat([sigma, rgb, sh], dim=1)
           
            out_rgb = torch.full((N_rays, N_samples_, 3), 1.0, device=device)
            out_sigma = torch.full((N_rays, N_samples_, 1), system_dict['sigma_default'], device=device)
            out_sh = torch.full((N_rays, N_samples_, system_dict['dim_sh']), 0.0, device=device)
            out_defaults = torch.cat([out_sigma, out_rgb, out_sh], dim=2)
            out_defaults[idx_render[:, 0], idx_render[:, 1]] = out
            out = out_defaults

            sigmas, rgbs, shs = torch.split(out, (1, 3, system_dict['dim_sh']), dim=-1)
            del out
            sigmas = sigmas.squeeze(-1)
                    
            # Convert these values using volume rendering (Section 4)
            deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples_-1)
            delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
            deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)
            
            weights, alphas = sigma2weights(deltas, sigmas)

            weights_sum = weights.sum(1) # (N_rays), the accumulated opacity along the rays
                                        # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically

            # compute final weighted outputs
            rgb_final = torch.sum(weights.unsqueeze(-1)*rgbs, -2) # (N_rays, 3)
            depth_final = torch.sum(weights*z_vals, -1) # (N_rays)

            if white_back:
                rgb_final = rgb_final + 1-weights_sum.unsqueeze(-1)

            return rgb_final, depth_final, weights, sigmas, shs

        
        # Extract models from lists
        model_coarse = models[0]
        embedding_xyz = embeddings[0]
        device = rays.device

        result = {}

        # Decompose the inputs
        N_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)

        # Embed direction
        dir_embedded = None
        
        N_samples_coarse = system_dict['N_samples_coarse']
        z_vals_coarse = system_dict['z_vals_coarse'].clone().expand(N_rays, -1)
        
        xyz_sampled_coarse = rays_o.unsqueeze(1) + \
                             rays_d.unsqueeze(1) * z_vals_coarse.unsqueeze(2) # (N_rays, N_samples_coarse, 3)

        xyz_coarse = xyz_sampled_coarse.reshape(-1, 3)

        # valid sampling
        sigmas = nerf_tree.query_coarse(xyz_coarse, type='sigma').reshape(N_rays, N_samples_coarse)
        
        
        # deltas_coarse = self.deltas_coarse
        with torch.no_grad():
            deltas_coarse = z_vals_coarse[:, 1:] - z_vals_coarse[:, :-1] # (N_rays, N_samples_-1)
            delta_inf = 1e10 * torch.ones_like(deltas_coarse[:, :1]) # (N_rays, 1) the last delta is infinity
            deltas_coarse = torch.cat([deltas_coarse, delta_inf], -1)  # (N_rays, N_samples_)
            weights_coarse, _ = sigma2weights(deltas_coarse, sigmas)
            weights_coarse = weights_coarse.detach()

        # pivotal sampling
        idx_render = torch.nonzero(weights_coarse >= min(hparams.weight_threashold, weights_coarse.max().item()))
        scale = N_importance
        z_vals_fine = system_dict['z_vals_fine'].clone()


        idx_render = idx_render.unsqueeze(1).expand(-1, scale, -1)  # (B, scale, 2)
        idx_render_fine = idx_render.clone()
        idx_render_fine[..., 1] = idx_render[..., 1] * scale + (torch.arange(scale, device=device)).reshape(1, scale)
        idx_render_fine = idx_render_fine.reshape(-1, 2)

        #if idx_render_fine.shape[0] > N_rays * 64:
        #    indices = torch.randperm(idx_render_fine.shape[0])[:N_rays * 64]
        #    idx_render_fine = idx_render_fine[indices]
        
        xyz_sampled_fine = rays_o.unsqueeze(1) + \
                            rays_d.unsqueeze(1) * z_vals_fine.unsqueeze(2) # (N_rays, N_samples*scale, 3)

        # if self.nerf_tree.voxels_fine != None:
        #     xyz_norm = (xyz_sampled_fine - self.xyz_min) / self.xyz_scope
        #     xyz_norm = (xyz_norm * self.res_fine).long().float() / float(self.res_fine)
        #     xyz_sampled_fine = xyz_norm * self.xyz_scope + self.xyz_min

        model_fine = models[1]
        model_fine.to(device)
        rgb_fine, depth_fine, _, sigmas_fine, shs_fine = \
            inference(model_fine, embedding_xyz, xyz_sampled_fine, rays_d,
                    dir_embedded, z_vals_fine, idx_render_fine)

        result['rgb_fine'] = rgb_fine
        result['depth_fine'] = depth_fine
        result['num_samples_fine'] = torch.FloatTensor([idx_render_fine.shape[0] / N_rays])

        return result





def forward(rays):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    results = defaultdict(list)
    # if self.nerf_tree.voxels_fine == None or self.models[0].training:
    #     chunk = self.hparams.chunk
    # else:
    #     chunk = B // 8
    chunk = hparams.chunk
    for i in range(0, B, chunk):
        rendered_ray_chunks = \
            render_rays(system_dict['models'],
                        system_dict['embeddings'],
                        rays[i:i+chunk],
                        hparams.N_samples,
                        hparams.use_disp,
                        hparams.noise_std,
                        hparams.N_importance,
                        chunk, # chunk size is effective in val mode
                        True
                        )
                            
        for k, v in rendered_ray_chunks.items():
            results[k] += [v]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results


def batchify_rays(rays_flat, chunk=1024*32):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    
    for i in range(0, rays_flat.shape[0], chunk):
        ret = forward(rays_flat[i:i+chunk])
        #print(i, time.time())
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret
print(time.time())
results = batchify_rays(rays)
typ = 'fine' if 'rgb_fine' in results else 'coarse'
W, H = hparams.img_wh
img = results[f'rgb_{typ}'].view(H, W, 3)
img = img.permute(2, 0, 1)
print(time.time())
transforms.ToPILImage()(img).convert("RGB").save('test2.png')


