import torch
import os
exp_name = 'lego_coarse128_fine5_V384'
system_dict={}
os.makedirs(f'logs/{exp_name}/ckpts', exist_ok=True)
system_dict['nerftree_path'] = os.path.join(f'logs/{exp_name}/ckpts', 'nerftree.pt')
voxels_dict = torch.load(system_dict['nerftree_path'], map_location=torch.device('cpu'))
print(voxels_dict['sigma_voxels_coarse'].shape)
print(voxels_dict['index_voxels_coarse'].shape)
print(voxels_dict['voxels_fine'].shape)

class Container(torch.nn.Module):
    def __init__(self, my_values):
        super().__init__()
        for key in my_values:
            setattr(self, key, my_values[key])

voxels_dict = {
    'sigma_voxels_coarse': voxels_dict['sigma_voxels_coarse'],
    'index_voxels_coarse': voxels_dict['index_voxels_coarse'],
    'voxels_fine': voxels_dict['voxels_fine']
}

# Save arbitrary values supported by TorchScript
# https://pytorch.org/docs/master/jit.html#supported-type
container = torch.jit.script(Container(voxels_dict))
container.save("voxels_dict.pt")