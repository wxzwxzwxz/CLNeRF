import os
import torch

import numpy as np
import imageio
import mcubes
import pprint

import matplotlib.pyplot as plt

import run_nerf
import load_blender

# General setup for GPU device and default tensor type.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.cuda.FloatTensor')

parser = run_nerf.config_parser()
args = parser.parse_args()

basedir = './logs'
expname = args.expname
config = args.config

args.n_gpus = torch.cuda.device_count()

# Create nerf model
_, render_kwargs_test, _, _, _, _ = run_nerf.create_nerf(args, ckpt_path=args.ckpt_path)

if args.near:
    near = args.near
    far = args.far
else:
    near = 2.
    far = 6.
    
bds_dict = {
    'near' : near,
    'far' : far,
}
render_kwargs_test.update(bds_dict)

net_fn = render_kwargs_test['network_query_fn']

N = 256
# x = np.linspace(-3, 3, N+1)
# y = np.linspace(-3, 3, N+1)
# z = np.linspace(-3, 3, N+1)
x = np.linspace(-5, 5, N+1)
y = np.linspace(-5, 5, N+1)
z = np.linspace(-5, 5, N+1)

query_pts = np.stack(np.meshgrid(x, y, z), -1).astype(np.float32)
print(query_pts.shape)
sh = query_pts.shape
flat = torch.from_numpy(query_pts.reshape([-1,3]))

with torch.no_grad():
    fn = lambda i0, i1 : net_fn(flat[i0:i1,None,:].to(device), viewdirs=torch.zeros_like(flat[i0:i1]).to(device), network_fn=render_kwargs_test['network_fine'])
    chunk = 2048*640
    raw = np.concatenate([fn(i, i+chunk)[0].cpu().numpy() for i in range(0, flat.shape[0], chunk)], 0)
    raw = np.reshape(raw, list(sh[:-1]) + [-1])
    sigma = np.maximum(raw[...,-1], 0.)
    
threshold = 5 # 5 # 50.
print('fraction occupied', np.sum(sigma > threshold), np.mean(sigma > threshold))
vertices, triangles = mcubes.marching_cubes(sigma, threshold)
print('done', vertices.shape, triangles.shape, expname+"_{}.obj".format(N))

## Uncomment to save out the mesh
mcubes.export_obj(vertices, triangles, os.path.join("logs", expname, expname+"_{}.obj".format(N)))