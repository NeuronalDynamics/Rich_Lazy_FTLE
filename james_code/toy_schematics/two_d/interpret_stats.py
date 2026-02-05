import matplotlib.pyplot as plt
import numpy as np
import torch, os, glob
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from tqdm import tqdm
from itertools import product
from train import GetTask, GetModel
from kpflow.grad_op import HiddenNTKOperator as NTK
import sys
sys.path.append('../../')
from common import set_mpl_defaults, project
set_mpl_defaults(14)

def get_dist(x): 
    q = torch.abs(x) - np.sqrt(np.pi) / 2.
    d = torch.linalg.norm(torch.maximum(q, 0.*q), axis = 1) + torch.minimum(torch.max(q,1)[0], 0.*q[:,0]) # SDF of square.
    return d

def GetTaskUniform(nsamp): # nsamp should be a square.
    width = np.sqrt(2 * np.pi) # So area of +1 and -1 regions are same.
    nsamp_x = int(nsamp ** 0.5)
    X = torch.linspace(-width/2, width/2, nsamp_x)
    inps = torch.stack(torch.meshgrid(X, X)).reshape((2, nsamp)).T
    targs = get_dist(inps)[:, None]
    return inps, targs

class EvalAll(nn.Module): # Evaluate all activations of model and output it, not traditional output.
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        states = []
        for layer in self.net:
            states.append(layer(x))
            x = states[-1]
        return torch.stack(states[:-1]) # Hidden activations.

files = glob.glob('data/sweep/model*.pt')
print('Interpreting the data :).')
plot_idx = []

nsamp_x = 30
nsamp = nsamp_x ** 2 
inps, targs = GetTaskUniform(nsamp)

# All the data I've generated (combining two sweeps).
# To visualize a specific sweep, just remove any data that's not in it during the loop below.

for fl in tqdm(files):
    ld = torch.load(fl)

    config, sd = ld['config'], ld['model']
    L, N, gain, seed = config['depth'], config['width'], config['gain'], config['seed']

    model = GetModel(L, N, gain, True)
    model.load_state_dict(sd)
    model_all = EvalAll(model) # return ALL hidden activations 
    hidd = model_all(inps)

    w_out = model[-1]
    out = w_out(hidd).detach().numpy()

    model_out_all = lambda x : w_out(model_all(x))[:, :, 0].sum(1) # Sum along batches since no cross talk between distinct inputs (no grad info lost)
    pi = torch.autograd.functional.jacobian(model_out_all, inps) # [layer, batch, 2].

    plt.figure(figsize = (4*3, 3))
    plt.subplot(1,3,1)
    grid = out[-1, :, 0].reshape((nsamp_x, nsamp_x))
    plt.imshow(grid)
    plt.colorbar()
    plt.title('Output')
    plt.subplot(1,3,2)
    grad = np.zeros((grid.shape[0]-2,grid.shape[1]-2, 2))
    grad[:, :, 0] = grid[2:, 1:-1] - grid[:-2, 1:-1]
    grad[:, :, 1] = grid[1:-1, 2:] - grid[1:-1, :-2]
    grad = grad / (2. * (inps[1, 1] - inps[0, 1]))
    plt.imshow(np.linalg.norm(grad, axis = -1))
    plt.colorbar()
    plt.title('Finite Grad')
    plt.subplot(1,3,3)
    plt.imshow(np.linalg.norm(pi[-1], axis=-1).reshape(grid.shape))
    plt.colorbar()
    plt.title('Auto Grad')

    nm = np.linalg.norm(inps, axis = -1)
    colors = plt.get_cmap('viridis')(nm / max(nm))

    true_pi = torch.autograd.functional.jacobian((lambda x : get_dist(x).sum(0)), inps) # [layer, batch, 2].
    inds = np.arange(pi.shape[1])
    np.random.shuffle(inds)
    plt.figure()
    for idx in inds[:40]:
        plt.plot(pi[:, idx, 0], pi[:, idx, 1], color = colors[idx])

    plt.plot(np.cos(np.linspace(0., 2 * np.pi, 20)), np.sin(np.linspace(0., 2 * np.pi, 20)), color = 'black')

    plt.figure()
    plt.scatter(pi[0, :, 0], pi[0, :,1], c = colors)
    plt.scatter(pi[-1, :, 0], pi[-1, :,1], c = colors)
    plt.scatter(true_pi[:, 0], true_pi[:, 1])

    plt.figure()
    plt.plot(np.linalg.norm(pi, axis = -1))
    plt.show()
    
    plt.figure()
    normalized_dist = np.abs(1. - np.abs(inps)[:, 0]).detach().numpy()
    normalized_dist = np.zeros_like(normalized_dist)
    colors = plt.get_cmap('hot')(normalized_dist)
    for idx in range(out.shape[1]):
        plt.plot(pi[:, idx, 0], color = colors[idx])

#    plt.figure()
#    c0 = (model[0][0].weight.data.T @  model[-1].weight.data.T).item()
#    plt.axhline(c0)
#    plt.plot(inps[:,0], pi[-1, :, 0], c = 'black')


plt.show()
