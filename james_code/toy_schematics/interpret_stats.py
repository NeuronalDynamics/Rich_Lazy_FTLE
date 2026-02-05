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
sys.path.append('../')
from common import set_mpl_defaults, project
set_mpl_defaults(14)

def GetTaskUniform(nsamp): # nsamp should be a square.
    inps = torch.linspace(-2., 2., nsamp)[:, None]
    targs = torch.where(torch.abs(inps) > 1, 0., 1.)
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

nsamp = 40 
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

    colors = np.where(targs.numpy(), 'red', 'blue')[:,0]
    for idx in range(out.shape[1]):
        plt.plot(out[:, idx, 0], color = str(colors[idx]))
    plt.xlabel('Time, $t$')
    plt.ylabel('Output, $y(x,t)$')
    plt.tight_layout()
    plt.savefig('learned_occ.png')

    plt.figure()
    pcs = project(hidd.detach().numpy())[1]
    plt.subplot(1,1,1,projection = '3d')
    for idx in range(out.shape[1]):
        plt.plot(out[:, idx, 0], pcs[:, idx, 1], pcs[:, idx, 2], color = str(colors[idx]))

    model_out_all = lambda x : w_out(model_all(x))[:, :, 0].sum(1) # Sum along batches since no cross talk between distinct inputs (no grad info lost)
    pi = torch.autograd.functional.jacobian(model_out_all, inps)

    plt.figure()
    normalized_dist = np.abs(1. - np.abs(inps)[:, 0]).detach().numpy()
    colors = plt.get_cmap('hot')(normalized_dist)
    for idx in range(out.shape[1]):
        plt.plot(pi[:, idx, 0], color = colors[idx])

    plt.figure()
    c0 = (model[0][0].weight.data.T @  model[-1].weight.data.T).item()
    plt.axhline(c0)
    plt.plot(inps[:,0], pi[-1, :, 0], c = 'black')


plt.show()
