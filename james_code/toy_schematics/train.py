import matplotlib.pyplot as plt
import numpy as np
import torch, os
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from tqdm import tqdm
from itertools import product
import torch_optimizer as toptim

residual = True 

def GetModel(L, N, gain = 1., residual = residual):
    ls = []

    class Residual(nn.Module):
        def __init__(self, block, alpha=1.0):
            super().__init__()
            self.block = block
            self.alpha = alpha  # can be a float or nn.Parameter if you want it learnable

        def forward(self, x):
            return x + self.alpha * self.block(x)

    ls.append(nn.Sequential(nn.Linear(1, N), nn.Tanh()))  # group 1
    for l in range(L-1):
        if residual:
            ls.append(Residual(nn.Sequential(nn.Linear(N, N), nn.Tanh())))  # group 1
        else:
            ls.append(nn.Sequential(nn.Linear(N, N), nn.Tanh())) # group 1
    ls.append(nn.Linear(N, 1))

    # gain parameter
    for layer in ls:
        lin = layer
        if isinstance(layer, Residual):
            lin = layer.block[0]
        elif isinstance(layer, nn.Sequential):
            lin = layer[0]
        lin.weight.data *= gain
        lin.bias.data *= 0.

    return nn.Sequential(*ls)

def GetTask(nsamp, batch_size, shuffle = True):
    inps = torch.rand((nsamp, 1)) * 4 - 2.
    targs = torch.where(torch.abs(inps) > 1, 0, 1).float()
    dataset = TensorDataset(inps, targs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataset, dataloader

WIDTHS = [100]
DEPTHS = [20]
GAINS = [1.]
BASE_LRS = [1e-3]
SEEDS    = [0]
grid = [
    {"width": w, "depth": d, "gain": g, "base_lr": lr, "seed": s}
    for w, d, g, lr, s in product(WIDTHS, DEPTHS, GAINS, BASE_LRS, SEEDS)
]

if __name__ == '__main__':
    nsamp, B = 10000, 1000
    nepoch = 100

    _, dataloader = GetTask(nsamp, B)
    test_dataset, _ = GetTask(5000, 5000)
    test_inp, test_targ = test_dataset[:]

    mse = nn.MSELoss()
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/sweep/', exist_ok=True)
    for config in tqdm(grid):
        L, N, gain, lr, seed = config['depth'], config['width'], config['gain'], config['base_lr'], config['seed']
        torch.manual_seed(seed)
        model = GetModel(L, N, gain)

        losses = []
        optim = toptim.Shampoo(model.parameters(), lr = lr)
        pbar = tqdm(range(nepoch))
        for epoch in pbar:
            for inps, targs in dataloader:
                inps.requires_grad_(True)
                optim.zero_grad()
                y = model(inps)
#                out_grad = torch.autograd.grad(y.sum(), inps, create_graph = True, retain_graph = True)[0]
                loss = mse(y, targs) #+ 1e-1 * (out_grad ** 2).mean()
                loss.backward()
                optim.step()
            losses.append(loss.item())
            pbar.set_description(f'{loss:.3e}')

            if epoch % 10 == 0:
                with torch.no_grad():
                    y_test = model(test_inp)
                    acc_test = torch.mean((torch.abs(y_test - test_targ) < 1e-2).float())
                    if acc_test > 0.98:
                        break

            if epoch == nepoch - 1:
                print("Could not reach target accuracy in epochs!")
                config['invalid'] = True

        torch.save({'config': config, 'model': model.state_dict()}, f'data/sweep/model_L={L}_N={N}_g={gain}_lr={lr}_seed={seed}.pt')

        plt.plot(losses)
        plt.show()
