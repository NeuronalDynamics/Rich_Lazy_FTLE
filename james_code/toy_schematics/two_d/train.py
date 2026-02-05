import matplotlib.pyplot as plt
import numpy as np
import torch, os
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from tqdm import tqdm
from itertools import product
import torch_optimizer as toptim

residual = True 

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

def GetTask(nsamp, batch_size, shuffle = True):
    width = np.sqrt(2 * np.pi) # So area of +1 and -1 regions are same.
    inps = torch.rand((nsamp, 2)) * width - width/2

    q = torch.abs(inps) - np.sqrt(np.pi) / 2.
    d = torch.linalg.norm(torch.maximum(q, 0.*q), axis = 1) + torch.minimum(torch.max(q,1)[0], 0.*q[:,0]) # SDF of square.
    targs = d[:, None]

    dataset = TensorDataset(inps, targs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataset, dataloader

def GetModel(L, N, gain = 1., residual = residual):
    ls = []

    class Residual(nn.Module):
        def __init__(self, block, alpha=0.1):
            super().__init__()
            self.block = block
            self.alpha = alpha  # can be a float or nn.Parameter if you want it learnable

        def forward(self, x):
            return (1 - self.alpha) * x + self.alpha * 10 * self.block(x)

    class SphereSDF(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return (torch.linalg.norm(x, axis = -1) - 1.).unsqueeze(-1)

    ls.append(nn.Linear(2, N))
    ls.append(nn.Tanh())
    for l in range(L-1):
        if residual:
            ls.append(Residual(nn.Sequential(nn.Linear(N, N), nn.Tanh())))  # group 1
        else:
            ls.append(nn.Sequential(nn.Linear(N, N), nn.Tanh())) # group 1
    ls.append(nn.Sequential(nn.Linear(N, 2), SphereSDF()))

    # gain parameter
    for layer in ls:
        lin = layer
        if isinstance(layer, Residual):
            lin = layer.block[0]
        elif isinstance(layer, nn.Sequential):
            lin = layer[0]
        else:
            continue
        lin.weight.data *= gain
        lin.bias.data *= 0.

    return nn.Sequential(*ls)

WIDTHS = [100]
DEPTHS = [30]
GAINS = [1.5]
BASE_LRS = [1e-3]
SEEDS    = [0]
grid = [
    {"width": w, "depth": d, "gain": g, "base_lr": lr, "seed": s}
    for w, d, g, lr, s in product(WIDTHS, DEPTHS, GAINS, BASE_LRS, SEEDS)
]

if __name__ == '__main__':
    nsamp, B = 10000, 5000
    nepoch = 500

    _, dataloader = GetTask(nsamp, B)
    test_dataset, _ = GetTask(5000, 5000)
    test_inp, test_targ = test_dataset[:]

    mse = nn.MSELoss()
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/sweep/', exist_ok=True)
    for config in grid:
        L, N, gain, lr, seed = config['depth'], config['width'], config['gain'], config['base_lr'], config['seed']
        torch.manual_seed(seed)
        model = GetModel(L, N, gain)

        with torch.no_grad():
            model[0].weight.data /= torch.linalg.norm(model[0].weight.data)
            model[-1][0].weight.data /= torch.linalg.norm(model[-1][0].weight.data)

        losses = []
#        optim = torch.optim.Adam(model.parameters(), lr = lr)
        optim = toptim.Shampoo(model.parameters(), lr = lr)
        for epoch in tqdm(range(nepoch)):
            for inps, targs in dataloader:
                optim.zero_grad()
                y = model(inps)
                loss = mse(y, targs)
                loss.backward()
                optim.step()
            losses.append(loss.item())

            inc = ((nepoch+9) // 10)
            if epoch % inc == 0:
                plt.subplot(2, 5, epoch // inc + 1)
                nsamp_x = 20
                test_inps, _ = GetTaskUniform(nsamp_x**2)
                test_out = model(test_inps)
                plt.imshow(test_out[:, 0].reshape((nsamp_x, nsamp_x)).detach().numpy())

            if epoch == nepoch - 1:
                print("Could not reach target accuracy in epochs!")
                config['invalid'] = True

            with torch.no_grad():
                model[0].weight.data /= torch.linalg.norm(model[0].weight.data)
                model[-1][0].weight.data /= torch.linalg.norm(model[-1][0].weight.data)

        torch.save({'config': config, 'model': model.state_dict()}, f'data/sweep/model_L={L}_N={N}_g={gain}_lr={lr}_seed={seed}.pt')
        print('Saved to', f'data/sweep/model_L={L}_N={N}_g={gain}_lr={lr}_seed={seed}.pt')
        plt.figure()
        plt.plot(losses)
        plt.show()
