import numpy as np, matplotlib.pyplot as plt

x0 = np.linspace(-1., 1., 30)
xf = np.where(np.abs(x0) < .5, 1., 0.)
ts = np.linspace(0., 10.1, 10)
X = x0[None] * (1-ts[:,None]) + xf[None] * ts[:,None]

plt.figure(figsize = (4, 3), dpi = 300)
colors = np.where(xf, 'red', 'blue')
for x, c in zip(X.T, colors):
    plt.plot(ts, x, c=c)

plt.xlabel('Time, $t$')
plt.ylabel('Model Output, $y(t)$')
plt.tight_layout()
plt.savefig('occ.png')

H = np.stack([X, xf[None] * ts[:,None]], 0)

plt.figure()
plt.subplot(1,1,1,projection = '3d')
for h, c in zip(np.moveaxis(H,-1,0), colors):
    plt.plot(ts, h[0], h[1], c=c)

plt.xlabel('Time, $t$')
plt.ylabel('Hidden, $h_1(t)$')
plt.gca().set_zlabel('Hidden, $h_2(t)$')
plt.savefig('occ_3d.png')

plt.figure(figsize = (4,3), dpi = 300)
for h, c in zip(np.moveaxis(H,-1,0), colors):
    plt.plot(h[0], h[1], c=c)

plt.xlabel('Hidden, $h_1(t)$')
plt.ylabel('Hidden, $h_2(t)$')
plt.tight_layout()
plt.savefig('occ_3d.png')

xf = (x0**2 - .5**2)
X = x0[None] * (1-ts[:,None]) + xf[None] * ts[:,None]
plt.figure()
for x, c in zip(X.T, colors):
    plt.plot(x, c=c)


plt.xlabel('Time, $t$')
plt.ylabel('Model Output, $y(t)$')
plt.savefig('sdf.png')
plt.show()
