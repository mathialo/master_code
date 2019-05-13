from plotstyle import *

xs = np.linspace(-4, 4, 500)

fig = plt.figure(figsize=(3.5, 3))
relu = np.zeros_like(xs)
relu[250:] = xs[250:]
plt.plot(xs, relu)
plt.tight_layout()
plt.xlim([-4, 4])
plt.ylim([-1, 4])
plt.savefig("relu.pdf", facecolor='none')
# plt.show()

plt.figure(figsize=(3.5, 3))
leaky_relu = np.zeros_like(xs)
leaky_relu[0:250] = 0.2*xs[0:250]
leaky_relu[250:] = xs[250:]
plt.plot(xs, leaky_relu)
plt.tight_layout()
plt.xlim([-4, 4])
plt.ylim([-2, 4])
plt.savefig("leaky_relu.pdf", facecolor='none')
# plt.show()

plt.figure(figsize=(3.5, 3))
tanh = np.tanh(xs)
plt.plot(xs, tanh)
plt.tight_layout()
plt.xlim([-4, 4])
plt.ylim([-1.5, 1.5])
plt.savefig("tanh.pdf", facecolor='none')
# plt.show()

plt.figure(figsize=(3.5, 3))
xs = np.linspace(-8, 8, 500)
sig = lambda x: 1 / (1 + np.exp(-x))
sigmoid = sig(xs)
plt.plot(xs, sigmoid)
plt.tight_layout()
plt.xlim([-8, 8])
plt.ylim([-.2, 1.2])
plt.savefig("sigmoid.pdf", facecolor='none')
# plt.show()
