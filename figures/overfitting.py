from plotstyle import *
import splinelib as spl


# xs = np.linspace(0, 4, 25)
# np.save("overfit_xs.npy", xs)
# ys = -xs**2 + 4*xs + np.random.normal(loc=0, scale=.4, size=xs.size)
# np.save("overfit_ys.npy", ys)

xs = np.load("overfit_xs.npy")
ys = np.load("overfit_ys.npy")

plt.figure(figsize=[4, 3])
plt.scatter(xs, ys)

plotx = np.linspace(0, 4, 500)
ploty = -plotx**2 + 4*plotx
plt.plot(plotx, ploty)
plt.tight_layout()
plt.savefig("overfit_true.pdf", facecolor='none')
#plt.show()


data = np.vstack([xs, ys]).T
degree = 3
knots = spl.fit.generate_uniform_knots(
	spl.fit.cord_length(data),
	degree,
	83
)

spline = spl.fit.least_squares(
	data,
	knots,
	degree
)

plt.figure(figsize=[4, 3])
plt.scatter(xs, ys)
plt.plot(plotx, np.squeeze(spline(plotx)))
plt.tight_layout()
plt.savefig("overfit_fitted.pdf", facecolor='none')
#plt.show()

