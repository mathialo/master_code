import mastl
import numpy as np
import sigpy.imaging as img
import matplotlib.pyplot as plt


shepp_logan = img.map_to_01(img.to_grayscale(plt.imread("shepp_logan_512.png"))).astype(np.complex64)
# img.display(np.abs(shepp_logan))


# sampling_pattern = mastl.patterns.gaussian_sampling(*shepp_logan.shape, int(shepp_logan.size*0.20), 4.5, 25)
# np.save("gaussian_pattern_20.npy", sampling_pattern)
sampling_pattern = np.load("gaussian_pattern_20.npy")

# print("Sampling rate: {}".format(np.mean(sampling_pattern)))
img.display(sampling_pattern, show=False, filename="gaussian_pattern_20.png")
sampling_pattern = np.fft.fftshift(sampling_pattern)

samples = np.fft.fft2(shepp_logan)
samples[np.logical_not(sampling_pattern)] = 0

recon_adj = np.fft.ifft2(samples)

img.display(np.abs(recon_adj), show=False, filename="recon_adj_gaussian.png")


sampling_pattern = mastl.patterns.radial_sampling(*shepp_logan.shape, 64, 1)
np.save("line_pattern_20.npy", sampling_pattern)
sampling_pattern = np.load("line_pattern_20.npy")
img.display(sampling_pattern, show=False, filename="line_pattern_20.png")
sampling_pattern = np.fft.fftshift(sampling_pattern)

samples = np.fft.fft2(shepp_logan)
samples[np.logical_not(sampling_pattern)] = 0

recon_adj = np.fft.ifft2(samples)

img.display(np.abs(recon_adj), show=False, filename="recon_adj_line.png")


# sampling_pattern = mastl.patterns.level_sampling(*shepp_logan.shape, [1, .9, .5, .3, .15])
# np.save("level_pattern_20.npy", sampling_pattern)
sampling_pattern = np.load("level_pattern_20.npy")
# print(np.mean(sampling_pattern))
img.display(sampling_pattern, show=False, filename="level_pattern_20.png")
sampling_pattern = np.fft.fftshift(sampling_pattern)

samples = np.fft.fft2(shepp_logan)
samples[np.logical_not(sampling_pattern)] = 0

recon_adj = np.fft.ifft2(samples)

img.display(np.abs(recon_adj), show=False, filename="recon_adj_level.png")

sampling_pattern = mastl.patterns.level_sampling(*shepp_logan.shape, [.2])
np.save("uniform_pattern_20.npy", sampling_pattern)
sampling_pattern = np.load("uniform_pattern_20.npy")
# print(np.mean(sampling_pattern))
img.display(sampling_pattern, show=False, filename="uniform_pattern_20.png")
sampling_pattern = np.fft.fftshift(sampling_pattern)

samples = np.fft.fft2(shepp_logan)
samples[np.logical_not(sampling_pattern)] = 0

recon_adj = np.fft.ifft2(samples)

img.display(np.abs(recon_adj), show=False, filename="recon_adj_uniform.png")

