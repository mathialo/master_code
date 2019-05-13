import numpy as np 
import sigpy.imaging as img
import matplotlib.pyplot as plt

image_orig = np.abs(np.squeeze(np.load("advattacks/originals_perturbs/2_original.npy")))
image_vanilla = np.abs(np.squeeze(np.load("advattacks/results_vanilla/2_reconstruction_original.npy")))
image_parseval = np.abs(np.squeeze(np.load("advattacks/results_parseval/2_reconstruction_original.npy")))

upper_left = (70, 100)
size = 100

hspan = (upper_left[0], upper_left[0]+size)
vspan = (upper_left[1], upper_left[1]+size)

img.display(image_orig, show=False, filename="reconcomp_orig_full.png")
img.display(image_orig[vspan[0]:vspan[1], hspan[0]:hspan[1]], show=False, filename="reconcomp_orig.png")
img.display(image_vanilla[vspan[0]:vspan[1], hspan[0]:hspan[1]], show=False, filename="reconcomp_vanilla.png")
img.display(image_parseval[vspan[0]:vspan[1], hspan[0]:hspan[1]], show=False, filename="reconcomp_parseval.png")


