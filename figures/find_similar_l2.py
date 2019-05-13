import numpy as np 
import sigpy.imaging as img 
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import sys

image_size = (350, 400)

baseimage = img.map_to_01(plt.imread("ILSVRC/ILSVRC2012_test_00000002.JPEG"))
image1 = img.map_to_01(plt.imread("ILSVRC/ILSVRC2012_test_00000019.JPEG"))
image2 = img.map_to_01(plt.imread("ILSVRC/ILSVRC2012_test_00000022.JPEG"))


def loss(point1, point2):
	return np.sqrt(np.sum(np.square(np.abs(point1 - point2)), axis=None))


def crop(image, upper_left=(0,0)):
	return image[
		upper_left[0]:(upper_left[0] + image_size[0]),
		upper_left[1]:(upper_left[1] + image_size[1]),
		:
	]


def max_crop_loss(image, ground_truth=None):
	if ground_truth is None:
		ground_truth = image
	v_range = image.shape[0] - image_size[0]
	h_range = image.shape[1] - image_size[1]
	
	max_so_far = 0
	argmax_so_far = (0, 0)

	it = 0

	for i in range(v_range):
		for j in range(h_range):
			if it % 50 == 0: print("\rFinding best crop distance: {:03.1f} %".format(it*100/(v_range*h_range)), end="")
			temp = loss(crop(ground_truth), crop(image, (i, j)))
			if temp > max_so_far:
				max_so_far = temp
				argmax_so_far = (i, j)

			it += 1

	print()

	return max_so_far, argmax_so_far


def add_noise(image, goal, maxit = 1000):
	print("Creating noisy version")
	sigma = goal / np.sqrt(np.prod(image.shape))
	noise = 0.68*gaussian_filter(np.random.normal(0, sigma, image.shape), sigma=0)
	noise -= 1.5*gaussian_filter(np.random.normal(0, sigma, image.shape), sigma=1)
	noise += 2*gaussian_filter(np.random.normal(0, sigma, image.shape), sigma=2)
	noise -= 3*gaussian_filter(np.random.normal(0, sigma, image.shape), sigma=3)
	noise += 3*gaussian_filter(np.random.normal(0, sigma, image.shape), sigma=4)
	noise += 5*gaussian_filter(np.random.normal(0, sigma, image.shape), sigma=5)
	noise += 6*gaussian_filter(np.random.normal(0, sigma, image.shape), sigma=6)
	noise += 12*gaussian_filter(np.random.normal(0, sigma, image.shape), sigma=7)

	return loss(image, image+noise), noise



def find_add(image, goal):
	print("Finding add distance")
	add = goal / np.sqrt(np.prod(image.shape))
	return loss(image, image+add), add


def main():
	bird_loss = loss(crop(baseimage), crop(image1))
	dog_loss = loss(crop(baseimage), crop(image2))

	crop_loss, max_crop = max_crop_loss(baseimage)
	add_loss, add = find_add(crop(baseimage), np.mean([dog_loss, bird_loss]))
	noise_loss, noise = add_noise(crop(baseimage), np.mean(dog_loss))

	img.display(crop(baseimage), show=False, filename="l2demo_base.png")
	img.display(crop(baseimage, max_crop), show=False, filename="l2demo_translate.png")
	img.display(crop(baseimage)+add, show=False, filename="l2demo_add.png")
	img.display(crop(baseimage) + noise, show=False, filename="l2demo_noise.png")
	img.display(crop(image1), show=False, filename="l2demo_other1.png")
	img.display(crop(image2), show=False, filename="l2demo_other2.png")

	print("Translated distance:   {:.2f}".format(crop_loss))
	print("Add distance:          {:.2f} (by adding {:.2f})".format(add_loss, add))
	print("Noise distance:        {:.2f}".format(noise_loss))
	print("Other bird distance:   {:.2f}".format(bird_loss))
	print("Dog distance:          {:.2f}".format(dog_loss))


if __name__ == '__main__':
	if len(sys.argv) > 1 and sys.argv[1] == "recompute":
		main()
	else:
		print("Doing nothing. Run with 'recompute' arg to run computations")
