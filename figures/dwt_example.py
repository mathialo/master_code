import tfwavelets as tfw
import numpy as np 
import matplotlib.pyplot as plt
import sigpy.imaging as img


def main():
	image = img.to_grayscale(img.map_to_01(plt.imread("lily.jpg")))

	level1 = tfw.wrappers.dwt2d(image, "haar", 1)
	level2 = tfw.wrappers.dwt2d(image, "haar", 2)

	img.display(img.to_grayscale(img.map_to_01(image)), show=False, filename="dwtexample_level0.png")
	img.display(level1, show=False, filename="dwtexample_level1.png")
	img.display(level2, show=False, filename="dwtexample_level2.png")


if __name__ == '__main__':
	main()