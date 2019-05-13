import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.font_manager._rebuild()

# import seaborn as sns
# sns.set()

plt.style.use("ggplot")

with open("mode") as f:
	mode = f.read().strip()

if mode == "thesis":
	plt.rcParams['font.family'] = 'Palatino'
elif mode == "presentation":
	plt.rcParams['font.family'] = 'Fira Sans Light'
elif mode == "paper":
	plt.rcParams['font.family'] = 'CMU Serif'
else:
	import sys
	print("Invalid mode", file=sys.stderr)
	sys.exit(1)

print("  - plotmode: {}".format(mode))

plt.rcParams['axes.unicode_minus'] = False