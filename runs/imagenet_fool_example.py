import foolbox
import keras
import numpy as np
from labeldict import labels
import sys

# Get pretrained resnet model from TF
model = keras.applications.ResNet50()

# ResNet requires this weird preprocessing:
subtract = np.array([104, 116, 123])

# Create fooling object
foolmodel = foolbox.models.KerasModel(model, bounds=(0, 255), preprocessing=(subtract, 1))

# Get sample image
image, label = foolbox.utils.imagenet_example()

# Apply attack (-1 is to reverse channel numbering to BGR as demanded by ResNet)
attack = foolbox.attacks.LBFGSAttack(foolmodel, criterion=foolbox.criteria.TargetClass(22))
adversarial = attack(image[:, :, ::-1], label)
new_label = np.argmax(model.predict(np.expand_dims(adversarial-subtract, 0)))

# Print labels
print("Original label:  {}".format(labels[label]))
print("Perturbed label: {}".format(labels[new_label]))
print("Norms:")
print("     |x| = {:.2f}".format(np.linalg.norm(np.ravel(image), ord=2)))
print("     |d| = {:.2f}".format(np.linalg.norm(np.ravel(adversarial - image), ord=2)))
print(" |d|/|x| = {:.2f}".format(np.linalg.norm(np.ravel(adversarial - image), ord=2) / np.linalg.norm(np.ravel(image), ord=2)))

# Save results
np.save("deepfool_original.npy", image)
np.save("deepfool_perturbed.npy", adversarial[:,:,::-1])
