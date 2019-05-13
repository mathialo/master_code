# Code from 'Parseval Reconstruction Networks'

This repository contains all the code necessary to reproduce any of results presented in the thesis 'Parseval Reconstruction Networks - Improving Robustness of Deep Learning Based MRI Reconstruction Towards Adversarial Attacks'

## Structure of this repository
This repository consists of the following parts:

 * **mastl_package** is a Python package with all the model specifications, sampling patterns, loss functions, dataset class, etc. It also includes a copy of the _parsnet_ package
 * **parsnet** is the code for the parsnet package, simply a copy of [this git repo](https://github.com/mathialo/parsnet).
 * **figures** contains scripts that reproduces figures from the thesis
 * **runs** contains runnable scripts for model training, inference, etc. Built on the _mastl_ package.


## Licensing
All the code is licensed under the permissive MIT license, _except_ for the parsnet package, which is licensed under the free LPGLv3 license. 
