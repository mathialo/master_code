from setuptools import setup

# Install python package
setup(
    name="mastl",
    version=0.1,
    author="Mathias Lohne",
    author_email="mathialo@ifi.uio.no",
    license="MIT, LPGLv3",
    description="Code from my master thesis",
    install_requires=["tensorflow>=1.5", "numpy", "sigpy"],
    packages=["mastl"])
