# Copyright (C) 2019  Mathias Lohne

# This library is free software; you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 3.0 of the License, or (at
# your option) any later version.

# This library is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
# License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with this library.  If not, see <http://www.gnu.org/licenses/>.

from setuptools import setup

# Install python package
setup(
    name="parsnet",
    version=0.1,
    author="Mathias Lohne",
    author_email="mathialo@ifi.uio.no",
    license="LGPLv3",
    description="TensorFlow implementation of the constraints necessary for parseval networks",
    url="https://github.com/mathialo/parsnet",
    install_requires=["tensorflow>=1.5"],
    packages=["parsnet"],
    zip_safe=False)
