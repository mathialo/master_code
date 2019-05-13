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

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.framework import dtypes
import numpy as _np


def convex_add(input1, input2, initial_convex_par=0.5, trainable=False):
	"""
	Do a convex combination of input1 and input2. That is, return the output of

	    lam * input1 + (1 - lam) * input2

	Where lam is a number in the unit interval. 

	Args:
		input1 (tf.Tensor):			Input to take convex combinatio of
		input2 (tf.Tensor):			Input to take convex combinatio of
		initial_convex_par (float):	Initial value for convex parameter. Must be
									in [0, 1].
		trainable (bool):			Whether convex parameter should be trainable
									or not. 

	Returns:
		tf.Tensor: Result of convex combination

	Raises:
		ValueError:		If initial_convex_par is outside of legal limit.
		TypeError:		If types are incorrect
	"""
	# Will implement this as sigmoid(p)*input1 + (1-sigmoid(p))*input2 to ensure
	# convex parameter to be in the unit interval without constraints during
	# optimization

	# Find value for p, also check for legal initial_convex_par
	if initial_convex_par < 0:
		raise ValueError("Convex parameter must be >=0")

	elif initial_convex_par == 0:
		# sigmoid(-16) is approximately a 32bit roundoff error, practically 0
		initial_p_value = -16

	elif initial_convex_par < 1:
		# Compute inverse of sigmoid to find initial p value
		initial_p_value = -_np.log(1/initial_convex_par - 1) 

	elif initial_convex_par == 1:
		# Same argument as for 0
		initial_p_value = 16

	else:
		raise ValueError("Convex parameter must be <=1")

	p = variables.Variable(
		initial_value = initial_p_value,
		dtype=dtypes.float32,
		trainable=trainable
	)

	lam = math_ops.sigmoid(p)
	return input1 * lam + (1 - lam)*input2


