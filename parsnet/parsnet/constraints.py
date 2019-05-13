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

from tensorflow.python.keras.constraints import Constraint
from tensorflow.python.ops import math_ops, array_ops

class TightFrame(Constraint):
    """
    Parseval (tight) frame contstraint, as introduced in https://arxiv.org/abs/1704.08847

    Constraints the weight matrix to be a tight frame, so that the Lipschitz
    constant of the layer is <= 1. This increases the robustness of the network
    to adversarial noise.

    Warning: This constraint simply performs the update step on the weight matrix
    (or the unfolded weight matrix for convolutional layers). Thus, it does not
    handle the necessary scalings for convolutional layers.

    Args:
        scale (float):    Retraction parameter (length of retraction step).
        num_passes (int): Number of retraction steps.

    Returns:
        Weight matrix after applying regularizer.
    """


    def __init__(self, scale, num_passes=1):
        self.scale = scale

        if num_passes < 1:
            raise ValueError("Number of passes cannot be non-positive! (got {})".format(num_passes))
        self.num_passes = num_passes


    def __call__(self, w):
        transpose_channels = (len(w.shape) == 4)

        # Move channels_num to the front in order to make the dimensions correct for matmul
        if transpose_channels:
            w_reordered = array_ops.reshape(w, (-1, w.shape[3].value))

        else:
            w_reordered = w

        last = w_reordered
        for i in range(self.num_passes):
            temp1 = math_ops.matmul(last, last, transpose_a=True)
            temp2 = (1 + self.scale) * w_reordered - self.scale * math_ops.matmul(w_reordered, temp1)

            last = temp2

        # Move channels_num to the back again
        if transpose_channels:
            return array_ops.reshape(last, w.shape)
        else:
            return last


    def get_config(self):
        return {'scale': self.scale, 'num_passes': self.num_passes}


# Alias
tight_frame = TightFrame
