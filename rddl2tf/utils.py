# This file is part of rddl2tf.

# rddl2tf is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# rddl2tf is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with rddl2tf. If not, see <http://www.gnu.org/licenses/>.


from typing import Optional

import tensorflow as tf


def range_type_to_dtype(range_type: str) -> Optional[tf.DType]:
    '''Maps RDDL range types to TensorFlow dtypes.'''
    range2dtype = {
        'real': tf.float32,
        'int': tf.int32,
        'bool': tf.bool
    }
    return range2dtype[range_type]


def python_type_to_dtype(python_type: type) -> Optional[tf.DType]:
    '''Maps python types to TensorFlow dtypes.'''
    dtype = None
    if python_type == float:
        dtype = tf.float32
    elif python_type == int:
        dtype = tf.int32
    elif python_type == bool:
        dtype = tf.bool
    return dtype


def identifier(name):
    name = name.replace("'", '')
    name = name.replace('/', '-')
    return name
