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


import tensorflow as tf

from typing import List, Optional, Sequence, Tuple

Reshaping = Optional[List[int]]


class TensorFluentShape(object):
    '''TensorFluentShape manages the fluent's shape in operations.

    Each RDDL fluent is a parameterized variable. The arity and
    the number of objects corresponding to the type of each parameter
    of a fluent are reflected in a ``rddl2tf.TensorFluentShape``
    object (i.e., the rank of a ``rddl2tf.TensorFluent`` corresponds
    to the fluent arity and the size of its dimensions corresponds
    to the number of objects of each type).

    Also, a ``rddl2tf.TensorFluentShape`` manages batch sizes
    when evaluating operations in batch mode.

    If in batch mode, the first dimension always corresponds
    to the batch size.

    Args:
        shape: The list of dimensions of the fluent.
        batch: The batch mode flag.
    '''

    def __init__(self, shape: List[int], batch: bool) -> None:
        self._shape = tf.TensorShape(shape)
        self._batch = batch

    def as_list(self) -> List[str]:
        '''Returns the fluent's shape as a list.'''
        return self._shape.as_list()

    def __getitem__(self, i: int) -> int:
        '''Returns the size of dimension `i`.'''
        return self._shape[i]

    def __eq__(self, other: 'TensorFluentShape') -> bool:
        '''Returns True if fluent shapes are equal. False, otherwise.'''
        return self._shape == other._shape and self._batch == other._batch

    def __ne__(self, other: 'TensorFluentShape') -> bool:
        '''Returns True if fluent shapes are not equal. False, otherwise.'''
        return self._shape != other._shape or self._batch != other._batch

    def __str__(self) -> str:
        '''Returns the string represenation of a TensorFluentShape object.'''
        return 'TensorFluentShape({}, batch={})'.format(self.as_list(), self._batch)

    @property
    def batch(self) -> bool:
        '''Returns True if fluent is in batch mode. False, otherwise.'''
        return self._batch

    @property
    def batch_size(self) -> int:
        '''Returns the batch size if in batch mode. Otherwise, returns 1.'''
        return self._shape.as_list()[0] if self._batch else 1

    @property
    def fluent_shape(self) -> Sequence[int]:
        '''Returns a copy of the fluent shape, ignoring batch size if in batch mode.'''
        return tuple(self._shape.as_list()[1:] if self._batch else self._shape.as_list()[:])

    @property
    def fluent_size(self) -> int:
        '''Returns the size of fluent shape (i.e., the fluent's arity).'''
        return len(self.fluent_shape)

    @classmethod
    def broadcast(cls,
            shape1: 'TensorFluentShape',
            shape2: 'TensorFluentShape') -> Tuple[Reshaping, Reshaping]:
        '''It broadcasts the fluent shapes if any input is in batch mode.

        It handles input shapes in different modes, expanding its
        dimensions if necessary. It outputs a tuple with new shapes.
        If no input shape is in batch mode, return (None, None).
        If an input shape does not need to be changed, return None.

        Args:
            shape1: A fluent's shape.
            shape2: A fluent's shape.

        Returns:
            A pair of new shapes.
        '''
        reshape_1, reshape_2 = None, None

        if not (shape1._batch or shape2._batch):
            return reshape_1, reshape_2

        size_1, size_2 = shape1.fluent_size, shape2.fluent_size
        size_diff = abs(size_1 - size_2)
        if size_diff == 0:
            return reshape_1, reshape_2

        if size_2 > size_1 and not (size_1 == 0 and not shape1._batch):
            reshape_1 = [1] * size_diff + list(shape1.fluent_shape)
            if shape1._batch:
                reshape_1 = [shape1.batch_size] + reshape_1
        elif size_1 > size_2 and not (size_2 == 0 and not shape2._batch):
            reshape_2 = [1] * size_diff + list(shape2.fluent_shape)
            if shape2._batch:
                reshape_2 = [shape2.batch_size] + reshape_2
        return reshape_1, reshape_2
