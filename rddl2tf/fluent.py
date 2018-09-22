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


from rddl2tf.fluentscope import TensorFluentScope
from rddl2tf.fluentshape import TensorFluentShape

import tensorflow as tf

from typing import Callable, List, Optional, Sequence, Union

Value = Union[bool, int, float]


class TensorFluent(object):
    '''TensorFluent is a wrapper for ``tf.Tensor`` objects.

    Each RDDL fluent is compiled to a ``TensorFluent`` after instantiation.
    A ``TensorFluent`` stores metadata needed for compiling RDDL expressions
    into TensorFlow's graph ops and for evaluating fluents in batch mode.

    Args:
        tensor: A tensor op representing the fluent in the graph.
        scope: The fluent's argument scope in an expression.
        batch: Batch mode flag.

    Attributes:
        tensor (:obj:`tf.Tensor`): A tensor op representing the fluent in the graph.
        scope (:obj:`rddl2tf.fluentscope.TensorFluentScope`): The argument scope of the fluent in the expression.
        shape (:obj:`rddl2tf.fluentshape.TensorFluentShape`): The fluent shape and dimensions.
    '''

    def __init__(self, tensor: tf.Tensor, scope: List[str], batch: bool = False) -> None:
        self.tensor = tensor
        self.scope = TensorFluentScope(scope)
        self.shape = TensorFluentShape(tensor.shape, batch)

    @property
    def batch(self) -> bool:
        '''Returns True if in batch mode. False, otherwise.'''
        return self.shape.batch

    @property
    def dtype(self) -> tf.DType:
        '''Returns the fluent's data type.'''
        return self.tensor.dtype

    @property
    def name(self) -> str:
        '''Returns the tensor's name.'''
        return self.tensor.name

    @classmethod
    def constant(cls,
            value: Value,
            dtype: tf.DType = tf.float32) -> 'TensorFluent':
        '''Returns a constant `value` TensorFluent with given `dtype`.

        Args:
            value: The constant value.
            dtype: The output's data type.

        Returns:
            A constant TensorFluent.
        '''
        t = tf.constant(value, dtype=dtype)
        scope = [] # type: List
        batch = False
        return TensorFluent(t, scope, batch=batch)

    @classmethod
    def Normal(cls,
            mean: 'TensorFluent', variance: 'TensorFluent',
            batch_size: Optional[int] = None) -> 'TensorFluent':
        '''Returns a TensorFluent for the Normal sampling op with given mean and variance.

        Args:
            mean: The mean parameter of the Normal distribution.
            variance: The variance parameter of the Normal distribution.
            batch_size: The size of the batch (optional).

        Returns:
            A TensorFluent sample drawn from the Normal distribution.

        Raises:
            ValueError: If parameters do not have the same scope.
        '''
        if mean.scope != variance.scope:
            raise ValueError('Normal distribution: parameters must have same scope!')
        loc = mean.tensor
        scale = tf.sqrt(variance.tensor)
        dist = tf.distributions.Normal(loc, scale)
        batch = mean.batch or variance.batch
        if not batch and batch_size is not None:
            t = dist.sample(batch_size)
            batch = True
        else:
            t = dist.sample()
        scope = mean.scope.as_list()
        return TensorFluent(t, scope, batch=batch)

    @classmethod
    def Uniform(cls,
            low: 'TensorFluent', high: 'TensorFluent',
            batch_size: Optional[int] = None) -> 'TensorFluent':
        '''Returns a TensorFluent for the Uniform sampling op with given low and high parameters.

        Args:
            low: The low parameter of the Uniform distribution.
            high: The high parameter of the Uniform distribution.
            batch_size: The size of the batch (optional).

        Returns:
            A TensorFluent sample drawn from the Uniform distribution.

        Raises:
            ValueError: If parameters do not have the same scope.
        '''
        if low.scope != high.scope:
            raise ValueError('Uniform distribution: parameters must have same scope!')
        dist = tf.distributions.Uniform(low.tensor, high.tensor)
        batch = low.batch or high.batch
        if not batch and batch_size is not None:
            t = dist.sample(batch_size)
            batch = True
        else:
            t = dist.sample()
        scope = low.scope.as_list()
        return TensorFluent(t, scope, batch=batch)

    @classmethod
    def Exponential(cls,
            mean: 'TensorFluent',
            batch_size: Optional[int] = None) -> 'TensorFluent':
        '''Returns a TensorFluent for the Exponential sampling op with given mean parameter.

        Args:
            mean: The mean parameter of the Exponential distribution.
            batch_size: The size of the batch (optional).

        Returns:
            A TensorFluent sample drawn from the Exponential distribution.
        '''
        rate = 1 / mean.tensor
        dist = tf.distributions.Exponential(rate)
        batch = mean.batch
        if not batch and batch_size is not None:
            t = dist.sample(batch_size)
            batch = True
        else:
            t = dist.sample()
        scope = mean.scope.as_list()
        return TensorFluent(t, scope, batch=batch)

    @classmethod
    def Gamma(cls,
            shape: 'TensorFluent',
            scale: 'TensorFluent',
            batch_size: Optional[int] = None) -> 'TensorFluent':
        '''Returns a TensorFluent for the Gamma sampling op with given shape and scale parameters.

        Args:
            shape: The shape parameter of the Gamma distribution.
            scale: The scale parameter of the Gamma distribution.
            batch_size: The size of the batch (optional).

        Returns:
            A TensorFluent sample drawn from the Uniform distribution.

        Raises:
            ValueError: If parameters do not have the same scope.
        '''
        if shape.scope != scale.scope:
            raise ValueError('Gamma distribution: parameters must have same scope!')
        concentration = shape.tensor
        rate = 1 / scale.tensor
        dist = tf.distributions.Gamma(concentration, rate)
        batch = shape.batch or scale.batch
        if not batch and batch_size is not None:
            t = dist.sample(batch_size)
            batch = True
        else:
            t = dist.sample()
        scope = shape.scope.as_list()
        return TensorFluent(t, scope, batch=batch)

    @classmethod
    def abs(cls, x: 'TensorFluent') -> 'TensorFluent':
        '''Returns a TensorFluent for the abs function.

        Args:
            x: The input fluent.

        Returns:
            A TensorFluent wrapping the abs function.
        '''
        return cls._unary_op(x, tf.abs, tf.float32)

    @classmethod
    def exp(cls, x: 'TensorFluent') -> 'TensorFluent':
        '''Returns a TensorFluent for the exp function.

        Args:
            x: The input fluent.

        Returns:
            A TensorFluent wrapping the exp function.
        '''
        return cls._unary_op(x, tf.exp, tf.float32)

    @classmethod
    def log(cls, x: 'TensorFluent') -> 'TensorFluent':
        '''Returns a TensorFluent for the log function.

        Args:
            x: The input fluent.

        Returns:
            A TensorFluent wrapping the log function.
        '''
        return cls._unary_op(x, tf.log, tf.float32)

    @classmethod
    def sqrt(cls, x: 'TensorFluent') -> 'TensorFluent':
        '''Returns a TensorFluent for the sqrt function.

        Args:
            x: The input fluent.

        Returns:
            A TensorFluent wrapping the sqrt function.
        '''
        return cls._unary_op(x, tf.sqrt, tf.float32)

    @classmethod
    def cos(cls, x: 'TensorFluent') -> 'TensorFluent':
        '''Returns a TensorFluent for the cos function.

        Args:
            x: The input fluent.

        Returns:
            A TensorFluent wrapping the cos function.
        '''
        return cls._unary_op(x, tf.cos, tf.float32)

    @classmethod
    def sin(cls, x: 'TensorFluent') -> 'TensorFluent':
        '''Returns a TensorFluent for the sin function.

        Args:
            x: The input fluent.

        Returns:
            A TensorFluent wrapping the sin function.
        '''
        return cls._unary_op(x, tf.sin, tf.float32)

    @classmethod
    def tan(cls, x: 'TensorFluent') -> 'TensorFluent':
        '''Returns a TensorFluent for the tan function.

        Args:
            x: The input fluent.

        Returns:
            A TensorFluent wrapping the tan function.
        '''
        return cls._unary_op(x, tf.tan, tf.float32)

    @classmethod
    def round(cls, x: 'TensorFluent') -> 'TensorFluent':
        '''Returns a TensorFluent for the round function.

        Args:
            x: The input fluent.

        Returns:
            A TensorFluent wrapping the round function.
        '''
        return cls._unary_op(x, tf.round, tf.float32)

    @classmethod
    def ceil(cls, x: 'TensorFluent') -> 'TensorFluent':
        '''Returns a TensorFluent for the ceil function.

        Args:
            x: The input fluent.

        Returns:
            A TensorFluent wrapping the ceil function.
        '''
        return cls._unary_op(x, tf.ceil, tf.float32)

    @classmethod
    def floor(cls, x: 'TensorFluent') -> 'TensorFluent':
        '''Returns a TensorFluent for the floor function.

        Args:
            x: The input fluent.

        Returns:
            A TensorFluent wrapping the floor function.
        '''
        return cls._unary_op(x, tf.floor, tf.float32)

    @classmethod
    def pow(cls, x: 'TensorFluent', y: 'TensorFluent') -> 'TensorFluent':
        '''Returns a TensorFluent for the pow function.TensorFluent

        Args:
            x: The first operand.
            y: The second operand.

        Returns:
            A TensorFluent wrapping the pow function.
        '''
        return cls._binary_op(x, y, tf.pow, tf.float32)

    @classmethod
    def max(cls, x: 'TensorFluent', y: 'TensorFluent') -> 'TensorFluent':
        '''Returns a TensorFluent for the maximum function.TensorFluent

        Args:
            x: The first operand.
            y: The second operand.

        Returns:
            A TensorFluent wrapping the maximum function.
        '''
        return cls._binary_op(x, y, tf.maximum, tf.float32)

    @classmethod
    def min(cls, x: 'TensorFluent', y: 'TensorFluent') -> 'TensorFluent':
        '''Returns a TensorFluent for the minimum function.

        Args:
            x: The first operand.
            y: The second operand.

        Returns:
            A TensorFluent wrapping the minimum function.
        '''
        return cls._binary_op(x, y, tf.minimum, tf.float32)

    @classmethod
    def if_then_else(cls,
            condition: 'TensorFluent',
            true_case: 'TensorFluent',
            false_case: 'TensorFluent') -> 'TensorFluent':
        '''Returns a TensorFluent for the control op if-then-else.

        Args:
            condition: Boolean fluent for the if condition.
            true_case: Fluent returned in the true clause.
            false_case: Fluent returned in the false clause.

        Returns:
            A TensorFluent wrapping the if-then-else control statement.

        Raises:
            ValueError: If cases don't have same shape.
        '''
        condition_tensor = condition.tensor
        true_case_tensor = true_case.tensor
        false_case_tensor = false_case.tensor

        if true_case.shape != false_case.shape:
            if true_case.shape.as_list() == []:
                true_case_tensor = tf.fill(false_case.shape.as_list(), true_case.tensor)
            elif false_case.shape.as_list() == []:
                false_case_tensor = tf.fill(true_case.shape.as_list(), false_case.tensor)

        if true_case_tensor.shape != false_case_tensor.shape:
            raise ValueError('TensorFluent.if_then_else: cases must be of same shape!')

        t = tf.where(condition_tensor, x=true_case_tensor, y=false_case_tensor)
        scope = condition.scope.as_list()

        batch = condition.batch
        # if (not batch) and (condition.batch or true_case.batch or false_case.batch):
        #     raise ValueError('TensorFluent.if_then_else: cases must be batch compatible!')

        return TensorFluent(t, scope, batch=batch)

    @classmethod
    def _binary_op(cls,
            x: 'TensorFluent',
            y: 'TensorFluent',
            op: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
            dtype: tf.DType) -> 'TensorFluent':
        '''Returns a TensorFluent for the binary `op` applied to fluents `x` and `y`.

        Args:
            x: The first operand.
            y: The second operand.
            op: The binary operator.
            dtype: The output's data type.

        Returns:
            A TensorFluent wrapping the binary operator's output.
        '''
        # scope
        s1 = x.scope.as_list()
        s2 = y.scope.as_list()
        scope, perm1, perm2 = TensorFluentScope.broadcast(s1, s2)
        if x.batch and perm1 != []:
            perm1 = [0] + [p+1 for p in perm1]
        if y.batch and perm2 != []:
            perm2 = [0] + [p+1 for p in perm2]
        x = x.transpose(perm1)
        y = y.transpose(perm2)

        # shape
        reshape1, reshape2 = TensorFluentShape.broadcast(x.shape, y.shape)
        if reshape1 is not None:
            x = x.reshape(reshape1)
        if reshape2 is not None:
            y = y.reshape(reshape2)

        # dtype
        x = x.cast(dtype)
        y = y.cast(dtype)

        # operation
        t = op(x.tensor, y.tensor)

        # batch
        batch = x.batch or y.batch

        return TensorFluent(t, scope, batch=batch)

    @classmethod
    def _unary_op(cls,
            x: 'TensorFluent',
            op: Callable[[tf.Tensor], tf.Tensor],
            dtype: tf.DType) -> 'TensorFluent':
        '''Returns a TensorFluent for the unary `op` applied to fluent `x`.

        Args:
            x: The input fluent.
            op: The unary operation.
            dtype: The output's data type.

        Returns:
            A TensorFluent wrapping the unary operator's output.
        '''
        x = x.cast(dtype)
        t = op(x.tensor)
        scope = x.scope.as_list()
        batch = x.batch
        return TensorFluent(t, scope, batch=batch)

    @classmethod
    def _aggregation_op(cls,
            op: Callable[[tf.Tensor, Optional[Sequence[int]]], tf.Tensor],
            x: 'TensorFluent',
            vars_list: List[str]) -> 'TensorFluent':
        '''Returns a TensorFluent for the aggregation `op` applied to fluent `x`.

        Args:
            op: The aggregation operation.
            x: The input fluent.
            vars_list: The list of variables to be aggregated over.

        Returns:
            A TensorFluent wrapping the aggregation operator's output.
        '''
        axis = []
        for var in vars_list:
            if var in x.scope.as_list():
                ax = x.scope.index(var)
                if x.batch:
                    ax += 1
                axis.append(ax)
        t = op(x.tensor, axis)

        scope = []
        for var in x.scope.as_list():
            if var not in vars_list:
                scope.append(var)

        batch = x.batch

        return TensorFluent(t, scope, batch=batch)

    def cast(self, dtype: tf.DType) -> 'TensorFluent':
        '''Returns a TensorFluent for the cast operation with given `dtype`.

        Args:
            dtype: The output's data type.

        Returns:
            A TensorFluent wrapping the cast operation.
        '''
        t = self.tensor if self.tensor.dtype == dtype else tf.cast(self.tensor, dtype)
        scope = self.scope.as_list()
        batch = self.batch
        return TensorFluent(t, scope, batch=batch)

    def reshape(self, shape: tf.TensorShape) -> 'TensorFluent':
        '''Returns a TensorFluent for the reshape operation with given `shape`.

        Args:
            shape: The output's shape.

        Returns:
            A TensorFluent wrapping the reshape operation.
        '''
        t = tf.reshape(self.tensor, shape)
        scope = self.scope.as_list()
        batch = self.batch
        return TensorFluent(t, scope, batch=batch)

    def transpose(self, permutation: Optional[List[int]] = None) -> 'TensorFluent':
        '''Returns a TensorFluent for the transpose operation with given `permutation`.

        Args:
            permutation: The output's shape permutation.

        Returns:
            A TensorFluent wrapping the transpose operation.
        '''
        t = tf.transpose(self.tensor, permutation) if permutation != [] else self.tensor
        scope = self.scope.as_list()
        batch = self.batch
        return TensorFluent(t, scope, batch=batch)

    def sum(self, vars_list: List[str]) -> 'TensorFluent':
        '''Returns the TensorFluent for the sum aggregation function.

        Args:
            vars_list: The list of variables to be aggregated over.

        Returns:
            A TensorFluent wrapping the sum aggregation function.
        '''
        operand = self
        if operand.dtype == tf.bool:
            operand = operand.cast(tf.float32)
        return self._aggregation_op(tf.reduce_sum, operand, vars_list)

    def avg(self, vars_list: List[str]) -> 'TensorFluent':
        '''Returns the TensorFluent for the avg aggregation function.

        Args:
            vars_list: The list of variables to be aggregated over.

        Returns:
            A TensorFluent wrapping the avg aggregation function.
        '''
        operand = self
        if operand.dtype == tf.bool:
            operand = operand.cast(tf.float32)
        return self._aggregation_op(tf.reduce_mean, operand, vars_list)

    def prod(self, vars_list: List[str]) -> 'TensorFluent':
        '''Returns the TensorFluent for the prod aggregation function.

        Args:
            vars_list: The list of variables to be aggregated over.

        Returns:
            A TensorFluent wrapping the prod aggregation function.
        '''
        operand = self
        if operand.dtype == tf.bool:
            operand = operand.cast(tf.float32)
        return self._aggregation_op(tf.reduce_prod, operand, vars_list)

    def maximum(self, vars_list: List[str]) -> 'TensorFluent':
        '''Returns the TensorFluent for the maximum aggregation function.

        Args:
            vars_list: The list of variables to be aggregated over.

        Returns:
            A TensorFluent wrapping the maximum aggregation function.
        '''
        return self._aggregation_op(tf.reduce_max, self, vars_list)

    def minimum(self, vars_list: List[str]) -> 'TensorFluent':
        '''Returns the TensorFluent for the minimum aggregation function.

        Args:
            vars_list: The list of variables to be aggregated over.

        Returns:
            A TensorFluent wrapping the minimum aggregation function.
        '''
        return self._aggregation_op(tf.reduce_min, self, vars_list)

    def forall(self, vars_list: List[str]) -> 'TensorFluent':
        '''Returns the TensorFluent for the forall aggregation function.

        Args:
            vars_list: The list of variables to be aggregated over.

        Returns:
            A TensorFluent wrapping the forall aggregation function.
        '''
        return self._aggregation_op(tf.reduce_all, self, vars_list)

    def exists(self, vars_list: List[str]) -> 'TensorFluent':
        '''Returns the TensorFluent for the exists aggregation function.

        Args:
            vars_list: The list of variables to be aggregated over.

        Returns:
            A TensorFluent wrapping the exists aggregation function.
        '''
        return self._aggregation_op(tf.reduce_any, self, vars_list)

    def __neg__(self):
        '''Returns a TensorFluent for the unary negative operator.

        Args:
            self: The operand

        Returns:
            A TensorFluent wrapping the operator's output.
        '''
        return self._unary_op(self, tf.negative, tf.float32)

    def __add__(self, other: 'TensorFluent') -> 'TensorFluent':
        '''Returns a TensorFluent for the addition arithmetic operator.

        Args:
            self: The first operand.
            other: The second operand.

        Returns:
            A TensorFluent wrapping the operator's output.
        '''
        return self._binary_op(self, other, tf.add, tf.float32)

    def __sub__(self, other: 'TensorFluent') -> 'TensorFluent':
        '''Returns a TensorFluent for the subtraction arithmetic operator.

        Args:
            self: The first operand.
            other: The second operand.

        Returns:
            A TensorFluent wrapping the operator's output.
        '''
        return self._binary_op(self, other, tf.subtract, tf.float32)

    def __mul__(self, other: 'TensorFluent') -> 'TensorFluent':
        '''Returns a TensorFluent for the multiplication arithmetic operator.

        Args:
            self: The first operand.
            other: The second operand.

        Returns:
            A TensorFluent wrapping the operator's output.
        '''
        return self._binary_op(self, other, tf.multiply, tf.float32)

    def __truediv__(self, other: 'TensorFluent') -> 'TensorFluent':
        '''Returns a TensorFluent for the division arithmetic operator.

        Args:
            self: The first operand.
            other: The second operand.

        Returns:
            A TensorFluent wrapping the operator's output.
        '''
        return self._binary_op(self, other, tf.divide, tf.float32)

    def __and__(self, other: 'TensorFluent') -> 'TensorFluent':
        '''Returns a TensorFluent for the and logical operator.

        Args:
            self: The first operand.
            other: The second operand.

        Returns:
            A TensorFluent wrapping the operator's output.
        '''
        return self._binary_op(self, other, tf.logical_and, tf.bool)

    def __or__(self, other: 'TensorFluent') -> 'TensorFluent':
        '''Returns a TensorFluent for the or logical operator.

        Args:
            self: The first operand.
            other: The second operand.

        Returns:
            A TensorFluent wrapping the operator's output.
        '''
        return self._binary_op(self, other, tf.logical_or, tf.bool)

    def __xor__(self, other: 'TensorFluent') -> 'TensorFluent':
        '''Returns a TensorFluent for the xor logical operator.

        Args:
            self: The first operand.
            other: The second operand.

        Returns:
            A TensorFluent wrapping the operator's output.
        '''
        return self._binary_op(self, other, tf.logical_xor, tf.bool)

    def __invert__(self) -> 'TensorFluent':
        '''Returns a TensorFluent for the not logical operator.

        Args:
            self: The operand.

        Returns:
            A TensorFluent wrapping the operator's output.
        '''
        return self._unary_op(self, tf.logical_not, tf.bool)

    def __le__(self, other: 'TensorFluent') -> 'TensorFluent':
        '''Returns a TensorFluent for the less-than-or-equal relational operator.

        Args:
            self: The first operand.
            other: The second operand.
        '''
        return self._binary_op(self, other, tf.less_equal, tf.float32)

    def __lt__(self, other: 'TensorFluent') -> 'TensorFluent':
        '''Returns a TensorFluent for the less-then relational operator.

        Args:
            self: The first operand.
            other: The second operand.
        '''
        return self._binary_op(self, other, tf.less, tf.float32)

    def __ge__(self, other: 'TensorFluent') -> 'TensorFluent':
        '''Returns a TensorFluent for the greater-then-or-equal relational operator.

        Args:
            self: The first operand.
            other: The second operand.
        '''
        return self._binary_op(self, other, tf.greater_equal, tf.float32)

    def __gt__(self, other: 'TensorFluent') -> 'TensorFluent':
        '''Returns a TensorFluent for the greater-than relational operator.

        Args:
            self: The first operand.
            other: The second operand.
        '''
        return self._binary_op(self, other, tf.greater, tf.float32)

    def __eq__(self, other: 'TensorFluent') -> 'TensorFluent':
        '''Returns a TensorFluent for the equal relational operator.

        Args:
            self: The first operand.
            other: The second operand.
        '''
        return self._binary_op(self, other, tf.equal, tf.float32)

    def __ne__(self, other: 'TensorFluent') -> 'TensorFluent':
        '''Returns a TensorFluent for the not-equal relational operator.

        Args:
            self: The first operand.
            other: The second operand.
        '''
        return self._binary_op(self, other, tf.not_equal, tf.float32)

    def __str__(self) -> str:
        '''Returns TensorFluent's string representation.'''
        return 'TensorFluent("{}", dtype={}, {}, {})'.format(self.name, repr(self.dtype), self.scope, self.shape)
