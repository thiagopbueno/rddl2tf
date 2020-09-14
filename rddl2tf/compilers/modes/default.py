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

from typing import Dict, Optional
import tensorflow as tf

from pyrddl.rddl import RDDL
from pyrddl.expr import Expression

from rddl2tf.compilers.compiler import Compiler
from rddl2tf.core.fluent import TensorFluent
from rddl2tf import utils


class DefaultCompiler(Compiler):
    '''Default RDDL2TF compiler.

    This compiler supports compilation of constants, random variables,
    functions and operators used in most RDDL expressions.

    Random variables are compiled to sample ops in TensorFlow under the hood.

    Args:
        rddl (:obj:`pyrddl.rddl.RDDL`): The RDDL model.
        batch_size (int): The batch size of all compiled TensorFluent objects.
    '''

    def __init__(self, rddl: RDDL, batch_size: Optional[int] = 1) -> None:
        super(DefaultCompiler, self).__init__(rddl, batch_size)

    def _compile_constant_expression(self,
                                    expr: Expression,
                                    scope: Dict[str, TensorFluent],
                                    **kwargs) -> TensorFluent:
        '''Compile a constant expression `expr` into a TensorFluent
        in the given `scope`. The resulting TensorFluent will have
        batch dimension given by `batch_size`.

        Args:
            expr (:obj:`rddl2tf.expr.Expression`): A RDDL constant expression.
            scope (Dict[str, :obj:`rddl2tf.core.fluent.TensorFluent`]): A fluent scope.
            kwargs: Additional keyword arguments.

        Returns:
            :obj:`rddl2tf.core.fluent.TensorFluent`: The compiled expression as a TensorFluent.
        '''
        etype = expr.etype
        args = expr.args
        dtype = utils.python_type_to_dtype(etype[1])
        fluent = TensorFluent.constant(args, dtype=dtype)
        return fluent

    def _compile_pvariable_expression(self,
                                     expr: Expression,
                                     scope: Dict[str, TensorFluent],
                                     **kwargs) -> TensorFluent:
        '''Compile a pvariable expression `expr` into a TensorFluent
        in the given `scope`. The resulting TensorFluent will have
        batch dimension given by `batch_size`.

        Args:
            expr (:obj:`rddl2tf.expr.Expression`): A RDDL pvariable expression.
            scope (Dict[str, :obj:`rddl2tf.core.fluent.TensorFluent`]): A fluent scope.
            kwargs: Additional keyword arguments.

        Returns:
            :obj:`rddl2tf.core.fluent.TensorFluent`: The compiled expression as a TensorFluent.
        '''
        etype = expr.etype
        args = expr.args
        name = expr._pvar_to_name(args)
        if name not in scope:
            raise ValueError('Variable {} not in scope.'.format(name))
        fluent = scope[name]
        scope = args[1] if args[1] is not None else []
        if isinstance(fluent, TensorFluent):
            fluent = TensorFluent(fluent.tensor, scope, batch=fluent.batch)
        elif isinstance(fluent, tf.Tensor):
            fluent = TensorFluent(fluent, scope, batch=True)
        else:
            raise ValueError('Variable in scope must be TensorFluent-like: {}'.format(fluent))
        return fluent

    def _compile_random_variable_expression(self,
                                            expr: Expression,
                                            scope: Dict[str, TensorFluent],
                                            **kwargs) -> TensorFluent:
        '''Compile a random variable expression `expr` into a TensorFluent
        in the given `scope` with optional batch size.

        If `reparam` tensor is given, then it conditionally stops gradient
        backpropagation at the batch level where `reparam` is False.

        Args:
            expr (:obj:`rddl2tf.expr.Expression`): A RDDL random variable expression.
            scope (Dict[str, :obj:`rddl2tf.core.fluent.TensorFluent`]): A fluent scope.
            kwargs: Additional keyword arguments.

        Returns:
            :obj:`rddl2tf.core.fluent.TensorFluent`: The compiled expression as a TensorFluent.
        '''
        etype = expr.etype
        args = expr.args

        if etype[1] == 'KronDelta':
            sample = self._compile_expression(args[0], scope, **kwargs)
        elif etype[1] == 'Bernoulli':
            mean = self._compile_expression(args[0], scope, **kwargs)
            dist, sample = TensorFluent.Bernoulli(mean, self.batch_size)
        elif etype[1] == 'Uniform':
            low = self._compile_expression(args[0], scope, **kwargs)
            high = self._compile_expression(args[1], scope, **kwargs)
            dist, sample = TensorFluent.Uniform(low, high, self.batch_size)
        elif etype[1] == 'Normal':
            mean = self._compile_expression(args[0], scope, **kwargs)
            variance = self._compile_expression(args[1], scope, **kwargs)
            dist, sample = TensorFluent.Normal(mean, variance, self.batch_size)
        elif etype[1] == 'Laplace':
            mean = self._compile_expression(args[0], scope, **kwargs)
            variance = self._compile_expression(args[1], scope, **kwargs)
            dist, sample = TensorFluent.Laplace(mean, variance, self.batch_size)
        elif etype[1] == 'Gamma':
            shape = self._compile_expression(args[0], scope, **kwargs)
            scale = self._compile_expression(args[1], scope, **kwargs)
            dist, sample = TensorFluent.Gamma(shape, scale, self.batch_size)
        elif etype[1] == 'Exponential':
            rate = self._compile_expression(args[0], scope, **kwargs)
            dist, sample = TensorFluent.Exponential(rate, self.batch_size)
        else:
            raise ValueError('Invalid random variable expression:\n{}.'.format(expr))

        return sample

    def _compile_arithmetic_expression(self,
                                      expr: Expression,
                                      scope: Dict[str, TensorFluent],
                                      **kwargs) -> TensorFluent:
        '''Compile an arithmetic expression `expr` into a TensorFluent
        in the given `scope`. The resulting TensorFluent will have
        batch dimension given by `batch_size`.

        Args:
            expr (:obj:`rddl2tf.expr.Expression`): A RDDL arithmetic expression.
            scope (Dict[str, :obj:`rddl2tf.core.fluent.TensorFluent`]): A fluent scope.
            kwargs: Additional keyword arguments.

        Returns:
            :obj:`rddl2tf.core.fluent.TensorFluent`: The compiled expression as a TensorFluent.
        '''
        etype = expr.etype
        args = expr.args

        if len(args) == 1:
            etype2op = {
                '+': lambda x: x,
                '-': lambda x: -x
            }

            if etype[1] not in etype2op:
                raise ValueError('Invalid binary arithmetic expression:\n{}'.format(expr))

            op = etype2op[etype[1]]
            x = self._compile_expression(args[0], scope, **kwargs)
            fluent = op(x)

        else:
            etype2op = {
                '+': lambda x, y: x + y,
                '-': lambda x, y: x - y,
                '*': lambda x, y: x * y,
                '/': lambda x, y: x / y,
            }

            if etype[1] not in etype2op:
                raise ValueError('Invalid binary arithmetic expression:\n{}'.format(expr))

            op = etype2op[etype[1]]
            x = self._compile_expression(args[0], scope, **kwargs)
            y = self._compile_expression(args[1], scope, **kwargs)
            fluent = op(x, y)

        return fluent

    def _compile_boolean_expression(self,
                                   expr: Expression,
                                   scope: Dict[str, TensorFluent],
                                   **kwargs) -> TensorFluent:
        '''Compile a boolean/logical expression `expr` into a TensorFluent
        in the given `scope`. The resulting TensorFluent will have
        batch dimension given by `batch_size`.

        Args:
            expr (:obj:`rddl2tf.expr.Expression`): A RDDL boolean expression.
            scope (Dict[str, :obj:`rddl2tf.core.fluent.TensorFluent`]): A fluent scope.
            kwargs: Additional keyword arguments.

        Returns:
            :obj:`rddl2tf.core.fluent.TensorFluent`: The compiled expression as a TensorFluent.
        '''
        etype = expr.etype
        args = expr.args

        if len(args) == 1:
            etype2op = {
                '~': lambda x: ~x
            }

            if etype[1] not in etype2op:
                raise ValueError('Invalid unary boolean expression:\n{}'.format(expr))

            op = etype2op[etype[1]]
            x = self._compile_expression(args[0], scope, **kwargs)
            fluent = op(x)

        else:
            etype2op = {
                '^':   lambda x, y: x & y,
                '&':   lambda x, y: x & y,
                '|':   lambda x, y: x | y,
                '=>':  lambda x, y: ~x | y,
                '<=>': lambda x, y: (x & y) | (~x & ~y)
            }

            if etype[1] not in etype2op:
                raise ValueError('Invalid binary boolean expression:\n{}'.format(expr))

            op = etype2op[etype[1]]
            x = self._compile_expression(args[0], scope, **kwargs)
            y = self._compile_expression(args[1], scope, **kwargs)
            fluent = op(x, y)

        return fluent

    def _compile_relational_expression(self,
                                      expr: Expression,
                                      scope: Dict[str, TensorFluent],
                                      **kwargs) -> TensorFluent:
        '''Compile a relational expression `expr` into a TensorFluent
        in the given `scope`. The resulting TensorFluent will have
        batch dimension given by `batch_size`.

        Args:
            expr (:obj:`rddl2tf.expr.Expression`): A RDDL relational expression.
            scope (Dict[str, :obj:`rddl2tf.core.fluent.TensorFluent`]): A fluent scope.
            kwargs: Additional keyword arguments.

        Returns:
            :obj:`rddl2tf.core.fluent.TensorFluent`: The compiled expression as a TensorFluent.
        '''
        etype = expr.etype
        args = expr.args

        etype2op = {
            '<=': lambda x, y: x <= y,
            '<':  lambda x, y: x < y,
            '>=': lambda x, y: x >= y,
            '>':  lambda x, y: x > y,
            '==': lambda x, y: x == y,
            '~=': lambda x, y: x != y
        }

        if etype[1] not in etype2op:
            raise ValueError('Invalid relational expression:\n{}'.format(expr))

        op = etype2op[etype[1]]
        x = self._compile_expression(args[0], scope, **kwargs)
        y = self._compile_expression(args[1], scope, **kwargs)
        fluent = op(x, y)

        return fluent

    def _compile_function_expression(self,
                                    expr: Expression,
                                    scope: Dict[str, TensorFluent],
                                    **kwargs) -> TensorFluent:
        '''Compile a function expression `expr` into a TensorFluent
        in the given `scope`. The resulting TensorFluent will have
        batch dimension given by `batch_size`.

        Args:
            expr (:obj:`rddl2tf.expr.Expression`): A RDDL function expression.
            scope (Dict[str, :obj:`rddl2tf.core.fluent.TensorFluent`]): A fluent scope.
            kwargs: Additional keyword arguments.

        Returns:
            :obj:`rddl2tf.core.fluent.TensorFluent`: The compiled expression as a TensorFluent.
        '''
        etype = expr.etype
        args = expr.args

        if len(args) == 1:

            etype2func = {
                'abs':    TensorFluent.abs,
                'exp':    TensorFluent.exp,
                'log':    TensorFluent.log,
                'sqrt':   TensorFluent.sqrt,
                'cos':    TensorFluent.cos,
                'sin':    TensorFluent.sin,
                'tan':    TensorFluent.tan,
                'acos':   TensorFluent.acos,
                'arccos': TensorFluent.acos,
                'asin':   TensorFluent.asin,
                'arcsin': TensorFluent.asin,
                'atan':   TensorFluent.atan,
                'arctan': TensorFluent.atan,
                'round':  TensorFluent.round,
                'ceil':   TensorFluent.ceil,
                'floor':  TensorFluent.floor
            }

            if etype[1] not in etype2func:
                raise ValueError('Invalid unary function expression:\n{}'.format(expr))

            op = etype2func[etype[1]]
            x = self._compile_expression(args[0], scope, **kwargs)
            fluent = op(x)

        else:
            etype2func = {
                'pow': TensorFluent.pow,
                'max': TensorFluent.max,
                'min': TensorFluent.min
            }

            if etype[1] not in etype2func:
                raise ValueError('Invalid binary function expression:\n{}'.format(expr))

            op = etype2func[etype[1]]
            x = self._compile_expression(args[0], scope, **kwargs)
            y = self._compile_expression(args[1], scope, **kwargs)
            fluent = op(x, y)

        return fluent

    def _compile_control_flow_expression(self,
                                        expr: Expression,
                                        scope: Dict[str, TensorFluent],
                                        **kwargs) -> TensorFluent:
        '''Compile a control flow expression `expr` into a TensorFluent
        in the given `scope`. The resulting TensorFluent will have
        batch dimension given by `batch_size`.

        Args:
            expr (:obj:`rddl2tf.expr.Expression`): A RDDL control flow expression.
            scope (Dict[str, :obj:`rddl2tf.core.fluent.TensorFluent`]): A fluent scope.
            kwargs: Additional keyword arguments.

        Returns:
            :obj:`rddl2tf.core.fluent.TensorFluent`: The compiled expression as a TensorFluent.
        '''
        etype = expr.etype
        args = expr.args
        if etype[1] == 'if':
            condition = self._compile_expression(args[0], scope, **kwargs)
            true_case = self._compile_expression(args[1], scope, **kwargs)
            false_case = self._compile_expression(args[2], scope, **kwargs)
            fluent = TensorFluent.if_then_else(condition, true_case, false_case)
        else:
            raise ValueError('Invalid control flow expression:\n{}'.format(expr))
        return fluent

    def _compile_aggregation_expression(self,
                                       expr: Expression,
                                       scope: Dict[str, TensorFluent],
                                       **kwargs) -> TensorFluent:
        '''Compile an aggregation expression `expr` into a TensorFluent
        in the given `scope`. The resulting TensorFluent will have
        batch dimension given by `batch_size`.

        Args:
            expr (:obj:`rddl2tf.expr.Expression`): A RDDL aggregation expression.
            scope (Dict[str, :obj:`rddl2tf.core.fluent.TensorFluent`]): A fluent scope.
            kwargs: Additional keyword arguments.

        Returns:
            :obj:`rddl2tf.core.fluent.TensorFluent`: The compiled expression as a TensorFluent.
        '''
        etype = expr.etype
        args = expr.args

        typed_var_list = args[:-1]
        vars_list = [var for _, (var, _) in typed_var_list]
        expr = args[-1]

        x = self._compile_expression(expr, scope)

        etype2aggr = {
            'sum':     x.sum,
            'prod':    x.prod,
            'avg':     x.avg,
            'maximum': x.maximum,
            'minimum': x.minimum,
            'exists':  x.exists,
            'forall':  x.forall
        }

        if etype[1] not in etype2aggr:
            raise ValueError('Invalid aggregation expression {}.'.format(expr))

        aggr = etype2aggr[etype[1]]
        fluent = aggr(vars_list=vars_list)

        return fluent
