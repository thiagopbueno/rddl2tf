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


from pyrddl.expr import Expression
from pyrddl.rddl import RDDL

from rddl2tf.compilers.modes.default import DefaultCompiler
from rddl2tf.core.fluent import TensorFluent
from rddl2tf.core.fluentshape import TensorFluentShape
from rddl2tf import utils

import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional, Tuple


CPFPair = Tuple[str, TensorFluent]
ShapeAsList = List[int]
ShapeScope = Dict[str, TensorFluentShape]
Noise = Tuple[tf.distributions.Distribution, ShapeAsList]
NoiseList = List[Noise]
NoiseMap = List[Tuple[str, NoiseList]]


class ReparameterizationCompiler(DefaultCompiler):
    """Reparameterization-based Compiler class.

    This extends the DefaultCompiler in order to support the compilation of
    constants, functions and operators used in most RDDL expressions.

    Random variables are compiled to deterministic ops in TensorFlow
    under the hood via the re-parameterization trick.

    Args:
        rddl (:obj:`pyrddl.rddl.RDDL`): The RDDL model.
        batch_size (int): The batch size of all compiled TensorFluent objects.
    """

    def __init__(self, rddl: RDDL, batch_size: Optional[int] = 1) -> None:
        super(DefaultCompiler, self).__init__(rddl, batch_size)

    def _compile_intermediate_cpfs(
        self, scope: Dict[str, TensorFluent], **kwargs
    ) -> List[CPFPair]:
        """Compiles the intermediate fluent CPFs given the current `state` and `action` scope.

        Args:
            scope (Dict[str, :obj:`rddl2tf.core.fluent.TensorFluent`]): The fluent scope for CPF evaluation.
            batch_size (Optional[int]): The batch size.

        Returns:
            A list of intermediate fluent CPFs compiled to :obj:`rddl2tf.core.fluent.TensorFluent`.
        """
        noise = kwargs["noise"]

        interm_fluents = []

        with self.graph.as_default():
            with tf.name_scope("intermediate_cpfs"):

                for cpf in self.rddl.domain.intermediate_cpfs:
                    cpf_noise = noise.get(cpf.name)

                    name_scope = utils.identifier(cpf.name)
                    with tf.name_scope(name_scope):
                        t = self._compile_expression(cpf.expr, scope, noise=cpf_noise)

                    interm_fluents.append((cpf.name, t))
                    scope[cpf.name] = t

        return interm_fluents

    def _compile_state_cpfs(
        self, scope: Dict[str, TensorFluent], **kwargs
    ) -> List[CPFPair]:
        """Compiles the next state fluent CPFs given the current `state` and `action` scope.

        Args:
            scope (Dict[str, :obj:`rddl2tf.core.fluent.TensorFluent`]): The fluent scope for CPF evaluation.
            batch_size (Optional[int]): The batch size.

        Returns:
            A list of state fluent CPFs compiled to :obj:`rddl2tf.core.fluent.TensorFluent`.
        """
        noise = kwargs["noise"]

        next_state_fluents = []

        with self.graph.as_default():
            with tf.name_scope("state_cpfs"):

                for cpf in self.rddl.domain.state_cpfs:
                    cpf_noise = noise.get(cpf.name)

                    name_scope = utils.identifier(cpf.name)
                    with tf.name_scope(name_scope):
                        t = self._compile_expression(cpf.expr, scope, noise=cpf_noise)

                    next_state_fluents.append((cpf.name, t))

                key = lambda f: self.rddl.domain.next_state_fluent_ordering.index(f[0])
                next_state_fluents = sorted(next_state_fluents, key=key)

        return next_state_fluents

    def _compile_random_variable_expression(
        self, expr: Expression, scope: Dict[str, TensorFluent], **kwargs
    ) -> TensorFluent:
        """Compile a random variable expression `expr` into a TensorFluent
        in the given `scope` with optional batch size.

        If `reparam` tensor is given, then it conditionally stops gradient
        backpropagation at the batch level where `reparam` is False.

        Args:
            expr (:obj:`rddl2tf.expr.Expression`): A RDDL random variable expression.
            scope (Dict[str, :obj:`rddl2tf.core.fluent.TensorFluent`]): A fluent scope.
            kwargs: Additional keyword arguments.

        Keyword Args:
            noise (List[tf.Tensor]): Noise sample ops for reparameterization.

        Returns:
            :obj:`rddl2tf.core.fluent.TensorFluent`: The compiled expression as a TensorFluent.
        """
        etype = expr.etype
        args = expr.args

        noise = kwargs["noise"]

        if etype[1] == "KronDelta":
            sample = self._compile_expression(args[0], scope, **kwargs)
        elif etype[1] == "Bernoulli":
            mean = self._compile_expression(args[0], scope, **kwargs)
            dist, sample = TensorFluent.Bernoulli(mean, self.batch_size)
        elif etype[1] == "Uniform":
            xi = noise.pop()
            xi = TensorFluent(tf.sigmoid(xi), scope=[], batch=True)
            low = self._compile_expression(args[0], scope, **kwargs)
            high = self._compile_expression(args[0], scope, **kwargs)
            sample = low + (high - low) * xi
        elif etype[1] == "Normal":
            xi = noise.pop()
            xi = TensorFluent(2.0 * tf.tanh(xi / 2.0), scope=[], batch=True)
            mean = self._compile_expression(args[0], scope, **kwargs)
            variance = self._compile_expression(args[1], scope, **kwargs)
            sample = mean + TensorFluent.sqrt(variance) * xi
        elif etype[1] == "Laplace":
            mean = self._compile_expression(args[0], scope, **kwargs)
            variance = self._compile_expression(args[1], scope, **kwargs)
            dist, sample = TensorFluent.Laplace(mean, variance, self.batch_size)
        elif etype[1] == "Gamma":
            sample = TensorFluent(noise.pop(), scope=[], batch=True)
        elif etype[1] == "Exponential":
            xi = noise.pop()
            xi = TensorFluent(tf.sigmoid(xi), scope=[], batch=True)
            rate = self._compile_expression(args[0], scope, **kwargs)
            sample = -(TensorFluent.constant(1.0) / rate) * TensorFluent.log(xi)
        else:
            raise ValueError("Invalid random variable expression:\n{}.".format(expr))

        return sample

    def get_cpfs_reparameterization(self) -> NoiseMap:
        noise = self.get_intermediate_cpfs_reparameterization()
        noise += self.get_state_cpfs_reparameterization()
        return noise

    def get_state_cpfs_reparameterization(self) -> NoiseMap:
        scope = self._get_reparameterization_shape_scope()
        noise = []
        for cpf in self.rddl.domain.state_cpfs:
            noise.append(
                (cpf.name, self._get_expression_reparameterization(cpf.expr, scope))
            )
        return noise

    def get_intermediate_cpfs_reparameterization(self) -> NoiseMap:
        scope = self._get_reparameterization_shape_scope()
        noise = []
        for cpf in self.rddl.domain.intermediate_cpfs:
            noise.append(
                (cpf.name, self._get_expression_reparameterization(cpf.expr, scope))
            )
        return noise

    def _get_reparameterization_shape_scope(self) -> ShapeScope:
        scope = {
            name: TensorFluentShape(size, batch=False)
            for name, (_, size) in self.rddl.fluent_table.items()
        }
        return scope

    def _get_expression_reparameterization(
        self, expr: Expression, scope: ShapeScope
    ) -> NoiseList:
        noise = []
        self._get_reparameterization(expr, scope, noise)
        return noise

    def _get_reparameterization(
        self, expr: Expression, scope: ShapeScope, noise: NoiseList
    ) -> TensorFluentShape:
        etype = expr.etype
        args = expr.args

        if etype[0] == "constant":
            return TensorFluentShape([1], batch=False)
        elif etype[0] == "pvar":
            name = expr._pvar_to_name(args)
            if name not in scope:
                raise ValueError("Variable {} not in scope.".format(name))
            shape = scope[name]
            return shape
        elif etype[0] == "randomvar":
            if etype[1] == "Normal":
                mean_shape = self._get_reparameterization(args[0], scope, noise)
                var_shape = self._get_reparameterization(args[1], scope, noise)
                shape = ReparameterizationCompiler._broadcast(mean_shape, var_shape)
                dist = tf.distributions.Normal(loc=0.0, scale=1.0)
                noise.append((dist, shape.as_list()))
                return shape
            elif etype[1] == "Exponential":
                rate_shape = self._get_reparameterization(args[0], scope, noise)
                dist = tf.distributions.Uniform(low=0.0, high=1.0)
                noise.append((dist, rate_shape.as_list()))
                return rate_shape
            elif etype[1] == "Gamma":

                for fluent in self.rddl.get_dependencies(expr):
                    if fluent.is_state_fluent() or fluent.is_action_fluent():
                        raise ValueError(
                            f"Expression is not an exogenous event: {expr}"
                        )

                shape = []

                with self.graph.as_default():
                    scope = self._scope.non_fluents(self.non_fluents)
                    shape_fluent = self._compile_expression(args[0], scope, noise=None)
                    scale_fluent = self._compile_expression(args[1], scope, noise=None)
                    concentration = shape_fluent.tensor
                    rate = 1 / scale_fluent.tensor
                    dist = tf.distributions.Gamma(concentration, rate)

                noise.append((dist, shape))

                return shape

            elif etype[1] == "Uniform":
                low_shape = self._get_reparameterization(args[0], scope, noise)
                high_shape = self._get_reparameterization(args[1], scope, noise)
                shape = ReparameterizationCompiler._broadcast(low_shape, high_shape)
                dist = tf.distributions.Uniform(low=0.0, high=1.0)
                noise.append((dist, shape.as_list()))
                return shape
        elif etype[0] in ["arithmetic", "boolean", "relational"]:
            op1_shape = self._get_reparameterization(args[0], scope, noise)
            shape = op1_shape
            if len(args) > 1:
                op2_shape = self._get_reparameterization(args[1], scope, noise)
                shape = ReparameterizationCompiler._broadcast(op1_shape, op2_shape)
            return shape
        elif etype[0] == "func":
            op1_shape = self._get_reparameterization(args[0], scope, noise)
            shape = op1_shape
            if len(args) > 1:
                if len(args) == 2:
                    op2_shape = self._get_reparameterization(args[1], scope, noise)
                    shape = ReparameterizationCompiler._broadcast(op1_shape, op2_shape)
                else:
                    raise ValueError("Invalid function:\n{}".format(expr))
            return shape
        elif etype[0] == "control":
            if etype[1] == "if":
                condition_shape = self._get_reparameterization(args[0], scope, noise)
                true_case_shape = self._get_reparameterization(args[1], scope, noise)
                false_case_shape = self._get_reparameterization(args[2], scope, noise)
                shape = ReparameterizationCompiler._broadcast(
                    condition_shape, true_case_shape
                )
                shape = ReparameterizationCompiler._broadcast(shape, false_case_shape)
                return shape
            else:
                raise ValueError("Invalid control flow expression:\n{}".format(expr))
        elif etype[0] == "aggregation":
            return self._get_reparameterization(args[-1], scope, noise)

        raise ValueError("Expression type unknown: {}".format(etype))

    @classmethod
    def _broadcast(
        cls, shape1: TensorFluentShape, shape2: TensorFluentShape
    ) -> TensorFluentShape:
        s1, s2 = TensorFluentShape.broadcast(shape1, shape2)
        s1 = s1 if s1 is not None else shape1.as_list()
        s2 = s2 if s2 is not None else shape2.as_list()
        x1, x2 = np.zeros(s1), np.zeros(s2)
        y = np.broadcast(x1, x2)
        return TensorFluentShape(y.shape, batch=(shape1.batch or shape2.batch))
