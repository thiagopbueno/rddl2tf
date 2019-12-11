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

import abc
import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional, Sequence, Tuple

from pyrddl.expr import Expression
from pyrddl.rddl import RDDL

from rddl2tf.compilers.scope import CompilationScope
from rddl2tf.compilers.initializer import CompilationInitializer
from rddl2tf.core.fluent import TensorFluent
from rddl2tf import utils


Bounds = Tuple[Optional[TensorFluent], Optional[TensorFluent]]
CPFPair = Tuple[str, TensorFluent]


class Compiler(metaclass=abc.ABCMeta):
    '''Compiler base class.

    It defines the required methods all RDDL2TF compilers must implement
    in order to compile RDDL expressions to TensorFlow tensors wrapped as
    :obj:`rddl2tf.core.fluent.TensorFluent` objects.

    It supports constants, random variables, functions, and operations
    used in most RDDL expressions.

    Args:
        rddl (:obj:`pyrddl.rddl.RDDL`): The RDDL model.
        batch_size (Optional[int]): The batch size of all compiled TensorFluent objects.
    '''

    def __init__(self, rddl: RDDL, batch_size: Optional[int] = 128) -> None:
        self.rddl = rddl
        self.batch_size = batch_size

        self._initializer = CompilationInitializer(rddl)
        self._scope = CompilationScope(rddl)

    def init(self):
        self.graph = tf.Graph()
        pvariables = self._initializer.initialize_all_pvariables()
        self.non_fluents = [fluent for _, fluent in self._compile_pvariables(pvariables[0])]
        self.initial_state_fluents = self._compile_pvariables(pvariables[1])
        self.default_action_fluents = self._compile_pvariables(pvariables[2])

    def initial_state(self) -> Sequence[tf.Tensor]:
        '''Returns a tuple of tensors representing the initial state fluents.

        Returns:
            Sequence[tf.Tensor]: A tuple of tensors.
        '''
        with self.graph.as_default():
            with tf.name_scope('initial_state'):
                return self._compile_batch_fluents(self.initial_state_fluents)

    def default_action(self) -> Sequence[tf.Tensor]:
        '''Returns a tuple of tensors representing the default action fluents.

        Returns:
            Sequence[tf.Tensor]: A tuple of tensors.
        '''
        with self.graph.as_default():
            with tf.name_scope('default_action'):
                return self._compile_batch_fluents(self.default_action_fluents)

    def cpfs(self,
            state: Sequence[tf.Tensor],
            action: Sequence[tf.Tensor],
            **kwargs) -> Tuple[Sequence[TensorFluent], Sequence[TensorFluent]]:
        '''Compiles the intermediate and next state fluent CPFs given
        the current `state` and `action`.

        Args:
            state (Sequence[tf.Tensor]): A tuple of state tensors.
            action (Sequence[tf.Tensor]): A tuple of action tensors.

        Returns:
            Tuple[List[TensorFluent], List[TensorFluent]]: A pair of lists of TensorFluent
            representing the intermediate and state CPFs.
        '''
        scope = self._scope.transition(self.non_fluents, state, action)
        interm_fluents, next_state_fluents = self._compile_cpfs(scope, **kwargs)
        interms = [fluent for _, fluent in interm_fluents]
        next_state = [fluent for _, fluent in next_state_fluents]
        return interms, next_state

    def reward(self,
               state: Sequence[tf.Tensor],
               action: Sequence[tf.Tensor],
               next_state: Sequence[tf.Tensor],
               **kwargs) -> tf.Tensor:
        '''Compiles the reward function given the current `state`, `action` and
        `next_state`.

        Args:
            state (Sequence[tf.Tensor]): A tuple of current state tensors.
            action (Sequence[tf.Tensor]): A tuple of action tensors.
            next_state (Sequence[tf.Tensor]): A tuple of next state tensors.

        Returns:
            (:obj:`tf.Tensor`): A tensor representing the reward function.
        '''
        scope = self._scope.reward(self.non_fluents, state, action, next_state)
        r = self._compile_reward(scope, **kwargs).tensor
        with self.graph.as_default():
            with tf.name_scope('reward'):
                return tf.expand_dims(r, -1)

    def state_action_constraints(self,
            state: Sequence[tf.Tensor],
            action: Sequence[tf.Tensor]) -> List[TensorFluent]:
        '''Compiles the state-action constraints given current `state` and `action` fluents.

        Args:
            state (Sequence[tf.Tensor]): The current state fluents.
            action (Sequence[tf.Tensor]): The action fluents.

        Returns:
            A list of :obj:`rddl2tf.core.fluent.TensorFluent`.
        '''
        scope = self._scope.transition(self.non_fluents, state, action)
        constraints = []
        with self.graph.as_default():
            with tf.name_scope('state_action_constraints'):
                for p in self.rddl.domain.constraints:
                    fluent = self._compile_expression(p, scope)
                    constraints.append(fluent)
                return constraints

    def action_preconditions(self,
            state: Sequence[tf.Tensor],
            action: Sequence[tf.Tensor]) -> List[TensorFluent]:
        '''Compiles the action preconditions given current `state` and `action` fluents.

        Args:
            state (Sequence[tf.Tensor]): The current state fluents.
            action (Sequence[tf.Tensor]): The action fluents.

        Returns:
            A list of :obj:`rddl2tf.core.fluent.TensorFluent`.
        '''
        scope = self._scope.action_precondition(self.non_fluents, state, action)
        preconds = []
        with self.graph.as_default():
            with tf.name_scope('action_preconditions'):
                for p in self.rddl.domain.preconds:
                    fluent = self._compile_expression(p, scope)
                    preconds.append(fluent)
                return preconds

    def state_invariants(self,
            state: Sequence[tf.Tensor]) -> List[TensorFluent]:
        '''Compiles the state invarints given current `state` fluents.

        Args:
            state (Sequence[tf.Tensor]): The current state fluents.

        Returns:
            A list of :obj:`rddl2tf.core.fluent.TensorFluent`.
        '''
        scope = self._scope.state_invariant(self.non_fluents, state)
        invariants = []
        with self.graph.as_default():
            with tf.name_scope('state_invariants'):
                for p in self.rddl.domain.invariants:
                    fluent = self._compile_expression(p, scope)
                    invariants.append(fluent)
                return invariants

    def action_bound_constraints(self,
            state: Sequence[tf.Tensor]) -> Dict[str, Bounds]:
        '''Compiles all actions bounds for the given `state`.

        Args:
            state (Sequence[tf.Tensor]): The current state fluents.

        Returns:
            A mapping from action names to a pair of
            :obj:`rddl2tf.core.fluent.TensorFluent` representing
            its lower and upper bounds.
        '''
        scope = self._scope.action_precondition(self.non_fluents, state)

        lower_bounds = self.rddl.domain.action_lower_bound_constraints
        upper_bounds = self.rddl.domain.action_upper_bound_constraints

        with self.graph.as_default():
            with tf.name_scope('action_bound_constraints'):

                bounds = {}
                for name in self.rddl.domain.action_fluent_ordering:

                    lower_expr = lower_bounds.get(name)
                    lower = None
                    if lower_expr is not None:
                        with tf.name_scope('lower_bound'):
                            lower = self._compile_expression(lower_expr, scope)

                    upper_expr = upper_bounds.get(name)
                    upper = None
                    if upper_expr is not None:
                        with tf.name_scope('upper_bound'):
                            upper = self._compile_expression(upper_expr, scope)

                    bounds[name] = (lower, upper)

                return bounds

    def _compile_pvariables(self,
            pvariables: List[Tuple[str, str, np.array]]) -> List[Tuple[str, TensorFluent]]:

        fluents = []
        with self.graph.as_default():
            for name, pvar_range, fluent in pvariables:
                dtype = utils.range_type_to_dtype(pvar_range)
                t = tf.constant(fluent, dtype=dtype, name=utils.identifier(name))
                scope = [None] * len(t.shape)
                fluent = TensorFluent(t, scope, batch=False)
                fluents.append((name, fluent))
        return fluents


    def _compile_batch_fluents(self,
            fluents: List[Tuple[str, TensorFluent]]) -> Sequence[tf.Tensor]:
        '''Compiles `fluents` into tensors with given `batch_size`.

        Returns:
            Sequence[tf.Tensor]: A tuple of tensors with first dimension
            corresponding to the batch size.
        '''
        batch_fluents = []
        with self.graph.as_default():
            for name, fluent in fluents:
                name_scope = utils.identifier(name)
                with tf.name_scope(name_scope):
                    t = tf.stack([fluent.tensor] * self.batch_size)
                batch_fluents.append(t)
        return tuple(batch_fluents)

    def _compile_cpfs(self,
                     scope: Dict[str, TensorFluent],
                     **kwargs) -> Tuple[List[CPFPair], List[CPFPair]]:
        '''Compiles the intermediate and next state fluent CPFs given the current `state` and `action` scope.

        Args:
            scope (Dict[str, :obj:`rddl2tf.core.fluent.TensorFluent`]): The fluent scope for CPF evaluation.
            batch_size (Optional[int]): The batch size.

        Returns:
            Tuple[List[CPFPair], List[CPFPair]]: A pair of lists of TensorFluent
            representing the intermediate and state CPFs.
        '''
        interm_fluents = self._compile_intermediate_cpfs(scope, **kwargs)
        scope.update(dict(interm_fluents))
        next_state_fluents = self._compile_state_cpfs(scope, **kwargs)
        return interm_fluents, next_state_fluents

    def _compile_intermediate_cpfs(self,
                                  scope: Dict[str, TensorFluent],
                                  **kwargs) -> List[CPFPair]:
        '''Compiles the intermediate fluent CPFs given the current `state` and `action` scope.

        Args:
            scope (Dict[str, :obj:`rddl2tf.core.fluent.TensorFluent`]): The fluent scope for CPF evaluation.
            batch_size (Optional[int]): The batch size.

        Returns:
            A list of intermediate fluent CPFs compiled to :obj:`rddl2tf.core.fluent.TensorFluent`.
        '''
        interm_fluents = []

        with self.graph.as_default():
            with tf.name_scope('intermediate_cpfs'):

                for cpf in self.rddl.domain.intermediate_cpfs:
                    name_scope = utils.identifier(cpf.name)

                    with tf.name_scope(name_scope):
                        t = self._compile_expression(cpf.expr, scope, **kwargs)
                        interm_fluents.append((cpf.name, t))
                        scope[cpf.name] = t

        return interm_fluents

    def _compile_state_cpfs(self,
                           scope: Dict[str, TensorFluent],
                           **kwargs) -> List[CPFPair]:
        '''Compiles the next state fluent CPFs given the current `state` and `action` scope.

        Args:
            scope (Dict[str, :obj:`rddl2tf.core.fluent.TensorFluent`]): The fluent scope for CPF evaluation.
            batch_size (Optional[int]): The batch size.

        Returns:
            A list of state fluent CPFs compiled to :obj:`rddl2tf.core.fluent.TensorFluent`.
        '''
        next_state_fluents = []

        with self.graph.as_default():
            with tf.name_scope('state_cpfs'):

                for cpf in self.rddl.domain.state_cpfs:
                    name_scope = utils.identifier(cpf.name)

                    with tf.name_scope(name_scope):
                        t = self._compile_expression(cpf.expr, scope, **kwargs)
                        next_state_fluents.append((cpf.name, t))

                key = lambda f: self.rddl.domain.next_state_fluent_ordering.index(f[0])
                next_state_fluents = sorted(next_state_fluents, key=key)

        return next_state_fluents

    def _compile_reward(self,
            scope: Dict[str, TensorFluent],
            **kwargs) -> TensorFluent:
        '''Compiles the reward function given the fluent `scope`.

        Args:
            scope (Dict[str, :obj:`rddl2tf.core.fluent.TensorFluent`]): The fluent scope for reward evaluation.

        Returns:
            A :obj:`rddl2tf.core.fluent.TensorFluent` representing the reward function.
        '''
        reward_expr = self.rddl.domain.reward
        with self.graph.as_default():
            with tf.name_scope('reward'):
                return self._compile_expression(reward_expr, scope, **kwargs)

    def _compile_expression(self,
                           expr: Expression,
                           scope: Dict[str, TensorFluent],
                           **kwargs) -> TensorFluent:
        '''Compile the expression `expr` into a TensorFluent
        in the given `scope`. The resulting TensorFluent will have
        batch dimension given by `batch_size`.

        Args:
            expr (:obj:`rddl2tf.expr.Expression`): A RDDL expression.
            scope (Dict[str, :obj:`rddl2tf.core.fluent.TensorFluent`]): A fluent scope.
            kwargs: Additional keyword arguments.

        Returns:
            :obj:`rddl2tf.core.fluent.TensorFluent`: The compiled TensorFluent.
        '''
        etype2compiler = {
            'constant':    self._compile_constant_expression,
            'pvar':        self._compile_pvariable_expression,
            'randomvar':   self._compile_random_variable_expression,
            'arithmetic':  self._compile_arithmetic_expression,
            'boolean':     self._compile_boolean_expression,
            'relational':  self._compile_relational_expression,
            'func':        self._compile_function_expression,
            'control':     self._compile_control_flow_expression,
            'aggregation': self._compile_aggregation_expression
        }

        etype = expr.etype
        if etype[0] not in etype2compiler:
            raise ValueError('Expression type unknown: {}'.format(etype))

        with self.graph.as_default():
            compiler_fn = etype2compiler[etype[0]]
            return compiler_fn(expr, scope, **kwargs)

    @abc.abstractmethod
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
        raise NotImplementedError

    @abc.abstractmethod
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
        raise NotImplementedError

    @abc.abstractmethod
    def _compile_random_variable_expression(self,
                                           expr: Expression,
                                           scope: Dict[str, TensorFluent],
                                           **kwargs) -> TensorFluent:
        '''Compile a random variable expression `expr` into a TensorFluent
        in the given `scope`. The resulting TensorFluent will have
        batch dimension given by `batch_size`.

        If `reparam` tensor is given, then it conditionally stops gradient
        backpropagation at the batch level where `reparam` is False.

        Args:
            expr (:obj:`rddl2tf.expr.Expression`): A RDDL random variable expression.
            scope (Dict[str, :obj:`rddl2tf.core.fluent.TensorFluent`]): A fluent scope.
            kwargs: Additional keyword arguments.

        Returns:
            :obj:`rddl2tf.core.fluent.TensorFluent`: The compiled expression as a TensorFluent.
        '''
        raise NotImplementedError

    @abc.abstractmethod
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
        raise NotImplementedError

    @abc.abstractmethod
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
        raise NotImplementedError

    @abc.abstractmethod
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
        raise NotImplementedError

    @abc.abstractmethod
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
        raise NotImplementedError

    @abc.abstractmethod
    def _compile_control_flow_expression(self,
                                        expr: Expression,
                                        scope: Dict[str, TensorFluent],
                                        **kwargs) -> TensorFluent:
        '''Compile a control flow expression `expr` into a TensorFluent
        in the given `scope`. The resulting TensorFluent will have
        batch dimension given by `batch_size`

        Args:
            expr (:obj:`rddl2tf.expr.Expression`): A RDDL control flow expression.
            scope (Dict[str, :obj:`rddl2tf.core.fluent.TensorFluent`]): A fluent scope.
            kwargs: Additional keyword arguments.

        Returns:
            :obj:`rddl2tf.core.fluent.TensorFluent`: The compiled expression as a TensorFluent.
        '''
        raise NotImplementedError

    @abc.abstractmethod
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
        raise NotImplementedError
