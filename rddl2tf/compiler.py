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


from pyrddl.rddl import RDDL
from pyrddl.pvariable import PVariable
from pyrddl.expr import Expression

from rddl2tf.fluent import TensorFluent
from rddl2tf import utils

import numpy as np
import tensorflow as tf

from typing import Dict, List, Optional, Sequence, Tuple, Union

CPFPair = Tuple[str, TensorFluent]
CPFTriple = Tuple[str, TensorFluent, TensorFluent]
FluentList = List[Tuple[str, TensorFluent]]
Bounds = Tuple[Optional[TensorFluent], Optional[TensorFluent]]
Value = Union[bool, int, float]
ArgsList = Optional[List[str]]
InitializerPair = Tuple[Tuple[str, ArgsList], Value]
InitializerList = List[InitializerPair]

Noise = Dict[str, Sequence[tf.Tensor]]


class Compiler(object):
    '''RDDL2TensorFlow compiler.

    This is the core component of rddl2tf package.

    Its API provides methods to compile RDDL fluents and expressions
    to TensorFlow tensors wrapped as :obj:`rddl2tf.fluent.TensorFluent` objects.
    It supports constants, random variables, functions and operators
    used in most RDDL expressions. Also, it can handle next state
    and intermediate fluent CPFs, and rewards and action constraints.

    Args:
        rddl (:obj:`pyrddl.rddl.RDDL`): The RDDL model.
        batch_mode (bool): The batch mode flag.

    Attributes:
        rddl (:obj:`pyrddl.rddl.RDDL`): The RDDL model.
        batch_mode (bool): The batch mode flag.
        graph (:obj:`tensorflow.python.framework.ops.Graph`): The computation graph.
    '''

    def __init__(self, rddl: RDDL, batch_mode: bool = False) -> None:
        self.rddl = rddl
        self.batch_mode = batch_mode
        self.graph = tf.Graph()

    def batch_mode_on(self):
        '''Sets on the batch mode flag.'''
        self.batch_mode = True

    def batch_mode_off(self):
        '''Sets off the batch mode flag.'''
        self.batch_mode = False

    def compile_non_fluents(self):
        '''Returns a tuple of tensors representing the non fluents.

        Returns:
            Sequence[tf.Tensor]: A tuple of tensors.
        '''
        with self.graph.as_default():
            with tf.name_scope('non_fluents'):
                self._initialize_non_fluents()
                return self.non_fluents

    def compile_initial_state(self, batch_size: Optional[int] = None) -> Sequence[tf.Tensor]:
        '''Returns a tuple of tensors representing the initial state fluents.

        Args:
            batch_size (Optional[int]): The batch size.

        Returns:
            Sequence[tf.Tensor]: A tuple of tensors.
        '''
        with self.graph.as_default():
            with tf.name_scope('initial_state'):
                self._initialize_initial_state_fluents()
                if batch_size is None:
                    return self.initial_state_fluents
                return self._compile_batch_fluents(self.initial_state_fluents, batch_size)

    def compile_default_action(self, batch_size: Optional[int] = None) -> Sequence[tf.Tensor]:
        '''Returns a tuple of tensors representing the default action fluents.

        Args:
            batch_size (int): The batch size.

        Returns:
            Sequence[tf.Tensor]: A tuple of tensors.
        '''
        with self.graph.as_default():
            with tf.name_scope('default_action'):
                self._initialize_default_action_fluents()
                if batch_size is None:
                    return self.default_action_fluents
                return self._compile_batch_fluents(self.default_action_fluents, batch_size)

    def cpfs(self,
             state: Sequence[tf.Tensor],
             action: Sequence[tf.Tensor],
             noise: Optional[Noise] = None) -> Tuple[List[TensorFluent], List[TensorFluent]]:
        '''Compiles the intermediate and next state fluent CPFs given
        the current `state` and `action`.

        Args:
            state (Sequence[tf.Tensor]): A tuple of state tensors.
            action (Sequence[tf.Tensor]): A tuple of action tensors.

        Returns:
            Tuple[List[TensorFluent], List[TensorFluent]]: A pair of lists of TensorFluent
            representing the intermediate and state CPFs.
        '''
        scope = self.transition_scope(state, action)
        batch_size = int(state[0].shape[0])
        interm_fluents, next_state_fluents = self.compile_cpfs(scope, batch_size, noise)
        interms = [fluent for _, fluent in interm_fluents]
        next_state = [fluent for _, fluent in next_state_fluents]
        return interms, next_state

    def reward(self,
               state: Sequence[tf.Tensor],
               action: Sequence[tf.Tensor],
               next_state: Sequence[tf.Tensor]) -> tf.Tensor:
        '''Compiles the reward function given the current `state`, `action` and
        `next_state`.

        Args:
            state (Sequence[tf.Tensor]): A tuple of current state tensors.
            action (Sequence[tf.Tensor]): A tuple of action tensors.
            next_state (Sequence[tf.Tensor]): A tuple of next state tensors.

        Returns:
            (:obj:`tf.Tensor`): A tensor representing the reward function.
        '''
        scope = self.reward_scope(state, action, next_state)
        r = self.compile_reward(scope).tensor
        with self.graph.as_default():
            with tf.name_scope('reward'):
                return tf.expand_dims(r, -1)

    def compile_cpfs(self,
                     scope: Dict[str, TensorFluent],
                     batch_size: Optional[int] = None,
                     noise: Optional[Noise] = None) -> Tuple[List[CPFPair], List[CPFPair]]:
        '''Compiles the intermediate and next state fluent CPFs given the current `state` and `action` scope.

        Args:
            scope (Dict[str, :obj:`rddl2tf.fluent.TensorFluent`]): The fluent scope for CPF evaluation.
            batch_size (Optional[int]): The batch size.

        Returns:
            Tuple[List[CPFPair], List[CPFPair]]: A pair of lists of TensorFluent
            representing the intermediate and state CPFs.
        '''
        interm_fluents = self.compile_intermediate_cpfs(scope, batch_size, noise)
        scope.update(dict(interm_fluents))
        next_state_fluents = self.compile_state_cpfs(scope, batch_size, noise)
        return interm_fluents, next_state_fluents

    def compile_intermediate_cpfs(self,
                                  scope: Dict[str, TensorFluent],
                                  batch_size: Optional[int] = None,
                                  noise: Optional[Noise] = None) -> List[CPFPair]:
        '''Compiles the intermediate fluent CPFs given the current `state` and `action` scope.

        Args:
            scope (Dict[str, :obj:`rddl2tf.fluent.TensorFluent`]): The fluent scope for CPF evaluation.
            batch_size (Optional[int]): The batch size.

        Returns:
            A list of intermediate fluent CPFs compiled to :obj:`rddl2tf.fluent.TensorFluent`.
        '''
        interm_fluents = []

        with self.graph.as_default():
            with tf.name_scope('intermediate_cpfs'):

                for cpf in self.rddl.domain.intermediate_cpfs:
                    cpf_noise = noise.get(cpf.name, None) if noise is not None else None

                    name_scope = utils.identifier(cpf.name)
                    with tf.name_scope(name_scope):
                        t = self._compile_expression(cpf.expr, scope, batch_size, cpf_noise)

                    interm_fluents.append((cpf.name, t))
                    scope[cpf.name] = t

        return interm_fluents

    def compile_state_cpfs(self,
                           scope: Dict[str, TensorFluent],
                           batch_size: Optional[int] = None,
                           noise: Optional[Noise] = None) -> List[CPFPair]:
        '''Compiles the next state fluent CPFs given the current `state` and `action` scope.

        Args:
            scope (Dict[str, :obj:`rddl2tf.fluent.TensorFluent`]): The fluent scope for CPF evaluation.
            batch_size (Optional[int]): The batch size.

        Returns:
            A list of state fluent CPFs compiled to :obj:`rddl2tf.fluent.TensorFluent`.
        '''
        next_state_fluents = []

        with self.graph.as_default():
            with tf.name_scope('state_cpfs'):

                for cpf in self.rddl.domain.state_cpfs:
                    cpf_noise = noise.get(cpf.name, None) if noise is not None else None

                    name_scope = utils.identifier(cpf.name)
                    with tf.name_scope(name_scope):
                        t = self._compile_expression(cpf.expr, scope, batch_size, cpf_noise)

                    next_state_fluents.append((cpf.name, t))

                key = lambda f: self.rddl.domain.next_state_fluent_ordering.index(f[0])
                next_state_fluents = sorted(next_state_fluents, key=key)

        return next_state_fluents

    def compile_reward(self, scope: Dict[str, TensorFluent]) -> TensorFluent:
        '''Compiles the reward function given the fluent `scope`.

        Args:
            scope (Dict[str, :obj:`rddl2tf.fluent.TensorFluent`]): The fluent scope for reward evaluation.

        Returns:
            A :obj:`rddl2tf.fluent.TensorFluent` representing the reward function.
        '''
        reward_expr = self.rddl.domain.reward
        with self.graph.as_default():
            with tf.name_scope('reward'):
                return self._compile_expression(reward_expr, scope)

    def compile_state_action_constraints(self,
            state: Sequence[tf.Tensor],
            action: Sequence[tf.Tensor]) -> List[TensorFluent]:
        '''Compiles the state-action constraints given current `state` and `action` fluents.

        Args:
            state (Sequence[tf.Tensor]): The current state fluents.
            action (Sequence[tf.Tensor]): The action fluents.

        Returns:
            A list of :obj:`rddl2tf.fluent.TensorFluent`.
        '''
        scope = self.transition_scope(state, action)
        constraints = []
        with self.graph.as_default():
            with tf.name_scope('state_action_constraints'):
                for p in self.rddl.domain.constraints:
                    fluent = self._compile_expression(p, scope)
                    constraints.append(fluent)
                return constraints

    def compile_action_preconditions(self,
            state: Sequence[tf.Tensor],
            action: Sequence[tf.Tensor]) -> List[TensorFluent]:
        '''Compiles the action preconditions given current `state` and `action` fluents.

        Args:
            state (Sequence[tf.Tensor]): The current state fluents.
            action (Sequence[tf.Tensor]): The action fluents.

        Returns:
            A list of :obj:`rddl2tf.fluent.TensorFluent`.
        '''
        scope = self.action_precondition_scope(state, action)
        preconds = []
        with self.graph.as_default():
            with tf.name_scope('action_preconditions'):
                for p in self.rddl.domain.preconds:
                    fluent = self._compile_expression(p, scope)
                    preconds.append(fluent)
                return preconds

    def compile_state_invariants(self,
            state: Sequence[tf.Tensor]) -> List[TensorFluent]:
        '''Compiles the state invarints given current `state` fluents.

        Args:
            state (Sequence[tf.Tensor]): The current state fluents.

        Returns:
            A list of :obj:`rddl2tf.fluent.TensorFluent`.
        '''
        scope = self.state_invariant_scope(state)
        invariants = []
        with self.graph.as_default():
            with tf.name_scope('state_invariants'):
                for p in self.rddl.domain.invariants:
                    fluent = self._compile_expression(p, scope)
                    invariants.append(fluent)
                return invariants

    def compile_action_preconditions_checking(self,
            state: Sequence[tf.Tensor],
            action: Sequence[tf.Tensor]) -> tf.Tensor:
        '''Combines the action preconditions into an applicability checking op.

        Args:
            state (Sequence[tf.Tensor]): The current state fluents.
            action (Sequence[tf.Tensor]): The action fluents.

        Returns:
            A boolean tensor for checking if `action` is application in `state`.
        '''
        with self.graph.as_default():
            with tf.name_scope('action_preconditions_checking'):
                preconds = self.compile_action_preconditions(state, action)
                all_preconds = tf.stack([p.tensor for p in preconds], axis=1)
                checking = tf.reduce_all(all_preconds, axis=1)
                return checking

    def compile_action_bound_constraints(self,
            state: Sequence[tf.Tensor]) -> Dict[str, Bounds]:
        '''Compiles all actions bounds for the given `state`.

        Args:
            state (Sequence[tf.Tensor]): The current state fluents.

        Returns:
            A mapping from action names to a pair of
            :obj:`rddl2tf.fluent.TensorFluent` representing
            its lower and upper bounds.
        '''
        scope = self.action_precondition_scope(state)

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

    def non_fluents_scope(self) -> Dict[str, TensorFluent]:
        '''Returns a partial scope with non-fluents.

        Returns:
            A mapping from non-fluent names to :obj:`rddl2tf.fluent.TensorFluent`.
        '''
        if self.__dict__.get('non_fluents') is None:
            self._initialize_non_fluents()
        return dict(self.non_fluents)

    def state_scope(self, state_fluents: Sequence[tf.Tensor]) -> Dict[str, TensorFluent]:
        '''Returns a partial scope with current state-fluents.

        Args:
            state_fluents (Sequence[tf.Tensor]): The current state fluents.

        Returns:
            A mapping from state fluent names to :obj:`rddl2tf.fluent.TensorFluent`.
        '''
        return dict(zip(self.rddl.domain.state_fluent_ordering, state_fluents))

    def action_scope(self, action_fluents: Sequence[tf.Tensor]) -> Dict[str, TensorFluent]:
        '''Returns a partial scope with current action-fluents.

        Args:
            action_fluents (Sequence[tf.Tensor]): The action fluents.

        Returns:
            A mapping from action fluent names to :obj:`rddl2tf.fluent.TensorFluent`.
        '''
        return dict(zip(self.rddl.domain.action_fluent_ordering, action_fluents))

    def next_state_scope(self, next_state_fluents: Sequence[tf.Tensor]) -> Dict[str, TensorFluent]:
        '''Returns a partial scope with current next state-fluents.

        Args:
            next_state_fluents (Sequence[tf.Tensor]): The next state fluents.

        Returns:
            A mapping from next state fluent names to :obj:`rddl2tf.fluent.TensorFluent`.
        '''
        return dict(zip(self.rddl.domain.next_state_fluent_ordering, next_state_fluents))

    def transition_scope(self,
        state: Sequence[tf.Tensor],
        action: Sequence[tf.Tensor]) -> Dict[str, TensorFluent]:
        '''Returns the complete transition fluent scope
        for the current `state` and `action` fluents.

        Args:
            state (Sequence[tf.Tensor]): The current state fluents.
            action (Sequence[tf.Tensor]): The action fluents.

        Returns:
            A mapping from fluent names to :obj:`rddl2tf.fluent.TensorFluent`.
        '''
        scope = {}
        scope.update(self.non_fluents_scope())
        scope.update(self.state_scope(state))
        scope.update(self.action_scope(action))
        return scope

    def reward_scope(self,
                     state: Sequence[tf.Tensor],
                     action: Sequence[tf.Tensor],
                     next_state: Sequence[tf.Tensor]) -> Dict[str, TensorFluent]:
        '''Returns the complete reward fluent scope for the
        current `state`, `action` fluents, and `next_state` fluents.

        Args:
            state (Sequence[tf.Tensor]): The current state fluents.
            action (Sequence[tf.Tensor]): The action fluents.
            next_state (Sequence[tf.Tensor]): The next state fluents.

        Returns:
            A mapping from fluent names to :obj:`rddl2tf.fluent.TensorFluent`.
        '''
        scope = {}
        scope.update(self.non_fluents_scope())
        scope.update(self.state_scope(state))
        scope.update(self.action_scope(action))
        scope.update(self.next_state_scope(next_state))
        return scope

    def state_invariant_scope(self, state: Sequence[tf.Tensor]):
        '''Returns the state invariant fluent scope for the current `state`.

        Args:
            state (Sequence[tf.Tensor]): The current state fluents.

        Returns:
            A mapping from fluent names to :obj:`rddl2tf.fluent.TensorFluent`.
        '''
        scope = {}
        scope.update(self.non_fluents_scope())
        scope.update(self.state_scope(state))
        return scope

    def action_precondition_scope(self,
            state: Sequence[tf.Tensor],
            action: Optional[Sequence[tf.Tensor]] = None) -> Dict[str, TensorFluent]:
        '''Returns the action precondition fluent scope
        for the current `state` and `action` fluents.

        Args:
            state (Sequence[tf.Tensor]): The current state fluents.
            action (Sequence[tf.Tensor]): The action fluents.

        Returns:
            A mapping from fluent names to :obj:`rddl2tf.fluent.TensorFluent`.
        '''
        scope = {}
        scope.update(self.non_fluents_scope())
        scope.update(self.state_scope(state))
        if action is not None:
            scope.update(self.action_scope(action))
        return scope

    def _initialize_pvariables(self,
            pvariables: Dict[str, PVariable],
            ordering: List[str],
            initializer: Optional[InitializerList] = None) -> List[Tuple[str, TensorFluent]]:
        '''Instantiates `pvariables` given an initialization list and
        returns a list of TensorFluents in the given `ordering`.

        Returns:
            List[Tuple[str, TensorFluent]]: A list of pairs of fluent name and fluent tensor.
        '''
        if initializer is not None:
            init = dict()
            for ((name, args), value) in initializer:
                arity = len(args) if args is not None else 0
                name = '{}/{}'.format(name, arity)
                init[name] = init.get(name, [])
                init[name].append((args, value))

        fluents = []

        for name in ordering:
            pvar = pvariables[name]
            shape = self.rddl._param_types_to_shape(pvar.param_types)
            dtype = utils.range_type_to_dtype(pvar.range)
            fluent = np.full(shape, pvar.default)

            if initializer is not None:
                for args, val in init.get(name, []):
                    if args is not None:
                        idx = []
                        for ptype, arg in zip(pvar.param_types, args):
                            idx.append(self.rddl.object_table[ptype]['idx'][arg])
                        idx = tuple(idx)
                        fluent[idx] = val
                    else:
                        fluent = val

            with self.graph.as_default():
                t = tf.constant(fluent, dtype=dtype, name=utils.identifier(name))
                scope = [None] * len(t.shape)
                fluent = TensorFluent(t, scope, batch=False)
                fluent_pair = (name, fluent)
                fluents.append(fluent_pair)

        return fluents

    def _initialize_non_fluents(self):
        '''Returns the non-fluents instantiated.'''
        non_fluents = self.rddl.domain.non_fluents
        initializer = self.rddl.non_fluents.init_non_fluent
        self.non_fluents = self._initialize_pvariables(
            non_fluents,
            self.rddl.domain.non_fluent_ordering,
            initializer)
        return self.non_fluents

    def _initialize_initial_state_fluents(self):
        '''Returns the initial state-fluents instantiated.'''
        state_fluents = self.rddl.domain.state_fluents
        initializer = self.rddl.instance.init_state
        self.initial_state_fluents = self._initialize_pvariables(
            state_fluents,
            self.rddl.domain.state_fluent_ordering,
            initializer)
        return self.initial_state_fluents

    def _initialize_default_action_fluents(self):
        '''Returns the default action-fluents instantiated.'''
        action_fluents = self.rddl.domain.action_fluents
        self.default_action_fluents = self._initialize_pvariables(
            action_fluents,
            self.rddl.domain.action_fluent_ordering)
        return self.default_action_fluents

    def _compile_batch_fluents(self,
            fluents: List[Tuple[str, TensorFluent]],
            batch_size: int) -> Sequence[tf.Tensor]:
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
                    t = tf.stack([fluent.tensor] * batch_size)
                batch_fluents.append(t)
        return tuple(batch_fluents)

    def _compile_expression(self,
                            expr: Expression,
                            scope: Dict[str, TensorFluent],
                            batch_size: Optional[int] = None,
                            noise: Optional[List[tf.Tensor]] = None) -> TensorFluent:
        '''Compile the expression `expr` into a TensorFluent
        in the given `scope` with optional batch size.

        Args:
            expr (:obj:`rddl2tf.expr.Expression`): A RDDL expression.
            scope (Dict[str, :obj:`rddl2tf.fluent.TensorFluent`]): A fluent scope.
            batch_size (Optional[size]): The batch size.

        Returns:
            :obj:`rddl2tf.fluent.TensorFluent`: The compiled TensorFluent.
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
            return compiler_fn(expr, scope, batch_size, noise)

    def _compile_constant_expression(self,
                                     expr: Expression,
                                     scope: Dict[str, TensorFluent],
                                     batch_size: Optional[int] = None,
                                     noise: Optional[List[tf.Tensor]] = None) -> TensorFluent:
        '''Compile a constant expression `expr` into a TensorFluent
        in the given `scope` with optional batch size.

        Args:
            expr (:obj:`rddl2tf.expr.Expression`): A RDDL constant expression.
            scope (Dict[str, :obj:`rddl2tf.fluent.TensorFluent`]): A fluent scope.
            batch_size (Optional[size]): The batch size.

        Returns:
            :obj:`rddl2tf.fluent.TensorFluent`: The compiled expression as a TensorFluent.
        '''
        etype = expr.etype
        args = expr.args
        dtype = utils.python_type_to_dtype(etype[1])
        fluent = TensorFluent.constant(args, dtype=dtype)
        return fluent

    def _compile_pvariable_expression(self,
                                      expr: Expression,
                                      scope: Dict[str, TensorFluent],
                                      batch_size: Optional[int] = None,
                                      noise: Optional[List[tf.Tensor]] = None) -> TensorFluent:
        '''Compile a pvariable expression `expr` into a TensorFluent
        in the given `scope` with optional batch size.

        Args:
            expr (:obj:`rddl2tf.expr.Expression`): A RDDL pvariable expression.
            scope (Dict[str, :obj:`rddl2tf.fluent.TensorFluent`]): A fluent scope.
            batch_size (Optional[size]): The batch size.

        Returns:
            :obj:`rddl2tf.fluent.TensorFluent`: The compiled expression as a TensorFluent.
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
            fluent = TensorFluent(fluent, scope, batch=self.batch_mode)
        else:
            raise ValueError('Variable in scope must be TensorFluent-like: {}'.format(fluent))
        return fluent

    def _compile_random_variable_expression(self,
                                            expr: Expression,
                                            scope: Dict[str, TensorFluent],
                                            batch_size: Optional[int] = None,
                                            noise: Optional[List[tf.Tensor]] = None) -> TensorFluent:
        '''Compile a random variable expression `expr` into a TensorFluent
        in the given `scope` with optional batch size.

        If `reparam` tensor is given, then it conditionally stops gradient
        backpropagation at the batch level where `reparam` is False.

        Args:
            expr (:obj:`rddl2tf.expr.Expression`): A RDDL random variable expression.
            scope (Dict[str, :obj:`rddl2tf.fluent.TensorFluent`]): A fluent scope.
            batch_size (Optional[size]): The batch size.

        Returns:
            :obj:`rddl2tf.fluent.TensorFluent`: The compiled expression as a TensorFluent.
        '''
        etype = expr.etype
        args = expr.args

        if etype[1] == 'KronDelta':
            sample = self._compile_expression(args[0], scope, batch_size, noise)
        elif etype[1] == 'Bernoulli':
            mean = self._compile_expression(args[0], scope, batch_size, noise)
            dist, sample = TensorFluent.Bernoulli(mean, batch_size)
        elif etype[1] == 'Uniform':
            low = self._compile_expression(args[0], scope, batch_size, noise)
            high = self._compile_expression(args[1], scope, batch_size, noise)
            dist, sample = TensorFluent.Uniform(low, high, batch_size)
        elif etype[1] == 'Normal':
            if noise is None:
                mean = self._compile_expression(args[0], scope, batch_size, noise)
                variance = self._compile_expression(args[1], scope, batch_size, noise)
                dist, sample = TensorFluent.Normal(mean, variance, batch_size)
            else:
                xi = noise.pop()
                xi = TensorFluent(xi, scope=[], batch=True)
                mean = self._compile_expression(args[0], scope, batch_size, noise)
                variance = self._compile_expression(args[1], scope, batch_size, noise)
                sample = mean + TensorFluent.sqrt(variance) * xi
        elif etype[1] == 'Laplace':
            mean = self._compile_expression(args[0], scope, batch_size, noise)
            variance = self._compile_expression(args[1], scope, batch_size, noise)
            dist, sample = TensorFluent.Laplace(mean, variance, batch_size)
        elif etype[1] == 'Gamma':
            shape = self._compile_expression(args[0], scope, batch_size, noise)
            scale = self._compile_expression(args[1], scope, batch_size, noise)
            dist, sample = TensorFluent.Gamma(shape, scale, batch_size)
        elif etype[1] == 'Exponential':
            mean = self._compile_expression(args[0], scope, batch_size, noise)
            dist, sample = TensorFluent.Exponential(mean, batch_size)
        else:
            raise ValueError('Invalid random variable expression:\n{}.'.format(expr))

        return sample

    def _compile_arithmetic_expression(self,
                                       expr: Expression,
                                       scope: Dict[str, TensorFluent],
                                       batch_size: Optional[int] = None,
                                       noise: Optional[List[tf.Tensor]] = None) -> TensorFluent:
        '''Compile an arithmetic expression `expr` into a TensorFluent
        in the given `scope` with optional batch size.

        Args:
            expr (:obj:`rddl2tf.expr.Expression`): A RDDL arithmetic expression.
            scope (Dict[str, :obj:`rddl2tf.fluent.TensorFluent`]): A fluent scope.
            batch_size (Optional[size]): The batch size.

        Returns:
            :obj:`rddl2tf.fluent.TensorFluent`: The compiled expression as a TensorFluent.
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
            x = self._compile_expression(args[0], scope, batch_size, noise)
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
            x = self._compile_expression(args[0], scope, batch_size, noise)
            y = self._compile_expression(args[1], scope, batch_size, noise)
            fluent = op(x, y)

        return fluent

    def _compile_boolean_expression(self,
                                    expr: Expression,
                                    scope: Dict[str, TensorFluent],
                                    batch_size: Optional[int] = None,
                                    noise: Optional[List[tf.Tensor]] = None) -> TensorFluent:
        '''Compile a boolean/logical expression `expr` into a TensorFluent
        in the given `scope` with optional batch size.

        Args:
            expr (:obj:`rddl2tf.expr.Expression`): A RDDL boolean expression.
            scope (Dict[str, :obj:`rddl2tf.fluent.TensorFluent`]): A fluent scope.
            batch_size (Optional[size]): The batch size.

        Returns:
            :obj:`rddl2tf.fluent.TensorFluent`: The compiled expression as a TensorFluent.
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
            x = self._compile_expression(args[0], scope, batch_size, noise)
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
            x = self._compile_expression(args[0], scope, batch_size, noise)
            y = self._compile_expression(args[1], scope, batch_size, noise)
            fluent = op(x, y)

        return fluent

    def _compile_relational_expression(self,
                                       expr: Expression,
                                       scope: Dict[str, TensorFluent],
                                       batch_size: Optional[int] = None,
                                       noise: Optional[List[tf.Tensor]] = None) -> TensorFluent:
        '''Compile a relational expression `expr` into a TensorFluent
        in the given `scope` with optional batch size.

        Args:
            expr (:obj:`rddl2tf.expr.Expression`): A RDDL relational expression.
            scope (Dict[str, :obj:`rddl2tf.fluent.TensorFluent`]): A fluent scope.
            batch_size (Optional[size]): The batch size.

        Returns:
            :obj:`rddl2tf.fluent.TensorFluent`: The compiled expression as a TensorFluent.
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
        x = self._compile_expression(args[0], scope, batch_size, noise)
        y = self._compile_expression(args[1], scope, batch_size, noise)
        fluent = op(x, y)

        return fluent

    def _compile_function_expression(self,
                                     expr: Expression,
                                     scope: Dict[str, TensorFluent],
                                     batch_size: Optional[int] = None,
                                     noise: Optional[List[tf.Tensor]] = None) -> TensorFluent:
        '''Compile a function expression `expr` into a TensorFluent
        in the given `scope` with optional batch size.

        Args:
            expr (:obj:`rddl2tf.expr.Expression`): A RDDL function expression.
            scope (Dict[str, :obj:`rddl2tf.fluent.TensorFluent`]): A fluent scope.
            batch_size (Optional[size]): The batch size.

        Returns:
            :obj:`rddl2tf.fluent.TensorFluent`: The compiled expression as a TensorFluent.
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
            x = self._compile_expression(args[0], scope, batch_size, noise)
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
            x = self._compile_expression(args[0], scope, batch_size, noise)
            y = self._compile_expression(args[1], scope, batch_size, noise)
            fluent = op(x, y)

        return fluent

    def _compile_control_flow_expression(self,
                                         expr: Expression,
                                         scope: Dict[str, TensorFluent],
                                         batch_size: Optional[int] = None,
                                         noise: Optional[List[tf.Tensor]] = None) -> TensorFluent:
        '''Compile a control flow expression `expr` into a TensorFluent
        in the given `scope` with optional batch size.

        Args:
            expr (:obj:`rddl2tf.expr.Expression`): A RDDL control flow expression.
            scope (Dict[str, :obj:`rddl2tf.fluent.TensorFluent`]): A fluent scope.
            batch_size (Optional[size]): The batch size.

        Returns:
            :obj:`rddl2tf.fluent.TensorFluent`: The compiled expression as a TensorFluent.
        '''
        etype = expr.etype
        args = expr.args
        if etype[1] == 'if':
            condition = self._compile_expression(args[0], scope, batch_size, noise)
            true_case = self._compile_expression(args[1], scope, batch_size, noise)
            false_case = self._compile_expression(args[2], scope, batch_size, noise)
            fluent = TensorFluent.if_then_else(condition, true_case, false_case)
        else:
            raise ValueError('Invalid control flow expression:\n{}'.format(expr))
        return fluent

    def _compile_aggregation_expression(self,
                                        expr: Expression,
                                        scope: Dict[str, TensorFluent],
                                        batch_size: Optional[int] = None,
                                        noise: Optional[List[tf.Tensor]] = None) -> TensorFluent:
        '''Compile an aggregation expression `expr` into a TensorFluent
        in the given `scope` with optional batch size.

        Args:
            expr (:obj:`rddl2tf.expr.Expression`): A RDDL aggregation expression.
            scope (Dict[str, :obj:`rddl2tf.fluent.TensorFluent`]): A fluent scope.
            batch_size (Optional[size]): The batch size.

        Returns:
            :obj:`rddl2tf.fluent.TensorFluent`: The compiled expression as a TensorFluent.
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
