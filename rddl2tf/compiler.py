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

import itertools
import numpy as np
import tensorflow as tf

from typing import Dict, List, Optional, Sequence, Tuple, Union

CPFPair = Tuple[str, TensorFluent]
FluentList = List[Tuple[str, TensorFluent]]
Bounds = Tuple[Optional[TensorFluent], Optional[TensorFluent]]
ObjectStruct = Dict[str, Union[int, Dict[str, int], List[str]]]
ObjectTable = Dict[str, ObjectStruct]
FluentParamsList = Sequence[Tuple[str, List[str]]]
Value = Union[bool, int, float]
ArgsList = Optional[List[str]]
InitializerPair = Tuple[Tuple[str, ArgsList], Value]
InitializerList = List[InitializerPair]


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

    def compile_initial_state(self, batch_size: int) -> Sequence[tf.Tensor]:
        '''Returns a tuple of tensors representing the initial state fluents.

        Args:
            batch_size (int): The batch size.

        Returns:
            A tuple of tensors.
        '''
        with self.graph.as_default():
            with tf.name_scope('initial_state'):
                return self._compile_batch_fluents(self.initial_state_fluents, batch_size)

    def compile_default_action(self, batch_size: int) -> Sequence[tf.Tensor]:
        '''Returns a tuple of tensors representing the default action fluents.

        Args:
            batch_size (int): The batch size.

        Returns:
            A tuple of tensors.
        '''
        with self.graph.as_default():
            with tf.name_scope('default_action'):
                return self._compile_batch_fluents(self.default_action_fluents, batch_size)

    def compile_cpfs(self,
            scope: Dict[str, TensorFluent],
            batch_size: Optional[int] = None) -> Tuple[List[CPFPair], List[CPFPair]]:
        '''Compiles the intermediate and next state fluent CPFs.

        Args:
            scope (Dict[str, :obj:`rddl2tf.fluent.TensorFluent`]): The fluent scope for CPF evaluation.
            batch_size (Optional[int]): The batch size.

        Returns:
            Tuple[List[CPFPair], List[CPFPair]]: A pair of lists of TensorFluent
            representing the intermediate and state CPFs.
        '''
        interm_fluents = self.compile_intermediate_cpfs(scope, batch_size)
        scope.update(dict(interm_fluents))
        next_state_fluents = self.compile_state_cpfs(scope, batch_size)
        return interm_fluents, next_state_fluents

    def compile_intermediate_cpfs(self,
            scope: Dict[str, TensorFluent],
            batch_size: Optional[int] = None) -> Tuple[List[CPFPair], List[CPFPair]]:
        '''Compiles the intermediate fluent CPFs.

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
                    name_scope = self._identifier(cpf.name)
                    with tf.name_scope(name_scope):
                        t = self._compile_expression(cpf.expr, scope, batch_size)
                    interm_fluents.append((cpf.name, t))
                    scope[cpf.name] = t
                return interm_fluents

    def compile_state_cpfs(self,
            scope: Dict[str, TensorFluent],
            batch_size: Optional[int] = None) -> Tuple[List[CPFPair], List[CPFPair]]:
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
                    name_scope = self._identifier(cpf.name)
                    with tf.name_scope(name_scope):
                        t = self._compile_expression(cpf.expr, scope, batch_size)
                    next_state_fluents.append((cpf.name, t))
                key = lambda f: self.next_state_fluent_ordering.index(f[0])
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
                t = self._compile_expression(reward_expr, scope)
                tensor = tf.expand_dims(t.tensor, -1)
                return TensorFluent(tensor, t.scope[:], t.batch)

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
                for p in self.action_preconditions:
                    t = self._compile_expression(p, scope)
                    tensor = t.tensor
                    if t.shape.fluent_shape == ():
                        tensor = tf.expand_dims(tensor, -1)
                    fluent = TensorFluent(tensor, t.scope[:], t.batch)
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
                for p in self.state_invariants:
                    t = self._compile_expression(p, scope)
                    tensor = t.tensor
                    if t.shape.fluent_shape == ():
                        tensor = tf.expand_dims(tensor, -1)
                    fluent = TensorFluent(tensor, t.scope[:], t.batch)
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
        preconds = self.compile_action_preconditions(state, action)
        all_preconds = tf.concat([p.tensor for p in preconds], axis=1)
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

        lower_bounds = self.action_lower_bound_constraints
        upper_bounds = self.action_upper_bound_constraints

        with self.graph.as_default():
            with tf.name_scope('action_bound_constraints'):

                bounds = {}
                for name in self.action_fluent_ordering:

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
        '''Returns a partial scope with non-fluents.'''
        return dict(self.non_fluents)

    def state_scope(self, state_fluents: Sequence[tf.Tensor]) -> Dict[str, TensorFluent]:
        '''Returns a partial scope with current state-fluents.

        Args:
            state_fluents (Sequence[tf.Tensor]): The current state fluents.

        Returns:
            A mapping from state fluent names to :obj:`rddl2tf.fluent.TensorFluent`.
        '''
        return dict(zip(self.state_fluent_ordering, state_fluents))

    def action_scope(self, action_fluents: Sequence[tf.Tensor]) -> Dict[str, TensorFluent]:
        '''Returns a partial scope with current action-fluents.

        Args:
            action_fluents (Sequence[tf.Tensor]): The action fluents.

        Returns:
            A mapping from action fluent names to :obj:`rddl2tf.fluent.TensorFluent`.
        '''
        return dict(zip(self.action_fluent_ordering, action_fluents))

    def next_state_scope(self, next_state_fluents: Sequence[tf.Tensor]) -> Dict[str, TensorFluent]:
        '''Returns a partial scope with current next state-fluents.

        Args:
            next_state_fluents (Sequence[tf.Tensor]): The next state fluents.

        Returns:
            A mapping from next state fluent names to :obj:`rddl2tf.fluent.TensorFluent`.
        '''
        return dict(zip(self.next_state_fluent_ordering, next_state_fluents))

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

    @property
    def object_table(self) -> ObjectTable:
        '''The object table for each RDDL type.

        Returns:
            A mapping from type name to the type size,
            objects index and objects list.
        '''
        if self.__dict__.get('_object_table') is None:
            self._build_object_table()
        return self._object_table

    @property
    def non_fluents(self) -> FluentList:
        '''The list of non-fluents instantiated for a given RDDL non-fluents.

        Returns:
            List[Tuple[str, TensorFluent]]: the list of non-fluents.
        '''
        if self.__dict__.get('_non_fluents') is None:
            self._instantiate_non_fluents()
        return self._non_fluents

    @property
    def initial_state_fluents(self) -> FluentList:
        '''The list of initial state-fluents instantiated for a given RDDL instance.

        Returns:
            List[Tuple[str, TensorFluent]]: the list of state fluents.
        '''
        if self.__dict__.get('_initial_state_fluents') is None:
            self._instantiate_initial_state_fluents()
        return self._initial_state_fluents

    @property
    def default_action_fluents(self) -> FluentList:
        '''The list of non-fluents instantiated for a given RDDL domain.

        Returns:
            List[Tuple[str, TensorFluent]]: the list of action fluents.
        '''
        if self.__dict__.get('_default_action_fluents') is None:
            self._instantiate_default_action_fluents()
        return self._default_action_fluents

    @property
    def action_preconditions(self) -> Dict[str, List[Expression]]:
        '''The action precondition expressions.

        Returns:
            Dict[str, List[Expression]]: A mapping from fluent name to a list of Expressions.'''
        return self.rddl.domain.preconds

    @property
    def local_action_preconditions(self) -> Dict[str, List[Expression]]:
        '''The local action precondition expressions.

        Returns:
            Dict[str, List[Expression]]: A mapping from fluent name to a list of Expressions.'''
        if self.__dict__.get('_local_action_preconditions') is None:
            self._build_preconditions_table()
        return self._local_action_preconditions

    @property
    def global_action_preconditions(self) -> Dict[str, List[Expression]]:
        '''The global action precondition expressions.

        Returns:
            Dict[str, List[Expression]]: A mapping from fluent name to a list of Expressions.'''
        if self.__dict__.get('_global_action_preconditions') is None:
            self._build_preconditions_table()
        return self._global_action_preconditions

    @property
    def state_invariants(self) -> Dict[str, List[Expression]]:
        '''The state invariant expressions.

        Returns:
            Dict[str, List[Expression]]: A mapping from fluent name to a list of Expressions.'''
        return self.rddl.domain.invariants

    @property
    def action_lower_bound_constraints(self) -> Dict[str, Expression]:
        '''The action lower bound constraint expressions.

        Returns:
            Dict[str, Expression]: A mapping from fluent name to an Expression.'''
        if self.__dict__.get('_action_lower_bound_constraints') is None:
            self._build_action_bound_constraints_table()
        return self._action_lower_bound_constraints

    @property
    def action_upper_bound_constraints(self) -> Dict[str, Expression]:
        '''The action upper bound constraint expressions.

        Returns:
            Dict[str, Expression]: A mapping from fluent name to an Expression.'''
        if self.__dict__.get('_action_upper_bound_constraints') is None:
            self._build_action_bound_constraints_table()
        return self._action_upper_bound_constraints

    @property
    def non_fluent_ordering(self) -> List[str]:
        '''The list of non-fluent names in canonical order.

        Returns:
            List[str]: A list of fluent names.
        '''
        return [name for name in sorted(self.rddl.domain.non_fluents)]

    @property
    def state_fluent_ordering(self) -> List[str]:
        '''The list of state-fluent names in canonical order.

        Returns:
            List[str]: A list of fluent names.
        '''
        return [name for name in sorted(self.rddl.domain.state_fluents)]

    @property
    def action_fluent_ordering(self) -> List[str]:
        '''The list of action-fluent names in canonical order.

        Returns:
            List[str]: A list of fluent names.
        '''
        return [name for name in sorted(self.rddl.domain.action_fluents)]

    @property
    def next_state_fluent_ordering(self) -> List[str]:
        '''The list of next state-fluent names in canonical order.

        Returns:
            List[str]: A list of fluent names.
        '''
        key = lambda x: x.name
        return [cpf.name for cpf in sorted(self.rddl.domain.state_cpfs, key=key)]

    @property
    def interm_fluent_ordering(self) -> List[str]:
        '''The list of intermediate-fluent names in canonical order.

        Returns:
            List[str]: A list of fluent names.
        '''
        interm_fluents = self.rddl.domain.intermediate_fluents.values()
        key = lambda pvar: (pvar.level, pvar.name)
        return [str(pvar) for pvar in sorted(interm_fluents, key=key)]

    @property
    def state_size(self) -> Sequence[Sequence[int]]:
        '''The size of each state fluent in canonical order.

        Returns:
            Sequence[Sequence[int]]: A tuple of tuple of integers
            representing the shape and size of each fluent.
        '''
        return self._fluent_size(self.initial_state_fluents, self.state_fluent_ordering)

    @property
    def action_size(self) -> Sequence[Sequence[int]]:
        '''The size of each action fluent in canonical order.

        Returns:
            Sequence[Sequence[int]]: A tuple of tuple of integers
            representing the shape and size of each fluent.
        '''
        return self._fluent_size(self.default_action_fluents, self.action_fluent_ordering)

    @property
    def interm_size(self)-> Sequence[Sequence[int]]:
        '''The size of each intermediate fluent in canonical order.

        Returns:
            Sequence[Sequence[int]]: A tuple of tuple of integers
            representing the shape and size of each fluent.
        '''
        interm_fluents = self.rddl.domain.intermediate_fluents
        shapes = []
        for name in self.interm_fluent_ordering:
            fluent = interm_fluents[name]
            shape = self._param_types_to_shape(fluent.param_types)
            shapes.append(shape)
        return tuple(shapes)

    @property
    def state_dtype(self) -> Sequence[tf.DType]:
        '''The data type of each state fluent in canonical order.

        Returns:
            Sequence[tf.DType]: A tuple of dtypes representing
            the range of each fluent.
        '''
        return self._fluent_dtype(self.initial_state_fluents, self.state_fluent_ordering)

    @property
    def action_dtype(self) -> Sequence[tf.DType]:
        '''The data type of each action fluent in canonical order.

        Returns:
            Sequence[tf.DType]: A tuple of dtypes representing
            the range of each fluent.
        '''
        return self._fluent_dtype(self.default_action_fluents, self.action_fluent_ordering)

    @property
    def interm_dtype(self) -> Sequence[tf.DType]:
        '''The data type of each intermediate fluent in canonical order.

        Returns:
            Sequence[tf.DType]: A tuple of dtypes representing
            the range of each fluent.
        '''
        interm_fluents = self.rddl.domain.intermediate_fluents
        dtypes = []
        for name in self.interm_fluent_ordering:
            fluent = interm_fluents[name]
            dtype = self._range_type_to_dtype(fluent.range)
            dtypes.append(dtype)
        return tuple(dtypes)

    @property
    def non_fluent_variables(self) -> FluentParamsList:
        '''Returns the instantiated non-fluents in canonical order.

        Returns:
            Sequence[Tuple[str, List[str]]]: A tuple of pairs of fluent name
            and a list of instantiated fluents represented as strings.
        '''
        fluents = self.rddl.domain.non_fluents
        ordering = self.non_fluent_ordering
        return self._fluent_params(fluents, ordering)

    @property
    def state_fluent_variables(self) -> FluentParamsList:
        '''Returns the instantiated state fluents in canonical order.

        Returns:
            Sequence[Tuple[str, List[str]]]: A tuple of pairs of fluent name
            and a list of instantiated fluents represented as strings.
        '''
        fluents = self.rddl.domain.state_fluents
        ordering = self.state_fluent_ordering
        return self._fluent_params(fluents, ordering)

    @property
    def interm_fluent_variables(self) -> FluentParamsList:
        '''Returns the instantiated intermediate fluents in canonical order.

        Returns:
            Sequence[Tuple[str, List[str]]]: A tuple of pairs of fluent name
            and a list of instantiated fluents represented as strings.
        '''
        fluents = self.rddl.domain.intermediate_fluents
        ordering = self.interm_fluent_ordering
        return self._fluent_params(fluents, ordering)

    @property
    def action_fluent_variables(self) -> FluentParamsList:
        '''Returns the instantiated action fluents in canonical order.

        Returns:
            Sequence[Tuple[str, List[str]]]: A tuple of pairs of fluent name
            and a list of instantiated fluents represented as strings.
        '''
        fluents = self.rddl.domain.action_fluents
        ordering = self.action_fluent_ordering
        return self._fluent_params(fluents, ordering)

    def _fluent_params(self, fluents, ordering) -> FluentParamsList:
        '''Returns the instantiated `fluents` for the given `ordering`.

        For each fluent in `fluents`, it instantiates each parameter
        type w.r.t. the contents of the object table.

        Returns:
            Sequence[Tuple[str, List[str]]]: A tuple of pairs of fluent name
            and a list of instantiated fluents represented as strings.
        '''
        variables = []
        for fluent_id in ordering:
            fluent = fluents[fluent_id]
            param_types = fluent.param_types
            objects = ()
            names = []
            if param_types is None:
                names = [fluent.name]
            else:
                objects = tuple(self.object_table[ptype]['objects'] for ptype in param_types)
                for values in itertools.product(*objects):
                    values = ','.join(values)
                    var_name = '{}({})'.format(fluent.name, values)
                    names.append(var_name)
            variables.append((fluent_id, names))
        return tuple(variables)

    @classmethod
    def _fluent_dtype(cls, fluents, ordering) -> Sequence[tf.DType]:
        '''Returns the data types of `fluents` following the given `ordering`.

        Returns:
            Sequence[tf.DType]: A tuple of dtypes representing
            the range of each fluent.
        '''
        dtype = []
        fluents = dict(fluents)
        for name in ordering:
            fluent_dtype = fluents[name].dtype
            dtype.append(fluent_dtype)
        return tuple(dtype)

    @classmethod
    def _fluent_size(cls, fluents, ordering) -> Sequence[Sequence[int]]:
        '''Returns the sizes of `fluents` following the given `ordering`.

        Returns:
            Sequence[Sequence[int]]: A tuple of tuple of integers
            representing the shape and size of each fluent.
        '''
        size = []
        fluents = dict(fluents)
        for name in ordering:
            fluent_shape = fluents[name].shape.fluent_shape
            if fluent_shape == ():
                fluent_shape = (1,)
            size.append(fluent_shape)
        return tuple(size)

    def _build_object_table(self):
        '''Builds the object table for each RDDL type.'''
        types = self.rddl.domain.types
        objects = dict(self.rddl.non_fluents.objects)
        self._object_table = dict()
        for name, value in self.rddl.domain.types:
            if value == 'object':
                objs = objects[name]
                idx = { obj: i for i, obj in enumerate(objs) }
                self._object_table[name] = {
                    'size': len(objs),
                    'idx': idx,
                    'objects': objs
                }

    def _build_preconditions_table(self):
        '''Builds the local action precondition expressions.'''
        self._local_action_preconditions = dict()
        self._global_action_preconditions = []
        action_fluents = self.rddl.domain.action_fluents
        for precond in self.rddl.domain.preconds:
            scope = precond.scope
            action_scope = [action for action in scope if action in action_fluents]
            if len(action_scope) == 1:
                name = action_scope[0]
                self._local_action_preconditions[name] = self._local_action_preconditions.get(name, [])
                self._local_action_preconditions[name].append(precond)
            else:
                self._global_action_preconditions.append(precond)

    def _build_action_bound_constraints_table(self):
        '''Builds the lower and upper action bound constraint expressions.'''
        self._action_lower_bound_constraints = {}
        self._action_upper_bound_constraints = {}

        for name, preconds in self.local_action_preconditions.items():

            for precond in preconds:
                expr_type = precond.etype
                expr_args = precond.args

                if expr_type == ('aggregation', 'forall'):

                    inner_expr = expr_args[1]
                    if inner_expr.etype[0] == 'relational':

                        # lower bound
                        bound = self._extract_lower_bound(name, inner_expr)
                        if bound is not None:
                            self._action_lower_bound_constraints[name] = bound
                            next

                        # upper bound
                        bound = self._extract_upper_bound(name, inner_expr)
                        if bound is not None:
                            self._action_upper_bound_constraints[name] = bound

    def _extract_lower_bound(self, name: str, expr: Expression) -> Optional[Expression]:
        '''Returns the lower bound expression of the action with given `name`.'''
        etype = expr.etype
        args = expr.args
        if etype[1] in ['<=', '<']:
            if args[1].is_pvariable_expression() and args[1].name == name:
                return args[0]
        elif etype[1] in ['>=', '>']:
            if args[0].is_pvariable_expression() and args[0].name == name:
                return args[1]
        return None

    def _extract_upper_bound(self, name: str, expr: Expression) -> Optional[Expression]:
        '''Returns the upper bound expression of the action with given `name`.'''
        etype = expr.etype
        args = expr.args
        if etype[1] in ['<=', '<']:
            if args[0].is_pvariable_expression() and args[0].name == name:
                return args[1]
        elif etype[1] in ['>=', '>']:
            if args[1].is_pvariable_expression() and args[1].name == name:
                return args[0]
        return None

    def _instantiate_pvariables(self,
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
            shape = self._param_types_to_shape(pvar.param_types)
            dtype = self._range_type_to_dtype(pvar.range)
            fluent = np.full(shape, pvar.default)

            if initializer is not None:
                for args, val in init.get(name, []):
                    if args is not None:
                        idx = []
                        for ptype, arg in zip(pvar.param_types, args):
                            idx.append(self.object_table[ptype]['idx'][arg])
                        idx = tuple(idx)
                        fluent[idx] = val
                    else:
                        fluent = val

            with self.graph.as_default():
                t = tf.constant(fluent, dtype=dtype, name=self._identifier(name))
                scope = [None] * len(t.shape)
                fluent = TensorFluent(t, scope, batch=False)
                fluent_pair = (name, fluent)
                fluents.append(fluent_pair)

        return fluents

    def _instantiate_non_fluents(self):
        '''Returns the non-fluents instantiated.'''
        non_fluents = self.rddl.domain.non_fluents
        initializer = self.rddl.non_fluents.init_non_fluent
        with self.graph.as_default():
            with tf.name_scope('non_fluents'):
                self._non_fluents = self._instantiate_pvariables(
                    non_fluents, self.non_fluent_ordering, initializer)
                return self._non_fluents

    def _instantiate_initial_state_fluents(self):
        '''Returns the initial state-fluents instantiated.'''
        state_fluents = self.rddl.domain.state_fluents
        initializer = self.rddl.instance.init_state
        self._initial_state_fluents = self._instantiate_pvariables(state_fluents, self.state_fluent_ordering, initializer)
        return self._initial_state_fluents

    def _instantiate_default_action_fluents(self):
        '''Returns the default action-fluents instantiated.'''
        action_fluents = self.rddl.domain.action_fluents
        self._default_action_fluents = self._instantiate_pvariables(action_fluents, self.action_fluent_ordering)
        return self._default_action_fluents

    def _compile_batch_fluents(self,
            fluents: List[Tuple[str, TensorFluent]],
            batch_size: int) -> Sequence[tf.Tensor]:
        '''Compiles `fluents` into tensors with given `batch_size`.

        Returns:
            Sequence[tf.Tensor]: A tuple of tensors with first dimensio
            corresping to the batch size.
        '''
        batch_fluents = []
        with self.graph.as_default():
            for name, fluent in fluents:
                name_scope = self._identifier(name)
                with tf.name_scope(name_scope):
                    t = tf.stack([fluent.tensor] * batch_size)
                    if t.shape.ndims == 1:
                        t = tf.expand_dims(t, -1)
                batch_fluents.append(t)
        return tuple(batch_fluents)

    def _compile_expression(self,
            expr: Expression,
            scope: Dict[str, TensorFluent],
            batch_size: Optional[int] = None) -> TensorFluent:
        '''Compile the expression `expr` into a TensorFluent
        in the given `scope` with optional batch size.

        Args:
            expr (:obj:`rddl2tf.expr.Expression`): A RDDL expression.
            scope (Dict[str, :obj:`rddl2tf.fluent.TensorFluent`]): A fluent scope.
            batch_size (Optional[size]): The batch size.

        Returns:
            :obj:`rddl2tf.fluent.TensorFluent`: A TensorFluent.
        '''
        etype = expr.etype
        args = expr.args

        with self.graph.as_default():

            if etype[0] == 'constant':
                dtype = self._python_type_to_dtype(etype[1])
                return TensorFluent.constant(args, dtype=dtype)
            elif etype[0] == 'pvar':
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
            elif etype[0] == 'randomvar':
                if etype[1] == 'KronDelta':
                    return self._compile_expression(args[0], scope)
                elif etype[1] == 'Bernoulli':
                    mean = self._compile_expression(args[0], scope)
                    return TensorFluent.Bernoulli(mean, batch_size)
                elif etype[1] == 'Normal':
                    mean = self._compile_expression(args[0], scope)
                    variance = self._compile_expression(args[1], scope)
                    return TensorFluent.Normal(mean, variance, batch_size)
                elif etype[1] == 'Uniform':
                    low = self._compile_expression(args[0], scope)
                    high = self._compile_expression(args[1], scope)
                    return TensorFluent.Uniform(low, high, batch_size)
                elif etype[1] == 'Exponential':
                    mean = self._compile_expression(args[0], scope)
                    return TensorFluent.Exponential(mean, batch_size)
                elif etype[1] == 'Gamma':
                    shape = self._compile_expression(args[0], scope)
                    scale = self._compile_expression(args[1], scope)
                    return TensorFluent.Gamma(shape, scale, batch_size)
            elif etype[0] == 'arithmetic':
                if etype[1] == '+':
                    if len(args) == 1:
                        op1 = self._compile_expression(args[0], scope)
                        return op1
                    else:
                        op1 = self._compile_expression(args[0], scope)
                        op2 = self._compile_expression(args[1], scope)
                        return op1 + op2
                elif etype[1] == '-':
                    if len(args) == 1:
                        op1 = self._compile_expression(args[0], scope)
                        return -op1
                    else:
                        op1 = self._compile_expression(args[0], scope)
                        op2 = self._compile_expression(args[1], scope)
                        return op1 - op2
                elif etype[1] == '*':
                    op1 = self._compile_expression(args[0], scope)
                    op2 = self._compile_expression(args[1], scope)
                    return op1 * op2
                elif etype[1] == '/':
                    op1 = self._compile_expression(args[0], scope)
                    op2 = self._compile_expression(args[1], scope)
                    return op1 / op2
            elif etype[0] == 'boolean':
                if etype[1] in ['^', '&']:
                    op1 = self._compile_expression(args[0], scope)
                    op2 = self._compile_expression(args[1], scope)
                    return op1 & op2
                elif etype[1] == '|':
                    op1 = self._compile_expression(args[0], scope)
                    op2 = self._compile_expression(args[1], scope)
                    return op1 | op2
                elif etype[1] == '=>':
                    op1 = self._compile_expression(args[0], scope)
                    op2 = self._compile_expression(args[1], scope)
                    return ~op1 | op2
                elif etype[1] == '<=>':
                    op1 = self._compile_expression(args[0], scope)
                    op2 = self._compile_expression(args[1], scope)
                    return (op1 & op2) | (~op1 & ~op2)
                elif etype[1] == '~':
                    op = self._compile_expression(args[0], scope)
                    return ~op
            elif etype[0] == 'relational':
                if etype[1] == '<=':
                    op1 = self._compile_expression(args[0], scope)
                    op2 = self._compile_expression(args[1], scope)
                    return op1 <= op2
                elif etype[1] == '<':
                    op1 = self._compile_expression(args[0], scope)
                    op2 = self._compile_expression(args[1], scope)
                    return op1 < op2
                elif etype[1] == '>=':
                    op1 = self._compile_expression(args[0], scope)
                    op2 = self._compile_expression(args[1], scope)
                    return op1 >= op2
                elif etype[1] == '>':
                    op1 = self._compile_expression(args[0], scope)
                    op2 = self._compile_expression(args[1], scope)
                    return op1 > op2
                elif etype[1] == '==':
                    op1 = self._compile_expression(args[0], scope)
                    op2 = self._compile_expression(args[1], scope)
                    return op1 == op2
                elif etype[1] == '~=':
                    op1 = self._compile_expression(args[0], scope)
                    op2 = self._compile_expression(args[1], scope)
                    return op1 != op2
            elif etype[0] == 'func':
                if etype[1] == 'abs':
                    x = self._compile_expression(args[0], scope)
                    return TensorFluent.abs(x)
                elif etype[1] == 'exp':
                    x = self._compile_expression(args[0], scope)
                    return TensorFluent.exp(x)
                elif etype[1] == 'log':
                    x = self._compile_expression(args[0], scope)
                    return TensorFluent.log(x)
                elif etype[1] == 'sqrt':
                    x = self._compile_expression(args[0], scope)
                    return TensorFluent.sqrt(x)
                elif etype[1] == 'cos':
                    x = self._compile_expression(args[0], scope)
                    return TensorFluent.cos(x)
                elif etype[1] == 'sin':
                    x = self._compile_expression(args[0], scope)
                    return TensorFluent.sin(x)
                elif etype[1] == 'round':
                    x = self._compile_expression(args[0], scope)
                    return TensorFluent.round(x)
                elif etype[1] == 'ceil':
                    x = self._compile_expression(args[0], scope)
                    return TensorFluent.ceil(x)
                elif etype[1] == 'floor':
                    x = self._compile_expression(args[0], scope)
                    return TensorFluent.floor(x)
                elif etype[1] == 'pow':
                    x = self._compile_expression(args[0], scope)
                    y = self._compile_expression(args[1], scope)
                    return TensorFluent.pow(x, y)
                elif etype[1] == 'max':
                    x = self._compile_expression(args[0], scope)
                    y = self._compile_expression(args[1], scope)
                    return TensorFluent.max(x, y)
                elif etype[1] == 'min':
                    x = self._compile_expression(args[0], scope)
                    y = self._compile_expression(args[1], scope)
                    return TensorFluent.min(x, y)
            elif etype[0] == 'control':
                if etype[1] == 'if':
                    condition = self._compile_expression(args[0], scope)
                    true_case = self._compile_expression(args[1], scope)
                    false_case = self._compile_expression(args[2], scope)
                    return TensorFluent.if_then_else(condition, true_case, false_case)
            elif etype[0] == 'aggregation':
                if etype[1] not in ['sum', 'prod', 'avg', 'maximum', 'minimum', 'exists', 'forall']:
                    raise ValueError('Unkown aggregation function {}.'.format(etype[1]))
                typed_var_list = args[:-1]
                vars_list = [var for _, (var, _) in typed_var_list]
                expr = args[-1]
                x = self._compile_expression(expr, scope)
                if etype[1] == 'sum':
                    return x.sum(vars_list=vars_list)
                elif etype[1] == 'prod':
                    return x.prod(vars_list=vars_list)
                elif etype[1] == 'avg':
                    return x.avg(vars_list=vars_list)
                elif etype[1] == 'maximum':
                    return x.maximum(vars_list=vars_list)
                elif etype[1] == 'minimum':
                    return x.minimum(vars_list=vars_list)
                elif etype[1] == 'exists':
                    return x.exists(vars_list=vars_list)
                elif etype[1] == 'forall':
                    return x.forall(vars_list=vars_list)

    @classmethod
    def _range_type_to_dtype(cls, range_type: str) -> Optional[tf.DType]:
        '''Maps RDDL range types to TensorFlow dtypes.'''
        range2dtype = {
            'real': tf.float32,
            'int': tf.int32,
            'bool': tf.bool
        }
        return range2dtype[range_type]

    @classmethod
    def _python_type_to_dtype(cls, python_type: type) -> Optional[tf.DType]:
        '''Maps python types to TensorFlow dtypes.'''
        dtype = None
        if python_type == float:
            dtype = tf.float32
        elif python_type == int:
            dtype = tf.int32
        elif python_type == bool:
            dtype = tf.bool
        return dtype

    @classmethod
    def _identifier(cls, name):
        name = name.replace("'", '')
        name = name.replace('/', '-')
        return name

    def _param_types_to_shape(self, param_types: Optional[str]) -> Sequence[int]:
        '''Returns the fluent shape given its `param_types`.'''
        param_types = [] if param_types is None else param_types
        shape = tuple(self.object_table[ptype]['size'] for ptype in param_types)
        return shape
