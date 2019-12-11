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
from typing import Dict, Sequence, Optional

from pyrddl.rddl import RDDL

from rddl2tf.core.fluent import TensorFluent


class CompilationScope(object):

    def __init__(self, rddl: RDDL) -> None:
        self.rddl = rddl

    def non_fluents(self, non_fluents: Sequence[tf.Tensor]) -> Dict[str, TensorFluent]:
        '''Returns a partial scope with non-fluents.

        Returns:
            A mapping from non-fluent names to :obj:`rddl2tf.core.fluent.TensorFluent`.
        '''
        return dict(zip(self.rddl.domain.non_fluent_ordering, non_fluents))

    def state(self, state_fluents: Sequence[tf.Tensor]) -> Dict[str, TensorFluent]:
        '''Returns a partial scope with current state-fluents.

        Args:
            state_fluents (Sequence[tf.Tensor]): The current state fluents.

        Returns:
            A mapping from state fluent names to :obj:`rddl2tf.core.fluent.TensorFluent`.
        '''
        return dict(zip(self.rddl.domain.state_fluent_ordering, state_fluents))

    def action(self, action_fluents: Sequence[tf.Tensor]) -> Dict[str, TensorFluent]:
        '''Returns a partial scope with current action-fluents.

        Args:
            action_fluents (Sequence[tf.Tensor]): The action fluents.

        Returns:
            A mapping from action fluent names to :obj:`rddl2tf.core.fluent.TensorFluent`.
        '''
        return dict(zip(self.rddl.domain.action_fluent_ordering, action_fluents))

    def next_state(self, next_state_fluents: Sequence[tf.Tensor]) -> Dict[str, TensorFluent]:
        '''Returns a partial scope with current next state-fluents.

        Args:
            next_state_fluents (Sequence[tf.Tensor]): The next state fluents.

        Returns:
            A mapping from next state fluent names to :obj:`rddl2tf.core.fluent.TensorFluent`.
        '''
        return dict(zip(self.rddl.domain.next_state_fluent_ordering, next_state_fluents))

    def transition(self,
            non_fluents: Sequence[tf.Tensor],
            state: Sequence[tf.Tensor],
            action: Sequence[tf.Tensor]) -> Dict[str, TensorFluent]:
        '''Returns the complete transition fluent scope
        for the current `state` and `action` fluents.

        Args:
            state (Sequence[tf.Tensor]): The current state fluents.
            action (Sequence[tf.Tensor]): The action fluents.

        Returns:
            A mapping from fluent names to :obj:`rddl2tf.core.fluent.TensorFluent`.
        '''
        scope = {}
        scope.update(self.non_fluents(non_fluents))
        scope.update(self.state(state))
        scope.update(self.action(action))
        return scope

    def reward(self,
            non_fluents: Sequence[tf.Tensor],
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
            A mapping from fluent names to :obj:`rddl2tf.core.fluent.TensorFluent`.
        '''
        scope = {}
        scope.update(self.non_fluents(non_fluents))
        scope.update(self.state(state))
        scope.update(self.action(action))
        scope.update(self.next_state(next_state))
        return scope

    def state_invariant(self,
            non_fluents: Sequence[tf.Tensor],
            state: Sequence[tf.Tensor]):
        '''Returns the state invariant fluent scope for the current `state`.

        Args:
            state (Sequence[tf.Tensor]): The current state fluents.

        Returns:
            A mapping from fluent names to :obj:`rddl2tf.core.fluent.TensorFluent`.
        '''
        scope = {}
        scope.update(self.non_fluents(non_fluents))
        scope.update(self.state(state))
        return scope

    def action_precondition(self,
            non_fluents: Sequence[tf.Tensor],
            state: Sequence[tf.Tensor],
            action: Optional[Sequence[tf.Tensor]] = None) -> Dict[str, TensorFluent]:
        '''Returns the action precondition fluent scope
        for the current `state` and `action` fluents.

        Args:
            state (Sequence[tf.Tensor]): The current state fluents.
            action (Sequence[tf.Tensor]): The action fluents.

        Returns:
            A mapping from fluent names to :obj:`rddl2tf.core.fluent.TensorFluent`.
        '''
        scope = {}
        scope.update(self.non_fluents(non_fluents))
        scope.update(self.state(state))
        if action is not None:
            scope.update(self.action(action))
        return scope
