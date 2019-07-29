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


import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional, Tuple, Union

from pyrddl.pvariable import PVariable
from pyrddl.rddl import RDDL

from rddl2tf.core.fluent import TensorFluent


Value = Union[bool, int, float]
ArgsList = Optional[List[str]]
InitializerPair = Tuple[Tuple[str, ArgsList], Value]
InitializerList = List[InitializerPair]


class CompilationInitializer(object):

    def __init__(self, rddl: RDDL) -> None:
        self.rddl = rddl

    def initialize_all_pvariables(self):
        non_fluents = self._initialize_non_fluents()
        initial_state_fluents = self._initialize_initial_state_fluents()
        default_action_fluents = self._initialize_default_action_fluents()
        return non_fluents, initial_state_fluents, default_action_fluents

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
            # dtype = utils.range_type_to_dtype(pvar.range)
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

            fluents.append((name, pvar.range, fluent))

        return fluents
