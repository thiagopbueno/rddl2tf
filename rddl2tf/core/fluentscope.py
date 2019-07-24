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


from typing import List, Iterator, Tuple

ParamsList = List[str]
Permutation = List[int]
BroadcastTuple = Tuple[ParamsList, Permutation, Permutation]


class TensorFluentScope(object):
    '''TensorFluentScope handles the ordering of the fluent's parameters.

    Each RDDL fluent is defined by a parameterized variable.
    The TensorFluentScope class manages to match the order of the
    fluent's parameters with the tensor's shape when evaluating
    operations and functions in expressions.

    Args:
        scope: The list of the parameters of the fluent
    '''

    def __init__(self, scope: List[str]) -> None:
        self._scope = scope

    def as_list(self) -> List[str]:
        '''Returns a copy of the fluent's scope.'''
        return self._scope[:]

    def index(self, p: str) -> int:
        '''Returns the position of parameter `p` in the scope.'''
        return self._scope.index(p)

    def __len__(self) -> int:
        '''Returns the number of parameters in the scope.'''
        return len(self._scope)

    def __getitem__(self, i: int) -> str:
        '''Returns the parameter at position `i`.'''
        return self._scope[i]

    def __eq__(self, other: 'TensorFluentScope') -> bool:
        '''Returns True if scopes are equal. False, otherwise.'''
        return self._scope == other._scope

    def __ne__(self, other: 'TensorFluentScope') -> bool:
        '''Returns True if scopes are not equal. False, otherwise.'''
        return self._scope != other._scope

    def __str__(self) -> str:
        '''Returns the string representation of a TensorFluentScope object.'''
        return 'TensorFluentScope({})'.format(str(self._scope))

    @classmethod
    def broadcast(cls, s1: ParamsList, s2: ParamsList) -> BroadcastTuple:
        '''It broadcasts the smaller scope over the larger scope.

        It handles scope intersection as well as differences in scopes
        in order to output a resulting scope so that input scopes are
        contained within it (i.e., input scopes are subscopes of the
        output scope). Also, if necessary, it outputs permutations of
        the input scopes so that tensor broadcasting invariants are
        not violated.

        Note:
            For more information on broadcasting, please report to
            NumPy's official documentation available at the following URLs:
            1. https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
            2. https://docs.scipy.org/doc/numpy/reference/generated/numpy.broadcast.html

        Args:
            s1: A fluent's scope.
            s2: A fluent's scope.

        Returns:
            A tuple with the output scope and permutations of the input scopes.
        '''
        if len(s1) == 0:
            return s2, [], []
        if len(s2) == 0:
            return s1, [], []

        subscope = list(set(s1) & set(s2))
        if len(subscope) == len(s1):
            subscope = s1
        elif len(subscope) == len(s2):
            subscope = s2

        perm1 = []
        if s1[-len(subscope):] != subscope:
            i = 0
            for var in s1:
                if var not in subscope:
                    perm1.append(i)
                    i += 1
                else:
                    j = subscope.index(var)
                    perm1.append(len(s1) - len(subscope) + j)
        perm2 = []
        if s2[-len(subscope):] != subscope:
            i = 0
            for var in s2:
                if var not in subscope:
                    perm2.append(i)
                    i += 1
                else:
                    j = subscope.index(var)
                    perm2.append(len(s2) - len(subscope) + j)

        scope = [] # type: ParamsList
        if len(s1) >= len(s2):
            if perm1 == []:
                scope = s1
            else:
                for i in range(len(s1)):
                    scope.append(s1[perm1.index(i)])
        else:
            if perm2 == []:
                scope = s2
            else:
                for i in range(len(s2)):
                    scope.append(s2[perm2.index(i)])

        return (scope, perm1, perm2)
