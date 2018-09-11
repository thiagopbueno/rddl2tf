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

import unittest


class TestTensorScope(unittest.TestCase):

    def test_broadcast(self):
        tests = [
            (([], []), ([], [], [])),
            ((['?r'], []), (['?r'], [], [])),
            (([], ['?r']), (['?r'], [], [])),
            ((['?r'], ['?r']), (['?r'], [], [])),
            ((['?s', '?r'], []), (['?s', '?r'], [], [])),
            (([], ['?s', '?r']), (['?s', '?r'], [], [])),
            ((['?s', '?r'], ['?r']), (['?s', '?r'], [], [])),
            ((['?r'], ['?s', '?r']), (['?s', '?r'], [], [])),
            ((['?r', '?s'], ['?r']), (['?s', '?r'], [1, 0], [])),
            ((['?r'], ['?r', '?s']), (['?s', '?r'], [], [1, 0])),
            ((['?r', '?s', '?t'], []), (['?r', '?s', '?t'], [], [])),
            (([], ['?r', '?s', '?t']), (['?r', '?s', '?t'], [], [])),
            ((['?r', '?s', '?t'], ['?r']), (['?s', '?t', '?r'], [2, 0, 1], [])),
            ((['?r'], ['?r', '?s', '?t']), (['?s', '?t', '?r'], [], [2, 0, 1])),
            ((['?r', '?s', '?t'], ['?s']), (['?r', '?t', '?s'], [0, 2, 1], [])),
            ((['?s'], ['?r', '?s', '?t']), (['?r', '?t', '?s'], [], [0, 2, 1])),
            ((['?r', '?s', '?t'], ['?t']), (['?r', '?s', '?t'], [], [])),
            ((['?t'], ['?r', '?s', '?t']), (['?r', '?s', '?t'], [], [])),
            ((['?r', '?s', '?t'], ['?s', '?t']), (['?r', '?s', '?t'], [], [])),
            ((['?s', '?t'], ['?r', '?s', '?t']), (['?r', '?s', '?t'], [], [])),
            ((['?r', '?s', '?t'], ['?t', '?s']), (['?r', '?t', '?s'], [0, 2, 1], [])),
            ((['?t', '?s'], ['?r', '?s', '?t']), (['?r', '?t', '?s'], [], [0, 2, 1])),
            ((['?r', '?s', '?t'], ['?t', '?r']), (['?s', '?t', '?r'], [2, 0, 1], [])),
            ((['?t', '?r'], ['?r', '?s', '?t']), (['?s', '?t', '?r'], [], [2, 0, 1])),
            ((['?r', '?s', '?t'], ['?r', '?t']), (['?s', '?r', '?t'], [1, 0, 2], [])),
            ((['?r', '?t'], ['?r', '?s', '?t']), (['?s', '?r', '?t'], [], [1, 0, 2])),
            ((['?r', '?s', '?t'], ['?r', '?s']), (['?t', '?r', '?s'], [1, 2, 0], [])),
            ((['?r', '?s'], ['?r', '?s', '?t']), (['?t', '?r', '?s'], [], [1, 2, 0])),
            ((['?r', '?s', '?t'], ['?s', '?r']), (['?t', '?s', '?r'], [2, 1, 0], [])),
            ((['?s', '?r'], ['?r', '?s', '?t']), (['?t', '?s', '?r'], [], [2, 1, 0])),
        ]

        for (s1, s2), (s, p1, p2) in tests:
            scope, perm1, perm2 = TensorFluentScope.broadcast(s1, s2)
            self.assertListEqual(perm1, p1)
            self.assertListEqual(perm2, p2)
            self.assertListEqual(scope, s)
