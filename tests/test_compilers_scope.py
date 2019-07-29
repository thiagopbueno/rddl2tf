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


import unittest

import rddlgym

from rddl2tf.compilers.modes.default import DefaultCompiler
from rddl2tf.compilers.scope import CompilationScope


class TestCompilationScope(unittest.TestCase):

    def setUp(self):
        self.batch_size = 64

        self.rddl1 = rddlgym.make('Reservoir-8', mode=rddlgym.AST)
        self.compiler1 = DefaultCompiler(self.rddl1, self.batch_size)
        self.compiler1.init()
        self.scope1 = CompilationScope(self.rddl1)

        self.rddl2 = rddlgym.make('Mars_Rover', mode=rddlgym.AST)
        self.compiler2 = DefaultCompiler(self.rddl2, self.batch_size)
        self.compiler2.init()
        self.scope2 = CompilationScope(self.rddl2)

        self.rddl3 = rddlgym.make('HVAC-v1', mode=rddlgym.AST)
        self.compiler3 = DefaultCompiler(self.rddl3, self.batch_size)
        self.compiler3.init()
        self.scope3 = CompilationScope(self.rddl3)

        self.rddl4 = rddlgym.make('CrossingTraffic-10', mode=rddlgym.AST)
        self.compiler4 = DefaultCompiler(self.rddl4, self.batch_size)
        self.compiler4.init()
        self.scope4 = CompilationScope(self.rddl4)

        self.rddl5 = rddlgym.make('GameOfLife-10', mode=rddlgym.AST)
        self.compiler5 = DefaultCompiler(self.rddl5, self.batch_size)
        self.compiler5.init()
        self.scope5 = CompilationScope(self.rddl5)

        self.rddl6 = rddlgym.make('CarParking-v1', mode=rddlgym.AST)
        self.compiler6 = DefaultCompiler(self.rddl6, self.batch_size)
        self.compiler6.init()
        self.scope6 = CompilationScope(self.rddl6)

        self.rddl7 = rddlgym.make('Navigation-v3', mode=rddlgym.AST)
        self.compiler7 = DefaultCompiler(self.rddl7, self.batch_size)
        self.compiler7.init()
        self.scope7 = CompilationScope(self.rddl7)

    def test_state_scope(self):
        compilers = [self.compiler1, self.compiler2, self.compiler3, self.compiler4, self.compiler5, self.compiler6, self.compiler7]
        scopes = [self.scope1, self.scope2, self.scope3, self.scope4, self.scope5, self.scope6, self.scope7]
        for compiler, scope in zip(compilers, scopes):
            fluents = compiler.initial_state()
            scope = scope.state(fluents)
            self.assertEqual(len(fluents), len(scope))
            for i, name in enumerate(compiler.rddl.domain.state_fluent_ordering):
                self.assertIs(scope[name], fluents[i])

    def test_action_scope(self):
        compilers = [self.compiler1, self.compiler2, self.compiler3, self.compiler4, self.compiler5, self.compiler6, self.compiler7]
        scopes = [self.scope1, self.scope2, self.scope3, self.scope4, self.scope5, self.scope6, self.scope7]
        for compiler, scope in zip(compilers, scopes):
            fluents = compiler.default_action()
            scope = scope.action(fluents)
            self.assertEqual(len(fluents), len(scope))
            for i, name in enumerate(compiler.rddl.domain.action_fluent_ordering):
                self.assertIs(scope[name], fluents[i])
