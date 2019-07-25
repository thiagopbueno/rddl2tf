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


import rddlgym

from pyrddl.expr import Expression
from pyrddl import utils

from rddl2tf.compilers.modes.default import DefaultCompiler
from rddl2tf.core.fluent import TensorFluent
from rddl2tf.core.fluentshape import TensorFluentShape

import numpy as np
import tensorflow as tf

import unittest


class TestDefaultCompiler(unittest.TestCase):

    def setUp(self):
        self.batch_size = 32

        self.rddl1 = rddlgym.make('Reservoir-8', mode=rddlgym.AST)
        self.compiler1 = DefaultCompiler(self.rddl1, self.batch_size)
        self.compiler1.init()

        self.rddl2 = rddlgym.make('Mars_Rover', mode=rddlgym.AST)
        self.compiler2 = DefaultCompiler(self.rddl2, self.batch_size)
        self.compiler2.init()

        self.rddl3 = rddlgym.make('HVAC-v1', mode=rddlgym.AST)
        self.compiler3 = DefaultCompiler(self.rddl3, self.batch_size)
        self.compiler3.init()

        self.rddl4 = rddlgym.make('CrossingTraffic-10', mode=rddlgym.AST)
        self.compiler4 = DefaultCompiler(self.rddl4, self.batch_size)
        self.compiler4.init()

        self.rddl5 = rddlgym.make('GameOfLife-10', mode=rddlgym.AST)
        self.compiler5 = DefaultCompiler(self.rddl5, self.batch_size)
        self.compiler5.init()

        self.rddl6 = rddlgym.make('CarParking-v1', mode=rddlgym.AST)
        self.compiler6 = DefaultCompiler(self.rddl6, self.batch_size)
        self.compiler6.init()

        self.rddl7 = rddlgym.make('Navigation-v3', mode=rddlgym.AST)
        self.compiler7 = DefaultCompiler(self.rddl7, self.batch_size)
        self.compiler7.init()

    def test_state_invariants(self):
        compilers = [self.compiler1, self.compiler2]
        expected_invariants = [2, 0]
        for compiler, expected in zip(compilers, expected_invariants):
            state = compiler.initial_state()
            invariants = compiler.state_invariants(state)
            self.assertIsInstance(invariants, list)
            self.assertEqual(len(invariants), expected)
            for p in invariants:
                self.assertIsInstance(p, TensorFluent)
                self.assertEqual(p.dtype, tf.bool)
                self.assertTupleEqual(p.shape.fluent_shape, ())

    def test_state_action_constraints(self):
        compilers = [self.compiler4, self.compiler5]
        expected_preconds = [(12, [True] + [False] * 11), (1, [False])]
        for compiler, expected in zip(compilers, expected_preconds):
            state = compiler.initial_state()
            action = compiler.default_action()
            constraints = compiler.state_action_constraints(state, action)
            self.assertIsInstance(constraints, list)
            self.assertEqual(len(constraints), expected[0])
            for c, batch_mode in zip(constraints, expected[1]):
                self.assertIsInstance(c, TensorFluent)
                self.assertEqual(c.dtype, tf.bool)
                if batch_mode:
                    self.assertEqual(c.shape.batch_size, self.batch_size)
                else:
                    self.assertEqual(c.shape.batch_size, 1)
                self.assertEqual(c.shape.batch, batch_mode)
                self.assertTupleEqual(c.shape.fluent_shape, ())

    def test_action_preconditions(self):
        compilers = [self.compiler1, self.compiler2]
        expected_preconds = [2, 1]
        for compiler, expected in zip(compilers, expected_preconds):
            state = compiler.initial_state()
            action = compiler.default_action()
            preconds = compiler.action_preconditions(state, action)
            self.assertIsInstance(preconds, list)
            self.assertEqual(len(preconds), expected)
            for p in preconds:
                self.assertIsInstance(p, TensorFluent)
                self.assertEqual(p.dtype, tf.bool)
                self.assertEqual(p.shape.batch_size, self.batch_size)
                self.assertTrue(p.shape.batch)
                self.assertTupleEqual(p.shape.fluent_shape, ())

    def test_compile_action_lower_bound_constraints(self):
        compilers = [self.compiler1, self.compiler3]
        expected = [[('outflow/1', [], tf.int32)], [('AIR/1', [], tf.int32)]]

        for compiler, expected_bounds in zip(compilers, expected):
            initial_state = compiler.initial_state()
            default_action_fluents = compiler.default_action()
            bounds = compiler.action_bound_constraints(initial_state)
            self.assertIsInstance(bounds, dict)

            self.assertEqual(len(bounds), len(expected_bounds))
            for fluent_name, shape, dtype in expected_bounds:
                self.assertIn(fluent_name, bounds)
                self.assertIsInstance(bounds[fluent_name], tuple)
                self.assertEqual(len(bounds[fluent_name]), 2)
                lower, _ = bounds[fluent_name]
                self.assertIsInstance(lower, TensorFluent)
                self.assertListEqual(lower.shape.as_list(), shape)
                self.assertEqual(lower.dtype, dtype)

    def test_compile_action_upper_bound_constraints(self):
        compilers = [self.compiler1, self.compiler3]
        expected = [[('outflow/1', [self.batch_size, 8], tf.float32)], [('AIR/1', [3], tf.float32)]]

        for compiler, expected_bounds in zip(compilers, expected):
            initial_state = compiler.initial_state()
            default_action_fluents = compiler.default_action()
            bounds = compiler.action_bound_constraints(initial_state)
            self.assertIsInstance(bounds, dict)

            self.assertEqual(len(bounds), len(expected_bounds))
            for fluent_name, shape, dtype in expected_bounds:
                self.assertIn(fluent_name, bounds)
                self.assertIsInstance(bounds[fluent_name], tuple)
                self.assertEqual(len(bounds[fluent_name]), 2)
                _, upper = bounds[fluent_name]
                self.assertIsInstance(upper, TensorFluent)
                self.assertEqual(upper.dtype, dtype)
                self.assertListEqual(upper.shape.as_list(), shape)

    def test_compile_expressions(self):
        expected = {
            # rddl1: RESERVOIR ====================================================
            'rainfall/1':   { 'shape': [self.batch_size, 8,], 'dtype': tf.float32, 'scope': ['?r'] },
            'evaporated/1': { 'shape': [self.batch_size, 8,], 'dtype': tf.float32, 'scope': ['?r'] },
            'overflow/1':   { 'shape': [self.batch_size, 8,], 'dtype': tf.float32, 'scope': ['?r'] },
            'inflow/1':     { 'shape': [self.batch_size, 8,], 'dtype': tf.float32, 'scope': ['?r'] },
            "rlevel'/1":    { 'shape': [self.batch_size, 8,], 'dtype': tf.float32, 'scope': ['?r'] },

            # rddl2: MARS ROVER ===================================================
            "xPos'/0":   { 'shape': [self.batch_size, ], 'dtype': tf.float32, 'scope': [] },
            "yPos'/0":   { 'shape': [self.batch_size, ], 'dtype': tf.float32, 'scope': [] },
            "time'/0":   { 'shape': [self.batch_size, ], 'dtype': tf.float32, 'scope': [] },
            "picTaken'/1": { 'shape': [self.batch_size, 3,], 'dtype': tf.bool, 'scope': ['?p'] }
        }

        compilers = [self.compiler1, self.compiler2]
        rddls = [self.rddl1, self.rddl2]
        for compiler, rddl in zip(compilers, rddls):
            state = compiler.initial_state()
            action = compiler.default_action()
            scope = compiler._scope.transition(compiler.non_fluents, state, action)

            _, cpfs = rddl.domain.cpfs
            for cpf in cpfs:
                name = cpf.name
                expr = cpf.expr
                t = compiler._compile_expression(expr, scope)
                scope[name] = t
                self.assertIsInstance(t, TensorFluent)
                self.assertEqual(t.shape.as_list(), expected[name]['shape'])
                self.assertEqual(t.dtype, expected[name]['dtype'])
                self.assertEqual(t.scope.as_list(), expected[name]['scope'])

            reward_expr = rddl.domain.reward
            t = compiler._compile_expression(reward_expr, scope)
            self.assertIsInstance(t, TensorFluent)
            self.assertEqual(t.shape.as_list(), [self.batch_size])
            self.assertEqual(t.dtype, tf.float32)
            self.assertEqual(t.scope.as_list(), [])

    def test_compile_cpfs(self):
        compilers = [self.compiler1, self.compiler2]

        expected = [
            (['evaporated/1', 'overflow/1', 'rainfall/1', 'inflow/1'], ["rlevel'/1"]),
            ([], ["picTaken'/1", "time'/0", "xPos'/0", "yPos'/0"]),
        ]

        for compiler, (expected_interm, expected_state) in zip(compilers, expected):
            state = compiler.initial_state()
            action = compiler.default_action()
            scope = compiler._scope.transition(compiler.non_fluents, state, action)

            interm_fluents, next_state_fluents = compiler._compile_cpfs(scope)

            self.assertIsInstance(interm_fluents, list)
            self.assertEqual(len(interm_fluents), len(expected_interm))
            for fluent, expected_fluent in zip(interm_fluents, expected_interm):
                self.assertEqual(fluent[0], expected_fluent)

            self.assertIsInstance(next_state_fluents, list)
            self.assertEqual(len(next_state_fluents), len(expected_state))
            for fluent, expected_fluent in zip(next_state_fluents, expected_state):
                self.assertEqual(fluent[0], expected_fluent)

    def test_compile_state_cpfs(self):
        compilers = [self.compiler1, self.compiler2, self.compiler3, self.compiler4, self.compiler5, self.compiler6, self.compiler7]
        for compiler in compilers:

            state = compiler.initial_state()
            action = compiler.default_action()
            scope = compiler._scope.transition(compiler.non_fluents, state, action)

            interm_fluents = compiler._compile_intermediate_cpfs(scope)
            scope.update(dict(interm_fluents))
            next_state_fluents = compiler._compile_state_cpfs(scope)

            self.assertIsInstance(next_state_fluents, list)
            for cpf in next_state_fluents:
                self.assertIsInstance(cpf, tuple)
            self.assertEqual(len(next_state_fluents), len(state))

            for name, fluent in next_state_fluents:
                self.assertIsInstance(fluent, TensorFluent)
                next_fluent = utils.rename_next_state_fluent(name)
                self.assertIn(next_fluent, scope)

    def test_compile_intermediate_cpfs(self):
        compilers = [self.compiler1, self.compiler2, self.compiler3, self.compiler4, self.compiler5, self.compiler6, self.compiler7]
        for compiler in compilers:
            fluents = compiler.rddl.domain.interm_fluent_ordering

            state = compiler.initial_state()
            action = compiler.default_action()
            scope = compiler._scope.transition(compiler.non_fluents, state, action)

            interm_fluents = compiler._compile_intermediate_cpfs(scope)
            self.assertIsInstance(interm_fluents, list)
            self.assertEqual(len(interm_fluents), len(fluents))
            for actual, expected in zip(interm_fluents, fluents):
                self.assertIsInstance(actual, tuple)
                self.assertEqual(len(actual), 2)
                self.assertIsInstance(actual[0], str)
                self.assertIsInstance(actual[1], TensorFluent)
                self.assertEqual(actual[0], expected)

    def test_compile_reward(self):
        # TODO: self.compiler4
        compilers = [self.compiler1, self.compiler2, self.compiler3, self.compiler5, self.compiler6, self.compiler7]
        for compiler in compilers:
            state = compiler.initial_state()
            action = compiler.default_action()
            scope = compiler._scope.transition(compiler.non_fluents, state, action)
            interm_fluents, next_state_fluents = compiler._compile_cpfs(scope)
            scope.update(next_state_fluents)
            reward = compiler._compile_reward(scope)
            self.assertIsInstance(reward, TensorFluent)
            self.assertEqual(reward.shape.as_list(), [self.batch_size])

    def test_compile_probabilistic_normal_random_variable(self):
        mean = Expression(('number', 0.0))
        var = Expression(('number', 1.0))
        normal = Expression(('randomvar', ('Normal', (mean, var))))

        expressions = [normal]
        self._test_random_variable_expressions(expressions)

    def test_compile_probabilistic_gamma_random_variable(self):
        shape = Expression(('number', 5.0))
        scale = Expression(('number', 1.0))
        gamma = Expression(('randomvar', ('Gamma', (shape, scale))))

        expressions = [gamma]
        self._test_random_variable_expressions(expressions)

    def __get_batch_compiler_with_state_action_scope(self):
        compilers = [self.compiler2]

        for compiler in compilers:
            state = compiler.initial_state()
            action = compiler.default_action()
            scope = compiler._scope.transition(compiler.non_fluents, state, action)

            yield (compiler, scope)

    def _test_random_variable_expressions(self, expressions):
        for compiler, scope in self.__get_batch_compiler_with_state_action_scope():
            for expr in expressions:
                self._test_random_variable_expression(expr, compiler, scope)

    def _test_random_variable_expression(self, expr, compiler, scope):
        sample = compiler._compile_random_variable_expression(expr, scope)
        self._test_sample_fluents(sample)
        self._test_sample_fluent(sample)

    def _test_sample_fluents(self, sample):
        self.assertIsInstance(sample, TensorFluent)
        if self.batch_size is not None:
            self.assertEqual(sample.shape[0], self.batch_size)

    def _test_sample_fluent(self, sample):
        self.assertTrue(sample.tensor.name.startswith('sample'), sample.tensor)

    def _test_conditional_sample(self, sample):
        inputs = sample.tensor.op.inputs
        self.assertEqual(len(inputs), 3)
        self.assertTrue(inputs[0].name.startswith('LogicalNot'), inputs[0])
        self.assertTrue(inputs[1].name.startswith('StopGradient'), inputs[1])
        self.assertTrue(inputs[2].name.startswith('sample'), inputs[2])
