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

from rddl2tf.compiler import Compiler
from rddl2tf.fluent import TensorFluent
from rddl2tf.fluentshape import TensorFluentShape

import numpy as np
import tensorflow as tf

import unittest


class TestCompiler(unittest.TestCase):

    def setUp(self):
        self.rddl1 = rddlgym.make('Reservoir-8', mode=rddlgym.AST)
        self.rddl2 = rddlgym.make('Mars_Rover', mode=rddlgym.AST)
        self.rddl3 = rddlgym.make('HVAC-v1', mode=rddlgym.AST)
        self.rddl4 = rddlgym.make('CrossingTraffic-10', mode=rddlgym.AST)
        self.rddl5 = rddlgym.make('GameOfLife-10', mode=rddlgym.AST)
        self.rddl6 = rddlgym.make('CarParking-v1', mode=rddlgym.AST)
        self.rddl7 = rddlgym.make('Navigation-v3', mode=rddlgym.AST)
        self.compiler1 = Compiler(self.rddl1)
        self.compiler2 = Compiler(self.rddl2)
        self.compiler3 = Compiler(self.rddl3)
        self.compiler4 = Compiler(self.rddl4)
        self.compiler5 = Compiler(self.rddl5)
        self.compiler6 = Compiler(self.rddl6)
        self.compiler7 = Compiler(self.rddl7)

    def test_compile_state_action_constraints(self):
        batch_size = 1000
        compilers = [self.compiler4, self.compiler5]
        expected_preconds = [(12, [True] + [False] * 11), (1, [False])]
        for compiler, expected in zip(compilers, expected_preconds):
            compiler.batch_mode_on()
            state = compiler.compile_initial_state(batch_size)
            action = compiler.compile_default_action(batch_size)
            constraints = compiler.compile_state_action_constraints(state, action)
            self.assertIsInstance(constraints, list)
            self.assertEqual(len(constraints), expected[0])
            for c, batch_mode in zip(constraints, expected[1]):
                self.assertIsInstance(c, TensorFluent)
                self.assertEqual(c.dtype, tf.bool)
                if batch_mode:
                    self.assertEqual(c.shape.batch_size, batch_size)
                else:
                    self.assertEqual(c.shape.batch_size, 1)
                self.assertEqual(c.shape.batch, batch_mode)
                self.assertTupleEqual(c.shape.fluent_shape, ())

    def test_compile_action_preconditions(self):
        batch_size = 1000
        compilers = [self.compiler1, self.compiler2]
        expected_preconds = [2, 1]
        for compiler, expected in zip(compilers, expected_preconds):
            compiler.batch_mode_on()
            state = compiler.compile_initial_state(batch_size)
            action = compiler.compile_default_action(batch_size)
            preconds = compiler.compile_action_preconditions(state, action)
            self.assertIsInstance(preconds, list)
            self.assertEqual(len(preconds), expected)
            for p in preconds:
                self.assertIsInstance(p, TensorFluent)
                self.assertEqual(p.dtype, tf.bool)
                self.assertEqual(p.shape.batch_size, batch_size)
                self.assertTrue(p.shape.batch)
                self.assertTupleEqual(p.shape.fluent_shape, ())

    def test_compile_state_invariants(self):
        batch_size = 1000
        compilers = [self.compiler1, self.compiler2]
        expected_invariants = [2, 0]
        for compiler, expected in zip(compilers, expected_invariants):
            compiler.batch_mode_on()
            state = compiler.compile_initial_state(batch_size)
            invariants = compiler.compile_state_invariants(state)
            self.assertIsInstance(invariants, list)
            self.assertEqual(len(invariants), expected)
            for p in invariants:
                self.assertIsInstance(p, TensorFluent)
                self.assertEqual(p.dtype, tf.bool)
                self.assertTupleEqual(p.shape.fluent_shape, ())

    def test_compile_action_preconditions_checking(self):
        batch_size = 1000
        compilers = [self.compiler1, self.compiler2]
        for compiler in compilers:
            compiler.batch_mode_on()
            state = compiler.compile_initial_state(batch_size)
            action = compiler.compile_default_action(batch_size)
            checking = compiler.compile_action_preconditions_checking(state, action)
            self.assertIsInstance(checking, tf.Tensor)
            self.assertEqual(checking.dtype, tf.bool)
            self.assertListEqual(checking.shape.as_list(), [batch_size])

    def test_compile_action_lower_bound_constraints(self):
        batch_size = 1000
        compilers = [self.compiler1, self.compiler3]
        expected = [[('outflow/1', [], tf.int32)], [('AIR/1', [], tf.int32)]]

        for compiler, expected_bounds in zip(compilers, expected):
            compiler.batch_mode_on()
            initial_state = compiler.compile_initial_state(batch_size)
            default_action_fluents = compiler.compile_default_action(batch_size)
            bounds = compiler.compile_action_bound_constraints(initial_state)
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

    def test_compile_action_lower_bound_constraints(self):
        batch_size = 1000
        compilers = [self.compiler1, self.compiler3]
        expected = [[('outflow/1', [], tf.int32)], [('AIR/1', [], tf.int32)]]

        for compiler, expected_bounds in zip(compilers, expected):
            compiler.batch_mode_on()
            initial_state = compiler.compile_initial_state(batch_size)
            default_action_fluents = compiler.compile_default_action(batch_size)
            bounds = compiler.compile_action_bound_constraints(initial_state)
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
        batch_size = 1000
        compilers = [self.compiler1, self.compiler3]
        expected = [[('outflow/1', [batch_size, 8], tf.float32)], [('AIR/1', [3], tf.float32)]]

        for compiler, expected_bounds in zip(compilers, expected):
            compiler.batch_mode_on()
            initial_state = compiler.compile_initial_state(batch_size)
            default_action_fluents = compiler.compile_default_action(batch_size)
            bounds = compiler.compile_action_bound_constraints(initial_state)
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

    def test_initialize_non_fluents(self):
        nf = dict(self.compiler1.compile_non_fluents())

        expected_non_fluents = {
            'MAX_RES_CAP/1': { 'shape': [8,], 'dtype': tf.float32 },
            'UPPER_BOUND/1': { 'shape': [8,], 'dtype': tf.float32 },
            'LOWER_BOUND/1': { 'shape': [8,], 'dtype': tf.float32 },
            'RAIN_SHAPE/1': { 'shape': [8,], 'dtype': tf.float32 },
            'RAIN_SCALE/1': { 'shape': [8,], 'dtype': tf.float32 },
            'DOWNSTREAM/2': { 'shape': [8,8], 'dtype': tf.bool },
            'SINK_RES/1': { 'shape': [8,], 'dtype': tf.bool },
            'MAX_WATER_EVAP_FRAC_PER_TIME_UNIT/0': { 'shape': [], 'dtype': tf.float32 },
            'LOW_PENALTY/1': { 'shape': [8,], 'dtype': tf.float32 },
            'HIGH_PENALTY/1': { 'shape': [8,], 'dtype': tf.float32 }
        }
        self.assertIsInstance(nf, dict)
        self.assertEqual(len(nf), len(expected_non_fluents))
        for name, fluent in nf.items():
            self.assertIn(name, expected_non_fluents)
            shape = expected_non_fluents[name]['shape']
            dtype = expected_non_fluents[name]['dtype']
            self.assertEqual(fluent.name, 'non_fluents/{}:0'.format(name.replace('/', '-')))
            self.assertIsInstance(fluent, TensorFluent)
            self.assertEqual(fluent.dtype, dtype)
            self.assertEqual(fluent.shape.as_list(), shape)

        expected_initializers = {
            'MAX_RES_CAP/1': [ 100.,  100.,  200.,  300.,  400.,  500.,  800., 1000.],
            'UPPER_BOUND/1': [ 80.,  80., 180., 280., 380., 480., 780., 980.],
            'LOWER_BOUND/1': [20., 20., 20., 20., 20., 20., 20., 20.],
            'RAIN_SHAPE/1': [1., 1., 1., 1., 1., 1., 1., 1.],
            'RAIN_SCALE/1': [ 5.,  3.,  9.,  7., 15., 13., 25., 30.],
            'DOWNSTREAM/2': [
                [False, False, False, False, False, True, False, False],
                [False, False, True, False, False, False, False, False],
                [False, False, False, False, True, False, False, False],
                [False, False, False, False, False, False, False, True],
                [False, False, False, False, False, False, True, False],
                [False, False, False, False, False, False, True, False],
                [False, False, False, False, False, False, False, True],
                [False, False, False, False, False, False, False, False]
            ],
            'SINK_RES/1': [False, False, False, False, False, False, False, True],
            'MAX_WATER_EVAP_FRAC_PER_TIME_UNIT/0': 0.05,
            'LOW_PENALTY/1': [-5., -5., -5., -5., -5., -5., -5., -5.],
            'HIGH_PENALTY/1': [-10., -10., -10., -10., -10., -10., -10., -10.]
        }
        with tf.Session(graph=self.compiler1.graph) as sess:
            for name, fluent in nf.items():
                value = sess.run(fluent.tensor)
                list1 = list(value.flatten())
                list2 = list(np.array(expected_initializers[name]).flatten())
                for v1, v2 in zip(list1, list2):
                    self.assertAlmostEqual(v1, v2)

    def test_initialize_initial_state_fluents(self):
        sf = dict(self.compiler1.compile_initial_state())

        expected_state_fluents = {
            'rlevel/1': { 'shape': [8,] , 'dtype': tf.float32 }
        }
        self.assertIsInstance(sf, dict)
        self.assertEqual(len(sf), len(expected_state_fluents))
        for name, fluent in sf.items():
            self.assertIn(name, expected_state_fluents)
            shape = expected_state_fluents[name]['shape']
            dtype = expected_state_fluents[name]['dtype']
            self.assertEqual(fluent.name, 'initial_state/{}:0'.format(name.replace('/', '-')))
            self.assertIsInstance(fluent, TensorFluent)
            self.assertEqual(fluent.dtype, dtype)
            self.assertEqual(fluent.shape.as_list(), shape)

        expected_initializers = {
            'rlevel/1': [75., 50., 50., 50., 50., 50., 50., 50.]
        }
        with tf.Session(graph=self.compiler1.graph) as sess:
            for name, fluent in sf.items():
                value = sess.run(fluent.tensor)
                list1 = list(value.flatten())
                list2 = list(np.array(expected_initializers[name]).flatten())
                for v1, v2 in zip(list1, list2):
                    self.assertAlmostEqual(v1, v2)

    def test_initialize_default_action_fluents(self):
        action_fluents = self.compiler1.compile_default_action()
        self.assertIsInstance(action_fluents, list)
        for fluent in action_fluents:
            self.assertIsInstance(fluent, tuple)
            self.assertEqual(len(fluent), 2)
            self.assertIsInstance(fluent[0], str)
            self.assertIsInstance(fluent[1], TensorFluent)

        af = dict(action_fluents)

        expected_action_fluents = {
            'outflow/1': { 'shape': [8,] , 'dtype': tf.float32 }
        }
        self.assertEqual(len(af), len(expected_action_fluents))
        for name, fluent in af.items():
            self.assertIn(name, expected_action_fluents)
            shape = expected_action_fluents[name]['shape']
            dtype = expected_action_fluents[name]['dtype']
            self.assertEqual(fluent.name, 'default_action/{}:0'.format(name.replace('/', '-')))
            self.assertIsInstance(fluent, TensorFluent)
            self.assertEqual(fluent.dtype, dtype)
            self.assertEqual(fluent.shape.as_list(), shape)

        expected_initializers = {
            'outflow/1': [0., 0., 0., 0., 0., 0., 0., 0.]
        }
        with tf.Session(graph=self.compiler1.graph) as sess:
            for name, fluent in af.items():
                value = sess.run(fluent.tensor)
                list1 = list(value.flatten())
                list2 = list(np.array(expected_initializers[name]).flatten())
                for v1, v2 in zip(list1, list2):
                    self.assertAlmostEqual(v1, v2)

    def test_state_scope(self):
        compilers = [self.compiler1, self.compiler2, self.compiler3, self.compiler4, self.compiler5, self.compiler6, self.compiler7]
        for compiler in compilers:
            fluents = compiler.compile_initial_state()
            scope = dict(fluents)
            self.assertEqual(len(fluents), len(scope))
            for i, name in enumerate(compiler.rddl.domain.state_fluent_ordering):
                self.assertIs(scope[name], fluents[i][1])

    def test_action_scope(self):
        compilers = [self.compiler1, self.compiler2, self.compiler3, self.compiler4, self.compiler5, self.compiler6, self.compiler7]
        for compiler in compilers:
            fluents = compiler.compile_default_action()
            scope = dict(fluents)
            self.assertEqual(len(fluents), len(scope))
            for i, name in enumerate(compiler.rddl.domain.action_fluent_ordering):
                self.assertIs(scope[name], fluents[i][1])

    def test_compile_expressions(self):
        expected = {
            # rddl1: RESERVOIR ====================================================
            'rainfall/1':   { 'shape': [8,], 'dtype': tf.float32, 'scope': ['?r'] },
            'evaporated/1': { 'shape': [8,], 'dtype': tf.float32, 'scope': ['?r'] },
            'overflow/1':   { 'shape': [8,], 'dtype': tf.float32, 'scope': ['?r'] },
            'inflow/1':     { 'shape': [8,], 'dtype': tf.float32, 'scope': ['?r'] },
            "rlevel'/1":    { 'shape': [8,], 'dtype': tf.float32, 'scope': ['?r'] },

            # rddl2: MARS ROVER ===================================================
            "xPos'/0":   { 'shape': [], 'dtype': tf.float32, 'scope': [] },
            "yPos'/0":   { 'shape': [], 'dtype': tf.float32, 'scope': [] },
            "time'/0":   { 'shape': [], 'dtype': tf.float32, 'scope': [] },
            "picTaken'/1": { 'shape': [3,], 'dtype': tf.bool, 'scope': ['?p'] }
        }

        compilers = [self.compiler1, self.compiler2]
        rddls = [self.rddl1, self.rddl2]
        for compiler, rddl in zip(compilers, rddls):
            nf = compiler.non_fluents_scope()
            sf = dict(compiler.compile_initial_state())
            af = dict(compiler.compile_default_action())
            scope = {}
            scope.update(nf)
            scope.update(sf)
            scope.update(af)

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
            self.assertEqual(t.shape.as_list(), [])
            self.assertEqual(t.dtype, tf.float32)
            self.assertEqual(t.scope.as_list(), [])

    def test_compile_cpfs(self):
        compilers = [self.compiler1, self.compiler2]
        expected = [
            (['evaporated/1', 'overflow/1', 'rainfall/1', 'inflow/1'], ["rlevel'/1"]),
            ([], ["picTaken'/1", "time'/0", "xPos'/0", "yPos'/0"]),
        ]
        for compiler, (expected_interm, expected_state) in zip(compilers, expected):
            nf = compiler.non_fluents_scope()
            sf = dict(compiler.compile_initial_state())
            af = dict(compiler.compile_default_action())
            scope = { **nf, **sf, **af }

            interm_fluents, next_state_fluents = compiler.compile_cpfs(scope)

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
            nf = compiler.non_fluents_scope()
            sf = dict(compiler.compile_initial_state())
            af = dict(compiler.compile_default_action())
            scope = { **nf, **sf, **af }

            interm_fluents = compiler.compile_intermediate_cpfs(scope)
            scope.update(dict(interm_fluents))
            next_state_fluents = compiler.compile_state_cpfs(scope)

            self.assertIsInstance(next_state_fluents, list)
            for cpf in next_state_fluents:
                self.assertIsInstance(cpf, tuple)
            self.assertEqual(len(next_state_fluents), len(sf))

            next_state_fluents = dict(next_state_fluents)
            for fluent in sf:
                next_fluent = utils.rename_state_fluent(fluent)
                self.assertIn(next_fluent, next_state_fluents)
                self.assertIsInstance(next_state_fluents[next_fluent], TensorFluent)

    def test_compile_intermediate_cpfs(self):
        compilers = [self.compiler1, self.compiler2, self.compiler3, self.compiler4, self.compiler5, self.compiler6, self.compiler7]
        for compiler in compilers:
            fluents = compiler.rddl.domain.interm_fluent_ordering

            nf = compiler.non_fluents_scope()
            sf = dict(compiler.compile_initial_state())
            af = dict(compiler.compile_default_action())
            scope = { **nf, **sf, **af }
            interm_fluents = compiler.compile_intermediate_cpfs(scope)
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
        batch_size = 32
        for compiler in compilers:
            compiler.batch_mode_on()
            state = compiler.compile_initial_state(batch_size)
            action = compiler.compile_default_action(batch_size)
            scope = compiler.transition_scope(state, action)
            interm_fluents, next_state_fluents = compiler.compile_cpfs(scope)
            scope.update(next_state_fluents)
            reward = compiler.compile_reward(scope)
            self.assertIsInstance(reward, TensorFluent)
            self.assertEqual(reward.shape.as_list(), [batch_size])

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
        batch_sizes = [8]

        for compiler, batch_size in zip(compilers, batch_sizes):
            compiler.batch_mode_on()

            nf = compiler.non_fluents_scope()
            sf = dict(compiler.compile_initial_state())
            af = dict(compiler.compile_default_action())
            scope = { **nf, **sf, **af }

            yield (compiler, batch_size, scope)

    def _test_random_variable_expressions(self, expressions):
        for compiler, batch_size, scope in self.__get_batch_compiler_with_state_action_scope():
            for expr in expressions:
                self._test_random_variable_expression(expr, compiler, scope, batch_size)

    def _test_random_variable_expression(self, expr, compiler, scope, batch_size):
        sample = compiler._compile_random_variable_expression(expr, scope, batch_size)
        self._test_sample_fluents(sample, batch_size)
        self._test_sample_fluent(sample)

    def _test_sample_fluents(self, sample, batch_size=None):
        self.assertIsInstance(sample, TensorFluent)
        if batch_size is not None:
            self.assertEqual(sample.shape[0], batch_size)

    def _test_sample_fluent(self, sample):
        self.assertTrue(sample.tensor.name.startswith('sample'), sample.tensor)

    def _test_conditional_sample(self, sample):
        inputs = sample.tensor.op.inputs
        self.assertEqual(len(inputs), 3)
        self.assertTrue(inputs[0].name.startswith('LogicalNot'), inputs[0])
        self.assertTrue(inputs[1].name.startswith('StopGradient'), inputs[1])
        self.assertTrue(inputs[2].name.startswith('sample'), inputs[2])
