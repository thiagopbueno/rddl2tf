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
        self.compiler1 = Compiler(self.rddl1)
        self.compiler2 = Compiler(self.rddl2)

    def test_build_object_table(self):
        self.assertIn('res', self.compiler1.object_table)
        size = self.compiler1.object_table['res']['size']
        idx = self.compiler1.object_table['res']['idx']
        self.assertEqual(size, 8)
        objs = ['t1', 't2', 't3', 't4', 't5', 't6', 't7', 't8']
        for i, obj in enumerate(objs):
            self.assertIn(obj, idx)
            self.assertEqual(idx[obj], i)

    def test_build_action_preconditions_table(self):
        local_preconds = self.compiler1.local_action_preconditions
        self.assertIsInstance(local_preconds, dict)
        self.assertEqual(len(local_preconds), 1)
        self.assertIn('outflow/1', local_preconds)
        self.assertEqual(len(local_preconds['outflow/1']), 2)

        global_preconds = self.compiler1.global_action_preconditions
        self.assertIsInstance(global_preconds, list)
        self.assertEqual(len(global_preconds), 0)

    def test_lower_bound_constraints(self):
        lower_bounds = self.compiler1.action_lower_bound_constraints
        self.assertIsInstance(lower_bounds, dict)
        self.assertIn('outflow/1', lower_bounds)
        lower = lower_bounds['outflow/1']
        self.assertIsInstance(lower, Expression)
        self.assertTrue(lower.is_constant_expression())
        self.assertEqual(lower.value, 0)

    def test_upper_bound_constraints(self):
        upper_bounds = self.compiler1.action_upper_bound_constraints
        self.assertIsInstance(upper_bounds, dict)
        self.assertIn('outflow/1', upper_bounds)
        upper = upper_bounds['outflow/1']
        self.assertIsInstance(upper, Expression)
        self.assertTrue(upper.is_pvariable_expression())
        self.assertEqual(upper.name, 'rlevel/1')

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
                self.assertTupleEqual(p.shape.fluent_shape, (1,))

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
                self.assertTupleEqual(p.shape.fluent_shape, (1,))

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

    def test_compile_action_bound_constraints(self):
        batch_size = 1000
        self.compiler1.batch_mode_on()
        initial_state = self.compiler1.compile_initial_state(batch_size)
        default_action_fluents = self.compiler1.compile_default_action(batch_size)
        bounds = self.compiler1.compile_action_bound_constraints(initial_state)
        self.assertIsInstance(bounds, dict)
        self.assertIn('outflow/1', bounds)
        self.assertIsInstance(bounds['outflow/1'], tuple)
        self.assertEqual(len(bounds['outflow/1']), 2)
        lower, upper = bounds['outflow/1']
        self.assertIsInstance(lower, TensorFluent)
        self.assertListEqual(lower.shape.as_list(), [])
        self.assertEqual(lower.dtype, tf.int32)
        self.assertIsInstance(upper, TensorFluent)
        self.assertEqual(upper.dtype, tf.float32)
        self.assertListEqual(upper.shape.as_list(), [batch_size, 8])

    def test_instantiate_non_fluents(self):
        nf = dict(self.compiler1.non_fluents)

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

    def test_instantiate_initial_state_fluents(self):
        sf = dict(self.compiler1.initial_state_fluents)

        expected_state_fluents = {
            'rlevel/1': { 'shape': [8,] , 'dtype': tf.float32 }
        }
        self.assertIsInstance(sf, dict)
        self.assertEqual(len(sf), len(expected_state_fluents))
        for name, fluent in sf.items():
            self.assertIn(name, expected_state_fluents)
            shape = expected_state_fluents[name]['shape']
            dtype = expected_state_fluents[name]['dtype']
            self.assertEqual(fluent.name, '{}:0'.format(name.replace('/', '-')))
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

    def test_instantiate_default_action_fluents(self):
        action_fluents = self.compiler1.default_action_fluents
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
            self.assertEqual(fluent.name, '{}:0'.format(name.replace('/', '-')))
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

    def test_non_fluent_ordering(self):
        compilers = [self.compiler1, self.compiler2]
        for compiler in compilers:
            non_fluents = dict(compiler.non_fluents)
            action_fluent_ordering = compiler.non_fluent_ordering
            self.assertEqual(len(action_fluent_ordering), len(non_fluents))
            for action_fluent in action_fluent_ordering:
                self.assertIn(action_fluent, non_fluents)

    def test_state_fluent_ordering(self):
        compilers = [self.compiler1, self.compiler2]
        for compiler in compilers:
            initial_state_fluents = dict(compiler.initial_state_fluents)
            current_state_ordering = compiler.state_fluent_ordering
            self.assertEqual(len(current_state_ordering), len(initial_state_fluents))
            for fluent in initial_state_fluents:
                self.assertIn(fluent, current_state_ordering)

            next_state_ordering = compiler.next_state_fluent_ordering
            self.assertEqual(len(current_state_ordering), len(next_state_ordering))

            for current_fluent, next_fluent in zip(current_state_ordering, next_state_ordering):
                self.assertEqual(utils.rename_state_fluent(current_fluent), next_fluent)
                self.assertEqual(utils.rename_next_state_fluent(next_fluent), current_fluent)

    def test_interm_fluent_ordering(self):
        compilers = [self.compiler1, self.compiler2]
        expected = [
            ['evaporated/1', 'overflow/1', 'rainfall/1', 'inflow/1'],
            []
        ]
        for compiler, expected_ordering in zip(compilers, expected):
            interm_fluent_ordering = compiler.interm_fluent_ordering
            self.assertListEqual(interm_fluent_ordering, expected_ordering)

    def test_action_fluent_ordering(self):
        compilers = [self.compiler1, self.compiler2]
        for compiler in compilers:
            default_action_fluents = dict(compiler.default_action_fluents)
            action_fluent_ordering = compiler.action_fluent_ordering
            self.assertEqual(len(action_fluent_ordering), len(default_action_fluents))
            for action_fluent in action_fluent_ordering:
                self.assertIn(action_fluent, default_action_fluents)

    def test_state_fluent_variables(self):
        compilers = [self.compiler1, self.compiler2]
        fluent_variables = [
            {
            'rlevel/1': ['rlevel(t1)', 'rlevel(t2)', 'rlevel(t3)', 'rlevel(t4)', 'rlevel(t5)', 'rlevel(t6)', 'rlevel(t7)', 'rlevel(t8)']
            },
            {
                'picTaken/1': ['picTaken(p1)', 'picTaken(p2)', 'picTaken(p3)'],
                'time/0': ['time'],
                'xPos/0': ['xPos'],
                'yPos/0': ['yPos']
            }
        ]
        for compiler, expected_variables in zip(compilers, fluent_variables):
            fluent_variables = compiler.state_fluent_variables
            self.assertEqual(len(fluent_variables), len(expected_variables))
            for name, actual_variables in fluent_variables:
                self.assertIn(name, expected_variables)
                self.assertListEqual(actual_variables, expected_variables[name])

    def test_interm_fluent_variables(self):
        compilers = [self.compiler1, self.compiler2]
        fluent_variables = [
            {
            'evaporated/1': ['evaporated(t1)', 'evaporated(t2)', 'evaporated(t3)', 'evaporated(t4)', 'evaporated(t5)', 'evaporated(t6)', 'evaporated(t7)', 'evaporated(t8)'],
            'rainfall/1': ['rainfall(t1)', 'rainfall(t2)', 'rainfall(t3)', 'rainfall(t4)', 'rainfall(t5)', 'rainfall(t6)', 'rainfall(t7)', 'rainfall(t8)'],
            'overflow/1': ['overflow(t1)', 'overflow(t2)', 'overflow(t3)', 'overflow(t4)', 'overflow(t5)', 'overflow(t6)', 'overflow(t7)', 'overflow(t8)'],
            'inflow/1': ['inflow(t1)', 'inflow(t2)', 'inflow(t3)', 'inflow(t4)', 'inflow(t5)', 'inflow(t6)', 'inflow(t7)', 'inflow(t8)']
            },
            {}
        ]
        for compiler, expected_variables in zip(compilers, fluent_variables):
            fluent_variables = compiler.interm_fluent_variables
            self.assertEqual(len(fluent_variables), len(expected_variables))
            for name, actual_variables in fluent_variables:
                self.assertIn(name, expected_variables)
                self.assertListEqual(actual_variables, expected_variables[name])

    def test_action_fluent_variables(self):
        compilers = [self.compiler1, self.compiler2]
        fluent_variables = [
            {
                'outflow/1': ['outflow(t1)', 'outflow(t2)', 'outflow(t3)', 'outflow(t4)', 'outflow(t5)', 'outflow(t6)', 'outflow(t7)', 'outflow(t8)']
            },
            {
                'snapPicture/0': ['snapPicture'],
                'xMove/0': ['xMove'],
                'yMove/0': ['yMove']
            }
        ]
        for compiler, expected_variables in zip(compilers, fluent_variables):
            fluent_variables = compiler.action_fluent_variables
            self.assertEqual(len(fluent_variables), len(expected_variables))
            for name, actual_variables in fluent_variables:
                self.assertIn(name, expected_variables)
                self.assertListEqual(actual_variables, expected_variables[name])

    def test_state_size(self):
        compilers = [self.compiler1, self.compiler2]
        for compiler in compilers:
            state_size = compiler.state_size
            initial_state_fluents = dict(compiler.initial_state_fluents)
            state_fluent_ordering = compiler.state_fluent_ordering
            next_state_fluent_ordering = compiler.next_state_fluent_ordering

            self.assertIsInstance(state_size, tuple)
            for shape in state_size:
                self.assertIsInstance(shape, tuple)
            self.assertEqual(len(state_size), len(initial_state_fluents))
            self.assertEqual(len(state_size), len(state_fluent_ordering))
            self.assertEqual(len(state_size), len(next_state_fluent_ordering))

            for shape, name in zip(state_size, state_fluent_ordering):
                actual = list(shape)
                expected = initial_state_fluents[name].shape.as_list()
                if expected == []:
                    expected = [1]
                self.assertListEqual(actual, expected)

            nf = compiler.non_fluents_scope()
            sf = dict(compiler.initial_state_fluents)
            af = dict(compiler.default_action_fluents)
            scope = {}
            scope.update(nf)
            scope.update(sf)
            scope.update(af)
            interm_fluents, next_state_fluents = compiler.compile_cpfs(scope)
            next_state_fluents = dict(next_state_fluents)
            for shape, name in zip(state_size, next_state_fluent_ordering):
                actual = list(shape)
                expected = next_state_fluents[name].shape.as_list()
                if expected == []:
                    expected = [1]
                self.assertListEqual(actual, expected)

    def test_action_size(self):
        compilers = [self.compiler1, self.compiler2]
        for compiler in compilers:
            action_size = compiler.action_size
            default_action_fluents = dict(compiler.default_action_fluents)
            action_fluent_ordering = compiler.action_fluent_ordering
            self.assertIsInstance(action_size, tuple)
            self.assertEqual(len(action_size), len(default_action_fluents))
            self.assertEqual(len(action_size), len(action_fluent_ordering))
            for shape in action_size:
                self.assertIsInstance(shape, tuple)

    def test_interm_size(self):
        compilers = [self.compiler1, self.compiler2]
        for compiler in compilers:
            interm_size = compiler.interm_size
            interm_ordering = compiler.interm_fluent_ordering
            self.assertIsInstance(interm_size, tuple)
            self.assertEqual(len(interm_size), len(interm_ordering))
            for shape in interm_size:
                self.assertIsInstance(shape, tuple)

    def test_state_dtype(self):
        compilers = [self.compiler1, self.compiler2]
        for compiler in compilers:
            state_dtype = compiler.state_dtype
            initial_state_fluents = compiler.initial_state_fluents
            state_fluent_ordering = compiler.state_fluent_ordering
            self.assertIsInstance(state_dtype, tuple)
            self.assertEqual(len(state_dtype), len(initial_state_fluents))
            self.assertEqual(len(state_dtype), len(state_fluent_ordering))
            for i, dtype in enumerate(state_dtype):
                self.assertIsInstance(dtype, tf.DType)
                self.assertEqual(dtype, initial_state_fluents[i][1].dtype)

    def test_interm_dtype(self):
        compilers = [self.compiler1, self.compiler2]
        for compiler in compilers:
            interm_dtype = compiler.interm_dtype
            interm_ordering = compiler.interm_fluent_ordering
            self.assertIsInstance(interm_dtype, tuple)
            self.assertEqual(len(interm_dtype), len(interm_ordering))
            for i, dtype in enumerate(interm_dtype):
                self.assertIsInstance(dtype, tf.DType)

    def test_action_dtype(self):
        compilers = [self.compiler1, self.compiler2]
        for compiler in compilers:
            action_dtype = compiler.action_dtype
            default_action_fluents = compiler.default_action_fluents
            action_fluent_ordering = compiler.action_fluent_ordering
            self.assertIsInstance(action_dtype, tuple)
            self.assertEqual(len(action_dtype), len(default_action_fluents))
            self.assertEqual(len(action_dtype), len(action_fluent_ordering))
            for i, dtype in enumerate(action_dtype):
                self.assertIsInstance(dtype, tf.DType)
                self.assertEqual(dtype, default_action_fluents[i][1].dtype)

    def test_state_scope(self):
        compilers = [self.compiler1, self.compiler2]
        for compiler in compilers:
            fluents = compiler.initial_state_fluents
            scope = dict(fluents)
            self.assertEqual(len(fluents), len(scope))
            for i, name in enumerate(compiler.state_fluent_ordering):
                self.assertIs(scope[name], fluents[i][1])

    def test_state_scope(self):
        compilers = [self.compiler1, self.compiler2]
        for compiler in compilers:
            fluents = compiler.default_action_fluents
            scope = dict(fluents)
            self.assertEqual(len(fluents), len(scope))
            for i, name in enumerate(compiler.action_fluent_ordering):
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
            sf = dict(compiler.initial_state_fluents)
            af = dict(compiler.default_action_fluents)
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
            sf = dict(compiler.initial_state_fluents)
            af = dict(compiler.default_action_fluents)
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
        compilers = [self.compiler1, self.compiler2]
        for compiler in compilers:
            nf = compiler.non_fluents_scope()
            sf = dict(compiler.initial_state_fluents)
            af = dict(compiler.default_action_fluents)
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
        compilers = [self.compiler1, self.compiler2]
        expected = [
            ['evaporated/1', 'overflow/1', 'rainfall/1', 'inflow/1'],
            []
        ]
        for compiler, fluents in zip(compilers, expected):
            nf = compiler.non_fluents_scope()
            sf = dict(compiler.initial_state_fluents)
            af = dict(compiler.default_action_fluents)
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
        compilers = [self.compiler1, self.compiler2]
        for compiler in compilers:
            nf = compiler.non_fluents_scope()
            sf = dict(compiler.initial_state_fluents)
            af = dict(compiler.default_action_fluents)
            scope = { **nf, **sf, **af }
            interm_fluents, next_state_fluents = compiler.compile_cpfs(scope)
            scope.update(dict(next_state_fluents))
            reward = compiler.compile_reward(scope)
            self.assertIsInstance(reward, TensorFluent)
            self.assertEqual(reward.shape.as_list(), [1])
