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
import unittest

import rddlgym

from pyrddl.expr import Expression
from pyrddl import utils

from rddl2tf.compilers.modes.default import DefaultCompiler
from rddl2tf.compilers.initializer import CompilationInitializer


class TestCompilationInitializer(unittest.TestCase):

    def setUp(self):
        self.batch_size = 64

        self.rddl1 = rddlgym.make('Reservoir-8', mode=rddlgym.AST)
        self.initializer1 = CompilationInitializer(self.rddl1)

        self.rddl2 = rddlgym.make('Mars_Rover', mode=rddlgym.AST)
        self.initializer2 = CompilationInitializer(self.rddl2)

        self.rddl3 = rddlgym.make('HVAC-v1', mode=rddlgym.AST)
        self.initializer3 = CompilationInitializer(self.rddl3)

        self.rddl4 = rddlgym.make('CrossingTraffic-10', mode=rddlgym.AST)
        self.initializer4 = CompilationInitializer(self.rddl4)

        self.rddl5 = rddlgym.make('GameOfLife-10', mode=rddlgym.AST)
        self.initializer5 = CompilationInitializer(self.rddl5)

        self.rddl6 = rddlgym.make('CarParking-v1', mode=rddlgym.AST)
        self.initializer6 = CompilationInitializer(self.rddl6)

        self.rddl7 = rddlgym.make('Navigation-v3', mode=rddlgym.AST)
        self.initializer7 = CompilationInitializer(self.rddl7)

    def test_initialize_non_fluents(self):
        non_fluents = self.initializer1._initialize_non_fluents()

        pvariables = {
            'MAX_RES_CAP/1': { 'shape': (8,), 'range_type': 'real' },
            'UPPER_BOUND/1': { 'shape': (8,), 'range_type': 'real' },
            'LOWER_BOUND/1': { 'shape': (8,), 'range_type': 'real' },
            'RAIN_SHAPE/1': { 'shape': (8,), 'range_type': 'real' },
            'RAIN_SCALE/1': { 'shape': (8,), 'range_type': 'real' },
            'DOWNSTREAM/2': { 'shape': (8,8), 'range_type': 'bool' },
            'SINK_RES/1': { 'shape': (8,), 'range_type': 'bool' },
            'MAX_WATER_EVAP_FRAC_PER_TIME_UNIT/0': { 'shape': (), 'range_type': 'real' },
            'LOW_PENALTY/1': { 'shape': (8,), 'range_type': 'real' },
            'HIGH_PENALTY/1': { 'shape': (8,), 'range_type': 'real' }
        }

        initializers = {
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

        self._test_initialized_pvariables(non_fluents, pvariables, initializers)

    def test_initialize_initial_state_fluents(self):
        initial_state_fluents = self.initializer1._initialize_initial_state_fluents()

        pvariables = {
            'rlevel/1': { 'shape': (8,) , 'range_type': 'real' }
        }

        initializers = {
            'rlevel/1': [75., 50., 50., 50., 50., 50., 50., 50.]
        }

        self._test_initialized_pvariables(initial_state_fluents, pvariables, initializers)

    def test_initialize_default_action_fluents(self):
        default_action_fluents = self.initializer1._initialize_default_action_fluents()

        pvariables = {
            'outflow/1': { 'shape': (8,) , 'range_type': 'real' }
        }

        initializers = {
            'outflow/1': [0., 0., 0., 0., 0., 0., 0., 0.]
        }

        self._test_initialized_pvariables(default_action_fluents, pvariables, initializers)

    def _test_initialized_pvariables(self, fluents, pvariables, initializers):
        self.assertEqual(len(fluents), len(pvariables))
        self.assertEqual(len(fluents), len(initializers))

        for name, range_type, value in fluents:
            self.assertIn(name, pvariables)
            self.assertIn(name, initializers)
            self.assertIsInstance(value, np.ndarray)
            self.assertEqual(range_type, pvariables[name]['range_type'])
            self.assertEqual(value.shape, pvariables[name]['shape'])
            self.assertTrue(np.allclose(value, np.array(initializers[name])))
