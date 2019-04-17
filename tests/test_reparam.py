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

from rddl2tf.fluent import TensorFluent
from rddl2tf.fluentshape import TensorFluentShape

from rddl2tf.reparam import *

import tensorflow as tf
import unittest


class TestReparameterization(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.zero = Expression(('number', 0.0))
        cls.one = Expression(('number', 1.0))
        cls.two = Expression(('+', (cls.one, cls.one)))

        cls.z = Expression(('randomvar', ('Normal', (cls.zero, cls.one))))
        cls.mu = Expression(('pvar_expr', ('mu', ['?x'])))
        cls.sigma = Expression(('pvar_expr', ('sigma', ['?x'])))

        cls.x1 = Expression(('randomvar', ('Normal', (cls.mu, cls.one))))
        cls.x2 = Expression(('randomvar', ('Normal', (cls.zero, cls.sigma))))
        cls.x3 = Expression(('randomvar', ('Normal', (cls.mu, cls.sigma))))

        cls.x4 = Expression(('randomvar', ('Normal', (cls.z, cls.one))))
        cls.x5 = Expression(('randomvar', ('Normal', (cls.x1, cls.one))))
        cls.x6 = Expression(('randomvar', ('Normal', (cls.x1, cls.sigma))))

        cls.mu_plus_z = Expression(('+', (cls.mu, cls.z)))
        cls.z_plus_mu = Expression(('+', (cls.z, cls.mu)))

        cls.mu_plus_x2 = Expression(('+', (cls.mu, cls.x2)))
        cls.x2_plus_mu = Expression(('+', (cls.x2, cls.mu)))

        cls.x1_plus_z = Expression(('+', (cls.x1, cls.z)))
        cls.z_plus_x1 = Expression(('+', (cls.z, cls.x1)))

        cls.z_times_z = Expression(('*', (cls.z, cls.z)))
        cls.x2_times_x2 = Expression(('*', (cls.x2, cls.x2)))

        cls.x7 = Expression(('randomvar', ('Normal', (cls.one, cls.z_times_z))))
        cls.x8 = Expression(('randomvar', ('Normal', (cls.mu, cls.z_times_z))))
        cls.x9 = Expression(('randomvar', ('Normal', (cls.x3, cls.z_times_z))))

        cls.exp_2 = Expression(('func', ('exp', [cls.two])))
        cls.exp_z = Expression(('func', ('exp', [cls.z])))
        cls.exp_x1 = Expression(('func', ('exp', [cls.x1])))

        cls.y1 = Expression(('randomvar', ('Normal', (cls.one, cls.exp_z))))
        cls.y2 = Expression(('randomvar', ('Normal', (cls.mu, cls.exp_z))))
        cls.y3 = Expression(('randomvar', ('Normal', (cls.mu, cls.exp_x1))))

    def setUp(self):
        self.compiler = rddlgym.make('Navigation-v2', mode=rddlgym.SCG)
        self.compiler.batch_mode_on()

    def test_get_shape_scope(self):
        scope = get_reparameterization_shape_scope(self.compiler.rddl)
        self.assertEqual(len(scope), len(self.compiler.rddl.fluent_table))
        for name, fluent_shape in scope.items():
            self.assertIsInstance(fluent_shape, TensorFluentShape)
            self.assertListEqual(fluent_shape.as_list(), list(self.compiler.rddl.fluent_table[name][1]))

    def test_get_state_cpfs_reparameterization(self):
        noise = get_state_cpfs_reparameterization(self.compiler.rddl)
        self._test_cpf_reparameterization_dist(noise, [("location'/1", [('Normal', [2])])])

    def test_get_intermediate_cpfs_reparameterization(self):
        noise = get_intermediate_cpfs_reparameterization(self.compiler.rddl)
        self._test_cpf_reparameterization_dist(noise, [('distance/1', []), ('deceleration/1', [])])

    def test_standard_normal(self):
        noise = get_reparameterization(self.z, scope={})
        self._test_reparameterization_dist(noise, [('Normal', [1])])
        self._test_reparameterized_expression(self.z, scope={}, noise=noise, name='noise')

    def test_multivariate_normal(self):
        with self.compiler.graph.as_default():
            shape_scope = {
                'mu/1': TensorFluentShape([32], batch=False),
                'sigma/1': TensorFluentShape([32], batch=False)
            }

            scope = {
                'mu/1': TensorFluent(tf.zeros([32]), scope=['?x'], batch=False),
                'sigma/1': TensorFluent(tf.ones([32]), scope=['?x'], batch=False)
            }

        noise1 = get_reparameterization(self.x1, scope=shape_scope)
        self._test_reparameterization_dist(noise1, [('Normal', [32])])
        self._test_reparameterized_expression(self.x1, scope=scope, noise=noise1, name='noise1')

        noise2 = get_reparameterization(self.x2, scope=shape_scope)
        self._test_reparameterization_dist(noise2, [('Normal', [32])])
        self._test_reparameterized_expression(self.x2, scope=scope, noise=noise2, name='noise2')

        noise3 = get_reparameterization(self.x3, scope=shape_scope)
        self._test_reparameterization_dist(noise3, [('Normal', [32])])
        self._test_reparameterized_expression(self.x3, scope=scope, noise=noise3, name='noise3')

        noise4 = get_reparameterization(self.x4, scope=shape_scope)
        self._test_reparameterization_dist(noise4, [('Normal', [1]), ('Normal', [1])])
        self._test_reparameterized_expression(self.x4, scope=scope, noise=noise4, name='noise4')

        noise5 = get_reparameterization(self.x5, scope=shape_scope)
        self._test_reparameterization_dist(noise5, [('Normal', [32]), ('Normal', [32])])
        self._test_reparameterized_expression(self.x5, scope=scope, noise=noise5, name='noise5')

        noise6 = get_reparameterization(self.x6, scope=shape_scope)
        self._test_reparameterization_dist(noise6, [('Normal', [32]), ('Normal', [32])])
        self._test_reparameterized_expression(self.x6, scope=scope, noise=noise6, name='noise6')

        noise7 = get_reparameterization(self.x7, scope=shape_scope)
        self._test_reparameterization_dist(noise7, [('Normal', [1]), ('Normal', [1]), ('Normal', [1])])
        self._test_reparameterized_expression(self.x7, scope=scope, noise=noise7, name='noise7')

        noise8 = get_reparameterization(self.x8, scope=shape_scope)
        self._test_reparameterization_dist(noise8, [('Normal', [1]), ('Normal', [1]), ('Normal', [32])])
        self._test_reparameterized_expression(self.x8, scope=scope, noise=noise8, name='noise8')

        noise9 = get_reparameterization(self.x9, scope=shape_scope)
        self._test_reparameterization_dist(noise9, [('Normal', [32]), ('Normal', [1]), ('Normal', [1]), ('Normal', [32])])
        self._test_reparameterized_expression(self.x9, scope=scope, noise=noise9, name='noise9')

    def test_batch_normal(self):
        with self.compiler.graph.as_default():
            shape_scope = {
                'mu/1': TensorFluentShape((64, 16), batch=True),
                'sigma/1': TensorFluentShape((64, 16), batch=True)
            }

            scope = {
                'mu/1': TensorFluent(tf.zeros([64, 16]), scope=['?x'], batch=True),
                'sigma/1': TensorFluent(tf.ones([64, 16]), scope=['?x'], batch=True)
            }

        noise1 = get_reparameterization(self.x1, scope=shape_scope)
        self._test_reparameterization_dist(noise1, [('Normal', [64, 16])])
        self._test_reparameterized_expression(self.x1, scope=scope, noise=noise1, name='noise1')

        noise2 = get_reparameterization(self.x2, scope=shape_scope)
        self._test_reparameterization_dist(noise2, [('Normal', [64, 16])])
        self._test_reparameterized_expression(self.x2, scope=scope, noise=noise2, name='noise2')

        noise3 = get_reparameterization(self.x3, scope=shape_scope)
        self._test_reparameterization_dist(noise3, [('Normal', [64, 16])])
        self._test_reparameterized_expression(self.x3, scope=scope, noise=noise3, name='noise3')

    def test_arithmetic(self):
        with self.compiler.graph.as_default():
            shape_scope = {
                'mu/1': TensorFluentShape([32], batch=False),
                'sigma/1': TensorFluentShape([32], batch=False)
            }

            scope = {
                'mu/1': TensorFluent(tf.zeros([32]), scope=['?x'], batch=False),
                'sigma/1': TensorFluent(tf.ones([32]), scope=['?x'], batch=False)
            }

        noise1 = get_reparameterization(self.two, scope={})
        self._test_reparameterization_dist(noise1, [])
        self._test_reparameterized_expression(self.two, scope={}, noise=noise1, name='noise1')

        noise2 = get_reparameterization(self.z_times_z, scope={})
        self._test_reparameterization_dist(noise2, [('Normal', [1]), ('Normal', [1])])
        self._test_reparameterized_expression(self.z_times_z, scope={}, noise=noise2, name='noise2')

        noise3 = get_reparameterization(self.x2_times_x2, scope=shape_scope)
        self._test_reparameterization_dist(noise3, [('Normal', [32]), ('Normal', [32])])
        self._test_reparameterized_expression(self.x2_times_x2, scope=scope, noise=noise3, name='noise3')

        noise4 = get_reparameterization(self.mu_plus_z, scope=shape_scope)
        self._test_reparameterization_dist(noise4, [('Normal', [1])])
        self._test_reparameterized_expression(self.mu_plus_z, scope=scope, noise=noise4, name='noise4')

        noise5 = get_reparameterization(self.z_plus_mu, scope=shape_scope)
        self._test_reparameterization_dist(noise5, [('Normal', [1])])
        self._test_reparameterized_expression(self.z_plus_mu, scope=scope, noise=noise5, name='noise5')

        noise6 = get_reparameterization(self.mu_plus_x2, scope=shape_scope)
        self._test_reparameterization_dist(noise6, [('Normal', [32])])
        self._test_reparameterized_expression(self.mu_plus_x2, scope=scope, noise=noise6, name='noise6')

        noise7 = get_reparameterization(self.x2_plus_mu, scope=shape_scope)
        self._test_reparameterization_dist(noise7, [('Normal', [32])])
        self._test_reparameterized_expression(self.x2_plus_mu, scope=scope, noise=noise7, name='noise7')

        noise8 = get_reparameterization(self.x1_plus_z, scope=shape_scope)
        self._test_reparameterization_dist(noise8, [('Normal', [32]), ('Normal', [1])])
        self._test_reparameterized_expression(self.x1_plus_z, scope=scope, noise=noise8, name='noise8')

        noise9 = get_reparameterization(self.z_plus_x1, scope=shape_scope)
        self._test_reparameterization_dist(noise9, [('Normal', [1]), ('Normal', [32])])
        self._test_reparameterized_expression(self.z_plus_x1, scope=scope, noise=noise9, name='noise9')

    def test_function(self):
        with self.compiler.graph.as_default():
            shape_scope = {
                'mu/1': TensorFluentShape([24], batch=False),
                'sigma/1': TensorFluentShape([24], batch=False)
            }

            scope = {
                'mu/1': TensorFluent(tf.zeros([24]), scope=['?x'], batch=False),
                'sigma/1': TensorFluent(tf.ones([24]), scope=['?x'], batch=False)
            }

        noise1 = get_reparameterization(self.exp_2, scope=shape_scope)
        self._test_reparameterization_dist(noise1, [])
        self._test_reparameterized_expression(self.exp_2, scope=scope, noise=noise1, name='noise1')

        noise2 = get_reparameterization(self.exp_z, scope=shape_scope)
        self._test_reparameterization_dist(noise2, [('Normal', [1])])
        self._test_reparameterized_expression(self.exp_z, scope=scope, noise=noise2, name='noise2')

        noise3 = get_reparameterization(self.exp_x1, scope=shape_scope)
        self._test_reparameterization_dist(noise3, [('Normal', [24])])
        self._test_reparameterized_expression(self.exp_x1, scope=scope, noise=noise3, name='noise3')

        noise4 = get_reparameterization(self.y1, scope=shape_scope)
        self._test_reparameterization_dist(noise4, [('Normal', [1]), ('Normal', [1])])
        self._test_reparameterized_expression(self.y1, scope=scope, noise=noise4, name='noise4')

        noise5 = get_reparameterization(self.y2, scope=shape_scope)
        self._test_reparameterization_dist(noise5, [('Normal', [1]), ('Normal', [24])])
        self._test_reparameterized_expression(self.y2, scope=scope, noise=noise5, name='noise5')

        noise6 = get_reparameterization(self.y3, scope=shape_scope)
        self._test_reparameterization_dist(noise6, [('Normal', [24]), ('Normal', [24])])
        self._test_reparameterized_expression(self.y3, scope=scope, noise=noise6, name='noise6')

    def _test_cpf_reparameterization_dist(self, cpf_noise, reparam_map):
        for (name1, noise1), (name2, noise2) in zip(cpf_noise, reparam_map):
            self.assertEqual(name1, name2)
            self._test_reparameterization_dist(noise1, noise2)

    def _test_reparameterization_dist(self, noise, reparam_map):
        noise = [(dist.name, shape) for dist, shape in noise]
        self.assertListEqual(noise, reparam_map)

    def _test_reparameterized_expression(self, expr, scope, noise, name):
        with self.compiler.graph.as_default():
            with tf.variable_scope(name):
                noise = [tf.get_variable('noise_{}'.format(i), shape=shape)
                            for i, (_, shape) in enumerate(noise)]
                fluent = self.compiler._compile_expression(expr, scope, batch_size=10, noise=noise)
        self.assertIsInstance(fluent, TensorFluent)
        self.assertListEqual(noise, [])
