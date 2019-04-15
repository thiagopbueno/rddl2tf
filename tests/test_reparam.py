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


from pyrddl.expr import Expression

from rddl2tf.fluent import TensorFluent
from rddl2tf.reparam import get_reparameterization

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

    def test_standard_normal(self):
        noise = get_reparameterization(self.z, scope={})
        self.assertListEqual(noise, [('Normal', [1])])

    def test_multivariate_normal(self):
        scope = {
            'mu/1': TensorFluent(tf.zeros(32), scope=['?x']),
            'sigma/1': TensorFluent(tf.ones(32), scope=['?x'])
        }

        noise1 = get_reparameterization(self.x1, scope=scope)
        self.assertListEqual(noise1, [('Normal', [32])])

        noise2 = get_reparameterization(self.x2, scope=scope)
        self.assertListEqual(noise2, [('Normal', [32])])

        noise3 = get_reparameterization(self.x3, scope=scope)
        self.assertListEqual(noise3, [('Normal', [32])])

        noise4 = get_reparameterization(self.x4, scope=scope)
        self.assertListEqual(noise4, [('Normal', [1]), ('Normal', [1])])

        noise5 = get_reparameterization(self.x5, scope=scope)
        self.assertListEqual(noise5, [('Normal', [32]), ('Normal', [32])])

        noise6 = get_reparameterization(self.x6, scope=scope)
        self.assertListEqual(noise6, [('Normal', [32]), ('Normal', [32])])

        noise7 = get_reparameterization(self.x7, scope=scope)
        self.assertListEqual(noise7, [('Normal', [1]), ('Normal', [1]), ('Normal', [1])])

        noise8 = get_reparameterization(self.x8, scope=scope)
        self.assertListEqual(noise8, [('Normal', [1]), ('Normal', [1]), ('Normal', [32])])

        noise9 = get_reparameterization(self.x9, scope=scope)
        self.assertListEqual(noise9, [('Normal', [32]), ('Normal', [1]), ('Normal', [1]), ('Normal', [32])])

    def test_batch_normal(self):
        scope = {
            'mu/1': TensorFluent(tf.zeros((64, 16)), scope=['?x'], batch=True),
            'sigma/1': TensorFluent(tf.ones((64, 16)), scope=['?x'], batch=True)
        }

        noise1 = get_reparameterization(self.x1, scope=scope)
        self.assertListEqual(noise1, [('Normal', [64, 16])])

        noise2 = get_reparameterization(self.x2, scope=scope)
        self.assertListEqual(noise2, [('Normal', [64, 16])])

        noise3 = get_reparameterization(self.x3, scope=scope)
        self.assertListEqual(noise3, [('Normal', [64, 16])])

    def test_arithmetic(self):
        scope = {
            'mu/1': TensorFluent(tf.zeros(32), scope=['?x']),
            'sigma/1': TensorFluent(tf.ones(32), scope=['?x'])
        }

        noise = get_reparameterization(self.two, scope={})
        self.assertListEqual(noise, [])

        noise = get_reparameterization(self.z_times_z, scope={})
        self.assertListEqual(noise, [('Normal', [1]), ('Normal', [1])])

        noise = get_reparameterization(self.x2_times_x2, scope=scope)
        self.assertListEqual(noise, [('Normal', [32]), ('Normal', [32])])

        noise = get_reparameterization(self.mu_plus_z, scope=scope)
        self.assertListEqual(noise, [('Normal', [1])])

        noise = get_reparameterization(self.z_plus_mu, scope=scope)
        self.assertListEqual(noise, [('Normal', [1])])

        noise = get_reparameterization(self.mu_plus_x2, scope=scope)
        self.assertListEqual(noise, [('Normal', [32])])

        noise = get_reparameterization(self.x2_plus_mu, scope=scope)
        self.assertListEqual(noise, [('Normal', [32])])

        noise = get_reparameterization(self.x1_plus_z, scope=scope)
        self.assertListEqual(noise, [('Normal', [32]), ('Normal', [1])])

        noise = get_reparameterization(self.z_plus_x1, scope=scope)
        self.assertListEqual(noise, [('Normal', [1]), ('Normal', [32])])

    def test_function(self):
        scope = {
            'mu/1': TensorFluent(tf.zeros(24), scope=['?x']),
            'sigma/1': TensorFluent(tf.ones(24), scope=['?x'])
        }

        noise = get_reparameterization(self.exp_2, scope=scope)
        self.assertListEqual(noise, [])

        noise = get_reparameterization(self.exp_z, scope=scope)
        self.assertListEqual(noise, [('Normal', [1])])

        noise = get_reparameterization(self.exp_x1, scope=scope)
        self.assertListEqual(noise, [('Normal', [24])])

        noise = get_reparameterization(self.y1, scope=scope)
        self.assertListEqual(noise, [('Normal', [1]), ('Normal', [1])])

        noise = get_reparameterization(self.y2, scope=scope)
        self.assertListEqual(noise, [('Normal', [1]), ('Normal', [24])])

        noise = get_reparameterization(self.y3, scope=scope)
        self.assertListEqual(noise, [('Normal', [24]), ('Normal', [24])])
