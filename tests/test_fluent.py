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


from rddl2tf.fluent import TensorFluent

import numpy as np
import tensorflow as tf

import unittest


class TestTensorFluent(unittest.TestCase):

    def setUp(self):
        self.batch_size = 8
        tf.reset_default_graph()
        self.zero = TensorFluent.constant(0.0)
        self.one = TensorFluent.constant(1)

    def test_constant_fluent(self):
        self._test_op_name(self.zero, r'^Const')
        self._test_op_name(self.one, r'^Const')
        self._test_fluent(self.zero, tf.float32, [], [], False)
        self._test_fluent(self.one, tf.float32, [], [], False)

    def test_normal_fluent(self):
        mean = self.zero
        variance = self.one
        dist, fluent = TensorFluent.Normal(mean, variance, self.batch_size)
        self.assertIsInstance(dist, tf.distributions.Normal)
        self._test_op_name(fluent, r'^Normal[^/]*/sample')
        self._test_fluent(fluent, tf.float32, [self.batch_size], [], True)

    def test_stop_gradient(self):
        mean = self.zero
        variance = self.one
        _, sample = TensorFluent.Normal(mean, variance, self.batch_size)
        fluent = TensorFluent.stop_gradient(sample)
        self._test_fluent(fluent, sample.dtype, sample.shape.as_list(), sample.scope.as_list(), sample.batch)
        self._test_op_name(fluent, r'^StopGradient')

        grad_before, = tf.gradients(ys=tf.reduce_sum(sample.tensor), xs=mean.tensor)
        grad_after, = tf.gradients(ys=tf.reduce_sum(fluent.tensor), xs=mean.tensor)

        self.assertIsInstance(grad_before, tf.Tensor)
        self.assertIsNone(grad_after)

        with tf.Session() as sess:
            f1, f2 = sess.run([sample.tensor, fluent.tensor])
            self.assertListEqual(list(f1), list(f2))

            g1 = sess.run(grad_before)
            self.assertEqual(g1, self.batch_size)

    def test_stop_batch_gradient(self):
        mean = TensorFluent(tf.zeros([self.batch_size]), [], True)
        variance = TensorFluent(tf.ones([self.batch_size]), [], True)
        _, sample = TensorFluent.Normal(mean, variance)

        stop_batch = tf.cast(tf.distributions.Bernoulli(probs=0.5).sample(self.batch_size), tf.bool)

        fluent = TensorFluent.stop_batch_gradient(sample, stop_batch)
        self._test_fluent(fluent, sample.dtype, sample.shape.as_list(), sample.scope.as_list(), sample.batch)
        self._test_op_name(fluent, r'^Select')

        grad_before, = tf.gradients(ys=tf.reduce_sum(sample.tensor), xs=mean.tensor)
        grad_after, = tf.gradients(ys=tf.reduce_sum(fluent.tensor), xs=mean.tensor)

        self.assertIsInstance(grad_before, tf.Tensor)
        self.assertIsInstance(grad_after, tf.Tensor)

        with tf.Session() as sess:
            f1, f2 = sess.run([sample.tensor, fluent.tensor])
            self.assertListEqual(list(f1), list(f2))

            g1, g2, stop = sess.run([grad_before, grad_after, stop_batch])
            self.assertTrue(all(g1 == np.ones([self.batch_size])))
            self.assertTrue(all(g2 == (~stop).astype(np.float32)))


    def _test_fluent(self, fluent, dtype, shape, scope, batch):
        self.assertIsInstance(fluent, TensorFluent)
        self.assertEqual(fluent.dtype, dtype)
        self.assertListEqual(fluent.shape.as_list(), shape)
        self.assertListEqual(fluent.scope.as_list(), scope)
        self.assertEqual(fluent.batch, batch)

    def _test_op_name(self, fluent, regex):
        self.assertRegex(fluent.tensor.op.name, regex, fluent.tensor)
