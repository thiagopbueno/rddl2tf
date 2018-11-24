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
        self.batch_size = 32
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

    def test_if_then_else(self):
        # if (abs[x - gx] > 0.0) then
        #     (u * u) * 0.01
        # else
        #     0.0
        x = tf.random_normal([self.batch_size, 1])
        x = TensorFluent(x, scope=[], batch=True)
        gx = tf.random_normal([])
        gx = TensorFluent(gx, scope=[], batch=False)
        u = tf.random_normal([self.batch_size,])
        u = TensorFluent(u, scope=[], batch=True)
        c0 = TensorFluent.constant(0.01)
        cond = TensorFluent.abs(x - gx) > self.zero
        true_case = (u * u) * c0
        false_case = self.zero
        ite = TensorFluent.if_then_else(cond, true_case, false_case)
        self._test_fluent(ite, tf.float32, [self.batch_size, 1], [], True)

        # if ((u * u) * 0.01 > 1.0) then
        #     abs[x - gx]
        # else
        #     0.0
        cond = ((u * u) * c0) > self.one
        true_case = TensorFluent.abs(x - gx)
        false_case = self.zero
        ite = TensorFluent.if_then_else(cond, true_case, false_case)
        self._test_fluent(ite, tf.float32, [self.batch_size, 1], [], True)

        # if (gx < 1.0) then
        #     1.0 + u
        # else
        #     0.05 * (u * u)
        c1 = TensorFluent.constant(0.05)
        cond = (gx < self.one)
        true_case = self.one + u
        false_case = c1 * (u * u)
        ite = TensorFluent.if_then_else(cond, true_case, false_case)
        self._test_fluent(ite, tf.float32, [self.batch_size], [], True)

        # if (gx < 0.0) then
        #     1.0
        # else
        #     abs[x - gx]
        cond = (gx < self.zero)
        true_case = self.one
        false_case = TensorFluent.abs(x - gx)
        ite = TensorFluent.if_then_else(cond, true_case, false_case)
        self._test_fluent(ite, tf.float32, [self.batch_size, 1], [], True)

        # if (visited(?wpt)) then
        #     (1.0)
        # else
        #     (0.0)
        visited = tf.stack([[True, False, True]] * self.batch_size, axis=0)
        visited = TensorFluent(visited, scope=['?wpt'], batch=True)
        cond = visited
        true_case = self.one
        false_case = self.zero
        ite = TensorFluent.if_then_else(cond, true_case, false_case)
        self._test_fluent(ite, tf.float32, [self.batch_size, 3], ['?wpt'], True)

        # if (rlevel'(?r)<=LOWER_BOUND(?r)) then
        #     LOW_PENALTY(?r)*(LOWER_BOUND(?r)-rlevel'(?r))
        # else
        #     HIGH_PENALTY(?r)*(rlevel'(?r)-UPPER_BOUND(?r))];
        r_size = 8
        low_penalty = TensorFluent.constant(-5.0)
        high_penalty = TensorFluent.constant(-10.0)
        lower_bound = tf.random_normal([r_size], mean=50.0, stddev=10.0)
        lower_bound = TensorFluent(lower_bound, scope=['?r'], batch=False)
        upper_bound = tf.random_normal([r_size], mean=200.0, stddev=10.0)
        upper_bound = TensorFluent(upper_bound, scope=['?r'], batch=False)
        rlevel = tf.random_normal([self.batch_size, r_size], mean=100.0, stddev=10.0)
        rlevel = TensorFluent(rlevel, scope=['?r'], batch=True)
        cond = (rlevel <= lower_bound)
        true_case = low_penalty * (lower_bound - rlevel)
        false_case = high_penalty * (rlevel - upper_bound)
        ite = TensorFluent.if_then_else(cond, true_case, false_case)
        self._test_fluent(ite, tf.float32, [self.batch_size, r_size], ['?r'], True)

        # if (snapPicture) then
        #    (time + 0.25)
        # else
        #    (time + abs[xMove] + abs[yMove]);
        time = tf.fill([self.batch_size, 1], 12.0)
        time = TensorFluent(time, scope=[], batch=True)
        snapPicture = tf.fill([self.batch_size], True)
        snapPicture = TensorFluent(snapPicture, scope=[], batch=True)
        xMove = tf.fill([self.batch_size], 10.0)
        xMove = TensorFluent(xMove, scope=[], batch=True)
        yMove = tf.fill([self.batch_size], -3.0)
        yMove = TensorFluent(yMove, scope=[], batch=True)
        cond = snapPicture
        true_case = (time + TensorFluent.constant(0.25))
        false_case = (time + TensorFluent.abs(xMove) + TensorFluent.abs(yMove))
        ite = TensorFluent.if_then_else(cond, true_case, false_case)
        self._test_fluent(ite, tf.float32, [self.batch_size, 1], [], True)

        # if (alive(?x,?y)) then
        #     Bernoulli(1.0 - NOISE-PROB(?x,?y))
        # else
        #     Bernoulli(NOISE-PROB(?x,?y));
        x_size, y_size = 3, 3
        alive = tf.distributions.Bernoulli(probs=0.7, dtype=tf.bool).\
            sample([self.batch_size, x_size, y_size])
        alive = TensorFluent(alive, scope=['?x', '?y'], batch=True)
        noise_prob = tf.random_uniform([x_size, y_size], dtype=tf.float32)
        noise_prob = TensorFluent(noise_prob, scope=['?x', '?y'], batch=False)
        cond = alive
        true_case = TensorFluent.Bernoulli(self.one - noise_prob, batch_size=self.batch_size)[1]
        false_case = TensorFluent.Bernoulli(noise_prob, batch_size=self.batch_size)[1]
        ite = TensorFluent.if_then_else(cond, true_case, false_case)
        self._test_fluent(ite, tf.bool, [self.batch_size, 3, 3], ['?x', '?y'], True)

        # if (reboot(?x)) then
        #     KronDelta(true)
        # else
        #     ~running(?x);
        x_size = 128
        mean = tf.random_uniform([x_size], dtype=tf.float32)
        mean = TensorFluent(mean, scope=['?x'])
        running = TensorFluent.Bernoulli(mean, batch_size=self.batch_size)[1]
        reboot = TensorFluent.Bernoulli(self.one - mean, batch_size=self.batch_size)[1]
        cond = reboot
        true_case = TensorFluent.constant(True, dtype=tf.bool)
        false_case = ~running
        ite = TensorFluent.if_then_else(cond, true_case, false_case)
        self._test_fluent(ite, tf.bool, [self.batch_size, x_size], ['?x'], True)

        # if (running(?x)) then
        #     Bernoulli(.45 + .5 * running(?x))
        # else
        #     Bernoulli(REBOOT-PROB);
        c0 = TensorFluent.constant(0.45)
        c1 = TensorFluent.constant(0.5)
        reboot_prob = TensorFluent.constant(0.15)
        cond = running
        true_case = TensorFluent.Bernoulli(c0 + c1 * running)[1]
        false_case = TensorFluent.Bernoulli(reboot_prob)[1]
        ite = TensorFluent.if_then_else(cond, true_case, false_case)
        self._test_fluent(ite, tf.bool, [self.batch_size, x_size], ['?x'], True)

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
