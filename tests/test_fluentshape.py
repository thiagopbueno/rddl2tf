from rddl2tf.fluentshape import TensorFluentShape

import unittest


class TestTensorFluentShape(unittest.TestCase):

    def test_broadcast(self):
        tests = [
            (TensorFluentShape([], False), TensorFluentShape([], False), None, None),
            (TensorFluentShape([8], False), TensorFluentShape([], False), None, None),
            (TensorFluentShape([], False), TensorFluentShape([8], False), None, None),
            (TensorFluentShape([8, 8], False), TensorFluentShape([8], False), None, None),
            (TensorFluentShape([8], False), TensorFluentShape([8, 8], False), None, None),
            (TensorFluentShape([100], True), TensorFluentShape([100], True), None, None),
            (TensorFluentShape([100, 8], True), TensorFluentShape([100], True), None, [100, 1]),
            (TensorFluentShape([100], True), TensorFluentShape([100, 8], True), [100, 1], None),
            (TensorFluentShape([100, 8, 8], True), TensorFluentShape([100], True), None, [100, 1, 1]),
            (TensorFluentShape([100], True), TensorFluentShape([100, 8, 8], True), [100, 1, 1], None),
            (TensorFluentShape([100, 8, 8], True), TensorFluentShape([100, 8], True), None, [100, 1, 8]),
            (TensorFluentShape([100, 8], True), TensorFluentShape([100, 8, 8], True), [100, 1, 8], None),
            (TensorFluentShape([100], True), TensorFluentShape([], False), None, None),
            (TensorFluentShape([], False), TensorFluentShape([], True), None, None),
            (TensorFluentShape([100], True), TensorFluentShape([], False), None, None),
            (TensorFluentShape([100], True), TensorFluentShape([8], False), [100, 1], None),
            (TensorFluentShape([8], False), TensorFluentShape([100], True), None, [100, 1]),
            (TensorFluentShape([100], True), TensorFluentShape([8, 7], False), [100, 1, 1], None),
            (TensorFluentShape([8, 7], False), TensorFluentShape([100], True), None, [100, 1, 1]),
            (TensorFluentShape([100, 8], True), TensorFluentShape([], False), None, None),
            (TensorFluentShape([], False), TensorFluentShape([100, 8], True), None, None),
            (TensorFluentShape([100, 8], True), TensorFluentShape([8], False), None, None),
            (TensorFluentShape([8], False), TensorFluentShape([100, 8], True), None, None),
            (TensorFluentShape([100, 8, 7], True), TensorFluentShape([7], False), None, [1, 7]),
            (TensorFluentShape([7], False), TensorFluentShape([100, 8, 7], True), [1, 7], None),
            (TensorFluentShape([100, 7, 8], True), TensorFluentShape([7, 8], False), None, None),
            (TensorFluentShape([7, 8], False), TensorFluentShape([100, 7, 8], True), None, None),
            (TensorFluentShape([8, 8], False), TensorFluentShape([100, 8], True), None, [100, 1, 8]),
            (TensorFluentShape([100, 8], True), TensorFluentShape([8, 8], False), [100, 1, 8], None),
            (TensorFluentShape([2, 2], False), TensorFluentShape([1, 2], True), None, [1, 1, 2]),
            (TensorFluentShape([1, 2], True), TensorFluentShape([2, 2], False), [1, 1, 2], None),
        ]

        for s1, s2, ss1, ss2 in tests:
            reshape1, reshape2 = TensorFluentShape.broadcast(s1, s2)
            if ss1 is None:
                self.assertIsNone(reshape1)
            else:
                self.assertListEqual(reshape1, ss1)

            if ss2 is None:
                self.assertIsNone(reshape2)
            else:
                self.assertListEqual(reshape2, ss2)
