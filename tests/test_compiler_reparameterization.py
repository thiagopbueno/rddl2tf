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

# pylint: disable=missing-docstring


import pytest
import tensorflow as tf

from pyrddl.expr import Expression
import rddlgym

from rddl2tf.compilers import ReparameterizationCompiler
from rddl2tf.core.fluent import TensorFluent
from rddl2tf.core.fluentshape import TensorFluentShape


NORMAL = tf.distributions.Normal
GAMMA = tf.distributions.Gamma
UNIFORM = tf.distributions.Uniform


ZERO = Expression(("number", 0.0))
ONE = Expression(("number", 1.0))
TWO = Expression(("+", (ONE, ONE)))

Z = Expression(("randomvar", ("Normal", (ZERO, ONE))))

MU = Expression(("pvar_expr", ("mu", ["?x"])))
SIGMA = Expression(("pvar_expr", ("sigma", ["?x"])))

X1 = Expression(("randomvar", ("Normal", (MU, ONE))))
X2 = Expression(("randomvar", ("Normal", (ZERO, SIGMA))))
X3 = Expression(("randomvar", ("Normal", (MU, SIGMA))))

X4 = Expression(("randomvar", ("Normal", (Z, ONE))))
X5 = Expression(("randomvar", ("Normal", (X1, ONE))))
X6 = Expression(("randomvar", ("Normal", (X1, SIGMA))))

MU_PLUS_Z = Expression(("+", (MU, Z)))
Z_PLUS_MU = Expression(("+", (Z, MU)))

MU_PLUS_X2 = Expression(("+", (MU, X2)))
X2_PLUS_MU = Expression(("+", (X2, MU)))

X1_PLUS_Z = Expression(("+", (X1, Z)))
Z_PLUS_X1 = Expression(("+", (Z, X1)))

Z_TIMES_Z = Expression(("*", (Z, Z)))
X2_TIMES_X2 = Expression(("*", (X2, X2)))

X7 = Expression(("randomvar", ("Normal", (ONE, Z_TIMES_Z))))
X8 = Expression(("randomvar", ("Normal", (MU, Z_TIMES_Z))))
X9 = Expression(("randomvar", ("Normal", (X3, Z_TIMES_Z))))

EXP_2 = Expression(("func", ("exp", [TWO])))
EXP_Z = Expression(("func", ("exp", [Z])))
EXP_X1 = Expression(("func", ("exp", [X1])))

Y1 = Expression(("randomvar", ("Normal", (ONE, EXP_Z))))
Y2 = Expression(("randomvar", ("Normal", (MU, EXP_Z))))
Y3 = Expression(("randomvar", ("Normal", (MU, EXP_X1))))

EXP_RATE = Expression(("pvar_expr", ("rate", ["?r"])))
EXP1 = Expression(("randomvar", ("Exponential", (EXP_RATE,))))


@pytest.fixture(scope="function", params=["Navigation-v3", "Reservoir-8"])
def compiler(request):
    rddl = rddlgym.make(request.param, mode=rddlgym.AST)
    compiler_ = ReparameterizationCompiler(rddl)
    compiler_.init()
    return compiler_


def test_get_shape_scope(compiler):
    scope = compiler._get_reparameterization_shape_scope()
    assert len(scope) == len(compiler.rddl.fluent_table)
    for name, fluent_shape in scope.items():
        assert isinstance(fluent_shape, TensorFluentShape)
        assert fluent_shape.as_list() == list(compiler.rddl.fluent_table[name][1])


def test_get_state_cpfs_reparameterization(compiler):
    noise = compiler.get_state_cpfs_reparameterization()

    if compiler.rddl.domain.name == "Navigation":
        _test_cpf_reparameterization_dist(
            noise, [("location'/1", [(tf.distributions.Normal, [2])])]
        )


def test_get_intermediate_cpfs_reparameterization(compiler):
    noise = compiler.get_intermediate_cpfs_reparameterization()

    if compiler.rddl.domain.name == "Navigation":
        _test_cpf_reparameterization_dist(
            noise, [("distance/1", []), ("deceleration/1", [])]
        )


def test_standard_normal(compiler):
    noise = compiler._get_expression_reparameterization(Z, scope={})
    _test_reparameterization_dist(noise, [(tf.distributions.Normal, [1])])
    _test_reparameterized_expression(compiler, Z, scope={}, noise=noise, name="noise")


def test_multivariate_normal(compiler):
    with compiler.graph.as_default():
        shape_scope = {
            "mu/1": TensorFluentShape([32], batch=False),
            "sigma/1": TensorFluentShape([32], batch=False),
        }

        scope = {
            "mu/1": TensorFluent(tf.zeros([32]), scope=["?x"], batch=False),
            "sigma/1": TensorFluent(tf.ones([32]), scope=["?x"], batch=False),
        }

    noise1 = compiler._get_expression_reparameterization(X1, scope=shape_scope)
    _test_reparameterization_dist(noise1, [(NORMAL, [32])])
    _test_reparameterized_expression(
        compiler, X1, scope=scope, noise=noise1, name="noise1"
    )

    noise2 = compiler._get_expression_reparameterization(X2, scope=shape_scope)
    _test_reparameterization_dist(noise2, [(NORMAL, [32])])
    _test_reparameterized_expression(
        compiler, X2, scope=scope, noise=noise2, name="noise2"
    )

    noise3 = compiler._get_expression_reparameterization(X3, scope=shape_scope)
    _test_reparameterization_dist(noise3, [(NORMAL, [32])])
    _test_reparameterized_expression(
        compiler, X3, scope=scope, noise=noise3, name="noise3"
    )

    noise4 = compiler._get_expression_reparameterization(X4, scope=shape_scope)
    _test_reparameterization_dist(noise4, [(NORMAL, [1]), (NORMAL, [1])])
    _test_reparameterized_expression(
        compiler, X4, scope=scope, noise=noise4, name="noise4"
    )

    noise5 = compiler._get_expression_reparameterization(X5, scope=shape_scope)
    _test_reparameterization_dist(noise5, [(NORMAL, [32]), (NORMAL, [32])])
    _test_reparameterized_expression(
        compiler, X5, scope=scope, noise=noise5, name="noise5"
    )

    noise6 = compiler._get_expression_reparameterization(X6, scope=shape_scope)
    _test_reparameterization_dist(noise6, [(NORMAL, [32]), (NORMAL, [32])])
    _test_reparameterized_expression(
        compiler, X6, scope=scope, noise=noise6, name="noise6"
    )

    noise7 = compiler._get_expression_reparameterization(X7, scope=shape_scope)
    _test_reparameterization_dist(noise7, [(NORMAL, [1]), (NORMAL, [1]), (NORMAL, [1])])
    _test_reparameterized_expression(
        compiler, X7, scope=scope, noise=noise7, name="noise7"
    )

    noise8 = compiler._get_expression_reparameterization(X8, scope=shape_scope)
    _test_reparameterization_dist(
        noise8, [(NORMAL, [1]), (NORMAL, [1]), (NORMAL, [32])]
    )
    _test_reparameterized_expression(
        compiler, X8, scope=scope, noise=noise8, name="noise8"
    )

    noise9 = compiler._get_expression_reparameterization(X9, scope=shape_scope)
    _test_reparameterization_dist(
        noise9, [(NORMAL, [32]), (NORMAL, [1]), (NORMAL, [1]), (NORMAL, [32])]
    )
    _test_reparameterized_expression(
        compiler, X9, scope=scope, noise=noise9, name="noise9"
    )


def test_batch_normal(compiler):
    with compiler.graph.as_default():
        shape_scope = {
            "mu/1": TensorFluentShape((64, 16), batch=True),
            "sigma/1": TensorFluentShape((64, 16), batch=True),
        }

        scope = {
            "mu/1": TensorFluent(tf.zeros([64, 16]), scope=["?x"], batch=True),
            "sigma/1": TensorFluent(tf.ones([64, 16]), scope=["?x"], batch=True),
        }

    noise1 = compiler._get_expression_reparameterization(X1, scope=shape_scope)
    _test_reparameterization_dist(noise1, [(NORMAL, [64, 16])])
    _test_reparameterized_expression(
        compiler, X1, scope=scope, noise=noise1, name="noise1"
    )

    noise2 = compiler._get_expression_reparameterization(X2, scope=shape_scope)
    _test_reparameterization_dist(noise2, [(NORMAL, [64, 16])])
    _test_reparameterized_expression(
        compiler, X2, scope=scope, noise=noise2, name="noise2"
    )

    noise3 = compiler._get_expression_reparameterization(X3, scope=shape_scope)
    _test_reparameterization_dist(noise3, [(NORMAL, [64, 16])])
    _test_reparameterized_expression(
        compiler, X3, scope=scope, noise=noise3, name="noise3"
    )


def test_exponential(compiler):
    # rainfall(?r) = Exponential(RAIN_RATE(?r));
    with compiler.graph.as_default():
        shape_scope = {"rate/1": TensorFluentShape((32, 8), batch=True)}

        scope = {"rate/1": TensorFluent(tf.ones((32, 8)), scope=["?r"], batch=True)}

    noise1 = compiler._get_expression_reparameterization(EXP1, scope=shape_scope)
    _test_reparameterization_dist(noise1, [(UNIFORM, [32, 8])])
    _test_reparameterized_expression(
        compiler, EXP1, scope=scope, noise=noise1, name="noise1"
    )


def test_gamma(compiler):
    if compiler.rddl.domain.name == "reservoir":
        shape_scope = compiler._get_reparameterization_shape_scope()

        shape = Expression(("pvar_expr", ("RAIN_SHAPE", ["?res"])))
        scale = Expression(("pvar_expr", ("RAIN_SCALE", ["?res"])))
        gamma = Expression(("randomvar", ("Gamma", (shape, scale))))

        noise_lst = compiler._get_expression_reparameterization(gamma, shape_scope)
        assert isinstance(noise_lst, list)
        assert len(noise_lst) == 1

        dist, shape = noise_lst[0]

        assert isinstance(dist, GAMMA)
        assert dist.batch_shape == (8,)
        assert dist.event_shape == ()
        assert shape == []


def test_arithmetic(compiler):
    with compiler.graph.as_default():
        shape_scope = {
            "mu/1": TensorFluentShape([32], batch=False),
            "sigma/1": TensorFluentShape([32], batch=False),
        }

        scope = {
            "mu/1": TensorFluent(tf.zeros([32]), scope=["?x"], batch=False),
            "sigma/1": TensorFluent(tf.ones([32]), scope=["?x"], batch=False),
        }

    noise1 = compiler._get_expression_reparameterization(TWO, scope={})
    _test_reparameterization_dist(noise1, [])
    _test_reparameterized_expression(
        compiler, TWO, scope={}, noise=noise1, name="noise1"
    )

    noise2 = compiler._get_expression_reparameterization(Z_TIMES_Z, scope={})
    _test_reparameterization_dist(noise2, [(NORMAL, [1]), (NORMAL, [1])])
    _test_reparameterized_expression(
        compiler, Z_TIMES_Z, scope={}, noise=noise2, name="noise2"
    )

    noise3 = compiler._get_expression_reparameterization(X2_TIMES_X2, scope=shape_scope)
    _test_reparameterization_dist(noise3, [(NORMAL, [32]), (NORMAL, [32])])
    _test_reparameterized_expression(
        compiler, X2_TIMES_X2, scope=scope, noise=noise3, name="noise3"
    )

    noise4 = compiler._get_expression_reparameterization(MU_PLUS_Z, scope=shape_scope)
    _test_reparameterization_dist(noise4, [(NORMAL, [1])])
    _test_reparameterized_expression(
        compiler, MU_PLUS_Z, scope=scope, noise=noise4, name="noise4"
    )

    noise5 = compiler._get_expression_reparameterization(Z_PLUS_MU, scope=shape_scope)
    _test_reparameterization_dist(noise5, [(NORMAL, [1])])
    _test_reparameterized_expression(
        compiler, Z_PLUS_MU, scope=scope, noise=noise5, name="noise5"
    )

    noise6 = compiler._get_expression_reparameterization(MU_PLUS_X2, scope=shape_scope)
    _test_reparameterization_dist(noise6, [(NORMAL, [32])])
    _test_reparameterized_expression(
        compiler, MU_PLUS_X2, scope=scope, noise=noise6, name="noise6"
    )

    noise7 = compiler._get_expression_reparameterization(X2_PLUS_MU, scope=shape_scope)
    _test_reparameterization_dist(noise7, [(NORMAL, [32])])
    _test_reparameterized_expression(
        compiler, X2_PLUS_MU, scope=scope, noise=noise7, name="noise7"
    )

    noise8 = compiler._get_expression_reparameterization(X1_PLUS_Z, scope=shape_scope)
    _test_reparameterization_dist(noise8, [(NORMAL, [32]), (NORMAL, [1])])
    _test_reparameterized_expression(
        compiler, X1_PLUS_Z, scope=scope, noise=noise8, name="noise8"
    )

    noise9 = compiler._get_expression_reparameterization(Z_PLUS_X1, scope=shape_scope)
    _test_reparameterization_dist(noise9, [(NORMAL, [1]), (NORMAL, [32])])
    _test_reparameterized_expression(
        compiler, Z_PLUS_X1, scope=scope, noise=noise9, name="noise9"
    )


def test_function(compiler):
    with compiler.graph.as_default():
        shape_scope = {
            "mu/1": TensorFluentShape([24], batch=False),
            "sigma/1": TensorFluentShape([24], batch=False),
        }

        scope = {
            "mu/1": TensorFluent(tf.zeros([24]), scope=["?x"], batch=False),
            "sigma/1": TensorFluent(tf.ones([24]), scope=["?x"], batch=False),
        }

    noise1 = compiler._get_expression_reparameterization(EXP_2, scope=shape_scope)
    _test_reparameterization_dist(noise1, [])
    _test_reparameterized_expression(
        compiler, EXP_2, scope=scope, noise=noise1, name="noise1"
    )

    noise2 = compiler._get_expression_reparameterization(EXP_Z, scope=shape_scope)
    _test_reparameterization_dist(noise2, [(NORMAL, [1])])
    _test_reparameterized_expression(
        compiler, EXP_Z, scope=scope, noise=noise2, name="noise2"
    )

    noise3 = compiler._get_expression_reparameterization(EXP_X1, scope=shape_scope)
    _test_reparameterization_dist(noise3, [(NORMAL, [24])])
    _test_reparameterized_expression(
        compiler, EXP_X1, scope=scope, noise=noise3, name="noise3"
    )

    noise4 = compiler._get_expression_reparameterization(Y1, scope=shape_scope)
    _test_reparameterization_dist(noise4, [(NORMAL, [1]), (NORMAL, [1])])
    _test_reparameterized_expression(
        compiler, Y1, scope=scope, noise=noise4, name="noise4"
    )

    noise5 = compiler._get_expression_reparameterization(Y2, scope=shape_scope)
    _test_reparameterization_dist(noise5, [(NORMAL, [1]), (NORMAL, [24])])
    _test_reparameterized_expression(
        compiler, Y2, scope=scope, noise=noise5, name="noise5"
    )

    noise6 = compiler._get_expression_reparameterization(Y3, scope=shape_scope)
    _test_reparameterization_dist(noise6, [(NORMAL, [24]), (NORMAL, [24])])
    _test_reparameterized_expression(
        compiler, Y3, scope=scope, noise=noise6, name="noise6"
    )


def _test_cpf_reparameterization_dist(cpf_noise, reparam_map):
    for (name1, noise1), (name2, noise2) in zip(cpf_noise, reparam_map):
        assert name1 == name2
        _test_reparameterization_dist(noise1, noise2)


def _test_reparameterization_dist(noise, reparam_map):
    noise = [(dist.__class__, shape) for dist, shape in noise]
    assert noise == reparam_map


def _test_reparameterized_expression(compiler, expr, scope, noise, name):
    with compiler.graph.as_default():
        with tf.variable_scope(name):
            noise = [
                tf.get_variable("noise_{}".format(i), shape=shape)
                for i, (_, shape) in enumerate(noise)
            ]
            fluent = compiler._compile_expression(
                expr, scope, batch_size=16, noise=noise
            )
    assert isinstance(fluent, TensorFluent)
    assert noise == []
