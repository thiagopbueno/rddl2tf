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

from pyrddl.rddl import RDDL

from rddl2tf.fluentshape import TensorFluentShape

import numpy as np
import tensorflow as tf

from typing import Dict, List, Tuple

Noise = List[Tuple[tf.distributions.Distribution, List[int]]]
ShapeScope = Dict[str, TensorFluentShape]


def get_reparameterization_shape_scope(rddl: RDDL) -> ShapeScope:
    scope = {
        name: TensorFluentShape(size, batch=False)
        for name, (_, size) in rddl.fluent_table.items()
    }
    return scope


def get_cpfs_reparameterization(rddl: RDDL) -> Noise:
    noise = get_intermediate_cpfs_reparameterization(rddl)
    noise += get_state_cpfs_reparameterization(rddl)
    return noise


def get_state_cpfs_reparameterization(rddl: RDDL) -> Noise:
    scope = get_reparameterization_shape_scope(rddl)
    noise = []
    for cpf in rddl.domain.state_cpfs:
        noise.append((cpf.name, get_reparameterization(cpf.expr, scope)))
    return noise


def get_intermediate_cpfs_reparameterization(rddl: RDDL) -> Noise:
    scope = get_reparameterization_shape_scope(rddl)
    noise = []
    for cpf in rddl.domain.intermediate_cpfs:
        noise.append((cpf.name, get_reparameterization(cpf.expr, scope)))
    return noise


def get_reparameterization(expr: Expression,
                           scope: ShapeScope) -> Noise:
    noise = []
    _get_reparameterization(expr, scope, noise)
    return noise


def _get_reparameterization(expr: Expression,
                            scope: ShapeScope,
                            noise: Noise) -> TensorFluentShape:
    etype = expr.etype
    args = expr.args

    if etype[0] == 'constant':
        return TensorFluentShape([1], batch=False)
    elif etype[0] == 'pvar':
        name = expr._pvar_to_name(args)
        if name not in scope:
            raise ValueError('Variable {} not in scope.'.format(name))
        shape = scope[name]
        return shape
    elif etype[0] == 'randomvar':
        if etype[1] == 'Normal':
            mean_shape = _get_reparameterization(args[0], scope, noise)
            var_shape = _get_reparameterization(args[1], scope, noise)
            shape = _broadcast(mean_shape, var_shape)
            dist = tf.distributions.Normal(loc=0.0, scale=1.0)
            noise.append((dist, shape.as_list()))
            return shape
    elif etype[0] in ['arithmetic', 'boolean', 'relational']:
        op1_shape = _get_reparameterization(args[0], scope, noise)
        shape = op1_shape
        if len(args) > 1:
            op2_shape = _get_reparameterization(args[1], scope, noise)
            shape = _broadcast(op1_shape, op2_shape)
        return shape
    elif etype[0] == 'func':
        op1_shape = _get_reparameterization(args[0], scope, noise)
        shape = op1_shape
        if len(args) > 1:
            if len(args) == 2:
                op2_shape = _get_reparameterization(args[1], scope, noise)
                shape = _broadcast(op1_shape, op2_shape)
            else:
                raise ValueError('Invalid function:\n{}'.format(expr))
        return shape
    elif etype[0] == 'control':
        if etype[1] == 'if':
            condition_shape = _get_reparameterization(args[0], scope, noise)
            true_case_shape = _get_reparameterization(args[1], scope, noise)
            false_case_shape = _get_reparameterization(args[2], scope, noise)
            shape = _broadcast(condition_shape, true_case_shape)
            shape = _broadcast(shape, false_case_shape)
            return shape
        else:
            raise ValueError('Invalid control flow expression:\n{}'.format(expr))
    elif etype[0] == 'aggregation':
        return _get_reparameterization(args[-1], scope, noise)

    raise ValueError('Expression type unknown: {}'.format(etype))


def _broadcast(shape1: TensorFluentShape, shape2: TensorFluentShape) -> TensorFluentShape:
    s1, s2 = TensorFluentShape.broadcast(shape1, shape2)
    s1 = s1 if s1 is not None else shape1.as_list()
    s2 = s2 if s2 is not None else shape2.as_list()
    x1, x2 = np.zeros(s1), np.zeros(s2)
    y = np.broadcast(x1, x2)
    return TensorFluentShape(y.shape, batch=(shape1.batch or shape2.batch))
