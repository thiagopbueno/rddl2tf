# rddl2tf [![Build Status](https://travis-ci.org/thiagopbueno/rddl2tf.svg?branch=master)](https://travis-ci.org/thiagopbueno/rddl2tf) [![License](https://img.shields.io/aur/license/yaourt.svg)](https://github.com/thiagopbueno/rddl2tf/blob/master/LICENSE)

RDDL2TensorFlow compiler and trajectory simulator in Python3.

# Quickstart

```text
$ pip3 install rddl2tf
```

# Compiler

## Parameterized Variables (pvariables)

Each RDDL fluent is compiled to a ``rddl2tf.TensorFluent`` after instantiation.

A ``rddl2tf.TensorFluent`` object wraps a ``tf.Tensor`` object. The arity and the number of objects corresponding to the type of each parameter of a fluent are reflected in a ``rddl2tf.TensorFluentShape`` object (the rank of a ``rddl2tf.TensorFluent`` corresponds to the fluent arity and the size of its dimensions corresponds to the number of objects of each type). Also, a ``rddl2tf.TensorFluentShape`` manages batch sizes when evaluating operations in batch mode.

Additionally, a ``rddl2tf.TensorFluent``keeps information about the ordering of the fluent parameters in a ``rddl2tf.TensorScope`` object.

The ``rddl2tf.TensorFluent`` abstraction is necessary in the evaluation of RDDL expressions due the broadcasting rules of operations in TensorFlow.


## Conditional Probability Functions (CPFs)

Each CPF expression is compiled into an operation in a ``tf.Graph``, possibly composed of many other operations. Typical RDDL operations, functions, and probability distributions are mapped to equivalent TensorFlow ops. These operations are added to a ``tf.Graph`` by recursively compiling the expressions in a CPF into wrapped operations and functions implemented at the ``rddl2tf.TensorFluent`` level.

Note that the RDDL2TensorFlow compiler currently only supports element-wise operations (e.g. ``a(?x, ?y) = b(?x) * c(?y)`` is not allowed). However, all compiled operations are vectorized, i.e., computations are done simultaneously for all object instantiations of a pvariable.

Optionally, during simulation operations can be evaluated in batch mode. In this case, state-action trajectories are generated in parallel by the ``rddl2tf.Simulator``.


# License

Copyright (c) 2018 Thiago Pereira Bueno All Rights Reserved.

rddl2tf is free software: you can redistribute it and/or modify it
under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at
your option) any later version.

rddl2tf is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser
General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with rddl2tf. If not, see http://www.gnu.org/licenses/.
