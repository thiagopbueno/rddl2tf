# rddl2tf [![Build Status](https://travis-ci.org/thiagopbueno/rddl2tf.svg?branch=master)](https://travis-ci.org/thiagopbueno/rddl2tf) [![Documentation Status](https://readthedocs.org/projects/rddl2tf/badge/?version=latest)](https://rddl2tf.readthedocs.io/en/latest/?badge=latest) [![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://github.com/thiagopbueno/rddl2tf/blob/master/LICENSE.txt)

RDDL2TensorFlow compiler in Python3.

# Quickstart

**rddl2tf** is a Python 3.5+ package available in PyPI.

```text
$ pip3 install rddl2tf
```


# Usage

rddl2tf can be used as a standalone script or programmatically.


## Script mode

```text
$ rddl2tf --help
usage: rddl2tf [-h] [-b BATCH_SIZE] [--logdir LOGDIR] rddl

rddl2tf (v0.5.1): RDDL2TensorFlow compiler in Python3.

positional arguments:
  rddl                  path to RDDL file or rddlgym problem id

optional arguments:
  -h, --help            show this help message and exit
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        number of fluents in a batch (default=256)
  --logdir LOGDIR       log directory for tensorboard graph visualization
                        (default=/tmp/rddl2tf)
```

### Examples

```text
$ rddl2tf Reservoir-8 --batch-size=1024 --logdir=/tmp/rddl2tf
tensorboard --logdir /tmp/rddl2tf/reservoir/inst_reservoir_res8
```

```text
$ rddl2tf Mars_Rover --batch-size=1024 --logdir=/tmp/rddl2tf
tensorboard --logdir /tmp/rddl2tf/simple_mars_rover/inst_simple_mars_rover_pics3
```


## Programmatic mode

```python
import rddlgym

from rddl2tf.compilers import DefaultCompiler


# parse RDDL into an AST
model_id = 'Reservoir-8'
model = rddlgym.make(model_id, mode=rddlgym.AST)

# create a RDDL-to-TF compiler
compiler = DefaultCompiler(model, batch_size=256)
compiler.init()

# compile initial state and default action fluents
state = compiler.initial_state()
action = compiler.default_action()

# compile state invariants and action preconditions
invariants = compiler.state_invariants(state)
preconditions = compiler.action_preconditions(state, action)

# compile action bounds
bounds = compiler.action_bound_constraints(state)

# compile intermediate fluents and next state fluents
interms, next_state = compiler.cpfs(state, action)

# compile reward function
reward = compiler.reward(state, action, next_state)

# save and visualize the computation graph
logdir = os.path.join(args.logdir, model.domain.name, model.instance.name)
file_writer = tf.summary.FileWriter(logdir, compiler.graph)
print('tensorboard --logdir {}\n'.format(logdir))
```


# Compiler

## Core API methods

- `rddl2tf.Compiler.initial_state`
- `rddl2tf.Compiler.default_action`
- `rddl2tf.Compiler.cpfs`
- `rddl2tf.Compiler.reward`
- `rddl2tf.Compiler.state_action_constraints`
- `rddl2tf.Compiler.action_preconditions`
- `rddl2tf.Compiler.state_invariants`
- `rddl2tf.Compiler.action_bound_constraints`


## Parameterized Variables (pvariables)

Each RDDL fluent is compiled to a ``rddl2tf.TensorFluent`` after instantiation.

A ``rddl2tf.TensorFluent`` object wraps a ``tf.Tensor`` object. The arity and the number of objects corresponding to the type of each parameter of a fluent are reflected in a ``rddl2tf.TensorFluentShape`` object (the rank of a ``rddl2tf.TensorFluent`` corresponds to the fluent arity and the size of its dimensions corresponds to the number of objects of each type). Also, a ``rddl2tf.TensorFluentShape`` manages batch sizes when evaluating operations in batch mode.

Additionally, a ``rddl2tf.TensorFluent``keeps information about the ordering of the fluent parameters in a ``rddl2tf.TensorScope`` object.

The ``rddl2tf.TensorFluent`` abstraction is necessary in the evaluation of RDDL expressions due the broadcasting rules of operations in TensorFlow.


## Conditional Probability Functions (CPFs)

Each CPF expression is compiled into an operation in a ``tf.Graph``, possibly composed of many other operations. Typical RDDL operations, functions, and probability distributions are mapped to equivalent TensorFlow ops. These operations are added to a ``tf.Graph`` by recursively compiling the expressions in a CPF into wrapped operations and functions implemented at the ``rddl2tf.TensorFluent`` level.

Note that the RDDL2TensorFlow compiler currently only supports element-wise operations (e.g. ``a(?x, ?y) = b(?x) * c(?y)`` is not allowed). However, all compiled operations are vectorized, i.e., computations are done simultaneously for all object instantiations of a pvariable.

Optionally, during simulation operations can be evaluated in batch mode. In this case, state-action trajectories are generated in parallel by the ``rddl2tf.Simulator``.


# Documentation

Please refer to [https://rddl2tf.readthedocs.io/](https://rddl2tf.readthedocs.io/en/latest/) for the code documentation.


# Support

If you are having issues with ``rddl2tf``, please let me know at: [thiago.pbueno@gmail.com](mailto://thiago.pbueno@gmail.com).


# License

Copyright (c) 2018-2020 Thiago Pereira Bueno All Rights Reserved.

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
