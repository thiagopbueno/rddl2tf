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
