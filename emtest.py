from EMCost import *
import pennylane as qlm
jax.config.update("jax_enable_x64", True)

num_input_qubits=4
output_qubits=4
n_trash_qubits=0
operator_support=[]
operator_support_probs=0
operator_translation_invariance_Q=0
operator_support_max_range=0

set_global( num_input_qubits,
            output_qubits,
            n_trash_qubits,
            operator_support,
            operator_support_probs,
            operator_translation_invariance_Q,
            operator_support_max_range,
            use_jax=False)

dvc = qml.device('default.mixed', wires=num_input_qubits, shots=None)

@qml.qnode(dvc,interface='jax')
def trainer(w,dm):
    qml.QubitDensityMatrix(dm, wires=dvc.wires)
    return qml.state()
input_states=[]



X=[]
for i in range(2**num_input_qubits):

    a=np.zeros((2**num_input_qubits))
    a[i]=1
    a=jnp.array(a)
    X.append(jnp.outer(a, a))
    input_states.append(jnp.outer(a, a))

f =cost_fn_EM(X,trainer,input_states)
c=f([1])


# print(format(a,f'0{num_input_qubits}b'),format(b,f'0{num_input_qubits}b'))

# cost_fn_EM(input_states,trainer,input_states)
