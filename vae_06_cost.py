# -*- coding: utf-8 -*-

import pennylane as qml
from pennylane import numpy as np
import numpy
import itertools
import numbers
import random
import copy
import jax
import jax.numpy as jnp
from jax.config import config
config.update('jax_enable_x64', True)
import cvxpy
from functools import partial
import vae_06_funcs as vaef
import vae_06_params as vaep


# --------------------------
# GET COMMAND LINE ARGUMENTS
# --------------------------
# Command line arguments are loaded both in vea_06.py and in vae_06_cost.py, and
# they are made available using the system_parameters dictionary global variable
parser = vaep.create_parser()
args = parser.parse_args()



# --------------------------------------------------
# DEFINE GLOBAL VARIABLES USED IN THE CURRENT MODULE
# --------------------------------------------------
system_params = vaep.args_to_dict(args)
input_states, data_parameters = vaep.load_states(data_type=system_params['data_type'],
                                                 num_input_qubits=system_params['num_input_qubits'],
                                                 frac_sampled=system_params['frac_sampled'],
                                                 param_rng_min=system_params['param_rng_min'],
                                                 param_rng_max=system_params['param_rng_max'],
                                                 random_seed=system_params['random_seed'],
                                                 save_Q=system_params['save_Q'],
                                                 save_opt_txt_Q=system_params['save_opt_txt_Q'],
                                                 plot_Q=system_params['plot_Q'],
                                                 print_opt_fname=system_params['print_opt_fname'],
                                                 plot_fname_states=system_params['plot_fname_states']
                                                 )
# When JAX is used, the arrays need to be in jnp format
if system_params['use_jax_Q']:
    input_states = jnp.array(input_states)

svd_unitary_compression = vaef.svd_unitary_dm_mixture(input_states, system_params)

# Calculate state fidelities (to be compared to the cost function)
input_state_fidelity_mx = vaep.get_pure_state_fidelity_mx(input_states, system_params['use_jax_Q'])

# Define quantum devices
if system_params['use_jax_Q']:
    vae_dev_pure_input = qml.device('default.qubit.jax', wires=system_params['input_qubits'])
else:
    vae_dev_pure_input = qml.device('default.qubit', wires=system_params['input_qubits'])
vae_dev_mixed_middle = qml.device('default.mixed', wires=system_params['middle_qubits'])
vae_dev_mixed_output = qml.device('default.mixed', wires=system_params['output_qubits'])



# ---------------------------------------------
# COMPRESSING AND EXPANDING PART OF THE NETWORK
# ---------------------------------------------
# Compress the input state into the reduced space but do not trace out the
# remaining qubits yet, just return the full product state
@qml.qnode(vae_dev_pure_input, interface='jax' if system_params['use_jax_Q'] else 'autograd')
def vae_compress_to_pure_state(compress_convolution_parameters,
                               compress_pooling_parameters,
                               state_vector_in,
                               decomposed_Q:bool=False):
    qml.QubitStateVector(state_vector_in, vae_dev_pure_input.wires)
    vaef.qcnn_periodic_compress(convolution_parameters=compress_convolution_parameters,
                                pooling_parameters=compress_pooling_parameters,
                                wires=vae_dev_pure_input.wires,
                                num_layers=system_params['num_pooling_layers'],
                                num_conv_per_layer=system_params['num_conv_per_layer'],
                                even_odd_symm_Q=system_params['even_odd_site_symmetry_Q'],
                                inverse_Q=False,
                                decomposed_Q=decomposed_Q)
    return qml.state()
    # Looked the code through, looks good

# Use JAX to speed up vae_compress_to_pure_state
@partial(jax.jit, static_argnames=['decomposed_Q'])
def jax__vae_compress_to_pure_state(compress_convolution_parameters,
                                    compress_pooling_parameters,
                                    state_vector_in,
                                    decomposed_Q:bool=False):
    return vae_compress_to_pure_state(compress_convolution_parameters=jnp.array(compress_convolution_parameters),
                                      compress_pooling_parameters=jnp.array(compress_pooling_parameters),
                                      state_vector_in=jnp.array(state_vector_in),
                                      decomposed_Q=decomposed_Q)
    # Looked the code through, looks good


# Get the Z-expectation values of the traced qubits in the state pure_state
@qml.qnode(vae_dev_pure_input, interface='jax' if system_params['use_jax_Q'] else 'autograd')
def measure_traced_qubits(pure_state):
    qml.QubitStateVector(pure_state, vae_dev_pure_input.wires)
    return [qml.expval(qml.PauliZ(i)) for i in system_params['traced_qubits']]
    # Looked the code through, looks good

@jax.jit
def jax__measure_traced_qubits(pure_state):
    return measure_traced_qubits(pure_state)
    # Looked the code through, looks good

@jax.jit
def jax__reduce_to_middle_qubits(pure_state):
    return qml.math.reduce_statevector(pure_state, indices=system_params['middle_qubits'])
    # Looked the code through, looks good


# Compress the input state into the reduced space and return the density
# matrix after tracing out the remaining qubits (middle_dm)
# Also return the norm square of the states whose traced out qubits
# are not in the |1...1111> form
def vae_compress(compress_convolution_parameters,
                 compress_pooling_parameters,
                 state_vector_in,
                 decomposed_Q=False,
                 measure_traced_qubits_Q:bool=True,
                 use_SVD_unitary_Q:bool=False):
    # Inputs:
    # - compress_convolution_parameters: the convolution parameters of the compressing part of the network
    # - compress_pooling_parameters: the pooling part of the network
    # - state_vector_in: input state in 2**num_input_qubits dimensions
    # - decomposed_Q: if set to True, then the gates are decomposed for easier interpretability
    # - measure_traced_qubits_Q: if yes, then the Z expectation values of the traced qubits is returned too
    # - use_SVD_unitary_Q: if yes, then instead of the QCNN compressing network, we use the unitary transformation
    #                      which diagonalizes the density matrix rho = sum_i |psi_i><psi_i| where psi_i runs over
    #                      all input states
    if not use_SVD_unitary_Q:
        if system_params['use_jax_Q']:
            middle_state = jax__vae_compress_to_pure_state(
                              compress_convolution_parameters=compress_convolution_parameters,
                              compress_pooling_parameters=compress_pooling_parameters,
                              state_vector_in=state_vector_in,
                              decomposed_Q=decomposed_Q)
        else:
            middle_state = vae_compress_to_pure_state(
                              compress_convolution_parameters=compress_convolution_parameters,
                              compress_pooling_parameters=compress_pooling_parameters,
                              state_vector_in=state_vector_in,
                              decomposed_Q=decomposed_Q)
    else:
        middle_state = svd_unitary_compression @ state_vector_in
        
    if system_params['use_jax_Q']:
        middle_reduced_dm = jax__reduce_to_middle_qubits(middle_state)
    else:
        middle_reduced_dm = qml.math.reduce_statevector(middle_state, indices=system_params['middle_qubits'])

    if measure_traced_qubits_Q:
        if system_params['use_jax_Q']:
            expval_Z_traced_qubits = jax__measure_traced_qubits(middle_state)
        else:
            expval_Z_traced_qubits = measure_traced_qubits(middle_state)
    else:
        expval_Z_traced_qubits = []
    
    return middle_reduced_dm, middle_state, expval_Z_traced_qubits
    # Loked the code through, looks good

# Expanding part of the VAE network
@qml.qnode(vae_dev_mixed_output)
def _vae_expand(expand_convolution_parameters, expand_pooling_parameters,
                density_matrix_mid, decomposed_Q:bool):
    qml.QubitDensityMatrix(density_matrix_mid, wires=system_params['middle_qubits'])
    vaef.qcnn_periodic_compress(convolution_parameters=expand_convolution_parameters,
                                pooling_parameters=expand_pooling_parameters,
                                wires=vae_dev_mixed_output.wires,
                                num_layers=system_params['num_pooling_layers'],
                                num_conv_per_layer=system_params['num_conv_per_layer'],
                                even_odd_symm_Q=system_params['even_odd_site_symmetry_Q'],
                                inverse_Q=True,
                                decomposed_Q=decomposed_Q)
                                
    return qml.density_matrix(wires=vae_dev_mixed_output.wires)
    # Loked the code through, looks good

# Wrapping up the previous function to avoid optimization error
def vae_expand(expand_convolution_parameters, expand_pooling_parameters,
               density_matrix_mid, decomposed_Q=False, use_SVD_unitary_Q:bool=False):
    if use_SVD_unitary_Q:
        # Expand the density matrix by adding a string of zeros to the traced qubits
        density_matrix_mid_x_zeros = add_zeros_to_dm(density_matrix_mid)
        return np.conjugate(svd_unitary_compression.T) @ density_matrix_mid_x_zeros @ svd_unitary_compression
        
    return _vae_expand(expand_convolution_parameters=expand_convolution_parameters,
                       expand_pooling_parameters=expand_pooling_parameters,
                       density_matrix_mid=density_matrix_mid,
                       decomposed_Q=decomposed_Q)
    # Loked the code through, looks good

@qml.qnode(vae_dev_mixed_output)
def vae_compress_interp(compress_convolution_parameters,
                        compress_pooling_parameters,
                        state_vector_in,
                        decomposed_Q):
    qml.QubitStateVector(state_vector_in, vae_dev_mixed_output.wires)
    vaef.qcnn_periodic_compress(convolution_parameters=compress_convolution_parameters,
                                pooling_parameters=compress_pooling_parameters,
                                wires=vae_dev_mixed_output.wires,
                                num_layers=system_params['num_pooling_layers'],
                                num_conv_per_layer=system_params['num_conv_per_layer'],
                                even_odd_symm_Q=system_params['even_odd_site_symmetry_Q'],
                                inverse_Q=False,
                                decomposed_Q=decomposed_Q)
    return qml.density_matrix(system_params['middle_qubits'])
    # Loked the code through, looks good


@qml.qnode(vae_dev_mixed_output)
def vae_expand_interp(expand_convolution_parameters,
                      expand_pooling_parameters,
                      density_matrix_mid,
                      decomposed_Q):
    # By putting the middle_qubits wires into the density_matrix_mid, PennyLane
    # automatically puts the remaining traced_qubits into a |1...111> state
    qml.QubitDensityMatrix(density_matrix_mid, wires=system_params['middle_qubits'])

    vaef.qcnn_periodic_compress(convolution_parameters=expand_convolution_parameters,
                                pooling_parameters=expand_pooling_parameters,
                                wires=vae_dev_mixed_output.wires,
                                num_layers=system_params['num_pooling_layers'],
                                num_conv_per_layer=system_params['num_conv_per_layer'],
                                even_odd_symm_Q=system_params['even_odd_site_symmetry_Q'],
                                inverse_Q=True,
                                decomposed_Q=decomposed_Q)

    return qml.density_matrix(wires=vae_dev_mixed_output.wires)
    # Loked the code through, looks good



# ---------------------------------------------------------
# AUXILIARY FUNCTIONS USED BY THE EARTH MOVER COST FUNCTION
# ---------------------------------------------------------
def get_site_combinations(n_system_sites:int, operator_support:int, support_prob:float,
                          translation_invariance_Q:bool, max_rng=None,
                          random_shift_sites_Q:bool=True):
    """ Get combinations of operator_support sites in a system of n_system_sites sites
        Args:
        - n_system_sites: number of sites
        - operator_support: span of operators
        - support_prob: probability of choosing any given site out of all of them
        - translation_invariance_Q: if this is set to True, then out of all the operators
                                    we will consider only 2: one which starts at site 0
                                    and another one starting at site 1.
    """
    # Check parameters and deal with corner cases
    assert operator_support >= 0
    assert n_system_sites >= 0
    if max_rng is None or max_rng > n_system_sites:
        max_rng = n_system_sites
    elif max_rng < operator_support:
        return []
    if support_prob == 0.:
        return []

    # Generate combinations
    if translation_invariance_Q:
        # if we assume translation invariance, we can assume two types of sites:
        # one where the first site is at the origin and anotherone where it
        # is at the second site
        arr =  [[0] + list(c) for c in list(itertools.combinations(list(
                range(1, min(max_rng, n_system_sites))), operator_support-1))]
        arr += [[1] + list(c) for c in list(itertools.combinations([i%n_system_sites
                for i in range(2, min(max_rng+1, n_system_sites+1))], operator_support-1))]

        # however, we shift the generated array around by a random vector just
        # to make sure that we are not pushing some unforeseen bias in the network
        # so that it generates non-translation invariant outputs when we
        # use these operators in the cost function
        arr = np.array(arr, dtype=np.int16)
        if random_shift_sites_Q:
            arr = (arr + random.randint(0, n_system_sites)) % n_system_sites
        arr = arr.reshape(-1, arr.shape[-1])
        arr = np.sort(arr, axis=-1)
        arr = [[int(i) for i in c] for c in arr]
    else:
        if max_rng == n_system_sites:
            arr = [list(c) for c in list(itertools.combinations(list(
                   range(n_system_sites)), operator_support))]
        else:
            # we create all combinations that start at site 0 and is within the range max_rng
            # then, we shift these around in all possible ways, then we remove duplicates
            arr = [[0] + list(c) for c in list(itertools.combinations(list(range(1, max_rng)),
                   operator_support-1))]
            arr = np.array(arr, dtype=np.int16)
            arr = np.array([(arr + i) % n_system_sites for i in range(n_system_sites)])
            arr = arr.reshape(-1, arr.shape[-1])
            arr = np.sort(arr, axis=-1)
            arr = np.unique(arr, axis=0)
            arr = [[int(i) for i in c] for c in arr]

    if support_prob == 1.:
        return arr
    n_combinations = len(arr)
    n_samples = int(np.round(n_combinations * support_prob))
    random.shuffle(arr)
    return arr[:n_samples]
    # Tested and works


def sites_to_site_op(sites):
    # Having a list of sites e.g. [[1,3,8], [2,4,7]], we add all possible
    # combinations of 'x', 'y', and 'z' operators to it.
    # E.g. [[1,3,8], ...] --> [[[1, 'y'], [3, 'z'], [8, 'z']], ...]
    def sites_to_site_op_iterative_fn(sites, ind):
        if len(sites) == 0 or ind < 0:
            return []
        if ind == len(sites[0]):
            return sites
        return sites_to_site_op_iterative_fn([s[:ind] + [(s[ind], op)] + s[(ind+1):]
                                             for s in sites for op in ['x', 'y', 'z']],
                                             ind+1)
    return sites_to_site_op_iterative_fn(sites, 0)
    # Tested and works


def get_Pauli_strings(n_system_sites:int, operator_support_list=None,
                      support_prob_list=None,
                      translation_invariance_Q:bool=True,
                      max_rng=None,
                      random_shift_sites_Q:bool=True):
    """ Get combinations of operator_support sites in a system of
        n_system_sites sites together with
        all combinations of 'x', 'y', and 'z' measurement axes.
        Output example: [[(1, 'y'), (3, 'z'), (8, 'z')], ...]

        Inputs:
        - n_system_sites: number of sites
        - operator_support_list: list of spans of operators
          (if set to None by default, it becomes a list of n_system_sites)
        - support_prob_list: list of probabilities of choosing any given
                             site out of all of them (if set to None by default,
                             it becomes 1 for all operator_support values)
        - translation_invariance_Q: if this is set to True, then the first
                                      site will be chosen 0
        - max_rng: the maximum range of Pauli strings
            - max_rng == None (default) sets it to the maximum range of n_system_sites
            - max_rng == int sets it to that value for all operator supports
            - max_rng == list sets it to the elements of that list
              (missing elements are replaced by n_system_sites)
    """
    # if operator_support_list==None, then we consider all ranges
    if operator_support_list is None:
        operator_support_list = np.array(list(range(1, n_system_sites + 1)))

    # if support_prob_list == None, then it is set to 1
    if support_prob_list is None:
        support_prob_list = np.ones(len(operator_support_list))

    # make sure all elements are unique
    assert len(list(set(operator_support_list))) == len(operator_support_list)

    # fill the missing element of support_prob_list with 1-s
    assert len(support_prob_list) <= len(operator_support_list)
    support_prob_list = [(min(max(p, 0.), 1.) if p is not None else 1.)
                         for p in support_prob_list] \
                         + list(np.ones(len(operator_support_list) - len(support_prob_list)))
    assert len(support_prob_list) == len(operator_support_list)

    if max_rng is None:
        max_rng = n_system_sites * np.ones(n_system_sites, dtype=np.int16)
    elif isinstance(max_rng, numbers.Number):
        max_rng = int(np.round(max_rng)) * np.ones(n_system_sites, dtype=np.int16)
    elif type(max_rng) is list or isinstance(max_rng, np.ndarray):
        max_rng = np.array(np.round(max_rng), dtype=np.int16)
        if len(max_rng.shape) <= 1. and len(max_rng) <= n_system_sites:
            max_rng = np.concatenate((max_rng, n_system_sites
                      * np.ones(n_system_sites - len(max_rng), dtype=np.int16)))
        else:
            raise Exception(f'max_rng = {max_rng} should be None, a number, ' \
                            +'a list or an array shorter than the number of sites')
    else:
        raise Exception(f'max_rng = {max_rng} should be None, a number, ' \
                        +'a list or an array shorter than the number of sites')
    max_rng = list(max_rng)
        
    # Generate all Pauli strings
    Pauli_string_lists = [sites_to_site_op(get_site_combinations(n_system_sites,
                                                       operator_support_list[i],
                                                       support_prob_list[i],
                                                       translation_invariance_Q,
                                                       max_rng[i],
                                                       random_shift_sites_Q))
                          for i in range(len(operator_support_list))]
    return Pauli_string_lists
    # Tested and works

# Return the Pauli strings corresponding to operators in the Hamiltonian
def get_Hamiltonian_Pauli_strings(n_system_sites, data_type, translation_invariance_Q, random_shift_sites_Q):
    Pauli_string_lists = []
    if data_type == 'Ising':
        if not translation_invariance_Q:
            for i in range(n_system_sites):
                Pauli_string_lists += [[(i, 'z'), ((i+1) % n_system_sites, 'z')]]
            for i in range(n_system_sites):
                Pauli_string_lists += [[(i, 'x')]]
        else:
            i = np.random.randint(n_system_sites) if random_shift_sites_Q else 0
            Pauli_string_lists += [[(i, 'z'), ((i+1) % n_system_sites, 'z')]]
            Pauli_string_lists += [[((i+1) % n_system_sites, 'z'), ((i+2) % n_system_sites, 'z')]]
            Pauli_string_lists += [[(i, 'x')]]
            Pauli_string_lists += [[((i+1) % n_system_sites, 'x')]]
    elif data_type == 'Heisenberg':
        if not translation_invariance_Q:
            for i in range(n_system_sites):
                Pauli_string_lists += [[(i, s), ((i+1) % n_system_sites, s)] for s in ['x', 'y', 'z']]
        else:
            i = np.random.randint(n_system_sites) if random_shift_sites_Q else 0
            Pauli_string_lists += [[(i, s), ((i+1) % n_system_sites, s)] for s in ['x', 'y', 'z']]
            Pauli_string_lists += [[((i+1) % n_system_sites, s), ((i+2) % n_system_sites, s)] for s in ['x', 'y', 'z']]
    else:
        raise Exception(f'Incorrect data type {data_type} in get_Hamiltonian_Pauli_strings().')
    return [Pauli_string_lists]
    # Tested and works


# ---------------------
# DEFINE COST FUNCTIONS
# ---------------------
# Auxiliary function that add zero states on the traced qubits to the density matrix on the middle qubits
@qml.qnode(vae_dev_mixed_output)
def add_zeros_to_dm(middle_reduced_dm):
    # By putting the middle_qubits wires into the density_matrix_mid, PennyLane
    # automatically puts the remaining traced_qubits into a |1...111> state
    qml.QubitDensityMatrix(middle_reduced_dm, wires=system_params['middle_qubits'])
    return qml.state()

# Cost function is the difference from the value 1 of the fidelity
# between all input and their corresponding output states
def cost_fn_fidelity(compressing_convolutional_params,
                     compressing_pooling_params,
                     expanding_convolutional_params,
                     expanding_pooling_params,
                     use_SVD_unitary_Q:bool=False):
    
    n_states = len(input_states)

    cost = 0.
    for input_state in input_states:
        middle_reduced_dm, middle_state, expval_Z_traced_qubits = vae_compress(
                                                compress_convolution_parameters=compressing_convolutional_params,
                                                compress_pooling_parameters=compressing_pooling_params,
                                                state_vector_in=input_state,
                                                measure_traced_qubits_Q=(abs(system_params['coeff_traced_cost'])>0),
                                                use_SVD_unitary_Q=use_SVD_unitary_Q)
        if system_params['use_jax_Q']:
            cost += system_params['coeff_traced_cost'] * jnp.sum(jnp.array([(z-1)**2 for z in expval_Z_traced_qubits])) / len(system_params['traced_qubits'])
        else:
            cost += system_params['coeff_traced_cost'] * np.sum(np.array([(z-1)**2 for z in expval_Z_traced_qubits])) / len(system_params['traced_qubits'])

        middle_reduced_dm_x_zeros = add_zeros_to_dm(middle_reduced_dm)

        if system_params['same_compress_expand_Q']:
            middle_state_dm = qml.math.dm_from_state_vector(middle_state)
            if system_params['use_jax_Q']:
                cost += jnp.abs(qml.math.fidelity(middle_reduced_dm_x_zeros, middle_state_dm) - 1)**2
            else:
                cost += np.abs(qml.math.fidelity(middle_reduced_dm_x_zeros, middle_state_dm) - 1)**2
        else:
            _, middle_state, _ = vae_compress(
                                    compress_convolution_parameters=expanding_convolutional_params,
                                    compress_pooling_parameters=expanding_pooling_params,
                                    state_vector_in=input_state,
                                    measure_traced_qubits_Q=False,
                                    use_SVD_unitary_Q=use_SVD_unitary_Q)
            middle_state_dm = qml.math.dm_from_state_vector(middle_state)
            if system_params['use_jax_Q']:
                cost += jnp.abs(qml.math.fidelity(middle_reduced_dm_x_zeros, middle_state_dm) - 1)**2
            else:
                cost += np.abs(qml.math.fidelity(middle_reduced_dm_x_zeros, middle_state_dm) - 1)**2
        
    cost /= n_states
    
    return cost
    # Looked the code through, looks good
    

# Cost function is the 2-norm of the difference between the fidelity
# matrices of the input and the output states
# The idea is that we are trying to preserve the geometry of the input data cloud
def cost_fn_geometry(compressing_convolutional_params,
                     compressing_pooling_params,
                     use_SVD_unitary_Q:bool=False):
    n_states = len(input_states)
    middle_dms = []
    
    cost = 0
    for psi in input_states:
        middle_reduced_dm, _, expval_Z_traced_qubits = vae_compress(
                                    compress_convolution_parameters=compressing_convolutional_params,
                                    compress_pooling_parameters=compressing_pooling_params,
                                    state_vector_in=psi,
                                    measure_traced_qubits_Q=(abs(system_params['coeff_traced_cost'])>0),
                                    use_SVD_unitary_Q=use_SVD_unitary_Q)
        middle_dms += [middle_reduced_dm]
        
        if system_params['use_jax_Q']:
            cost += system_params['coeff_traced_cost'] * jnp.sum(jnp.array([(z-1)**2 for z in expval_Z_traced_qubits])) / len(system_params['traced_qubits'])
        else:
            cost += system_params['coeff_traced_cost'] * np.sum(np.array([(z-1)**2 for z in expval_Z_traced_qubits])) / len(system_params['traced_qubits'])

    middle_state_fidelity_mx = vaep.get_mixed_state_fidelity_mx(middle_dms, system_params['use_jax_Q'])
    for i in range(n_states):
        for j in range(n_states):
            cost += (middle_state_fidelity_mx[i][j] - input_state_fidelity_mx[i][j])**2
    cost /= n_states

    return cost
    # Looked the code through, looks good


# Calculate the expectation value of Pauli strings on input states
@qml.qnode(vae_dev_pure_input, interface='jax' if system_params['use_jax_Q'] else 'autograd')
def _expval_operators_input(state, operators):
    state_op_expval = []
    qml.QubitStateVector(state, vae_dev_pure_input.wires)
    return [qml.expval(op) for op in operators]
    # Looked the code through, looks good

@jax.jit
def jax__expval_operators_input(state, operators):
    return _expval_operators_input(state, operators)
    # Looked the code through, looks good
    
def expval_operators_input(state_in, operators):
    if system_params['use_jax_Q']:
        return jax__expval_operators_input(state_in, operators)
    else:
        return _expval_operators_input(state_in, operators)
    # Looked the code through, looks good
    
def expval_operators_middle(dm_mid, operators):
    @qml.qnode(vae_dev_mixed_middle)
    def _expval_operators_middle(dm_mid, operators):
        state_op_expval = []
        qml.QubitDensityMatrix(dm_mid, vae_dev_mixed_middle.wires)
        return [qml.expval(op) for op in operators]
    return _expval_operators_middle(dm_mid, operators)
    # Looked the code through, looks good

def expval_operators_output(dm_out, operators):
    @qml.qnode(vae_dev_mixed_output)
    def _expval_operators_output(dm_out, operators):
        state_op_expval = []
        qml.QubitDensityMatrix(dm_out, vae_dev_mixed_output.wires)
        return [qml.expval(op) for op in operators]
    return _expval_operators_output(dm_out, operators)
    # Looked the code through, looks good

def operators_from_Pauli_strings(Pauli_string_list):
    # Inputs:
    # - state_in_list: list of pure state vectors
    # - Pauli_string e.g. [(2, 'x'), (5, 'y'), (1, 'y'), (4, 'z'), ...]
    # - density_matrix_Q = True if state_in is a density matrix and False if
    #                      it is a pure state in state vector format
    
    operators = []
    for Pauli_string in Pauli_string_list:
        if len(Pauli_string) == 0:
            Pauli_string_expectation_values += [1.]
        else:
            (s, op) = Pauli_string[0]
            if op == 'x':
                operator = qml.PauliX(s)
            elif op == 'y':
                operator = qml.PauliY(s)
            elif op == 'z':
                operator = qml.PauliZ(s)
            else:
                raise Exception('Invalid operator string.')

            for site_op in Pauli_string[1:]:
                (s, op) = site_op
                if op == 'x':
                    operator = operator @ qml.PauliX(s)
                elif op == 'y':
                    operator = operator @ qml.PauliY(s)
                elif op == 'z':
                    operator = operator @ qml.PauliZ(s)
                else:
                    raise Exception('Invalid operator string.')

            operators += [operator]
    return operators
    # Looked the code through, looks good

# Calculates the expectation values of the Pauli strings
# in each of the input states
# Inputs:
# - states_in_list: 1D list of pure states or density matrices
# - Pauli_string_list: 1D list of Pauli strings,
#   e.g. Pauli_string_lists = [[(0, 'x')],
#                              [(0, 'y')],
#                              [(0, 'z')],
#                              [(2, 'x'), (3, 'x')],
#                              [(2, 'x'), (3, 'y')],
#                              [(2, 'x'), (3, 'z')],
#                              [(2, 'y'), (3, 'x')],
#                              [(2, 'y'), (3, 'y')],
#                              [(2, 'y'), (3, 'z')],
#                              [(2, 'z'), (3, 'x')],
#                              [(2, 'z'), (3, 'y')],
#                              [(2, 'z'), (3, 'z')]]
# - in_mid_out_Q: which device to use to evaluate the operators
#                 Options: 'input', 'middle', 'output'
# Output: an array of size len(states_in_list) x len(Pauli_string_list)
#         state_op_expval[i_st, i_op] = <states_in_list[i_st]| Operator(Pauli_string_list[i_op]) |states_in_list[i_st]>
def expval_Pauli_strings(states_in_list,
                         Pauli_string_list,
                         in_mid_out_Q:str='input'):
    ops = operators_from_Pauli_strings(Pauli_string_list)
    state_op_expval = []
    if in_mid_out_Q=='input':
        for state in states_in_list:
            state_op_expval += [expval_operators_input(state, ops)]
    elif in_mid_out_Q=='middle':
        for state in states_in_list:
            state_op_expval += [expval_operators_middle(state, ops)]
    elif in_mid_out_Q=='output':
        for state in states_in_list:
            state_op_expval += [expval_operators_output(state, ops)]
    if system_params['use_jax_Q']:
        state_op_expval = jnp.array(state_op_expval)
    else:
        state_op_expval = np.array(state_op_expval)
    return state_op_expval
    # Looked the code through, looks good


# Combined function for calculating the operator based and the
# earth mover distance based cost function.
# cost_type == 'earth_mover', 'operator', or 'hamiltonian'
#
# Approximation of the earth mover distance using the approach of
# B. T. Kiani, G. De Palma, M. Marvian, Z.-W. Liu, and Seth Lloyd,
# Quant. Sci. Technol. 7, 045002 (2022), equation (20).
#
# Operator cost function: sum_i max_O(|<in_i|O|in_i> - <out_i|O|out_i>|)
def _cost_fn_earth_mover_and_operator(compressing_convolutional_params,
                                      compressing_pooling_params,
                                      expanding_convolutional_params,
                                      expanding_pooling_params,
                                      cost_type:str,
                                      use_SVD_unitary_Q:bool=False):
    if not (cost_type == 'earth_mover' or cost_type == 'operator' or cost_type == 'hamiltonian'):
        raise Exception(f'Incorrect cost type {cost_type} in _cost_fn_earth_mover_and_operator().')
    
    cost = 0.
    n_states = len(input_states)
    output_dms = []
    for input_state in input_states:
        middle_dm, _, expval_Z_traced_qubits = vae_compress(
                                                compress_convolution_parameters=compressing_convolutional_params,
                                                compress_pooling_parameters=compressing_pooling_params,
                                                state_vector_in=input_state,
                                                measure_traced_qubits_Q=(abs(system_params['coeff_traced_cost'])>0),
                                                use_SVD_unitary_Q=use_SVD_unitary_Q)

        if system_params['use_jax_Q']:
            cost += system_params['coeff_traced_cost'] * jnp.sum(jnp.array([(z-1)**2 for z in expval_Z_traced_qubits])) / len(system_params['traced_qubits'])
        else:
            cost += system_params['coeff_traced_cost'] * np.sum(np.array([(z-1)**2 for z in expval_Z_traced_qubits])) / len(system_params['traced_qubits'])

        if system_params['same_compress_expand_Q']:
            output_dm = vae_expand(compressing_convolutional_params,
                                   compressing_pooling_params,
                                   middle_dm,
                                   decomposed_Q=False,
                                   use_SVD_unitary_Q=use_SVD_unitary_Q)
        else:
            output_dm = vae_expand(expanding_convolutional_params,
                                   expanding_pooling_params,
                                   middle_dm,
                                   decomposed_Q=False,
                                   use_SVD_unitary_Q=use_SVD_unitary_Q)

        if system_params['use_jax_Q']:
            assert jnp.abs(jnp.trace(output_dm) - 1.) < 1e-6
            output_dm /= jnp.trace(output_dm)
        else:
            assert np.abs(np.trace(output_dm) - 1.) < 1e-6
            output_dm /= np.trace(output_dm)
        output_dms += [output_dm]

    if cost_type == 'earth_mover' or cost_type == 'operator':
        # Generate operators defined by earth_mover_cost_operator_support
        # and earth_mover_cost_operator_support_probs
        # This is re-generated every time since the operators are sampled if
        # support_prob_list is not identically 1., and the operators get
        # shifted around if translation_invariance_Q is True
        Pauli_string_lists = get_Pauli_strings(n_system_sites=system_params['num_input_qubits'],
                                               operator_support_list=system_params['operator_support'],
                                               support_prob_list=system_params['operator_support_probs'],
                                               translation_invariance_Q=system_params['operator_translation_invariance_Q'],
                                               max_rng=system_params['operator_support_max_range'],
                                               random_shift_sites_Q=True)
    elif cost_type == 'hamiltonian':
        Pauli_string_lists = get_Hamiltonian_Pauli_strings(n_system_sites=system_params['num_input_qubits'],
                                                           data_type=system_params['data_type'],
                                                           translation_invariance_Q=system_params['operator_translation_invariance_Q'],
                                                           random_shift_sites_Q=True)

    
    # Flatten the top level of Pauli_strings_lists
    # Example:
    # Pauli_string_lists = [[(0, 'x')],
    #                       [(0, 'y')],
    #                       [(0, 'z')],
    #                       [(2, 'x'), (3, 'x')],
    #                       [(2, 'x'), (3, 'y')],
    #                       [(2, 'x'), (3, 'z')],
    #                       [(2, 'y'), (3, 'x')],
    #                       [(2, 'y'), (3, 'y')],
    #                       [(2, 'y'), (3, 'z')],
    #                       [(2, 'z'), (3, 'x')],
    #                       [(2, 'z'), (3, 'y')],
    #                       [(2, 'z'), (3, 'z')]]
    Pauli_string_lists = [Pauli_string \
                          for Pauli_string_list in Pauli_string_lists \
                          for Pauli_string in Pauli_string_list]
    
    if cost_type == 'earth_mover':
        # Pauli_string_lists_indices contains only the index of the sites in Pauli_string_lists (auxiliary variable)
        # E.g. in the example above, it is
        # Pauli_string_lists = [[0], [0], [0], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]
        Pauli_string_lists_indices = [[P[0] for P in Ps] for Ps in Pauli_string_lists]
        
        # P_mx is the indicator matrix of whether a particular site is in Pauli_string_lists
        # Its size is num_input_qubits x len(Pauli_string_lists_indices)
        # It is used in the linear programming condition needed to determine the earth mover distance
        # E.g. in the example above,
        # P_mx = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                  [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #                  [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        P_mx = numpy.array([[1. if i in PSI else 0 for PSI in Pauli_string_lists_indices] for i in range(system_params['num_input_qubits'])])
    
    # Calculat the expectation value of each Pauli string in each
    # input state or output density matrix
    # Generates the following results:
    # - expval_input_list_list: an array of size len(input_states) x len(Pauli_string_list)
    # - expval_output_list_list: an array of size len(output_dms) x len(Pauli_string_list)
    expval_input_list_list = expval_Pauli_strings(input_states,
                                                  Pauli_string_lists,
                                                  in_mid_out_Q='input')

    expval_output_list_list = expval_Pauli_strings(output_dms,
                                                   Pauli_string_lists,
                                                   in_mid_out_Q='output')
    
    if cost_type == 'earth_mover':
        # The earth mover distance part of the cost function for each pair of input and output states is
        # max_{w} sum_op w_op * c_op, where c_op = Tr(op @ (out_dm - in_dm))
        # with the condition P_mx @ |w| <= 1
        # The condition defines a len(Pauli_string_lists) dimensional simplicial complex which is independent
        # of the parameters of the network, and it only depends on the choice of the correlators.
        # Within this simplex, we need to maximize the overlap between the vector c and w which
        # correspond to the operator expectation values and the weight coefficients, respectively.
        # This will be maximized by w values that are in the corner of this simplicial complex
        # closest to c.
        # For generic values of the network parameters and thus for generic values of c, small perturbations
        # in c will not change which w maximizes the overlap. Therefore, the derivative of c will
        # not depend on the linear constraints. We can simply take the resulting w as a constant
        # and take the derivative w.r.t. c only.
        #
        # Note, however, that the corner of the simplex maximizing c.T @ w can change during the
        # gradient descent learning procedure. This can lead to wiggles in the cost function during
        # minimization.
        w = cvxpy.Variable(len(Pauli_string_lists_indices))
    
        for i_state in range(n_states):
            # expval_output_list_list is of qml.ArrayBox type, hence it needs to be transformed to regular numpy arrays
            expval_diff = qml.math.toarray(expval_output_list_list[i_state]) - numpy.array(expval_input_list_list[i_state])
            lin_prog_problem = cvxpy.Problem(cvxpy.Maximize(expval_diff.T @ w), [P_mx @ cvxpy.abs(w) <= 1.])
            lin_prog_problem.solve()
            
            # Note that we cannot use the numpy vector expval_diff in the cost function
            # Instead, we need to use the pennylane.numpy or jax.numpy vectors that allow us to differentiate
            # the cost finction. The solution of the optimization, however, is a simple constant vector
            # that we don't take the gradient of w.r.t. the VAE parameters
            cost += (expval_output_list_list[i_state] - expval_input_list_list[i_state]).T @ w.value
    elif cost_type == 'operator' or cost_type == 'hamiltonian':
        if system_params['use_jax_Q']:
            cost += jnp.sum(jnp.array([abs(expval_input_list_list[i][j]-expval_output_list_list[i][j])
                                       for j in range(len(Pauli_string_lists))
                                       for i in range(len(input_states))]))
        else:
            cost += np.sum(np.array([abs(expval_input_list_list[i][j]-expval_output_list_list[i][j])
                                     for j in range(len(Pauli_string_lists))
                                     for i in range(len(input_states))]))

    cost /= n_states

    return cost
    # Looked the code through, looks good

# Earth mover cost function
# Approximation of the earth mover distance using the approach of
# B. T. Kiani, G. De Palma, M. Marvian, Z.-W. Liu, and Seth Lloyd,
# Quant. Sci. Technol. 7, 045002 (2022), equation (20).
def cost_fn_earth_mover(compressing_convolutional_params,
                        compressing_pooling_params,
                        expanding_convolutional_params,
                        expanding_pooling_params,
                        use_SVD_unitary_Q:bool=False):
    return _cost_fn_earth_mover_and_operator(compressing_convolutional_params,
                                     compressing_pooling_params,
                                     expanding_convolutional_params,
                                     expanding_pooling_params,
                                     'earth_mover',
                                     use_SVD_unitary_Q)

# Operator cost function: sum_i max_O(|<in_i|O|in_i> - <out_i|O|out_i>|)
def cost_fn_operator(compressing_convolutional_params,
                     compressing_pooling_params,
                     expanding_convolutional_params,
                     expanding_pooling_params,
                     use_SVD_unitary_Q:bool=False):
    return _cost_fn_earth_mover_and_operator(compressing_convolutional_params,
                                             compressing_pooling_params,
                                             expanding_convolutional_params,
                                             expanding_pooling_params,
                                             'operator',
                                             use_SVD_unitary_Q)

# Operator cost function: sum_i max_O(|<in_i|O|in_i> - <out_i|O|out_i>|)
#  where the operators are chosen from the Hamiltonian
def cost_fn_hamiltonian(compressing_convolutional_params,
                        compressing_pooling_params,
                        expanding_convolutional_params,
                        expanding_pooling_params,
                        use_SVD_unitary_Q:bool=False):
    return _cost_fn_earth_mover_and_operator(compressing_convolutional_params,
                                             compressing_pooling_params,
                                             expanding_convolutional_params,
                                             expanding_pooling_params,
                                             'hamiltonian',
                                             use_SVD_unitary_Q)

# Generate constant quantities used in cost_fn_earth_mover_compressed()
# All operators are considered in the compressed site
if system_params['cost_type'] == 'earth_mover_compressed' \
   or system_params['cost_type'] == 'operator_compressed':
    emoc_all_Pauli_strings_middle = get_Pauli_strings(n_system_sites=system_params['num_middle_qubits'],
                                                     operator_support_list=list(range(1, system_params['num_middle_qubits']+1)),
                                                     support_prob_list=[1. for _ in range(system_params['num_middle_qubits'])],
                                                     translation_invariance_Q=False,
                                                     max_rng=system_params['num_middle_qubits'],
                                                     random_shift_sites_Q=True)

    # Flatten the top level of Pauli_strings_lists
    emoc_Pauli_string_lists = [Pauli_string \
                          for Pauli_string_list in emoc_all_Pauli_strings_middle \
                          for Pauli_string in Pauli_string_list]
                          
    # Index the positions of the operators using the middle wires
    emoc_Pauli_string_lists_reduced = [[(system_params['middle_qubits'][p[0]], p[1]) for p in Pauli_string]
                          for Pauli_string in emoc_Pauli_string_lists]
    
    # Adding the string of Z operators on the traced qubits
    emoc_ZZ_string = [(i, 'z') for i in system_params['traced_qubits']]
    emoc_Pauli_string_lists_pure = [emoc_ZZ_string + Pauli_string for Pauli_string in emoc_Pauli_string_lists_reduced]

    emoc_Pauli_string_lists_pure_indices = [[P[0] for P in Ps] for Ps in emoc_Pauli_string_lists_pure]
    emoc_P_mx = numpy.array([[1. if i in PSI else 0 for PSI in emoc_Pauli_string_lists_pure_indices]
                        for i in range(system_params['num_input_qubits'])])
    
    del emoc_all_Pauli_strings_middle, emoc_Pauli_string_lists, emoc_ZZ_string, emoc_Pauli_string_lists_pure_indices
    
# Definition of the earth mover distance and the operator distance in the compressed space
# cost_type == 'earth_mover_compressed', or 'operator_compressed'
def _cost_fn_earth_mover_and_operator_compressed(compressing_convolutional_params,
                                                 compressing_pooling_params,
                                                 cost_type:str,
                                                 use_SVD_unitary_Q:bool=False):
    if not (cost_type == 'earth_mover_compressed' or cost_type == 'operator_compressed'):
        raise Exception(f'Incorrect cost type {cost_type} in _cost_fn_earth_mover_and_operator_compressed().')
    
    cost = 0.
    middle_reduced_dms = []
    middle_pure_states = []
    n_states = len(input_states)
    for input_state in input_states:
        middle_reduced_dm, middle_pure_state, expval_Z_traced_qubits = vae_compress(
                                                compress_convolution_parameters=compressing_convolutional_params,
                                                compress_pooling_parameters=compressing_pooling_params,
                                                state_vector_in=input_state,
                                                measure_traced_qubits_Q=(abs(system_params['coeff_traced_cost'])>0),
                                                use_SVD_unitary_Q=use_SVD_unitary_Q)
        middle_reduced_dms += [middle_reduced_dm]
        middle_pure_states += [middle_pure_state]

        if system_params['use_jax_Q']:
            cost += system_params['coeff_traced_cost'] * jnp.sum(jnp.array([(z-1)**2 for z in expval_Z_traced_qubits])) / len(system_params['traced_qubits'])
        else:
            cost += system_params['coeff_traced_cost'] * np.sum(np.array([(z-1)**2 for z in expval_Z_traced_qubits])) / len(system_params['traced_qubits'])
    
    # Array containing the expectation values of the middle qubits
    expval_middle_reduced_list_list = expval_Pauli_strings(middle_reduced_dms,
                                                           emoc_Pauli_string_lists_reduced,
                                                           in_mid_out_Q='middle')
    # Note that instead of what is in Pauli_string_lists, we are actually also
    # measureing the Z expectation value on the traced qubits. But since in these
    # qubits the state is |1..111>, the expectation value of the Z..ZZZ operator is 1.
    # Hence, it is enough to calculate the expectation value of the Pauli string
    expval_middle_pure_list_list = expval_Pauli_strings(middle_pure_states,
                                                        emoc_Pauli_string_lists_pure,
                                                        in_mid_out_Q='input')
    
    if cost_type == 'earth_mover_compressed':
        # From this point on, the calculation of the earth mover distance is
        # identical to that in the function cost_fn_earth_mover().
        # The comments related to that part still hold
        w = cvxpy.Variable(len(emoc_Pauli_string_lists_pure))
        for i_state in range(n_states):
            expval_diff = qml.math.toarray(expval_middle_reduced_list_list[i_state]) - qml.math.toarray(expval_middle_pure_list_list[i_state])
            lin_prog_problem = cvxpy.Problem(cvxpy.Maximize(expval_diff.T @ w), [emoc_P_mx @ cvxpy.abs(w) <= 1.])
            lin_prog_problem.solve()
            cost += (expval_middle_reduced_list_list[i_state] - expval_middle_pure_list_list[i_state]).T @ w.value
    elif cost_type == 'operator_compressed':
        if system_params['use_jax_Q']:
            cost += jnp.sum(jnp.array([max([abs(expval_middle_pure_list_list[i][j] - expval_middle_reduced_list_list[i][j])
                                       for j in range(len(emoc_Pauli_string_lists_pure))])
                                       for i in range(len(input_states))]))
        else:
            cost += np.sum(np.array([max([abs(expval_middle_pure_list_list[i][j] - expval_middle_reduced_list_list[i][j])
                                       for j in range(len(emoc_Pauli_string_lists_pure))])
                                       for i in range(len(input_states))]))

    cost /= n_states
    return cost
    # Looked the code through, looks good


# Earth mover cost function in the compressed space
# Approximation of the earth mover distance using the approach of
# B. T. Kiani, G. De Palma, M. Marvian, Z.-W. Liu, and Seth Lloyd,
# Quant. Sci. Technol. 7, 045002 (2022), equation (20).
def cost_fn_earth_mover_compressed(compressing_convolutional_params,
                                   compressing_pooling_params,
                                   use_SVD_unitary_Q:bool=False):
    return _cost_fn_earth_mover_and_operator_compressed(compressing_convolutional_params,
                                                        compressing_pooling_params,
                                                        'earth_mover_compressed',
                                                        use_SVD_unitary_Q)

# Operator cost function in the compressed space: sum_i max_O(|<in_i|O|in_i> - <out_i|O|out_i>|)
def cost_fn_operator_compressed(compressing_convolutional_params,
                                compressing_pooling_params,
                                use_SVD_unitary_Q:bool=False):
    return _cost_fn_earth_mover_and_operator_compressed(compressing_convolutional_params,
                                                        compressing_pooling_params,
                                                        'operator_compressed',
                                                        use_SVD_unitary_Q)


# -----------------------------
# TEST TRANSLATIONAL INVARIANCE
# -----------------------------
def test_trans_inv(compressing_convolutional_params,
                   compressing_pooling_params,
                   expanding_convolutional_params,
                   expanding_pooling_params,
                   n_random_Pauli_strings:int=20,
                   error_tolerance:float=1e-8):
    if not (system_params['operator_translation_invariance_Q'] \
            and (system_params['cost_type'] == 'earth_mover' \
                 or system_params['cost_type'] == 'operator')):
        return
    
    n_states = len(input_states)
    output_dms = []
    for input_state in input_states:
        middle_dm, _, _ = vae_compress(compress_convolution_parameters=compressing_convolutional_params,
                                       compress_pooling_parameters=compressing_pooling_params,
                                       state_vector_in=input_state,
                                       measure_traced_qubits_Q=False,
                                       use_SVD_unitary_Q=False)

        if system_params['same_compress_expand_Q']:
            output_dm = vae_expand(compressing_convolutional_params,
                                   compressing_pooling_params,
                                   middle_dm,
                                   decomposed_Q=False,
                                   use_SVD_unitary_Q=False)
        else:
            output_dm = vae_expand(expanding_convolutional_params,
                                   expanding_pooling_params,
                                   middle_dm,
                                   decomposed_Q=False,
                                   use_SVD_unitary_Q=False)
        output_dms += [output_dm]

    Pauli_string_lists = []
    shifted_Pauli_string_lists = []
    n_sites = system_params['num_input_qubits']
    sites = np.arange(n_sites)
    for i in range(n_random_Pauli_strings):
        n_ops = random.randint(1, n_sites)
        chosen_sites = list(np.random.choice(sites, size=n_ops, replace=False))
        chosen_sites = [int(i) for i in chosen_sites]
        shift = random.choice(list(range(0, n_sites, 2)))
        Pauli_string_lists += [[(s, random.choice(['x', 'y', 'z'])) for s in chosen_sites]]
        shifted_Pauli_string_lists += [[((s_op[0] + shift) % n_sites, s_op[1]) for s_op in Pauli_string_lists[-1]]]

    expval_Pauli_list_list = expval_Pauli_strings(output_dms,
                                                  Pauli_string_lists,
                                                  in_mid_out_Q='output')
    expval_shifted_Pauli_list_list = expval_Pauli_strings(output_dms,
                                                          shifted_Pauli_string_lists,
                                                          in_mid_out_Q='output')

    err = np.linalg.norm(np.array(expval_Pauli_list_list) - np.array(expval_shifted_Pauli_list_list))
    if err > error_tolerance:
        print('\n\n')
        print('#'*80)
        print(f'Error: Pauli string and its shifted version differs larger than {err} > {error_tolerance}.')
        for i in range(len(Pauli_string_lists)):
            diff = np.array(expval_Pauli_list_list).T[i] - np.array(expval_shifted_Pauli_list_list).T[i]
            diff = [float(d) for d in diff]
            if np.linalg.norm(diff) > error_tolerance:
                print(f'Pauli string list:\t\t{Pauli_string_lists[i]}')
                print(f'Pauli string list shifted:\t{shifted_Pauli_string_lists[i]}')
                print(f'<out|Pauli_string_list - shifted_Pauli_string_list|out> = \n{diff}\n')
        print('#'*80)
        print('\n\n')
    return
    # Looked the code through, looks good

# -----------------------
# PRINT RESULTING NETWORK
# -----------------------
def network_to_txt(ccps, cpps, ecps, epps):
    ccps_interp = np.array(copy.deepcopy(ccps), requires_grad=False)
    cpps_interp = np.array(copy.deepcopy(cpps), requires_grad=False)
    if not system_params['same_compress_expand_Q']:
        ecps_interp = np.array(copy.deepcopy(ecps), requires_grad=False)
        epps_interp = np.array(copy.deepcopy(epps), requires_grad=False)
    else:
        ecps_interp = np.array(copy.deepcopy(ccps), requires_grad=False)
        epps_interp = np.array(copy.deepcopy(cpps), requires_grad=False)

    # Create drawer of the networks
    drawer_compress_interp = qml.draw(vae_compress_interp)
    drawer_expand_interp = qml.draw(vae_expand_interp)
    
    # Choose any input state
    input_state = input_states[-1]
    middle_dm, _, _ = vae_compress(compress_convolution_parameters=ccps,
                                   compress_pooling_parameters=cpps,
                                   state_vector_in=input_state,
                                   decomposed_Q=False,
                                   measure_traced_qubits_Q=False)
    
    # Print networks
    net_str  = ''
    net_str += '\n\n\n'
    net_str += '#################\n'
    net_str += 'Resulting network\n'
    net_str += '#################\n'
    net_str += '\nCompress:\n'
    net_str += drawer_compress_interp(ccps_interp, cpps_interp,
                                      input_state,
                                      decomposed_Q=False)
    net_str += '\n\nExpand:'
    net_str += drawer_expand_interp(ecps_interp, epps_interp,
                                    middle_dm,
                                    decomposed_Q=False)
    net_str += '\n\n\n'
    
    net_str += '##################\n'
    net_str += 'Decomposed network\n'
    net_str += '##################\n'
    net_str += '\nCompress:\n'
    net_str += drawer_compress_interp(ccps_interp, cpps_interp,
                                      input_state,
                                      decomposed_Q=True)
    net_str += '\n\nExpand:\n'
    net_str += drawer_expand_interp(ecps_interp, epps_interp,
                                    middle_dm,
                                    decomposed_Q=True)
    net_str += '\n\n\n'
    
    return net_str
    # Looked the code through, looks good
