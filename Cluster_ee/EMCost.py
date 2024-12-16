import pennylane as qml
from pennylane import numpy as np
import os 
os.environ["JAX_PLATFORMS"] = "cpu"
import pennylane as qml
from pennylane import numpy as np
import numpy
import numbers
import cvxpy
import itertools
import random
from pennylane.math import reduce_dm
import jax
import jax.numpy as jnp


system_params={}
vae_dev_mixed_input=None
vae_dev_mixed_middle=None
vae_dev_mixed_output=None

def set_global( num_input_qubits,output_qubits,n_trash_qubits,operator_support,operator_support_probs,operator_translation_invariance_Q,operator_support_max_range,use_jax=True):
    global system_params
    global vae_dev_mixed_input
    global vae_dev_mixed_middle
    global vae_dev_mixed_output
    system_params={'num_input_qubits': num_input_qubits,
                #    'middle_qubits':middle_qubits,
                   'output_qubits':output_qubits,
                   'trash_qubits':n_trash_qubits,
                   'operator_support':operator_support,
                   'operator_support_probs':operator_support_probs,
                   'operator_translation_invariance_Q':operator_translation_invariance_Q,
                   'operator_support_max_range':operator_support_max_range,
                   'use_jax_Q':use_jax}
    
    vae_dev_mixed_input = qml.device('default.mixed', wires=system_params['num_input_qubits'])
    # vae_dev_mixed_middle = qml.device('default.mixed', wires=system_params['middle_qubits'])
    vae_dev_mixed_output = qml.device('default.mixed', wires=system_params['output_qubits'])
    # Calculate the expectation value of Pauli strings on input states
    # Looked the code through, looks good

def expval_operators_input(state_in, operators):
    @jax.jit
    def jax__expval_operators_input(state, operators):
        return _expval_operators_input(state, operators)
    @qml.qnode(vae_dev_mixed_input,interface='jax' if system_params['use_jax_Q'] else 'autograd')
    def _expval_operators_input(state, operators):
        # print(state.shape,vae_dev_mixed_input.wires)
        state_op_expval = []
        qml.QubitDensityMatrix(state, vae_dev_mixed_input.wires)
        return [qml.expval(op) for op in operators]


    if system_params['use_jax_Q']:
        return jax__expval_operators_input(state_in, operators)
    else:
        return _expval_operators_input(state_in, operators)
    # Looked the code through, looks good

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
 
    
# def expval_operators_middle(dm_mid, operators):
#     @qml.qnode(vae_dev_mixed_middle)
#     def _expval_operators_middle(dm_mid, operators):
#         state_op_expval = []
#         qml.QubitDensityMatrix(dm_mid, vae_dev_mixed_middle.wires)
#         return [qml.expval(op) for op in operators]
#     return _expval_operators_middle(dm_mid, operators)


def expval_operators_output(dm_out, operators):
    @qml.qnode(vae_dev_mixed_output,interface='jax' if system_params['use_jax_Q'] else 'autograd')
    def _expval_operators_output(dm_out, operators):
        state_op_expval = []
        qml.QubitDensityMatrix(dm_out, vae_dev_mixed_output.wires)
        return [qml.expval(op) for op in operators]
    return _expval_operators_output(dm_out, operators)
    
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

def expval_Pauli_strings(states_in_list,
                         Pauli_string_list,
                         in_mid_out_Q:str='input'):
    ops = operators_from_Pauli_strings(Pauli_string_list)
    state_op_expval = []
    if in_mid_out_Q=='input':
        for state in states_in_list:
            state_op_expval += [expval_operators_input(state, ops)]
    # elif in_mid_out_Q=='middle':
    #     for state in states_in_list:
    #         state_op_expval += [expval_operators_middle(state, ops)]
    elif in_mid_out_Q=='output':
        for state in states_in_list:
            state_op_expval += [expval_operators_output(state, ops)]
    state_op_expval = jnp.array(state_op_expval)
    return state_op_expval



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






def cost_fn_EM(X,trainer,input_states):
    def _cost_fn_EM(w):
        cost = 0.
        n_states = len(X)
        output_dms = jnp.array([trainer(w,x) for x in X])
        # return the density matrix of the autoencoder
        # for input_state in input_states:
        #     middle_dm, _, expval_Z_traced_qubits = vae_compress(
        #                                             compress_convolution_parameters=compressing_convolutional_params,
        #                                             compress_pooling_parameters=compres
        # sing_pooling_params,
        #                                             state_vector_in=input_state,
        #                                             measure_traced_qubits_Q=(abs(system_params['coeff_traced_cost'])>0),
        #                                             use_SVD_unitary_Q=use_SVD_unitary_Q)

        #     if system_params['use_jax_Q']:
        #         cost += system_params['coeff_traced_cost'] * jnp.sum(jnp.array([(z-1)**2 for z in expval_Z_traced_qubits])) / len(system_params['traced_qubits'])
        #     else:
        #         cost += system_params['coeff_traced_cost'] * np.sum(np.array([(z-1)**2 for z in expval_Z_traced_qubits])) / len(system_params['traced_qubits'])

        #     if system_params['same_compress_expand_Q']:
        #         output_dm = vae_expand(compressing_convolutional_params,
        #                                compressing_pooling_params,
        #                                middle_dm,
        #                                decomposed_Q=False,
        #                                use_SVD_unitary_Q=use_SVD_unitary_Q)
        #     else:
        #         output_dm = vae_expand(expanding_convolutional_params,
        #                                expanding_pooling_params,
        #                                middle_dm,
        #                                decomposed_Q=False,
        #                                use_SVD_unitary_Q=use_SVD_unitary_Q)

        #     if system_params['use_jax_Q']:
        #         assert jnp.abs(jnp.trace(output_dm) - 1.) < 1e-6
        #         output_dm /= jnp.trace(output_dm)
        #     else:
        #         assert np.abs(np.trace(output_dm) - 1.) < 1e-6
        #         output_dm /= np.trace(output_dm)
        #     output_dms += [output_dm]

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
        
        
        #FO this input state should be simply self.__sp(x,0)

        expval_input_list_list = expval_Pauli_strings(input_states,
                                                    Pauli_string_lists,
                                                    in_mid_out_Q='input')

        expval_output_list_list = expval_Pauli_strings(output_dms,
                                                    Pauli_string_lists,
                                                    in_mid_out_Q='output')
        
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

        cost /= n_states

        return cost

    return _cost_fn_EM



