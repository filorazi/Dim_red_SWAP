import pennylane as qml
from pennylane import numpy as np
import pandas as pd
import jax
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mse
from scipy.optimize import minimize
from IPython.display import clear_output
import random 
from pennylane.optimize import AdamOptimizer,QNSPSAOptimizer
from qutip import *
from qutip import gates
import matplotlib.pyplot as plt




def original_swap(wires):
  qml.Hadamard(wires=0)
  
  for target in wires:
    qml.CSWAP(wires=[0,target[0], target[1]])
  qml.Hadamard(wires=0)
  return qml

def isotropic_state( p, wires):
    qml.Hadamard(wires=wires[0])
    theta = p
    for i in wires[:-1]:
        qml.CNOT(wires=[i , 1+i])
    for i in wires:
        qml.RX(theta, wires=i)
    for i in wires[:-1]:
        qml.CNOT(wires=[i , 1+i])
  
def destructive_swap(n_qubit):
    for wires in range(n_qubit): 
        qml.CNOT(wires=[wires,wires+n_qubit])
        qml.Hadamard(wires)

    return qml

def interpret_results(data):

  def check_parity_bitwise_and(s):
    n = len(s)
    first_half = s[:n//2]
    second_half = s[n//2:]
    and_result = ''.join('1' if first_half[i] == '1' and second_half[i] == '1' else '0' for i in range(n//2))
    parity = and_result.count('1') % 2

    return parity
  
  comb = [bin(i)[2:].zfill(int(np.log2(len(data)))) for i in range(len(data))]
  dictdata = dict(zip(comb,data))
  kk={}
  for k,item in dictdata.items():
    if dictdata[k]>0.00000001:
      kk[k]=item
  fail =0
  for k,i in kk.items():
    if check_parity_bitwise_and(k):
      fail += i
  return fail

## TODO
'''
def train_log_depth(X,opt,n_qubit_autoencoder,repetition,epochs,visual=False):
    loss = []   
    layerparam=6
    n_qubit_swap=n_qubit_autoencoder-n_qubit_autoencoder//(2**(repetition)) +1
    n_qubit=n_qubit_autoencoder+n_qubit_swap 
    num_params=sum([layerparam*n_qubit_autoencoder//2**(i+1) for i in range(repetition)])
    num_params=2*np.sum([i for i in range(n_qubit_autoencoder-n_qubit_swap,n_qubit_autoencoder+1)])
    weights = np.array([random.uniform(0, np.pi) for _ in range(num_params)], requires_grad=True)
    wq=[weights]

    dvc=qml.device('default.qubit', wires=n_qubit, shots=None)
    @qml.qnode(dvc,diff_method='adjoint')
    def trainer(param,p):
        
        create_isotropic_state(p, n_qubit_autoencoder, n_qubit_swap)
        qml.Barrier()
        autoencoder_fulldense(param, n_qubit_autoencoder,n_qubit_swap)
        qml.Barrier()
        original_swap(n_qubit_swap)
        return qml.probs([0])
    
    def loss_function(w): 
        pred =np.array([trainer(w,x)[1] for x in X], requires_grad=True)
        current_loss = pred.mean()
        return current_loss

    for epoch in range(epochs):
        weights, loss_value = opt.step_and_cost(loss_function, wq[-1])
        print(f'Epoch {epoch}: Loss = {loss_value}',end='\r')

        loss.append(loss_value)
        wq.append(weights)
    if visual:
        fig, ax = qml.draw_mpl(trainer)(wq[-1],.56)
        plt.show(   )


    return loss, wq

'''

def compare_state_orig(n_qb_input):
    @qml.qnode(qml.device('default.qubit', wires=n_qb_input*2+1, shots=1000))
    def pio(param):
        isotropic_state(param[0],list(range(1,n_qb_input+1)))
        isotropic_state(param[1],list(range(n_qb_input+1,n_qb_input*2+1)))
        
        qml.Barrier()
        original_swap([(1+a,1+a+n_qb_input) for a in range(n_qb_input)])
        return qml.probs([0])
    return pio

def compare_state_ae(n_qb_input,n_qb_trash,ae):
    @qml.qnode(qml.device('default.qubit', wires=n_qb_input*2+1, shots=1000))
    def pio(param):
        isotropic_state(param[0],list(range(1,n_qb_input+1)))
        isotropic_state(param[1],list(range(n_qb_input+1,n_qb_input*2+1)))
        
        qml.Barrier()
        ae.get_cirq(1)
        ae.get_cirq(n_qb_input+1)

        qml.Barrier()
        original_swap([(1+n_qb_trash+a,1+n_qb_trash+a+n_qb_input) for a in range(n_qb_input-n_qb_trash)])
        return qml.probs([0])
    return pio


def compare_fidelity(n_qubit_autoencoder,n_trash_qubit,ae,loc=9):
    c1=[]
    c2=[]
    for a in np.linspace(0,np.pi,500):
        res1 = compare_state_ae(n_qubit_autoencoder,n_trash_qubit,ae)([a,0])
        res2 = compare_state_orig(n_qubit_autoencoder)([a,0])
        c1.append(res1[0])
        c2.append(res2[0])

    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    lns1=ax.plot( np.array(range(len(c1)))/100,c1,label=['Reduced'])
    lns2=ax.plot( np.array(range(len(c2)))/100,c2,label=['Original'])
    ax.set_ylim((0,1))
    errors = np.abs(np.array(c2)-np.array(c1))
    lns3=ax2.plot( np.array(range(len(c2)))/100,errors,label=['Relative error'],color='red')
    from sklearn.metrics import mean_squared_error as mse 
    print(f'MSE of the error is {mse(c1,c2)}')
    ax2.set_ylim([0,1])
    lns = lns1+lns2+lns3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs,loc=loc)
    ax.set_xlabel("p")
    ax.set_ylabel(r"Probability of passing the SWAP test")
    ax2.set_ylabel(r"Relative error")
    plt.show();



def sp_qutip(theta,n_qubit):
    qb=[basis(2, 0) for _ in range(n_qubit)]
    state = tensor(qb)  # Tensor product of the qubits
    H=gates.hadamard_transform()
    hi=[H] + [qeye(2)]*(n_qubit-1)
    state  = tensor(hi)*state
    CNOT = gates.cnot()
    for i in range(n_qubit-1):

        cnoti =[ qeye(2)]*(i) + [CNOT]
        cnoti += [qeye(2)]*(n_qubit-2-i)
        state = tensor(cnoti) * state
        
    rx = gates.rx(theta)
    rxx= [rx]*n_qubit
    state  = tensor(rxx)*state
    for i in range(n_qubit-1):
        cnoti =[ qeye(2)]*(i) + [CNOT]
        cnoti += [qeye(2)]*(n_qubit-2-i)
        state = tensor(cnoti) * state
    return state

def get_min_loss_fid(X,qb_input_state,qb_trash_state):
    a=[]
    for theta in X:
        a.append(sp_qutip(theta,qb_input_state))
    b =np.sum([tensor(c, c.dag()) for c in a])/len(a)
    rank=sum([a>1e-10 for a in b.eigenenergies()])
    c=np.sum(b.eigenenergies()[-pow(2,qb_input_state-qb_trash_state)-1:])
    return 1 - c,  rank


def get_eigen_loss_values(X,qb_input_state,qb_trash_state):
    a=[]
    for theta in X:
        a.append(sp_qutip(theta,qb_input_state))
    b =np.sum([tensor(c, c.dag()) for c in a])/len(a)
    return [1-np.sum(b.eigenenergies()[-a:]) for a in range(pow(2,qb_input_state-qb_trash_state))]

def delete_following_elements(lst, eta):
    result = [lst[0]]  
    i = 0  
    while i < len(lst) - 1:
        if abs(lst[i] - lst[i + 1]) > eta:
            result.append(lst[i + 1])  # Add the next element if the condition holds
        i += 1  # Move to the next element
    
    return [x for x in result if abs(x) >= 0.0001 ]

def get_eigen_loss_values(X,qb_input_state,qb_trash_state):
    a=[]
    for theta in X:
        a.append(sp_qutip(theta,qb_input_state))
    b =np.sum([tensor(c, c.dag()) for c in a])/len(a)
    
    c=[]
    for j in range(pow(2,qb_input_state-qb_trash_state)):
        c.append(1-np.sum(b.eigenenergies()[-j-1:]))


    return c

def compute_ovelap_matrix(n_qubit, segm=100):
    n=[]
    for a in np.linspace(0,np.pi,segm):
        nn=[]
        for b in np.linspace(0,np.pi,segm):
            nn.append(compare_state_orig(n_qubit)([a,b])[0])
  
        n.append(nn)
    return n 
