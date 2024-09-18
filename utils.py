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




def original_swap(wires):
  qml.Hadamard(wires=0)
  
  for target in wires:
    qml.CSWAP(wires=[0,target[0], target[1]])
  qml.Hadamard(wires=0)
  return qml

def isotropic_state( p, wires):
    qml.Hadamard(wires=wires[0])
    theta = 2 * np.arccos(np.sqrt(p))
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

