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


def autoencoder_fulldense(param,n_qubit,trash_qubit,original_swap=True):
    strprm=0
    for a in range(trash_qubit-1*original_swap):
        start=trash_qubit+a
        current_param=2*(n_qubit-a)
        fclayer(n_qubit-a,param[strprm:current_param+strprm],start=start)
        strprm=current_param+strprm

def create_isotropic_state(p, n_qubit, start):
    qml.Hadamard(wires=start)
    theta = 2 * np.arccos(np.sqrt(p))

    for i in range(n_qubit-1):
        qml.CNOT(wires=[start+i , start+1+i])
    
    for i in range(n_qubit):
        qml.RX(theta, wires=start+i)

    for i in range(n_qubit-1):
        qml.CNOT(wires=[start+i , start+1+i])
    return qml

def fclayer(qb,parameter,start):
    for i in range(start,qb+start):
       qml.RY(parameter[(i-start)*2+1],wires=i)
    for i in range(start,qb+start-1,2):
       qml.CNOT(wires=[i,i+1])
    for i in range(start+1,qb+start-1,2):
        qml.CNOT(wires=[i,i+1])
    for i in range(start,qb+start):
       qml.RX(parameter[(i-start)*2+qb],wires=i)
   
def dense(a,b,parameters):
    qml.RY(parameters[0],wires=a)
    qml.RY(parameters[1],wires=b)
    qml.RX(parameters[2],wires=a)
    qml.RX(parameters[3],wires=b)
    qml.CNOT(wires=[a,b])
    qml.RY(parameters[4],wires=a)
    qml.RY(parameters[5],wires=b)
    qml.CNOT(wires=[b,a])

def pool(a,b):
    qml.CRZ(np.pi,wires=[a,b])
    qml.X(a)  
    qml.CRX(np.pi,wires=[a,b])

def autoencoder(offset,param,repetition,n_qubit):
    start=0
    layerparam=6
    for i in range(repetition):
        if start % 2!=0:
            raise Exception('The number of qubits should be a power of 2 greater than 2 to the power of repetition')

        for a in range(start,( n_qubit-start)//2+start):
            param_corrente=sum([layerparam*n_qubit//2**(j+1) for j in range(i)])+(a-start)*layerparam            
            a+=offset
            dense(a,a+(n_qubit-start)//2,param[param_corrente:param_corrente +layerparam])
            pool(a,a+(n_qubit-start)//2)
        start+=n_qubit//(2**(i+1))


def original_swap(n_qubit_swap):
  qml.Hadamard(wires=0)
  for wires in range(1,n_qubit_swap):
    qml.CSWAP(wires=[0,wires,wires+n_qubit_swap-1])
  qml.Hadamard(wires=0)
  return qml


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


def train_full_dense(X,opt,n_qubit_autoencoder,n_qubit_swap,repetition,epochs,visual=False):
    loss = []   
    layerparam=4
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


