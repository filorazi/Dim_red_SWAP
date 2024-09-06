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



def create_isotropic_state(p, n_qubit, start):
    def circuit():
        qml.Hadamard(wires=start)
        theta = 2 * np.arccos(np.sqrt(p))

        for i in range(n_qubit-1):
            qml.CNOT(wires=[start+i , start+1+i])
        
        for i in range(n_qubit):
            qml.RX(theta, wires=start+i)

        for i in range(n_qubit-1):
            qml.CNOT(wires=[start+i , start+1+i])
        return qml

    return circuit


def dense(a,b,parameters):
    qml.RY(parameters[0],wires=a)
    qml.RY(parameters[1],wires=b)
    qml.CNOT(wires=[a,b])
    qml.RY(parameters[2],wires=a)
    qml.RY(parameters[3],wires=b)
    qml.CNOT(wires=[b,a])

def pool(a,b,parameters):
    qml.CRZ(parameters[0],wires=[a,b])
    qml.X(a)  
    qml.CRX(parameters[1],wires=[a,b])

def autoencoder(offset,param,repetition,n_qubit):
    start=0
    for i in range(repetition):
        if start % 2!=0:
            raise Exception('The number of qubits should be a power of 2 greater than 2 to the power of repetition')

        for a in range(start,( n_qubit-start)//2+start):
            param_corrente=sum([6*n_qubit//2**(j+1) for j in range(i)])+(a-start)*6
            a+=offset
            dense(a,a+(n_qubit-start)//2,param[param_corrente:param_corrente +4])
            pool(a,a+(n_qubit-start)//2,param[param_corrente+4:param_corrente+6])
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





def train_with_swap():
    loss = []   
    n_qubit_autoencoder=8
    repetition=1
    n_qubit_swap=n_qubit_autoencoder-n_qubit_autoencoder//(2**(repetition))
    n_qubit=n_qubit_autoencoder+n_qubit_swap
    num_params=sum([6*n_qubit_autoencoder//2**(i+1) for i in range(repetition)])
    epochs= 300
    device = 'default.qubit'
    dvc=qml.device(device, wires=n_qubit, shots=None)

    @qml.qnode(dvc,interface='jax', diff_method=None)
    def trainer(param,p):
        create_isotropic_state(p, n_qubit_autoencoder, n_qubit_swap)()
        qml.Barrier(dvc.wires)
        autoencoder(n_qubit_swap,param,repetition,n_qubit_autoencoder)
        qml.Barrier(dvc.wires)
        destructive_swap(n_qubit_swap)
        return qml.probs(list(range(n_qubit_swap*2)))

    n= 200
    P=np.random.rand(n)
    np.random.shuffle(P)
    y=[0]*n
    import random 
    random.seed(42)
    weights=[random.uniform(0, 1) for _ in range(num_params)]
    wq=[weights]

    def loss_function(w): 
        pred =[interpret_results(trainer(w,x)) for x in P]
        wq.append(w)
        loss.append(mse(pred,y))
        clear_output(wait=True)
        print(f'Loss for current iteration: {loss[-1]}')

        return mse(pred,y)


    for a in range(4):
        res=minimize(loss_function,wq[-1],method='COBYLA',options={'maxiter':epochs,'rhobeg':np.pi/(a+1),'disp':True,'tol':0.01})
    plt.plot(loss)
    plt.show()
    print(res)
    # weights,_,_,_,_= opt.step(loss_function, weights,trainer, n_qubit ,P, y)
