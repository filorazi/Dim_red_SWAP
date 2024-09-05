import pennylane as qml
import numpy as np
import pandas as pd
import jax
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
import matplotlib.pyplot as plt



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

        
def destructive_swap(n_qubit):
    for wires in range(n_qubit): 
        qml.CNOT(wires=[wires,wires+n_qubit])
        qml.Hadamard(wires)

    return qml

def interpret_results(data):
  def generate_combinations(n):
      # Generate all combinations of '0' and '1' of length n
      combinations = [''.join(comb) for comb in itertools.product('01', repeat=n)]
      return combinations

  def check_parity_bitwise_and(s):
    n = len(s)
    first_half = s[:n//2]
    second_half = s[n//2:]
    and_result = ''.join('1' if first_half[i] == '1' and second_half[i] == '1' else '0' for i in range(n//2))
    parity = and_result.count('1') % 2

    return parity

  dictdata = dict(zip(generate_combinations(len(data)),data))
  kk={}
  for k,item in dictdata.items():
    if dictdata[k]>0.00000001:
      kk[k]=item
  fail =0
  for k,i in kk.items():
    if check_parity_bitwise_and(k):
      fail += i
  return fail
