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
from utils import *
import os
from EMCost import *
from pennylane.math import reduce_dm
from jax import numpy as jnp
import optax 
from utils import *


class GHZ():
    def __init__(self,n_qubit,dvc,seed):
        self.n_qubit=n_qubit
        self.__dvc=dvc
        self.__seed=seed
        self.__num_params=n_qubit
        self.__wq=[jnp.array([random.uniform(0, 2*np.pi) for _ in range(self.__num_params)])]
        self.__GHZ = get_GHZ_state_matrix(n_qubit)
        self.__set_weights=None
        
    def circ(self,par):
        qml.RX(par[0],0)
        for a in range(1,self.n_qubit):
            qml.CRX(par[a],[a-1,a])
    
    def plot_cirq(self):
        @qml.qnode(self.__dvc)
        def node(param):
            self.circ(param)
        fig, ax = qml.draw_mpl(node)(self.__wq[-1])
        plt.show()

    def train(self,iterations,opt):      
        opt_state = opt.init(self.__wq[-1])
        final_epoch=-1
        train_loss=[]

        @qml.qnode(self.__dvc, interface="jax")
        def trainer(param,dm):
            self.circ(param)
            return qml.density_matrix(wires=self.__dvc.wires)
        opt_state = opt.init(self.__wq[-1])

        def train_step(weights,opt_state):
            loss_function = cost_fn_EM([0],trainer,[self.__GHZ])
            loss, grads = jax.value_and_grad(loss_function)(weights)
            updates, opt_state = opt.update(grads, opt_state)
            weights = optax.apply_updates(weights, updates)
            return weights, opt_state, loss

        for i in range(iterations):
            weights=self.__wq[-1]
            weights, opt_state, loss_value = train_step(weights, opt_state)
            train_loss.append(loss_value)
            self.__wq.append(weights)

            print(f'\rEpoch {i+1}/{iterations}, \tTrain Loss = {jnp.mean(loss_value):.6f}',end='')
            if i > 5 and np.mean(train_loss[-3:])<0.001:
                print(f'\nEarly stop at epoch {i} for perfect training')
                final_epoch = i
                break
            if i > 15 and np.std(train_loss[-1000:])<0.001:
                print(f'\nEarly stop at epoch {i} for plateau')
                final_epoch = i
                break
        if final_epoch ==-1:
            final_epoch=i
        try:
            console_size = os.get_terminal_size()[0]
        except OSError:
            console_size = 75
        print('\n')
        print('-'*console_size)
        self.__train_loss=train_loss
        self.final_epoch=final_epoch
        return train_loss, self.__wq.copy()
    
    def plot_loss(self):
        custom_palette =['#EABFCB','#C191A1','#A4508B','#5F0A87','#2F004F','#120021',]
        sns.set_palette(custom_palette)  

        plt.plot(list(range(len(self.__train_loss))),self.__train_loss, label='train loss')
        plt.legend()
        plt.show()

    def get_loss(self):
        return self.__train_loss
    
    def best_params(self):
        return self.__wq[np.argmin(self.__train_loss)+1] 

    def get_cirq(self,wire):
        if self.__set_weights is None:
            self.circ(self.best_params())
        else:
            self.circ(self.__set_weights)

    def plot_weights(self):
        i=0
        for a in np.array(self.__wq).T:
            plt.plot(range(len(a)),a,label=[i])
            i-=-1
        plt.legend()
