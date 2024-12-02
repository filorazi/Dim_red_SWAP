import pennylane as qml
from pennylane import numpy as np
import os 
os.environ["JAX_PLATFORMS"] = "cpu"
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

#autoencoder con jax


class JAxutoencoder():
   

    def __init__(self,n_qubit_autoencoder,n_qubit_trash,device,circ='c11',seed=None):

        # if seed is None:
        #     seed=random.random()
        #     self.__seed=seed
        # else:
        #     self.__seed=seed
        # random.seed(seed)
        if n_qubit_autoencoder not in [4,8]:
            raise Exception('either 4 or 8 qubit with this state prep')
        self.__layers=3
        self.__n_qubit_auto = n_qubit_autoencoder
        self.__n_qubit_trash = n_qubit_trash
        self.__n_qubit=n_qubit_autoencoder+n_qubit_trash
        self.__dvc=device
        self.__setup()
        self.__circ = circ
        self.__num_params= self.__circuits[circ]['n_par'](n_qubit_autoencoder)
        self.__set_weights =None
        
        #set parameter to random values for the first stage and 0 to all the following
        # self.__wq=[np.array([random.uniform(0, np.pi) for _ in range(self.__num_params_stages[0])]+[0]*(self.__num_params-self.__num_params_stages[0]), requires_grad=True)]
        # print(f'the device has {len(device.wires)} qubits')
    

    def __setup(self):
        self.__circuits = {
            'c6' : {'func':self.c6ansatz,
                    'n_par':lambda q: q**2 +3*q,
            },
            'c11' :{'func':self.c11ansatz,
                    'n_par':lambda q: (q*4 -4)*self.__layers,
            },
            'isin' : {'func':self.create_ising_state,
            'n_par': lambda q : 0
            }
        }
        self.__losses={
            'fidelity': {'func':fidelity},
            'EMdistance': {'func': cost_fn_EM}

        }
        self.__train_loss={}
        self.__val_loss= {}
        self.__sp = self.__circuits['isin']['func']
        self.__loss_name='EMdistance'
        self.__loss= self.__losses[self.__loss_name]['func']
        # self.__data = get_data(self.__n_qubit_auto)

    # @jax.jit
    # def get_input_state(self,p):
    #     return jnp.outer(self.__data.ground_states[p], self.__data.ground_states[p].conj())


    def create_ising_state(self,dm1,start=0):
        qml.QubitDensityMatrix(dm1, wires=range(start,self.__n_qubit_auto+start))

    

    def c6ansatz(self,param,start=0):
        raise Exception('Removed from the implementation, use circ 11')
    
    def create_circ(self,param,dm,start=0):
        self.__sp(dm,0)
        qml.Barrier()
        self.create_encoder(param,start)


    def create_encoder(self,params,start=0):
        self.__circuits[self.__circ]['func'](params,start)
            
    def create_decoder(self,params,start=0):
        wire_map=dict(zip(list(range(self.__n_qubit_trash)),list(range(self.__n_qubit_auto,self.__n_qubit))))

        def f():
            self.__circuits[self.__circ]['func'](params,start)
        qml.adjoint(qml.map_wires(f, wire_map))()


    def set_layers(self,layers):
        self.__layers = layers
        self.__num_params=self.__circuits[self.__circ]['n_par'](self.__n_qubit_auto)
        # random.seed(self.__seed)
        self.__wq=[jnp.array([random.uniform(0, np.pi) for _ in range(self.__num_params)])]

    def c11(self,parameter,qb,start):
        current_par =0
        for i in range(start,qb//2+start):
            qml.RY(parameter[current_par],wires=(i-start)*2+start)
            current_par-=-1
            qml.RY(parameter[current_par],wires=(i-start)*2+start+1)
            current_par-=-1

        for i in range(start,qb//2+start):
            qml.RZ(parameter[current_par],wires=(i-start)*2+start)
            current_par-=-1
            qml.RZ(parameter[current_par],wires=(i-start)*2+start+1)
            current_par-=-1

        for i in range(start,qb//2+start):
            qml.CNOT([(i-start)*2+start+1,(i-start)*2+start])

        qml.Barrier()
        for i in range(start,(qb-1)//2+start):        
            qml.RY(parameter[current_par],wires=(i-start)*2+start+1)
            current_par-=-1
            qml.RY(parameter[current_par],wires=(i-start)*2+start+2)
            current_par-=-1

        for i in range(start,(qb-1)//2+start):   
            qml.RZ(parameter[current_par],wires=(i-start)*2+start+1)
            current_par-=-1
            qml.RZ(parameter[current_par],wires=(i-start)*2+start+2)
            current_par-=-1


        for i in range(start,(qb-1)//2+start):
            qml.CNOT([(i-start)*2+start+2,(i-start)*2+start+1])
        qml.Barrier()

    def c11ansatz(self,param,start):
        parperlay = 4*self.__n_qubit_auto-4
        for l in range(self.__layers):
            self.c11(param[parperlay*l:parperlay*(l+1)],self.__n_qubit_auto,start) 
            qml.Barrier()

    def plot_cirq(self):

        @qml.qnode(self.__dvc)
        def trainer(param,p):
            self.create_circ(param,p)
        fig, ax = qml.draw_mpl(trainer)(self.__wq[-1],.5)
        plt.show()

    def train(self, X , opt,epochs,batch_size=None,warm_weights=None, val_split=0.0):
        train_loss = []   
        val_loss = [0]
        final_epoch=-1

        X_train = X[0:int(np.floor(len(X)*(1-val_split)))]
        X_val = X[int(np.floor(len(X)*(1-val_split))):]
        if batch_size is None:
            batch_size=len(X)
        if warm_weights is not None:
            if len(warm_weights)!= self.__num_params:
                raise ValueError(f'The weights for the warm start should have length {self.__num_params}, but {len(warm_weights)} where found.')
            self.__wq[-1]=warm_weights
        if type(epochs) is not int:
            raise ValueError(f'Epochs should be a integer not a {type(epochs)}')

        @qml.qnode(self.__dvc, interface="jax")
        def trainer(param,dm):
            self.create_circ(param,dm)
            return qml.state()
        opt_state = opt.init(self.__wq[-1])
        
        def train_step(weights,opt_state,data):
            #Remember to change the EM function
            loss_function = self.__loss(data,trainer,[1]+[0]*(2**self.__n_qubit_trash-1))
            # print(loss_function(weights))
            
            loss, grads = jax.value_and_grad(loss_function)(weights)
            # print(f'loss:\n{loss}')

            # print(f'grads:\n{grads}')
            updates, opt_state = opt.update(grads, opt_state)
            weights = optax.apply_updates(weights, updates)
            return weights, opt_state, loss


        for epoch in range(epochs):
            batch_loss=[]
            weights=jnp.array(self.__wq[-1])
            for i, X_batch in enumerate([X_train[i:i + batch_size] for i in range(0, len(X_train), batch_size)]):
                weights, opt_state, loss_value = train_step(weights, opt_state, X_batch)
                batch_loss.append(loss_value)
                print(f'\rEpoch {epoch+1}, \tBatch:{i}, \tTrain Loss = {np.mean(batch_loss):.6f}, \tVal Loss = {val_loss[-1]:.6f}',end='')
            self.__wq.append(weights)
            val_l=self.__loss(X_val,trainer,X_val) 
            val_loss.append(val_l(self.__wq[-1]))
            train_loss.append(np.average(batch_loss,weights=[len(X_batch) for X_batch in [X_train[i:i + batch_size] for i in range(0, len(X_train), batch_size)]]))
            if epoch > 5 and np.mean(val_loss[-3:])<0.001:
                print(f'\nEarly stop at epoch {epoch} for perfect training')
                final_epoch = epoch
                break
            if epoch > 15 and np.std(val_loss[-15:])<0.001:
                print(f'\nEarly stop at epoch {epoch} for plateau')
                final_epoch = epoch
                break
        if final_epoch ==-1:
            final_epoch=epochs
        try:
            console_size = os.get_terminal_size()[0]
        except OSError:
            console_size = 75
        print('\n')
        print('-'*console_size)
        self.__train_loss=train_loss
        self.__val_loss=val_loss[1:]
        self.final_epoch=final_epoch
        return train_loss,val_loss[1:], self.__wq.copy()

    def best_params(self):
        return self.__wq[np.argmin(self.__val_loss)+1] 

    def get_cirq(self,wire):
        if self.__set_weights is None:
            self.create_encoder(self.best_params(),wire)
        else:
            self.create_encoder(self.__set_weights,wire)
    
    def get_final_epoch(self):
        return self.final_epoch
    
    def plot_loss(self):
        custom_palette =['#EABFCB','#C191A1','#A4508B','#5F0A87','#2F004F','#120021',]
        sns.set_palette(custom_palette)  

        plt.set_cmap
        plt.plot(list(range(len(self.__train_loss))),self.__train_loss, label='train loss')
        plt.plot(list(range(len(self.__val_loss))),self.__val_loss, label='val loss')
        plt.legend()
        plt.show()

    def plot_weights(self):
        i=0
        for a in np.array(self.__wq).T:
            plt.plot(range(len(a)),a,label=[i])
            i-=-1
        plt.legend()

    def get_loss(self):
        return self.__train_loss,self.__val_loss
    
    def get_num_par(self):
        return self.__num_params
    
    def set_weights(self,param):
        self.__set_weights= param
    
    def load(self,path):
        self.__set_weights=np.load(path+'/weights.npy')
        self.__train_loss=np.load(path+'/train_loss.npy')
        self.__val_loss=np.load(path+'/val_loss.npy')

    def get_current_loss(self,X):
        @qml.qnode(self.__dvc,diff_method='adjoint')
        def trainer(param,p):
            self.create_circ(param,p)
            return qml.probs(list(range(self.__n_qubit_trash)))
        def loss_function():
            if self.__set_weights is not None:
                W=self.__set_weights
            else:
                W =self.__wq[-1]
            pred =np.array([1-trainer(W,x)[0] for x in X], requires_grad=True)
            current_loss = pred.mean()
            return current_loss
        return loss_function()


    def get_loss_name(self):
        return self.__loss_name