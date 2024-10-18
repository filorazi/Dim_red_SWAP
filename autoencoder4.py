import pennylane as qml
from pennylane import numpy as np
import jax
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mse
import random 
from utils import *
import os

class Autoencoder_iterative():
    def __init__(self, n_qubit_autoencoder, n_qubit_trash, device, arch='c11', seed=None):

        self.__setup()

        if seed is None:
            seed=random.random()
            self.__seed=seed
        else:
            self.__seed=seed
        random.seed(seed)

        self.__layers=3
        self.__n_qubit_auto = n_qubit_autoencoder
        self.__n_qubit_trash = n_qubit_trash
        self.__dvc=device
        self.__set_weights =None
        self.__arch=arch
        self.__fixed_w =[]
        self.__optimized_layers=0
        #set parameter to random values for the first stage and 0 to all the following
        self.__wq=[np.array([random.uniform(0, np.pi) for _ in range(self.__circuits[self.__arch]['n_par'](self.__n_qubit_auto))], requires_grad=True)]
        # print(f'the device has {len(device.wires)} qubits')
    
    def __setup(self):
        self.__circuits = {
            'c6' : {'func':self.c6ansatz,
                    'n_par':lambda q: q**2 +3*q,
            },
            'c11' :{'func':self.c11ansatz,
                    'n_par':lambda q: (q*4 -4)*self.__layers,
            },
            'iso' : {'func':self.create_isotropic_state,
                     'n_par': lambda q : 0
            },
        }
        self.__train_loss={}
        self.__val_loss= {}
        self.__sp = self.__circuits['iso']['func']

    def c6ansatz(self,param,start=0):
        self.original_auto(self.__n_qubit_auto,param,start=start)

    def create_circ(self,param,p,start=0):
        self.__sp(p,0)
        qml.Barrier()
        prv_w=0
        for i in range(self.__optimized_layers):
            crn_w=prv_w+self.__circuits[self.__arch]['n_par'](self.__n_qubit_auto-i)
            self.__circuits[self.__arch]['func'](self.__fixed_w[prv_w:crn_w],start=i)
            prv_w=crn_w
        self.create_ansatz(param,start)

    def create_ansatz(self,params,start=0):
        self.__circuits[self.__arch]['func'](params,start)
        qml.Barrier()

    def create_isotropic_state(self, p, start):
        qml.Hadamard(wires=start)
        theta = p
        for i in range(self.__n_qubit_auto-1):
            qml.CNOT(wires=[start+i , start+1+i])
        for i in range(self.__n_qubit_auto):
            qml.RX(theta, wires=start+i)
        for i in range(self.__n_qubit_auto-1):
            qml.CNOT(wires=[start+i , start+1+i])


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
        parperlay = 4*(self.__n_qubit_auto-self.__optimized_layers)-4
        for l in range(self.__layers):
            self.c11(param[parperlay*l:parperlay*(l+1)],self.__n_qubit_auto-self.__optimized_layers,start) 
            qml.Barrier()

    def plot_cirq(self):
        pass

    def train(self, X , opt,epochs,batch_size=None,warm_weights=None, val_split=0.0):

        X_train = X[0:int(np.floor(len(X)*(1-val_split)))]
        X_val = X[int(np.floor(len(X)*(1-val_split))):]

        if batch_size is None:
            batch_size=len(X)
        if warm_weights is not None:
            raise ValueError(f'Not yet implemented')
            if len(warm_weights)!= self.__num_params:
                raise ValueError(f'The weights for the warm start should have length {self.__num_params}, but {len(warm_weights)} where found.')
            self.__wq[-1]=warm_weights
        @qml.qnode(self.__dvc,diff_method='adjoint')
        def trainer(param,p):
            self.create_circ(param,p,self.__optimized_layers)
            return qml.probs([self.__optimized_layers])


        for s in range(self.__n_qubit_trash):
            train_loss = []   
            val_loss = [0]

            self.__wq=[np.array([random.uniform(0, np.pi) for _ in range(self.__circuits[self.__arch]['n_par'](self.__n_qubit_auto-s))], requires_grad=True)]

            for epoch in range(epochs):
                batch_loss=[]
                for i, X_batch in enumerate([X_train[i:i + batch_size] for i in range(0, len(X_train), batch_size)]):

                    def loss_function(w):
                        pred =np.array([1-trainer(w,x)[0] for x in X_batch], requires_grad=True)
                        current_loss = pred.mean()
                        return current_loss
                    weights, loss_value = opt.step_and_cost(loss_function, self.__wq[-1])
                    batch_loss.append(loss_value)
                    print(f'\rStage: {s}, \tEpoch {epoch+1}, \tBatch:{i}, \tTrain Loss = {np.mean(batch_loss):.6f}, \tVal Loss = {val_loss[-1]:.6f}',end='')
                
                self.__wq.append(weights)

                val_pred =[1-trainer(self.__wq[-1],x)[0] for x in X_val]
                val_loss.append(np.mean(val_pred))
                train_loss.append(np.average(batch_loss,weights=[len(X_batch) for X_batch in [X_train[i:i + batch_size] for i in range(0, len(X_train), batch_size)]]))

                if epoch > 5 and np.mean(val_loss[-3:])<0.001:
                    print('\nEarly stop')
                    break
            opt.reset()

            self.__fixed_w += self.best_params().tolist()
            self.__optimized_layers +=1
            self.__train_loss[s]=train_loss
            self.__val_loss[s]=val_loss[1:]

        try:
            console_size = os.get_terminal_size()
        except OSError:
            console_size = 50
        print('')
        print('-'*console_size)
        return self.__train_loss,self.__val_loss, self.__wq.copy()


    def best_params(self):
        if self.__optimized_layers != self.__n_qubit_trash:
            return self.__wq[np.argmin(self.__val_loss)+1] 
        else:
            return self.__fixed_w

    def get_cirq(self,wire):
        if self.__set_weights is None:
            self.create_ansatz(self.best_params(),wire)
        else:
            self.create_ansatz(self.__set_weights,wire)
    
    def plot_loss(self):
        fig,axs=plt.subplot(1,self.__n_qubit_trash)
        for a in range(self.__n_qubit_trash):
            custom_palette =['#EABFCB','#C191A1','#A4508B','#5F0A87','#2F004F','#120021',]
            sns.set_palette(custom_palette)  

            axs[0,a].plot(list(range(len(self.__train_loss))),self.__train_loss, label='train loss')
            axs[0,a].plot(list(range(len(self.__val_loss))),self.__val_loss, label='val loss')
            axs[0,a].legend()
            axs[0,a].title('training of the {a} stage')
        plt.show()

    # def plot_weights(self):
    #     i=0
    #     for a in np.array(self.__wq).T:
    #         plt.plot(range(len(a)),a,label=[i])
    #         i-=-1
    #     plt.legend()

    def get_loss(self):
        return self.__train_loss,self.__val_loss
    
    def get_num_par(self):
        return self.__num_params
    
    def set_weights(self,param):
        self.__set_weights= param
    
    # def load(self,path):
    #     self.__set_weights=np.load(path+'/weights.npy')
    #     self.__train_loss=np.load(path+'/train_loss.npy')
    #     self.__val_loss=np.load(path+'/val_loss.npy')

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
