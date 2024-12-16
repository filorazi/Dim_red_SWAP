import pennylane as qml
from pennylane import numpy as np
import os 
os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mse
import random 
from utils import *
import warnings
warnings.filterwarnings("ignore")
import os 
import time
import pennylane as qml
from pennylane import numpy as np
import pandas as pd
import jax
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mse
import random 
from utils import *
import os
from EMCost import *
from autoencoder6 import JAxutoencoder
jax.config.update("jax_enable_x64", True)
import optax 

custom_palette =[
    # '#C24AA2','#D6518F',
    '#EC5A77','#F57C73','#F69C6D','#F6BC66']

range_qubit_autoencoder=[4]
range_batches = [20,50,100]
seed=42
epochs=100 
n=100
stepsize=.2
opt = optax.adam(stepsize)
# X=np.random.rand(n)*np.pi/2

X=list(range(n))
random.shuffle(X)

for n_qubit_autoencoder in range_qubit_autoencoder:
    data=get_data(n_qubit_autoencoder)
    def get_input_state(p):
        return jnp.outer(jnp.conjugate(data.ground_states[p]), data.ground_states[p])
    X=[get_input_state(x) for x in X]

    for n_trash_qubit in range(2,n_qubit_autoencoder):
        train_batch_losses={}
        val_batch_losses={}
        batch_times={}
        batch_epochs={}
        img_folder=f'./runs/run_{n_qubit_autoencoder}to{n_qubit_autoencoder-n_trash_qubit}'
        os.makedirs(img_folder,exist_ok=True)
        for batch_size in range_batches:
            folder=img_folder+f'/{batch_size}'
            print(f"Running AE with {n_qubit_autoencoder} input qubit and {n_trash_qubit} trash qubit in batches of {batch_size}")
            n_qubit=n_qubit_autoencoder+n_trash_qubit
            dvc = qml.device('default.mixed', wires=n_qubit, shots=None)
            ae = JAxutoencoder(n_qubit_autoencoder,n_trash_qubit,dvc,'c11')
            ae.set_layers(3)
            start_time = time.time()
            ae.train(X,opt,epochs,batch_size,val_split=.20)
            end_time = time.time()

            os.makedirs(folder)
            train_loss,val_loss=ae.get_loss()
            train_batch_losses[batch_size]=train_loss
            val_batch_losses[batch_size]=val_loss
            batch_times[batch_size]=end_time-start_time
            batch_epochs[batch_size]=ae.get_final_epoch()
            weights=ae.best_params()
            np.save(folder+'/loss_train',np.array(train_loss))            
            np.save(folder+'/loss_val',np.array(val_loss))            
            np.save(folder+'/weights',np.array(weights))

        # Min loss
        min_val_found= {a:min(val_batch_losses[a]) for a in range_batches }
        min_train_found= {a:min(train_batch_losses[a]) for a in range_batches }
        # min_loss,rank =get_min_loss_fid_ising(X,n_qubit_autoencoder,n_trash_qubit)

        # Figure
        plt.figure()
        sns.set_palette(custom_palette)  
        for a,b,c,d in zip(list(train_batch_losses.values()),list(val_batch_losses.values()),range_batches,custom_palette):
            sns.lineplot(x=range(len(a)),y=a,label=f'train_{c}',color=d)
            sns.lineplot(x=range(len(b)),y=[l.item() for l in b],label=f'val_{c}', color=d,linestyle=':')
        plt.legend(title='Batch size')
        # if len(epochs)>1:
        #     plt.vlines(epochs[:-1],0,1,color='#C24AA2',linestyle='--')
        #     plt.text(epochs[-1]-0.02, y=0.85, fontsize='medium', s=f'stage\nchange', color='#973C7F', ha='center', va='center')

        # plt.hlines(min_loss,0,np.max(list(batch_epochs.values()))-1,color='#773344',linestyle='--')
        # plt.text(x=np.max(list(batch_epochs.values()))//3*2, y=min_loss+0.05, fontsize='medium', s=f'Min loss', color='#773344', ha='right', va='center')
        plt.ylim((0,1))

        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title(f'Loss on AE {n_qubit_autoencoder}->{n_qubit_autoencoder-n_trash_qubit}')
        exit()
        plt.savefig(img_folder+f'/{n_qubit_autoencoder}_{n_qubit_autoencoder-n_trash_qubit}')
        # Info file 
        with open(img_folder+f'/info.txt','a') as file:
            # file.write(f'RUN INFORMATION\nInput qubits={n_qubit_autoencoder}\nTrash qubit={n_trash_qubit}\nSeed={seed}\nOptimizer=Optax.adam(stepsize={stepsize})\nEpochs=\n{''.join([f'\t\t\t\t{a}\t:\t{b}\n' for a, b in batch_epochs.items()])}\nBatch sizes={range_batches}\nMin fidelity loss={min_loss}\nDensity matrix rank={rank}\nMin val loss found=\n{''.join([f'\t\t\t\t{a}\t:\t{b}\n' for a, b in min_val_found.items()])}\nExec (training) time=\n{''.join([f'\t\t\t\t{a}\t:\t{b}\n' for a, b in batch_times.items()])}')
            file.write(f'RUN INFORMATION\nInput qubits={n_qubit_autoencoder}\nTrash qubit={n_trash_qubit}\nSeed={seed}\nOptimizer=Optax.adam(stepsize={stepsize})\nEpochs=\n{''.join([f'\t\t\t\t{a}\t:\t{b}\n' for a, b in batch_epochs.items()])}\nBatch sizes={range_batches}\nDensity matrix rank={rank}\nMin val loss found=\n{''.join([f'\t\t\t\t{a}\t:\t{b}\n' for a, b in min_val_found.items()])}\nExec (training) time=\n{''.join([f'\t\t\t\t{a}\t:\t{b}\n' for a, b in batch_times.items()])}')