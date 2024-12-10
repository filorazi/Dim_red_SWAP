from parser import *
from utils import * 
import pennylane as qml
from pennylane import numpy as np
import pandas as pd
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
from autoencoder7 import Axutoencoder
from pennylane.optimize import AdamOptimizer,QNSPSAOptimizer

from autoencoder6 import JAxutoencoder
jax.config.update("jax_enable_x64", True)
import optax 

def main():
    param=parse()
    n=100
    X=list(range(n))
    random.shuffle(X)
    data=get_data(param.n_input_qubit)
    # def get_input_state(p):
    #     return np.outer(np.conjugate(data['ground_states'][p]), data['ground_states'][p])
    # X=[get_input_state(x) for x in X]
    X=[data['ground_states'][x] for x in X]
    
    

    n_qubit=param.n_input_qubit
    dvc = qml.device('default.mixed', wires=n_qubit, shots=None)
 

    set_global( param.n_input_qubit,
            param.n_input_qubit,
            param.n_trash_qubit,
            param.list_op_support,
            param.list_op_support_probs,
            True,
            param.list_op_support_max_range,
            use_jax=param.jax)
    # train_batch_losses={}
    # val_batch_losses={}
    # batch_times={}
    # batch_epochs={}
    dest_folder=param.output_folder
    os.makedirs(dest_folder,exist_ok=True)
    batch_folder=dest_folder+f'/{param.batch_size}'
    os.makedirs(batch_folder,exist_ok=True)

    # don't overwrite if a file with the same name already exists
    i = 0
    while os.path.exists(batch_folder+f'/loss_train{i}.npy'):
        i += 1

    loss_train_file_name=batch_folder+f'/loss_train{i}'
    loss_val_file_name=batch_folder+f'/loss_val{i}'
    weights_file_name=batch_folder+f'/weights{i}'
    print(f"Running AE with {param.n_input_qubit} input qubit and {param.n_trash_qubit} trash qubit in batches of {param.batch_size}")
    if param.jax:
        ae = JAxutoencoder(param.n_input_qubit,param.n_trash_qubit,dvc,'c11')
        opt = optax.adam(param.step_size)

    else:
        ae = Axutoencoder(param.n_input_qubit,param.n_trash_qubit,dvc,'c11')
        opt=AdamOptimizer(stepsize=param.step_size)


    ae.set_layers(3)


    start_time = time.time()
    _=ae.train(X,opt,param.epochs,param.batch_size,val_split=param.val_percentage)
    end_time = time.time()
    
    
    train_loss,val_loss=ae.get_loss()
    timek=end_time-start_time
    batch_epochs=ae.get_final_epoch()
    weights=ae.best_params()
    np.save(loss_train_file_name,np.array(train_loss))            
    np.save(loss_val_file_name,np.array(val_loss))            
    np.save(weights_file_name,np.array(weights))

    min_val_found= min(val_loss)

    if param.image_output:
        plt.figure()
        custom_palette =[
        # '#C24AA2','#D6518F',
        '#EC5A77','#F57C73','#F69C6D','#F6BC66']
        sns.set_palette(custom_palette)  

        sns.lineplot(x=range(len(train_loss)), y=train_loss, label='train loss', color='#009E81')
        sns.lineplot(x=range(len(val_loss)), y=val_loss, label='validation loss' ,color='#007162',linestyle=':')
        plt.legend(title='Batch size')

        plt.ylim((0,1))

        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title(f'Loss on AE {param.n_input_qubit}->{param.n_input_qubit-param.n_trash_qubit}, Batch size = {param.batch_size}')
        
        plt.savefig(batch_folder+f'/graph{param.n_input_qubit}_{param.n_input_qubit-param.n_trash_qubit}__{i}')
        # Info file 
    with open(batch_folder+f'/info_{i}.txt','a') as file:
        file.write(f'RUN INFORMATION\n'\
                    f'Input qubits={param.n_input_qubit}\n'\
                    f'Trash qubit={param.n_trash_qubit}\n'\
                    f'Seed={param.seed}\n'\
                    f'Optimizer=Optax.adam(stepsize={param.step_size})\n'\
                    f'Epochs={batch_epochs}\n'\
                    f'Batch sizes={param.batch_size}\n'\
                    f'Loss name={ae.get_loss_name()}\n'\
                    f'Val_percentage={param.val_percentage}\n'\
                    f'Min val loss found={min_val_found}\n'\
                    f'Exec (training) time={timek}')





























if __name__ == '__main__':
    main()
