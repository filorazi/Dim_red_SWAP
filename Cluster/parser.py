import argparse
import numpy





def parse():
    parser = argparse.ArgumentParser(
                        prog='ProgramName',
                        description='',
                        epilog='')
    parser.add_argument('-ni', '--n_input_qubit',dest='n_input_qubit' ,help='aaaaaaaaaaaaaaaa',type=int)           # positional argument
    parser.add_argument('-nt', '--n_trash_qubit',dest= 'n_trash_qubit',type=int)      # option that takes a value
    parser.add_argument('-b', '--batch_size',dest='batch_size',type=int)      # option that takes a value
    parser.add_argument('-e', '--epochs',dest= 'epochs',type=int)  # on/off flag
    parser.add_argument('--seed',default=42,dest= 'seed',type=int)
    parser.add_argument('-v', '--val_percentage',dest='val_percentage' ,help='validation set percentage', type=float)
    parser.add_argument('-sz', '--step_size',dest= 'step_size',type=float)
    parser.add_argument('-of','--output_folder',dest= 'output_folder',)
    parser.add_argument('-io','--image_output',dest='image_output' ,action='store_true')
    parser.add_argument('-r','--repetition',dest='repetition' ,type=int, default=1)

    return parser.parse_args()


if __name__ == '__main__':
    exec()












'''
& C:/Users/forazi/.conda/envs/forazi/python.exe //gess-fs.d.ethz.ch/home$/forazi/Desktop/multidestructiveSWAP/Cluster/single_run.py -ni 4 -nt 1 -b 50 -e 50 -v .2 -sz .2 -of 'Cluster\\runs' -r 3 
'''