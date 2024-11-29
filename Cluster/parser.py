import argparse





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
    parser.add_argument('-fs','--frac_sampled',dest='frac_sampled' ,type=int, default=1)
    parser.add_argument('-ls','--list-op-support', required=False, type=int, default=[],
                        nargs='*', dest='list_op_support',
                        help='operator ranges considered in the earth mover distance')

    parser.add_argument('-lp','--list-op-support-probs', required=False, type=float, default=[],
                        nargs='*', dest='list_op_support_probs',
                        help='fraction of the operators of samples defined by ' \
                        + '--list-op-support sampled')
    parser.add_argument('-lr','--list-op-support-max-range', required=False, type=int, default=[],
                    nargs='*', dest='list_op_support_max_range',
                    help='max range of operators defined by --list-op-support')


    parser.add_argument('-sr','--support_list_m_range',dest='support_list_m_range' ,type=int, default=1)
    return parser.parse_args()


if __name__ == '__main__':
    exec()












'''
module load stack/2024-06 python/3.11.6
python single_run.py -ni 8 -nt 1 -b 50 -e 50 -v .2 -sz .2 -of 'runs' -ls 1 2  -lp 1. 1.  -lr 1 5'''