import os
import json
import numpy as np
import pandas as pd

def init_trial_path(args,is_save=True):
    """Initialize the path for a hyperparameter setting

    Parameters
    ----------
    args : dict
        Settings

    Returns
    -------
    args : dict
        Settings with the trial path updated
    """
    prename = args.dataset + '_' + str(args.test_dataset)+ '_' +str(args.n_shot_test) + '_' + args.enc_gnn
    result_path = os.path.join(args.result_path, prename)
    os.makedirs(result_path, exist_ok=True)
    trial_id = 0
    path_exists = True
    while path_exists:
        trial_id += 1
        path_to_results = result_path + '/{:d}'.format(trial_id)
        path_exists = os.path.exists(path_to_results)
    args.trial_path = path_to_results
    os.makedirs(args.trial_path)
    if is_save:
        save_args(args)

    return args

def save_args(args):
    args = args.__dict__
    json.dump(args, open(args['trial_path'] + '/args.json', 'w'))
    prename=f"upt{args['update_step']}-{args['inner_lr']}_mod{args['batch_norm']}-{args['rel_node_concat']}"
    prename+=f"-{args['rel_hidden_dim']}-{args['rel_res']}"
    json.dump(args, open(args['trial_path'] +'/'+prename+ '.json', 'w'))

def count_model_params(model):
    print(model)
    param_size = {}
    cnt = 0
    for name, p in model.named_parameters():
        k = name.split('.')[0]
        if k not in param_size:
            param_size[k] = 0
        p_cnt = 1
        for j in p.size():
            p_cnt *= j
        param_size[k] += p_cnt
        cnt += p_cnt
    for k, v in param_size.items():
        print(f"Number of parameters for {k} = {round(v / 1024, 2)} k")
    print(f"Total parameters of model = {round(cnt / 1024, 2)} k")

class Logger(object):
    '''Save training process to log file with simple plot function.'''
    def __init__(self, fpath, title=None, resume=False):
        self.file = None
        self.resume = resume
        self.title = '' if title == None else title
        self.fpath = fpath
        if fpath is not None:
            if resume:
                self.file = open(fpath, 'r')
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')
            else:
                self.file = open(fpath, 'w')
        
    def set_names(self, names):
        if self.resume:
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()


    def append(self, numbers, verbose=True):
        #print(self.names,numbers)
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()
        if verbose:
            self.print()

    def print(self):
        log_str = ""
        for name, num in self.numbers.items():
            log_str += f"{name}: {num[-1]}, "
        print(log_str)

    def conclude(self, avg_k=3):
        avg_numbers={}
        best_numbers={}
        valid_name=[]
        for name, num in self.numbers.items():
            best_numbers[name] = np.max(num)
            avg_numbers[name] = np.mean(num[-avg_k:])
            if str.isdigit(name.split('-')[-1]):
                valid_name.append(name)
        vals=np.array([list(avg_numbers.values()),list(best_numbers.values())])
        cols = list(self.numbers.keys())
        rows = ['avg','best']
        df = pd.DataFrame(vals,index=rows, columns=cols)
        df['mid'] = df[valid_name].apply(lambda x: np.median(x),axis=1)
        df['mean'] = df[valid_name].apply(lambda x: np.mean(x),axis=1)
        save_path = self.fpath +'stats.csv'
        df.to_csv(save_path,sep='\t',index=True)
        return df

    def close(self):
        if self.file is not None:
            self.file.close()
