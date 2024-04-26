import random
import os
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.functional import auroc
from torch_geometric.data import DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

#from .maml import MAML
from ..datasets import sample_meta_datasets, sample_test_datasets, MoleculeDataset
from ..utils import Logger
torch.set_printoptions(threshold=np.inf)

class Meta_Trainer(nn.Module):
    def __init__(self, args, model):
        super(Meta_Trainer, self).__init__()

        self.args = args
        self.mod=args.mod
        self.model = model
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.meta_lr, weight_decay=args.weight_decay)
        if self.mod in [17]:
            self.optimizer_c = optim.AdamW(self.model.adapt_relation.ctrl0.parameters(), lr=args.meta_lr, weight_decay=args.weight_decay)
        self.criterion = nn.CrossEntropyLoss().to(args.device)
        self.criterion1 = nn.CrossEntropyLoss(reduction='none').to(args.device)

        self.dataset = args.dataset
        self.test_dataset = args.test_dataset if args.test_dataset is not None else args.dataset
        self.data_dir = args.data_dir
        self.train_tasks = args.train_tasks
        self.test_tasks = args.test_tasks
        self.n_shot_train = args.n_shot_train
        self.n_shot_test = args.n_shot_test
        self.n_query = args.n_query

        self.device = args.device

        self.emb_dim = args.emb_dim

        self.batch_task = args.batch_task

        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.inner_update_step = args.inner_update_step

        self.trial_path = args.trial_path
        
        trial_name = self.dataset + '_' + self.test_dataset + '@' + args.enc_gnn
        print(trial_name)
        logger = Logger(self.trial_path + '/results.txt', title=trial_name)
        log_names = ['Epoch']
        log_names += ['AUC-' + str(t) for t in args.test_tasks]
        log_names += ['AUC-Avg', 'AUC-Mid','AUC-Best']
        logger.set_names(log_names)
        self.logger = logger

        preload_train_data = {}
        if args.preload_train_data:
            print('preload train data')
            for task in self.train_tasks:
                dataset = MoleculeDataset(self.data_dir + self.dataset + "/new/" + str(task + 1),
                                          dataset=self.dataset)
                preload_train_data[task] = dataset
        preload_test_data = {}
        if args.preload_test_data:
            print('preload_test_data')
            for task in self.test_tasks:
                dataset = MoleculeDataset(self.data_dir + self.test_dataset + "/new/" + str(task + 1),
                                          dataset=self.test_dataset)
                preload_test_data[task] = dataset
        self.preload_train_data = preload_train_data
        self.preload_test_data = preload_test_data
        
        if 'train' in self.dataset and args.support_valid:
            val_data_name = self.dataset.replace('train','valid')
            print('preload_valid_data')
            preload_val_data = {}
            for task in self.train_tasks:
                dataset = MoleculeDataset(self.data_dir + val_data_name + "/new/" + str(task + 1),
                                          dataset=val_data_name)
                preload_val_data[task] = dataset
            self.preload_valid_data = preload_val_data

        self.train_epoch = 0
        self.best_auc=0 
        
        self.res_logs=[]

    def loader_to_samples(self, data):
        loader = DataLoader(data, batch_size=len(data), shuffle=False, num_workers=0)
        for samples in loader:
            samples=samples.to(self.device)
            return samples

    def get_data_sample(self, task_id, train=True):
        if train:
            task = self.train_tasks[task_id]
            if task in self.preload_train_data:
                dataset = self.preload_train_data[task]
            else:
                dataset = MoleculeDataset(self.data_dir + self.dataset + "/new/" + str(task + 1), dataset=self.dataset)

            s_data, q_data = sample_meta_datasets(dataset, self.dataset, task,self.n_shot_train, self.n_query)

            s_data = self.loader_to_samples(s_data)
            q_data = self.loader_to_samples(q_data)

            adapt_data = {'s_data': s_data, 's_label': s_data.y, 'q_data': q_data, 'q_label': q_data.y,
                            'label': torch.cat([s_data.y, q_data.y], 0)}
            eval_data = { }
        else:
            task = self.test_tasks[task_id]
            if task in self.preload_test_data:
                dataset = self.preload_test_data[task]
            else:
                dataset = MoleculeDataset(self.data_dir + self.test_dataset + "/new/" + str(task + 1),
                                          dataset=self.test_dataset)
            s_data, q_data, q_data_adapt = sample_test_datasets(dataset, self.test_dataset, task, self.n_shot_test, self.n_query, self.update_step_test)
            s_data = self.loader_to_samples(s_data)
            q_loader = DataLoader(q_data, batch_size=self.n_query, shuffle=True, num_workers=0)
            q_loader_adapt = DataLoader(q_data_adapt, batch_size=self.n_query, shuffle=True, num_workers=0)

            adapt_data = {'s_data': s_data, 's_label': s_data.y, 'data_loader': q_loader_adapt}
            eval_data = {'s_data': s_data, 's_label': s_data.y, 'data_loader': q_loader}

        return adapt_data, eval_data

    def get_prediction(self, model, data, train=True):
        if train:
            s_logits, q_logits, adj, node_emb = model(data['s_data'], data['q_data'], data['s_label'])
            pred_dict = {'s_logits': s_logits, 'q_logits': q_logits, 'adj': adj, 'node_emb': node_emb}

        else:
            s_logits, logits,labels,adj_list,sup_labels = model.forward_query_loader(data['s_data'], data['data_loader'], data['s_label'])
            pred_dict = {'s_logits':s_logits, 'logits': logits, 'labels': labels,'adj':adj_list,'sup_labels':sup_labels}

        return pred_dict

    def get_adaptable_weights(self, model, adapt_weight=None):
        if adapt_weight is None:
            adapt_weight = self.args.adapt_weight
        fenc = lambda x: x[0]== 'mol_encoder'
        frel = lambda x: x[0]== 'adapt_relation'
        fedge = lambda x: x[0]== 'adapt_relation' and 'edge_layer'  in x[1]
        fnode = lambda x: x[0]== 'adapt_relation' and 'node_layer'  in x[1]
        fclf = lambda x: x[0]== 'adapt_relation' and 'fc'  in x[1]
        if adapt_weight==0:
            flag=lambda x: not fenc(x)
        elif adapt_weight==1:
            flag=lambda x: not frel(x)
        elif adapt_weight==2:
            flag=lambda x: not (fenc(x) or frel(x))
        elif adapt_weight==3:
            flag=lambda x: not (fenc(x) or fedge(x))
        elif adapt_weight==4:
            flag=lambda x: not (fenc(x) or fnode(x))
        elif adapt_weight==5:
            flag=lambda x: not (fenc(x) or fnode(x) or fedge(x))
        elif adapt_weight==6:
            flag=lambda x: not (fenc(x) or fclf(x))
        else:
            flag= lambda x: True
        if self.train_epoch < self.args.meta_warm_step or self.train_epoch>self.args.meta_warm_step2:
            adaptable_weights = None
        else:
            adaptable_weights = []
            adaptable_names=[]
            for name, p in model.module.named_parameters():
                names=name.split('.')
                if flag(names):
                    adaptable_weights.append(p)
                    adaptable_names.append(name)
        return adaptable_weights

    def get_loss(self, model, batch_data, pred_dict, train=True, flag = 0):
        n_support_train = self.args.n_shot_train
        n_support_test = self.args.n_shot_test
        n_query = self.args.n_query
        if not train:
            losses_adapt = self.criterion(pred_dict['s_logits'].reshape(2*n_support_test*n_query,2), pred_dict['s_logits'].reshape(2*n_support_test*n_query,2))
        else:
            if flag:
                losses_adapt = self.criterion(pred_dict['s_logits'].reshape(2*n_support_train*n_query,2), batch_data['s_label'].repeat(n_query))
            else:
                if self.mod in [11,12,13,14,15,16,21,22,23,24,25,26]:
                    p=pred_dict['s_logits']#16,5
                    bz=p.size(0)
                    q=pred_dict['q_logits']#16,5,2
                    y=batch_data['q_label']#16
                    qt=q.view(-1,2)#80,2
                    yt=y.unsqueeze(1).repeat(1,p.size(1))#16,5
                    yt=yt.view(-1,1).squeeze(-1)
                    #print(qt.size(),yt.size())
                    los = self.criterion1(qt, yt)#80
                    p=torch.softmax(p,-1)
                    '''pf=torch.zeros_like(p)
                    #fair
                    for i in range(p.size(1)):
                        pf[:,i]=p[:,i]*p[:,i]/torch.sum(p[:,i:],-1)
                    p=pf.view(-1,1)'''
                    p=p.view(-1,1)
                    losses_adapt=torch.sum(los*p)/bz
                elif self.mod in [17]:
                    p=pred_dict['s_logits']#16,5
                    bz=p.size(0)
                    q=pred_dict['q_logits']#16,5,2
                    y=batch_data['q_label']#16
                    qt=q.view(-1,2)#80,2
                    yt=y.unsqueeze(1).repeat(1,p.size(1))#16,5
                    yt=yt.view(-1,1).squeeze(-1)
                    los = self.criterion1(qt, yt)#80
                    p=torch.softmax(p,-1).view(-1,1)
                    losses_adapt_c=torch.sum(los*p)/bz
                    losses_adapt=torch.mean(los)
                else:
                    losses_adapt = self.criterion(pred_dict['q_logits'], batch_data['q_label'])

        if torch.isnan(losses_adapt).any() or torch.isinf(losses_adapt).any():
            print('!!!!!!!!!!!!!!!!!!! Nan value for supervised CE loss', losses_adapt)
            print(pred_dict['s_logits'])
            losses_adapt = torch.zeros_like(losses_adapt)
        if self.args.reg_adj > 0:
            n_support = batch_data['s_label'].size(0)
            adj = pred_dict['adj'][-1]
            if train:
                if flag:
                    s_label = batch_data['s_label'].unsqueeze(0).repeat(n_query, 1)
                    n_d = n_query * n_support
                    label_edge = model.label2edge(s_label).reshape((n_d, -1))
                    pred_edge = adj[:,:,:-1,:-1].reshape((n_d, -1))
                else:
                    s_label = batch_data['s_label'].unsqueeze(0).repeat(n_query, 1)
                    q_label = batch_data['q_label'].unsqueeze(1)
                    total_label = torch.cat((s_label, q_label), 1)
                    label_edge = model.label2edge(total_label)[:,:,-1,:-1]
                    pred_edge = adj[:,:,-1,:-1]
            else:
                s_label = batch_data['s_label'].unsqueeze(0)
                n_d = n_support
                label_edge = model.label2edge(s_label).reshape((n_d, -1))
                pred_edge = adj[:, :, :n_support, :n_support].mean(0).reshape((n_d, -1))
            adj_loss_val = F.mse_loss(pred_edge, label_edge)
            if torch.isnan(adj_loss_val).any() or torch.isinf(adj_loss_val).any():
                print('!!!!!!!!!!!!!!!!!!!  Nan value for adjacency loss', adj_loss_val)
                adj_loss_val = torch.zeros_like(adj_loss_val)

            losses_adapt += self.args.reg_adj * adj_loss_val
        if self.mod in [17]:
            return losses_adapt,losses_adapt_c
        return losses_adapt

    def train_step(self):

        self.train_epoch += 1

        task_id_list = list(range(len(self.train_tasks)))
        if self.batch_task > 0:
            batch_task = min(self.batch_task, len(task_id_list))
            task_id_list = random.sample(task_id_list, batch_task)
        data_batches={}
        for task_id in task_id_list:
            db = self.get_data_sample(task_id, train=True)
            data_batches[task_id]=db

        for k in range(self.update_step):
            losses_eval = []
            losses_eval_c = []
            for task_id in task_id_list:
                train_data, _ = data_batches[task_id]
                model=self.model
                model.train()

                pred_eval = self.get_prediction(model, train_data, train=True)
                if self.mod in [17]:
                    loss_eval,loss_eval_c = self.get_loss(model, train_data, pred_eval, train=True, flag = 0)
                    losses_eval.append(loss_eval)
                    losses_eval_c.append(loss_eval_c)
                else:
                    loss_eval = self.get_loss(model, train_data, pred_eval, train=True, flag = 0)

                    losses_eval.append(loss_eval)
            if self.mod in [17]:
                losses_eval = torch.stack(losses_eval)

                losses_eval = torch.sum(losses_eval)

                losses_eval = losses_eval / len(task_id_list)
                losses_eval_c = torch.stack(losses_eval_c)
                losses_eval_c = torch.sum(losses_eval_c)
                losses_eval_c = losses_eval_c / len(task_id_list)
                
                self.optimizer_c.zero_grad()
                losses_eval_c.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.model.adapt_relation.ctrl0.parameters(), 1)
                self.optimizer_c.step()

                #xprint('Train Epoch:',self.train_epoch,'(2), train update step:', k, ', loss_eval:', losses_eval_c.item())
                self.optimizer.zero_grad()
                losses_eval.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()

                print('Train Epoch:',self.train_epoch,', train update step:', k, ', loss_eval:', losses_eval.item())
            else:
                losses_eval = torch.stack(losses_eval)

                losses_eval = torch.sum(losses_eval)

                losses_eval = losses_eval / len(task_id_list)
                self.optimizer.zero_grad()
                losses_eval.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()

                print('Train Epoch:',self.train_epoch,', train update step:', k, ', loss_eval:', losses_eval.item())

        return self.model

    def test_step(self):
        step_results={'query_preds':[], 'query_labels':[], 'query_adj':[],'task_index':[]}
        auc_scores = []
        for task_id in range(len(self.test_tasks)):
            adapt_data, eval_data = self.get_data_sample(task_id, train=False)
            '''m=eval_data['s_data'].smiles
            for i in range(len(m)):
                draw = Draw.MolToImage(Chem.MolFromSmiles(m[i]),size=(1000,1000))
                draw.save('./figs'+str(task_id)+'/s'+str(i)+'.jpg')'''
            #ss
            model = self.model
            model.eval()
            with torch.no_grad():
                pred_eval = self.get_prediction(model, eval_data, train=False)
                #print(pred_eval.keys())
                #ss
                if self.mod in [11,12,13,14,15,16,17,21,22,23,24,25,26]:
                    idx=torch.argmax(pred_eval['s_logits'],1)
                    #np.save('./figs'+str(task_id)+'layercount.npy',idx.cpu().detach().numpy())
                    #ss
                    '''for i in range(idx.size(0)):
                        print(i,':',idx[i])
                    #print(idx)
                    ss'''
                    y_score = F.softmax(pred_eval['logits'][list(range(pred_eval['logits'].size(0))),idx,:],dim=-1).detach()[:,1]
                    y_true = pred_eval['labels']
                else:
                    y_score = F.softmax(pred_eval['logits'],dim=-1).detach()[:,1]
                    y_true = pred_eval['labels']
                #false
                if self.args.eval_support:
                    y_s_score = F.softmax(pred_eval['s_logits'],dim=-1).detach()[:,1]
                    y_s_true = eval_data['s_label']
                    y_score=torch.cat([y_score, y_s_score])
                    y_true=torch.cat([y_true, y_s_true])
                #print(y_score.size(),y_true.size())torch.Size([6447]) torch.Size([6447])
                auc = auroc(y_score,y_true,pos_label=1).item()

            auc_scores.append(auc)

            print('Test Epoch:',self.train_epoch,', test for task:', task_id, ', AUC:', round(auc, 4))
            if self.args.save_logs:
                step_results['query_preds'].append(y_score.cpu().numpy())
                step_results['query_labels'].append(y_true.cpu().numpy())
                step_results['query_adj'].append(pred_eval['adj'].cpu().numpy())
                step_results['task_index'].append(self.test_tasks[task_id])

        mid_auc = np.median(auc_scores)
        avg_auc = np.mean(auc_scores)
        if avg_auc>=self.best_auc:
            self.save_model(best=True)
        self.best_auc = max(self.best_auc,avg_auc)
        self.logger.append([self.train_epoch] + auc_scores  +[avg_auc, mid_auc,self.best_auc], verbose=False)

        print('Test Epoch:', self.train_epoch, ', AUC_Mid:', round(mid_auc, 4), ', AUC_Avg: ', round(avg_auc, 4),
              ', Best_Avg_AUC: ', round(self.best_auc, 4),)
        
        if self.args.save_logs:
            self.res_logs.append(step_results)

        return self.best_auc

    def save_model(self,best=False):
        save_path = os.path.join(self.trial_path, f"step_{self.train_epoch}.pth")
        if best:
            save_path = os.path.join(self.trial_path, f"step_best.pth")
        torch.save(self.model.state_dict(), save_path)
        print(f"Checkpoint saved in {save_path}")

    def save_result_log(self):
        joblib.dump(self.res_logs,self.args.trial_path+'/logs.pkl',compress=6)

    def conclude(self):
        df = self.logger.conclude()
        self.logger.close()
        print(df)
