
import os
print('pid:', os.getpid())
import setproctitle
from time import time
from parser import get_args
import torch
from chem_lib.models import ContextAwareRelationNet, Meta_Trainer
from chem_lib.utils import count_model_params
setproctitle.setproctitle("fsgnn@wsg")
torch.set_num_threads(8)
def main():
    root_dir = '.'
    args = get_args(root_dir)

    model = ContextAwareRelationNet(args)
    count_model_params(model)
    model = model.to(args.device)
    trainer = Meta_Trainer(args, model)
    if args.resume>0:
        prename = args.dataset + '_' + str(args.test_dataset)+ '_' +str(args.n_shot_test) + '_' + args.enc_gnn
        result_path = os.path.join(args.result_path, prename)
        rpath0=result_path+'/{}'.format(args.eid)
        if args.resume==1:
            rpath=rpath0+"/step_best.pth"
        else:
            rpath=rpath0+"/step_{}.pth".format(args.resume)
        print(rpath)
        if os.path.exists(rpath):   
            trainer.model.load_state_dict(torch.load(rpath))
            print("resume at epoch :{}".format(args.resume))
        else:
            print("no state dict found!!!")

    t1=time()
    print('Initial Evaluation')
    best_avg_auc=0
    for epoch in range(1, args.epochs + 1):
        '''if epoch % args.eval_steps == 0 or epoch==1 or epoch==args.epochs:
            print('Evaluation on epoch',epoch)
            best_avg_auc = trainer.test_step()'''
        print('----------------- Epoch:', epoch,' -----------------')
        trainer.train_step()

        if epoch % args.eval_steps == 0 or epoch==1 or epoch==args.epochs:
            print('Evaluation on epoch',epoch)
            best_avg_auc = trainer.test_step()

        if epoch % args.save_steps == 0:
            trainer.save_model()
        print('Time cost (min):', round((time()-t1)/60,3))
        t1=time()

    print('Train done.')
    print('Best Avg AUC:',best_avg_auc)

    trainer.conclude()

    if args.save_logs:
        trainer.save_result_log()

if __name__ == "__main__":
    main()
