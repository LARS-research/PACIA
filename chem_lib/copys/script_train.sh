DATA=tox21
NS=10
NQ=16
pretrain=1
gpu_id=4
seed=0
mod=2
ar=1
att=0
regadj=0
adc=4

nohup python -u main.py --epochs 40000 --eval_steps 10 --pretrained $pretrain --meta-lr 0.0006 --adc $adc \
--n_shot_train $NS  --n-shot-test $NS --n-query $NQ --dataset $DATA --test-dataset $DATA --seed $seed --gpu_id $gpu_id --mod $mod --ar $ar --att $att --reg_adj $regadj \
> 0nohup_${DATA}-k${NS}q${NQ}-m${mod}-att${att}-adc${adc}-44x2s${seed} 2>&1 &
