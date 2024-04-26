DATA=tox21
NS=10
NQ=16
pretrain=1
gpu_id=2
seed=0
mod=16
ar=1
att=0
regadj=0
adc=0
rsm=0
eid=-1
encl=5

nohup python -u main.py --epochs 40000 --eval_steps 10 --pretrained $pretrain --meta-lr 0.0006 --adc $adc --resume $rsm --eid $eid --enc_layer $encl \
--n_shot_train $NS  --n-shot-test $NS --n-query $NQ --dataset $DATA --test-dataset $DATA --seed $seed --gpu_id $gpu_id --mod $mod --ar $ar --att $att --reg_adj $regadj \
> submi 2>&1 &
#abl_nohup_${DATA}-k${NS}q${NQ}-m${mod}h2l${encl}-adc${adc}knn-33x2s${seed}r${rsm}id${eid}