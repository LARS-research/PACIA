Code for PACIA: Parameter-Efficient Adapter for Few-Shot Molecular Property Prediction
## Environment  
We used the following Python packages for core development. We tested on `Python 3.7`.
```
- pytorch 1.7.0
- torch-geometric 1.7.0
```

## Datasets 
Tox21, SIDER, MUV and ToxCast are previously downloaded from [SNAP](http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip). You can download the data [here](https://drive.google.com/file/d/1K3c4iCFHEKUuDVSGBtBYr8EOegvIJulO/view?usp=sharing), unzip the file and put the resultant ``muv, sider, tox21, and toxcast" in the data folder. 

## Experiments
To run the experiments, use the command (please check and tune the hyper-parameters in [parser.py](parser.py):
```
python main.py
```

quick reproduce results on Tox-21 10-shot:
```
bash script_train.sh
```