# DEGNN

This is the official implementation for the following paper:

Relaxing Continuous Constraints of Equivariant Graph Neural Networks for Physical Dynamics Learning

Zinan Zheng^, Yang liu^, Jia Li*, Jianhua Yao, Yu Rong* 

KDD 2024

```bash
cd DEGNN
pip install -r requirements.txt
```

## **Quick Start**
Run the crowd datasets.
```bash
python main.py --dataset "indi_low" --lr "5e-4"
python main.py --dataset "indi_high" --lr "5e-4"
python main.py --dataset "group_low" --lr "5e-4"
python main.py --dataset "group_high" --lr "5e-4"
```
Run the vehicle datasets
```bash
python main.py --dataset "0_vehicle" --lr "5e-4"
python main.py --dataset "1_vehicle" --lr "5e-4"
python main.py --dataset "2_vehicle" --lr "5e-4"
python main.py --dataset "3_vehicle" --lr "5e-4"
python main.py --dataset "4_vehicle" --lr "5e-4"
python main.py --dataset "5_vehicle" --lr "5e-4"
```