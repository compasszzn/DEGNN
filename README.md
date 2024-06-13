# DEGNN

This is the official implementation for the following paper:

Relaxing Continuous Constraints of Equivariant Graph Neural Networks for Physical Dynamics Learning

Zinan Zheng^, Yang liu^, Jia Li*, Jianhua Yao, Yu Rong* 

KDD 2024

```bash
cd DEGNN
pip install -r requirements.txt
```

The nbody and molecular dataset is hosted on Hugging Face and can be accessed [here](https://huggingface.co/datasets/compasszzn/Molecular/tree/main). 

Please place the **nbodydata** and **molecular_dataset** dataset folder under the **dataset** folder
```bash
-dataset
 -crowd_dataset
 -molecular_dataset
 -nbodydata
 -vehical_dataset
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
Run the nbody datasets.
```bash
python main.py --dataset "nbody_charged_4_4_4" --lr "3e-4" --dataset_segment "1,10,10" --dataset_size "4200"
python main.py --dataset "nbody_charged_5_4_3" --lr "3e-4" --dataset_segment "1,10,10" --dataset_size "4200"
python main.py --dataset "nbody_charged_5_4_4" --lr "3e-4" --dataset_segment "1,10,10" --dataset_size "4200"
python main.py --dataset "nbody_charged_5_5_5" --lr "3e-4" --dataset_segment "1,10,10" --dataset_size "4200"
python main.py --dataset "nbody_gravity_4_4_4" --lr "3e-4" --dataset_segment "1,10,10" --dataset_size "4200"
python main.py --dataset "nbody_gravity_5_4_3" --lr "3e-4" --dataset_segment "1,10,10" --dataset_size "4200"
python main.py --dataset "nbody_gravity_5_4_4" --lr "3e-4" --dataset_segment "1,10,10" --dataset_size "4200"
python main.py --dataset "nbody_gravity_5_5_5" --lr "3e-4" --dataset_segment "1,10,10" --dataset_size "4200"
```

Run the molecular datasets.
```bash
python main.py --dataset "lips" --lr "3e-4" --dataset_segment "1,1,1" --dataset_size "6000"
python main.py --dataset "lipo" --lr "3e-4" --dataset_segment "1,1,1" --dataset_size "6000"
```

