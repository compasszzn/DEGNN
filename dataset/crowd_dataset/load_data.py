import pandas as pd
import numpy as np
import torch

# 指定CSV文件路径
csv_file_path = '/home/zinanzheng/project/KD/nbody/nbody/bidirectional_data/bidirectional_data/individuals/trajectories/4_2_D.csv'


# 使用pandas库读取CSV文件
df = pd.read_csv(csv_file_path)
time=np.sort(df['time (s)'].unique())
len_time=len(time)
interval=10
total_loc_vel={}
for i in range(len_time-interval):
    start_frame = df[df['time (s)'] == time[i]]
    start_frame_next = df[df['time (s)'] == time[i+1]]
    end_frame = df[df['time (s)'] == time[i+interval]]
    start_frame_ped=list(start_frame['ped. #'].values)
    end_frame_ped=list(end_frame['ped. #'].values)
    both_vel=list(set(start_frame_ped) & set(end_frame_ped))
    if len(both_vel)>=5:
        start_loc_x=np.expand_dims(start_frame[start_frame['ped. #'].isin(both_vel)]['x (cm)'].values,1)
        start_loc_y=np.expand_dims(start_frame[start_frame['ped. #'].isin(both_vel)]['y (cm)'].values,1)

        start_next_loc_x=np.expand_dims(start_frame_next[start_frame_next['ped. #'].isin(both_vel)]['x (cm)'].values,1)
        start_next_loc_y=np.expand_dims(start_frame_next[start_frame_next['ped. #'].isin(both_vel)]['y (cm)'].values,1)

        start_vel_x=start_next_loc_x-start_loc_x
        start_vel_y=start_next_loc_y-start_loc_y

        end_loc_x=np.expand_dims(end_frame[end_frame['ped. #'].isin(both_vel)]['x (cm)'].values,1)
        end_loc_y=np.expand_dims(end_frame[end_frame['ped. #'].isin(both_vel)]['y (cm)'].values,1)   
        loc_vel=torch.tensor(np.concatenate((start_loc_x,start_loc_y,start_vel_x,start_vel_y,end_loc_x,end_loc_y),1)/100, dtype=torch.float32)
        total_loc_vel[i]=loc_vel

np.save("/home/zinanzheng/project/KD/nbody/nbody/bidirectional_data/bidirectional_data/processed_data_indi_4_2_20/4_2_D_20.npy",total_loc_vel)