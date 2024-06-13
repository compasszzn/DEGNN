import numpy as np
import torch
import pickle as pkl
import os
from tqdm import tqdm

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import radius_graph

def create_dataloader(dataset,size,segment:float, batch_size=32, shuffle=True, num_workers=8,args=None):
    if 'vehicle' in dataset:
        train_par, val_par, test_par = 0.7, 0.1, 0.2
        Data_list = []
        traj_dir = "dataset/vehical_dataset/processed_data"
        file=dataset+".npy"
        samples = np.load(os.path.join(traj_dir, file), allow_pickle=True)
        frames = samples.item()
        print(file)
        for i in list(frames.keys())[0:2000]:

            start_pos, end_pos = frames[i][:, 0:2], frames[i][:, 4:6]
            start_vel = frames[i][:, 2:4]
            node_feat = frames[i][:, 6:10]

            if "0_vehicle" in dataset:
                start_pos[:, 1] = start_pos[:, 1] - 1.75    ## crossing
                end_pos[:, 1] = end_pos[:, 1] - 1.75   ###

                start_pos[:, 0] = start_pos[:, 0] - 20  ##T-junc
                end_pos[:, 0] = end_pos[:, 0] - 20 ###
            elif "1_vehicle" in dataset:
                start_pos[:, 1] = start_pos[:, 1] - 2.4    ## crossing
                end_pos[:, 1] = end_pos[:, 1] - 2.4   ###

                start_pos[:, 0] = start_pos[:, 0] - 20  ##T-junc
                end_pos[:, 0] = end_pos[:, 0] - 20 ###
            elif "2_vehicle" in dataset:
                start_pos[:, 1] = start_pos[:, 1] - 1.55    ## crossing
                end_pos[:, 1] = end_pos[:, 1] - 1.55   ###

                start_pos[:, 0] = start_pos[:, 0] - 20  ##T-junc
                end_pos[:, 0] = end_pos[:, 0] - 20 ###
            elif "3_vehicle" in dataset:
                start_pos[:, 1] = start_pos[:, 1] - 1.65    ## crossing
                end_pos[:, 1] = end_pos[:, 1] - 1.65   ###

                start_pos[:, 0] = start_pos[:, 0] - 20  ##T-junc
                end_pos[:, 0] = end_pos[:, 0] - 20 ###
            elif "4_vehicle" in dataset:
                start_pos[:, 1] = start_pos[:, 1] - 1.7    ##
                end_pos[:, 1] = end_pos[:, 1] - 1.7   ###

                start_pos[:, 0] = start_pos[:, 0] - 20  ##
                end_pos[:, 0] = end_pos[:, 0] - 20 ###

            elif "5_vehicle" in dataset:
                start_pos[:, 1] = start_pos[:, 1] - 2    ## 
                end_pos[:, 1] = end_pos[:, 1] - 2   ###

                start_pos[:, 0] = start_pos[:, 0] - 20  ##
                end_pos[:, 0] = end_pos[:, 0] - 20 ###

            ped = start_pos.shape[0]
            
            edges = radius_graph(start_pos, r=1000, max_num_neighbors=100, loop=False)
            # edges, edge_attr = create_graph(start_pos)
            graph = Data(x=start_vel, edge_index=edges, pos=start_pos,node_attr=node_feat, ped=ped, y=end_pos)
            Data_list.append(graph)
        dataset_size = len(Data_list)

    elif dataset in ['indi_low','indi_high','group_low','group_high']:
        train_par, val_par, test_par = 0.7, 0.1, 0.2
        Data_list = []
        if dataset == 'indi_low':
            traj_dir = "dataset/crowd_dataset/processed_data_indi_4_2"
        elif dataset == 'indi_high':
            traj_dir = "dataset/crowd_dataset/processed_data_indi_5_1"
        elif dataset == 'group_low':
            traj_dir = "dataset/crowd_dataset/processed_data_groups_4_2"
        elif dataset == 'group_high':
            traj_dir = "dataset/crowd_dataset/processed_data_groups_5_1"

        files = os.listdir(traj_dir)

        for file in files:
            samples = np.load(os.path.join(traj_dir, file), allow_pickle=True)
            frames = samples.item()

            for i in list(frames.keys()):

                start_pos, end_pos = frames[i][:, 0:2], frames[i][:, 4:6]
                start_vel = frames[i][:, 2:4]
                reflect_x = torch.FloatTensor([[-1, 0], [0, 1]])

                ped = start_pos.shape[0]
                
                edges = radius_graph(start_pos, r=1000, max_num_neighbors=100, loop=False)
                graph = Data(x=start_vel, edge_index=edges, pos=start_pos, ped=ped, y=end_pos)
                Data_list.append(graph)
        dataset_size = len(Data_list)

    elif dataset=='lipo':
        train_par, val_par, test_par = segment.split(',')
        train_par, val_par, test_par = int(train_par),int(val_par),int(test_par)
        Data_list = []

        traj_dir = "dataset/molecular_dataset/processed_data/lipo"
        interval=10
        files = os.listdir(traj_dir)
        print(files)
        for file in files:
            samples = np.load(os.path.join(traj_dir, file))
            interval_frame=interval*1
            frames = torch.Tensor(samples['loc_vel'])[0:size+interval_frame]
            node_feat = np.expand_dims(samples['atom_types'][0],-1)
            node_feat = torch.Tensor(node_feat).long()
            cell=samples['cell']
            mean_x = np.mean(samples['loc_vel'][0:size+interval_frame][:, :, 0])
            mean_y = np.mean(samples['loc_vel'][0:size+interval_frame][:, :, 1])
            mean_z = np.mean(samples['loc_vel'][0:size+interval_frame][:, :, 2])
            print(mean_x,mean_y,mean_z)

            
            for i in range(len(frames)):
                start_frame=i
                end_frame=i+interval_frame
                if end_frame<len(frames):
                    start_pos, end_pos = frames[start_frame][:, 0:3], frames[end_frame][:, 0:3]
                    start_vel, end_vel = frames[start_frame][:, 3:6], frames[end_frame][:, 3:6]

                    start_pos[:, 0] = start_pos[:, 0] - mean_x  ##T-junc
                    end_pos[:, 0] = end_pos[:, 0] - mean_x ###

                    start_pos[:, 1] = start_pos[:, 1] - mean_y    ## crossing
                    end_pos[:, 1] = end_pos[:, 1] - mean_y   ###

                    start_pos[:, 2] = start_pos[:, 2] - mean_z    ## crossing
                    end_pos[:, 2] = end_pos[:, 2] - mean_z   ###

                    ped = start_pos.shape[0]

                    generate_edges = radius_graph(start_pos, r=2, max_num_neighbors=100, loop=False)

                    graph = Data(x=start_vel, edge_index=generate_edges, pos=start_pos, node_attr=node_feat, ped=ped, y=end_pos)
                    Data_list.append(graph)
                else:
                    break
        dataset_size = size
    elif dataset=='lips':
        train_par, val_par, test_par = segment.split(',')
        train_par, val_par, test_par = int(train_par),int(val_par),int(test_par)
        Data_list = []

        traj_dir = "dataset/molecular_dataset/processed_data/lips"
        interval=10
        files = os.listdir(traj_dir)

        print(files)
        for file in files:
            samples = np.load(os.path.join(traj_dir, file))
            node_feat = np.expand_dims(samples['atom_types'][0],-1)
            node_feat = torch.Tensor(node_feat).long()
            interval_frame=interval*1
            frames = torch.Tensor(samples['loc_vel'])[0:size+interval_frame]

            mean_x = np.mean(samples['loc_vel'][0:size+interval_frame][:, :, 0])
            mean_y = np.mean(samples['loc_vel'][0:size+interval_frame][:, :, 1])
            mean_z = np.mean(samples['loc_vel'][0:size+interval_frame][:, :, 2])

            cell=samples['cell']

            
            for i in range(len(frames)):
                start_frame=i
                end_frame=i+interval_frame
                if end_frame<len(frames):
                    start_pos, end_pos = frames[start_frame][:, 0:3], frames[end_frame][:, 0:3]
                    start_vel, end_vel = frames[start_frame][:, 3:6], frames[end_frame][:, 3:6]

                    start_pos[:, 0] = start_pos[:, 0] - mean_x  ##
                    end_pos[:, 0] = end_pos[:, 0] - mean_x ###

                    start_pos[:, 1] = start_pos[:, 1] - mean_y    ## 
                    end_pos[:, 1] = end_pos[:, 1] - mean_y   ###

                    start_pos[:, 2] = start_pos[:, 2] - mean_z    ##
                    end_pos[:, 2] = end_pos[:, 2] - mean_z   ###

                    ped = start_pos.shape[0]

                    generate_edges = radius_graph(start_pos, r=3, max_num_neighbors=100, loop=False)

                    graph = Data(x=start_vel, edge_index=generate_edges, pos=start_pos, node_attr=node_feat, ped=ped, y=end_pos)
                    Data_list.append(graph)
                else:
                    break
        dataset_size = size

    print()

    np.random.seed(100)

    train_idx = np.random.choice(np.arange(dataset_size), size=int(train_par * dataset_size / (train_par+val_par+test_par)), replace=False)
    flag = np.zeros(dataset_size)
    for _ in train_idx:
        flag[_] = 1
    rest = [_ for _ in range(dataset_size) if not flag[_]]
    val_idx = np.random.choice(rest, size=int(val_par * dataset_size / (train_par+val_par+test_par)), replace=False)
    for _ in val_idx:
        flag[_] = 1
    rest = [_ for _ in range(dataset_size) if not flag[_]]
    test_idx = np.random.choice(rest, size=int(test_par * dataset_size / (train_par+val_par+test_par)), replace=False)
    print("new",len(train_idx), len(test_idx), len(val_idx))
    
    train_set = [Data_list[i] for i in train_idx]
    val_set = [Data_list[i] for i in val_idx]
    test_set = [Data_list[i] for i in test_idx]


    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    testloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return trainloader, validloader, testloader
