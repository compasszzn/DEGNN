import numpy as np
import torch
from torch import nn, optim
import time
from models.model import DEGNN_3D

from dataset.dataset_nbody import NBodyDataset, GravityNBodyDataset

time_exp_dic = {'time': 0, 'counter': 0}


def train(args):

    if args.gpus:
        device = torch.device('cuda:' + str(args.gpus_num))
    else:
        device = 'cpu'


    model = DEGNN_3D(in_node_nf=1, in_edge_nf=8, hidden_nf=64,model_type=args.dataset,pool_method=args.pool_method,device=device, n_layers=args.layers,
                        recurrent=True, norm_diff=False, tanh=False,embed_vel=True)

    
    train_par, val_par, test_par = args.dataset_segment.split(',')
    train_par, val_par, test_par = int(train_par),int(val_par),int(test_par)

    train_max_samples=train_par * args.dataset_size / (train_par+val_par+test_par)
    val_max_samples=val_par * args.dataset_size / (train_par+val_par+test_par)
    test_max_samples=test_par * args.dataset_size / (train_par+val_par+test_par)
    print(train_max_samples, val_max_samples, test_max_samples )

    if 'charged' in args.dataset:
        dataset_train = NBodyDataset(partition='train', dataset_name=args.nbody_name, dataset_type=args.dataset,
                                    max_samples=train_max_samples)
        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True)

        dataset_val = NBodyDataset(partition='val', dataset_name=args.nbody_name, dataset_type=args.dataset,
                                    max_samples=val_max_samples)
        loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, drop_last=False)
        
        dataset_test = NBodyDataset(partition='test', dataset_name=args.nbody_name, dataset_type=args.dataset,
                                    max_samples=test_max_samples)
        loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False)
    elif 'gravity' in args.dataset:
        dataset_train = GravityNBodyDataset(partition='train', dataset_name=args.nbody_name, dataset_type=args.dataset,
                                    max_samples=train_max_samples)
        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True)

        dataset_val = GravityNBodyDataset(partition='val', dataset_name=args.nbody_name, dataset_type=args.dataset,
                                    max_samples=val_max_samples)
        loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, drop_last=False)
        
        dataset_test = GravityNBodyDataset(partition='test', dataset_name=args.nbody_name, dataset_type=args.dataset,
                                    max_samples=test_max_samples)
        loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False)
    
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_mse = nn.MSELoss()

    best_val_loss = 1e8
    best_test_loss = 1e8
    best_epoch = 0
    for epoch in range(0, args.epochs):
        train_loss, _ = run_epoch(model, optimizer, loss_mse, epoch, loader_train, device, args)
        if epoch % args.test_interval == 0 or epoch == args.epochs-1:
            val_loss, res = run_epoch(model, optimizer, loss_mse, epoch, loader_val, device, args, backprop=False)
            test_loss, res = run_epoch(model, optimizer, loss_mse, epoch, loader_test, device, args, backprop=False)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_loss = test_loss
                best_epoch = epoch
                best = res
                #torch.save(model.state_dict(), args.save_dir + '/'+ + 'saved_model.pth')
            print("*** Best Val Loss: %.5f \t Best Test Loss: %.5f \t Best epoch %d" %
                  (best_val_loss, best_test_loss, best_epoch))
            
            if epoch - best_epoch > 100:
                break
            
    return best_val_loss, best_test_loss, best_epoch


def run_epoch(model, optimizer, criterion, epoch, loader, device, args, backprop=True):
    if backprop:
        model.train()
    else:
        model.eval()

    res = {'epoch': epoch, 'loss': 0, 'counter': 0, 'long_loss': {}}
    n_nodes = args.n_nodes
    batch_size = args.batch_size

    # edges = loader.dataset.get_edges(args.batch_size, n_nodes)
    # edges = [edges[0], edges[1]]
    # edge_index = torch.stack(edges)
    for batch_idx, data in enumerate(loader):
        edges = loader.dataset.get_edges(args.batch_size, n_nodes)
        edges = [edges[0], edges[1]]
        edge_index = torch.stack(edges)
        data = [d.to(device) for d in data]

        for i in range(len(data)):
            if len(data[i].shape) == 4:
                data[i] = data[i].transpose(0, 1).contiguous()
                data[i] = data[i].view(data[i].size(0), -1, data[i].size(-1))
            else:
                data[i] = data[i][:, :data[i].size(1), :].contiguous()
                data[i] = data[i].view(-1, data[i].size(-1))  
        locs, vels, edge_attr, charges, loc_ends = data
        if args.cut:
            rows, cols = edges
            cut_list = (torch.sum((locs[rows] - locs[cols])**2, 1) < args.cut_size).to('cpu')
            edges[0]=rows[cut_list]
            edges[1]=cols[cut_list]
            edge_attr=edge_attr[cut_list]
        edges = torch.stack(edges)
        edges = edges.to(device)


        if backprop:
            loc, loc_end, vel = locs, loc_ends, vels             
            optimizer.zero_grad()

            #EGNN
            nodes = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach() 
            rows, cols = edges
            loc_dist = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)  # relative distances among locations
            edge_attr = torch.cat([edge_attr, loc_dist], 1).detach()  # concatenate all edge properties
            loc_pred = model(nodes, loc.detach(), edges, vel, edge_attr)


            loss = criterion(loc_pred, loc_end)
            loss.backward()
            optimizer.step()

            res['loss'] += loss.item()*batch_size
            res['counter'] += batch_size
        else:

            loc, loc_end, vel = locs, loc_ends, vels
            #EGNN
            nodes = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach() 
            rows, cols = edges
            loc_dist = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)  # relative distances among locations
            edge_attr = torch.cat([edge_attr, loc_dist], 1).detach()  # concatenate all edge properties
            loc_pred = model(nodes, loc.detach(), edges, vel, edge_attr)            

            loss = criterion(loc_pred, loc_end)
            

            res['loss'] += loss.item()*batch_size

        res['counter'] += batch_size

    if not backprop:
        prefix = "==> "

    else:
        prefix = ""
    print('%s epoch %d avg loss: %.5f' % (prefix, epoch, res['loss'] / res['counter']))

    return res['loss'] / res['counter'], res
