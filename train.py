import numpy as np
import torch
from torch import nn, optim
from models.model import DEGNN_3D,DEGNN_2D
from dataset.dataset import create_dataloader


def train(args):
    if args.gpus:
        device = torch.device('cuda:' + str(args.gpus_num))
    else:
        device = 'cpu'
    
    if args.dataset in ['lips','lipo']:
        node_nf=1
        model = DEGNN_3D(in_node_nf=1+node_nf, in_edge_nf=7, hidden_nf=64,model_type=args.dataset,pool_method=args.pool_method,device=device, n_layers=args.layers,
                            recurrent=True, norm_diff=False, tanh=False,embed_vel=True)
    elif 'vehicle' in args.dataset:
        node_nf=4
        model = DEGNN_2D(in_node_nf=1+node_nf, in_edge_nf=3, hidden_nf=64,model_type=args.dataset,pool_method=args.pool_method,device=device, n_layers=args.layers,
                            recurrent=True, norm_diff=False, tanh=False)
    elif args.dataset in ['indi_low','indi_high','group_low','group_high']:
        model = DEGNN_2D(in_node_nf=1, in_edge_nf=3, hidden_nf=64,model_type=args.dataset,pool_method=args.pool_method,device=device, n_layers=args.layers,
                            recurrent=True, norm_diff=False, tanh=False)
    print("The model has {:,} parameters.".format(sum(p.numel() for p in model.parameters())))

    loader_train, loader_val, loader_test = create_dataloader(args.dataset,args.dataset_size,args.dataset_segment, batch_size=args.batch_size,
                                            shuffle=True,  num_workers=8,args=args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_mse = nn.MSELoss()


    best_val_loss = 1e8
    best_test_loss = 1e8
    best_epoch = 0
    for epoch in range(0, args.epochs):
        train_loss, _ = run_epoch(model, optimizer, loss_mse, epoch, loader_train, device, args)
        if epoch % args.test_interval == 0 or epoch == args.epochs-1:

            val_loss, res = run_epoch(model, optimizer, loss_mse, epoch, loader_val,  device, args, backprop=False)
            test_loss, res = run_epoch(model, optimizer, loss_mse, epoch, loader_test,
                                device, args, backprop=False)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_loss = test_loss
                best_epoch = epoch
            
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

    res = {'epoch': epoch, 'loss': 0, 'counter': 0}
    for batch_idx, data in enumerate(loader):
        loc, vel, loc_end= data.pos.to(device), data.x.to(device), data.y.to(device)
        if args.dataset in ['lipo','lips'] or 'vehicle' in args.dataset:
            node_attr = data.node_attr.to(device)
        edges = data.edge_index.to(device)
        batch_size = loc.shape[0]

        if backprop:
                            
            optimizer.zero_grad()

            if args.dataset in ['indi_low','indi_high','group_low','group_high']:
                nodes = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach() 
                rows, cols = edges
                loc_dist = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)  # relative distances among locations
                edge_attr=loc_dist.detach()
                loc_pred = model(nodes, loc.detach(), edges, vel, edge_attr)
            elif args.dataset in ['lipo','lips'] or "vehicle" in args.dataset:
                nodes = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach()
                nodes = torch.cat([nodes, node_attr], dim=1)
                rows, cols = edges
                loc_dist = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)
                edge_attr=loc_dist.detach()
                loc_pred = model(nodes, loc.detach(), edges, vel, edge_attr)
            else:
                raise Exception("Unknown dataset")
            loss = criterion(loc_pred, loc_end)
            loss.backward()
            optimizer.step()
            res['loss'] += loss.item()*batch_size
            res['counter'] += batch_size
        else:
            if args.dataset in ['indi_low','indi_high','group_low','group_high']:
                nodes = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach() 
                rows, cols = edges
                loc_dist = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)  # relative distances among locations
                edge_attr=loc_dist.detach()
                loc_pred = model(nodes, loc.detach(), edges, vel, edge_attr)
            elif args.dataset in ['lipo','lips'] or "vehicle" in args.dataset:
                nodes = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach()
                nodes = torch.cat([nodes, node_attr], dim=1)
                rows, cols = edges
                loc_dist = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)  # relative distances among locations
                edge_attr=loc_dist.detach()
                loc_pred = model(nodes, loc.detach(), edges, vel, edge_attr)
            else:
                raise Exception("Unknown dataset")
            loss = criterion(loc_pred, loc_end)
            res['loss'] += loss.item()*batch_size
            res['counter'] += batch_size


    if not backprop:
        prefix = "==> "
    else:
        prefix = ""
    print('%s epoch %d avg loss: %.5f' % (prefix, epoch, res['loss'] / res['counter']))

    return res['loss'] / res['counter'], res
