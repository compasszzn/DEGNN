import numpy as np
import torch
import random
import os.path as osp
import pathlib


class NBodyDataset():
    """
    NBodyDataset

    """

    def __init__(self, partition='train', max_samples=1e8, dataset_name="nbody_small",dataset_type='nbody_charged_3_3_3'):
        self.partition = partition
        if self.partition == 'val':
            self.suffix = 'valid'
        else:
            self.suffix = self.partition
        self.dataset_name = dataset_name
        if dataset_name == "nbody":
            self.suffix += "_charged50_initvel1"
        elif dataset_name == "nbody_small" or dataset_name == "nbody_small_out_dist":
            self.suffix += "_charged50_initvel1"
        else:
            raise Exception("Wrong dataset name %s" % self.dataset_name)
        self.dataset_type=dataset_type
        self.max_samples = int(max_samples)
        self.dataset_name = dataset_name
        self.data, self.edges = self.load()

    def load(self):
        dir = pathlib.Path(__file__).parent.absolute()
        print(dir)
        print("loading dataset ",self.dataset_type)
        loc = np.load(osp.join(dir, 'nbodydata',self.dataset_type, 'loc_' + self.suffix + '.npy'))
        vel = np.load(osp.join(dir, 'nbodydata',self.dataset_type, 'vel_' + self.suffix + '.npy'))
        edges = np.load(osp.join(dir, 'nbodydata',self.dataset_type, 'edges_' + self.suffix + '.npy'))
        charges = np.load(osp.join(dir, 'nbodydata',self.dataset_type, 'charges_' + self.suffix + '.npy'))

        loc, vel, edge_attr, edges, charges = self.preprocess(loc, vel, edges, charges)
        return (loc, vel, edge_attr, charges), edges


    def preprocess(self, loc, vel, edges, charges):
        # cast to torch and swap n_nodes <--> n_features dimensions
        loc, vel = torch.Tensor(loc).transpose(2, 3), torch.Tensor(vel).transpose(2, 3)
        n_nodes = loc.size(2)
        loc = loc[0:self.max_samples, :, :, :]  # limit number of samples
        vel = vel[0:self.max_samples, :, :, :]  # speed when starting the trajectory
        charges = charges[0:self.max_samples]
        edge_attr = []

        # Initialize edges and edge_attributes
        rows, cols = [], []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    edge_attr.append(edges[:, i, j])
                    rows.append(i)
                    cols.append(j)
        edges = [rows, cols]
        edge_attr = torch.Tensor(edge_attr).transpose(0, 1).unsqueeze(
            2)  # swap n_nodes <--> batch_size and add nf dimension

        return torch.Tensor(loc), torch.Tensor(vel), torch.Tensor(edge_attr), edges, torch.Tensor(charges)

    def set_max_samples(self, max_samples):
        self.max_samples = int(max_samples)
        self.data, self.edges = self.load()
    '''
    def preprocess_old(self, loc, vel, edges, charges):
        # cast to torch and swap n_nodes <--> n_features dimensions
        loc, vel = torch.Tensor(loc).transpose(2, 3), torch.Tensor(vel).transpose(2, 3)
        n_nodes = loc.size(2)
        loc0 = loc[0:self.max_samples, 0, :, :]  # first location from the trajectory
        loc_last = loc[0:self.max_samples, -1, :, :]  # last location from the trajectory
        vel = vel[0:self.max_samples, 0, :, :]  # speed when starting the trajectory
        charges = charges[0:self.max_samples]
        edge_attr = []

        # Initialize edges and edge_attributes
        rows, cols = [], []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    edge_attr.append(edges[:, i, j])
                    rows.append(i)
                    cols.append(j)
        edges = [rows, cols]
        edge_attr = torch.Tensor(edge_attr).transpose(0, 1).unsqueeze(
            2) # swap n_nodes <--> batch_size and add nf dimension

        return torch.Tensor(loc0), torch.Tensor(vel), torch.Tensor(edge_attr), loc_last, edges, torch.Tensor(charges)
    '''

    def get_n_nodes(self):
        return self.data[0].size(1)

    def __getitem__(self, i):
        loc, vel, edge_attr, charges = self.data
        loc, vel, edge_attr, charges = loc[i], vel[i], edge_attr[i], charges[i]

        if self.dataset_name == "nbody":
            frame_0, frame_T = 6, 8
        elif self.dataset_name == "nbody_small":
            frame_0, frame_T = 20, 40
        elif self.dataset_name == "nbody_small_out_dist":
            frame_0, frame_T = 20, 30
        else:
            raise Exception("Wrong dataset partition %s" % self.dataset_name)

        return loc[frame_0], vel[frame_0], edge_attr, charges, loc[frame_T]
        #return loc, vel, edge_attr, charges, loc

    def __len__(self):
        return len(self.data[0])

    def get_edges(self, batch_size, n_nodes):
        edges = [torch.LongTensor(self.edges[0]), torch.LongTensor(self.edges[1])]
        if batch_size == 1:
            return edges
        elif batch_size > 1:
            rows, cols = [], []
            for i in range(batch_size):
                rows.append(edges[0] + n_nodes * i)
                cols.append(edges[1] + n_nodes * i)
            edges = [torch.cat(rows), torch.cat(cols)]
        return edges

class SpringNBodyDataset():
    """
    NBodyDataset

    """

    def __init__(self, partition='train', max_samples=1e8, dataset_name="nbody_small",dataset_type='nbody_charged_3_3_3'):
        self.partition = partition
        if self.partition == 'val':
            self.suffix = 'valid'
        else:
            self.suffix = self.partition
        self.dataset_name = dataset_name
        if dataset_name == "nbody":
            self.suffix += "_springs50_initvel1"
        elif dataset_name == "nbody_small" or dataset_name == "nbody_small_out_dist":
            self.suffix += "_springs50_initvel1"
        else:
            raise Exception("Wrong dataset name %s" % self.dataset_name)
        self.dataset_type=dataset_type
        self.max_samples = int(max_samples)
        self.dataset_name = dataset_name
        self.data, self.edges = self.load()

    def load(self):
        dir = pathlib.Path(__file__).parent.absolute()
        print(dir)
        print("loading dataset ",self.dataset_type)
        loc = np.load(osp.join(dir, 'nbodydata',self.dataset_type, 'loc_' + self.suffix + '.npy'))
        vel = np.load(osp.join(dir, 'nbodydata',self.dataset_type, 'vel_' + self.suffix + '.npy'))
        edges = np.load(osp.join(dir, 'nbodydata',self.dataset_type, 'edges_' + self.suffix + '.npy'))

        loc, vel, edge_attr, edges = self.preprocess(loc, vel, edges)
        return (loc, vel, edge_attr), edges


    def preprocess(self, loc, vel, edges):
        # cast to torch and swap n_nodes <--> n_features dimensions
        loc, vel = torch.Tensor(loc).transpose(2, 3), torch.Tensor(vel).transpose(2, 3)
        n_nodes = loc.size(2)
        loc = loc[0:self.max_samples, :, :, :]  # limit number of samples
        vel = vel[0:self.max_samples, :, :, :]  # speed when starting the trajectory
        edge_attr = []

        # Initialize edges and edge_attributes
        rows, cols = [], []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    edge_attr.append(edges[:, i, j])
                    rows.append(i)
                    cols.append(j)
        edges = [rows, cols]
        edge_attr = torch.Tensor(edge_attr).transpose(0, 1).unsqueeze(
            2)  # swap n_nodes <--> batch_size and add nf dimension

        return torch.Tensor(loc), torch.Tensor(vel), torch.Tensor(edge_attr), edges

    def set_max_samples(self, max_samples):
        self.max_samples = int(max_samples)
        self.data, self.edges = self.load()
    '''
    def preprocess_old(self, loc, vel, edges, charges):
        # cast to torch and swap n_nodes <--> n_features dimensions
        loc, vel = torch.Tensor(loc).transpose(2, 3), torch.Tensor(vel).transpose(2, 3)
        n_nodes = loc.size(2)
        loc0 = loc[0:self.max_samples, 0, :, :]  # first location from the trajectory
        loc_last = loc[0:self.max_samples, -1, :, :]  # last location from the trajectory
        vel = vel[0:self.max_samples, 0, :, :]  # speed when starting the trajectory
        charges = charges[0:self.max_samples]
        edge_attr = []

        # Initialize edges and edge_attributes
        rows, cols = [], []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    edge_attr.append(edges[:, i, j])
                    rows.append(i)
                    cols.append(j)
        edges = [rows, cols]
        edge_attr = torch.Tensor(edge_attr).transpose(0, 1).unsqueeze(
            2) # swap n_nodes <--> batch_size and add nf dimension

        return torch.Tensor(loc0), torch.Tensor(vel), torch.Tensor(edge_attr), loc_last, edges, torch.Tensor(charges)
    '''

    def get_n_nodes(self):
        return self.data[0].size(1)

    def __getitem__(self, i):
        loc, vel, edge_attr = self.data
        loc, vel, edge_attr = loc[i], vel[i], edge_attr[i]

        if self.dataset_name == "nbody":
            frame_0, frame_T = 6, 8
        elif self.dataset_name == "nbody_small":
            frame_0, frame_T = 20, 40
        elif self.dataset_name == "nbody_small_out_dist":
            frame_0, frame_T = 20, 30
        else:
            raise Exception("Wrong dataset partition %s" % self.dataset_name)

        return loc[frame_0], vel[frame_0], edge_attr, loc[frame_T]
        #return loc, vel, edge_attr, charges, loc

    def __len__(self):
        return len(self.data[0])

    def get_edges(self, batch_size, n_nodes):
        edges = [torch.LongTensor(self.edges[0]), torch.LongTensor(self.edges[1])]
        if batch_size == 1:
            return edges
        elif batch_size > 1:
            rows, cols = [], []
            for i in range(batch_size):
                rows.append(edges[0] + n_nodes * i)
                cols.append(edges[1] + n_nodes * i)
            edges = [torch.cat(rows), torch.cat(cols)]
        return edges

class GravityNBodyDataset():
    """
    NBodyDataset

    """

    def __init__(self, partition='train', max_samples=1e8, dataset_name="nbody_small",dataset_type='nbody_charged_3_3_3'):
        self.partition = partition
        if self.partition == 'val':
            self.suffix = 'valid'
        else:
            self.suffix = self.partition
        self.dataset_name = dataset_name
        if dataset_name == "nbody":
            self.suffix += "_gravity50_initvel1"
        elif dataset_name == "nbody_small" or dataset_name == "nbody_small_out_dist":
            self.suffix += "_gravity50_initvel1"
        else:
            raise Exception("Wrong dataset name %s" % self.dataset_name)
        self.dataset_type=dataset_type
        self.max_samples = int(max_samples)
        self.dataset_name = dataset_name
        self.data, self.edges = self.load()

    def load(self):
        dir = pathlib.Path(__file__).parent.absolute()
        print(dir)
        print("loading dataset ",self.dataset_type)
        loc = np.load(osp.join(dir, 'nbodydata',self.dataset_type, 'loc_' + self.suffix + '.npy'))
        vel = np.load(osp.join(dir, 'nbodydata',self.dataset_type, 'vel_' + self.suffix + '.npy'))
        edges = np.load(osp.join(dir, 'nbodydata',self.dataset_type, 'force_' + self.suffix + '.npy'))
        charges = np.load(osp.join(dir, 'nbodydata',self.dataset_type, 'mass_' + self.suffix + '.npy'))

        loc, vel, edge_attr, edges, charges = self.preprocess(loc, vel, edges, charges)
        return (loc, vel, edge_attr, charges), edges


    def preprocess(self, loc, vel, edges, charges):
        # cast to torch and swap n_nodes <--> n_features dimensions
        loc, vel = torch.Tensor(loc), torch.Tensor(vel)
        n_nodes = loc.size(2)
        loc = loc[0:self.max_samples, :, :, :]  # limit number of samples
        vel = vel[0:self.max_samples, :, :, :]  # speed when starting the trajectory
        charges = charges[0:self.max_samples]
        edge_attr = []

        # Initialize edges and edge_attributes
        rows, cols = [], []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    edge_attr.append(np.zeros(2000))
                    rows.append(i)
                    cols.append(j)
        edges = [rows, cols]
        edge_attr = torch.Tensor(edge_attr).transpose(0, 1).unsqueeze(
            2)  # swap n_nodes <--> batch_size and add nf dimension

        return torch.Tensor(loc), torch.Tensor(vel), torch.Tensor(edge_attr), edges, torch.Tensor(charges)

    def set_max_samples(self, max_samples):
        self.max_samples = int(max_samples)
        self.data, self.edges = self.load()
    '''
    def preprocess_old(self, loc, vel, edges, charges):
        # cast to torch and swap n_nodes <--> n_features dimensions
        loc, vel = torch.Tensor(loc).transpose(2, 3), torch.Tensor(vel).transpose(2, 3)
        n_nodes = loc.size(2)
        loc0 = loc[0:self.max_samples, 0, :, :]  # first location from the trajectory
        loc_last = loc[0:self.max_samples, -1, :, :]  # last location from the trajectory
        vel = vel[0:self.max_samples, 0, :, :]  # speed when starting the trajectory
        charges = charges[0:self.max_samples]
        edge_attr = []

        # Initialize edges and edge_attributes
        rows, cols = [], []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    edge_attr.append(edges[:, i, j])
                    rows.append(i)
                    cols.append(j)
        edges = [rows, cols]
        edge_attr = torch.Tensor(edge_attr).transpose(0, 1).unsqueeze(
            2) # swap n_nodes <--> batch_size and add nf dimension

        return torch.Tensor(loc0), torch.Tensor(vel), torch.Tensor(edge_attr), loc_last, edges, torch.Tensor(charges)
    '''

    def get_n_nodes(self):
        return self.data[0].size(1)

    def __getitem__(self, i):
        loc, vel, edge_attr, charges = self.data
        loc, vel, edge_attr, charges = loc[i], vel[i], edge_attr[i], charges[i]

        if self.dataset_name == "nbody":
            frame_0, frame_T = 6, 8
        elif self.dataset_name == "nbody_small":
            frame_0, frame_T = 20, 40
        elif self.dataset_name == "nbody_small_out_dist":
            frame_0, frame_T = 20, 30
        else:
            raise Exception("Wrong dataset partition %s" % self.dataset_name)

        return loc[frame_0], vel[frame_0], edge_attr, charges, loc[frame_T]
        #return loc, vel, edge_attr, charges, loc

    def __len__(self):
        return len(self.data[0])

    def get_edges(self, batch_size, n_nodes):
        edges = [torch.LongTensor(self.edges[0]), torch.LongTensor(self.edges[1])]
        if batch_size == 1:
            return edges
        elif batch_size > 1:
            rows, cols = [], []
            for i in range(batch_size):
                rows.append(edges[0] + n_nodes * i)
                cols.append(edges[1] + n_nodes * i)
            edges = [torch.cat(rows), torch.cat(cols)]
        return edges


if __name__ == "__main__":
    NBodyDataset()
