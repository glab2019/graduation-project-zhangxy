import csv
import torch
import numpy as np
from torch.utils.data import Dataset

def get_adjacent_matrix(distance_file: str, num_nodes: int, id_file: str = None, graph_type="connect") -> np.array:
   
    A = np.zeros([int(num_nodes), int(num_nodes)]) 

    if id_file:
        with open(id_file, "r") as f_id:
          
            node_id_dict = {int(node_id): idx for idx, node_id in enumerate(f_id.read().strip().split("\n"))}

            with open(distance_file, "r") as f_d:
                f_d.readline()
                reader = csv.reader(f_d) 
                for item in reader:  
                    if len(item) != 3: 
                        continue
                    i, j, distance = int(item[0]), int(item[1]), float(item[2])
                    if graph_type == "connect":
                        A[node_id_dict[i], node_id_dict[j]] = 1.
                        A[node_id_dict[j], node_id_dict[i]] = 1.
                    elif graph_type == "distance": 
                        A[node_id_dict[i], node_id_dict[j]] = 1. / distance
                        A[node_id_dict[j], node_id_dict[i]] = 1. / distance
                    else:
                        raise ValueError("graph type is not correct (connect or distance)")
        return A

    with open(distance_file, "r") as f_d:
        f_d.readline() 
        reader = csv.reader(f_d) 
        for item in reader:  
            if len(item) != 3: 
                continue
            i, j, distance = int(item[0]), int(item[1]), float(item[2])
            if distance>2:
                continue
            if graph_type == "connect": 
                A[i, j], A[j, i] = 1., 1.
            elif graph_type == "distance": 
                A[i, j] = 1./distance
                A[j, i] = 1./distance
            else:
                raise ValueError("graph type is not correct (connect or distance)")

    return A

def get_flow_data(flow_file: str) -> np.array: 

    data = np.load(flow_file)

    flow_data = data['data'].transpose([0, 1, 2])[:, :, 0][:, :, np.newaxis] 
    return flow_data  # [N, T, D]


class LoadData(Dataset): 
    def __init__(self, data_path, num_nodes, divide_days, time_interval, history_length, train_mode):

        self.data_path = data_path
        self.num_nodes = num_nodes
        self.train_mode = train_mode
        self.train_days = divide_days[0] 
        self.test_days = divide_days[1] 
        self.history_length = history_length 
        self.time_interval = time_interval

        self.one_day_length = int(24 * 60 / self.time_interval) 

        self.graph = get_adjacent_matrix(distance_file=data_path[0], num_nodes=num_nodes)

        self.flow_norm, self.flow_data = self.pre_process_data(data=get_flow_data(data_path[1]), norm_dim=1) 

    def __len__(self):  

        if self.train_mode == "train":
            return self.train_days * self.one_day_length - self.history_length 
        elif self.train_mode == "test":
            return self.test_days * self.one_day_length 
        else:
            raise ValueError("train mode: [{}] is not defined".format(self.train_mode))

    def __getitem__(self, index):
        if self.train_mode == "train":
            index = index
        elif self.train_mode == "test":
            index += self.train_days * self.one_day_length
        else:
            raise ValueError("train mode: [{}] is not defined".format(self.train_mode))

        data_x, data_y = LoadData.slice_data(self.flow_data, self.history_length, index, self.train_mode)

        data_x = LoadData.to_tensor(data_x) 
        data_y = LoadData.to_tensor(data_y).unsqueeze(1)  # [N, 1, D]　

        return {"graph": LoadData.to_tensor(self.graph), "flow_x": data_x, "flow_y": data_y} 

    @staticmethod
    def slice_data(data, history_length, index, train_mode): 
       
        if train_mode == "train":
            start_index = index 
            end_index = index + history_length 
        elif train_mode == "test":
            start_index = index - history_length 
            end_index = index 
        else:
            raise ValueError("train model {} is not defined".format(train_mode))

        data_x = data[:, start_index: end_index]  
        data_y = data[:, end_index]  

        return data_x, data_y

    @staticmethod
    def pre_process_data(data, norm_dim): 
        norm_base = LoadData.normalize_base(data, norm_dim) 
        norm_data = LoadData.normalize_data(norm_base[0], norm_base[1], data) 

        return norm_base, norm_data  
    @staticmethod
    def normalize_base(data, norm_dim):
        max_data = np.max(data, norm_dim, keepdims=True) 
        min_data = np.min(data, norm_dim, keepdims=True)

        return max_data, min_data  
    @staticmethod
    def normalize_data(max_data, min_data, data):
        mid = min_data
        base = max_data - min_data
        normalized_data = (data - mid) / base

        return normalized_data

    @staticmethod
    def recover_data(max_data, min_data, data): 
        mid = min_data
        base = max_data - min_data

        recovered_data = data * base + mid

        return recovered_data #这个就是原始的数据

    @staticmethod
    def to_tensor(data):
        return torch.tensor(data, dtype=torch.float)
