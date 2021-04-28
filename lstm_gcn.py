import torch
import torch.nn as nn

class LSTM_GCN(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super(LSTM_GCN, self).__init__()
        self.in_c=in_c
        self.hid_c=hid_c
        self.out_c=out_c
        self.lstm = nn.LSTM(in_c,in_c)
        self.linear_1 = nn.Linear(in_c, hid_c)  
        self.linear_2 = nn.Linear(hid_c, out_c) 
        self.act = nn.ReLU() 
    def forward(self, data, device):
        graph_data = data["graph"].to(device)[0]  
        graph_data = LSTM_GCN.process_graph(graph_data)  
        flow_x = data["flow_x"].to(device) 

        B, N = flow_x.size(0), flow_x.size(1) 

        flow_x = flow_x.view(B, N, -1)  
        
        lstm_input1 = flow_x
        X_out1, X_out2 = self.lstm(lstm_input1)
        
        output_1 = self.linear_1(X_out1)  
        output_1 = self.act(torch.matmul(graph_data, output_1))  

       
        output_2 = self.linear_2(output_1)
        output_2 = self.act(torch.matmul(graph_data, output_2)) 

        return output_2.unsqueeze(2)  
    @staticmethod
    def process_graph(graph_data): 
        N = graph_data.size(0) 
        matrix_i = torch.eye(N, dtype=torch.float, device=graph_data.device)  
        graph_data += matrix_i  

        degree_matrix = torch.sum(graph_data, dim=1, keepdim=False) 
        degree_matrix = degree_matrix.pow(-1) 
        degree_matrix[degree_matrix == float("inf")] = 0.  

        degree_matrix = torch.diag(degree_matrix)

        return torch.mm(degree_matrix, graph_data) 