import numpy as np
from torch import nn
import torch
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
 
def tanh(x):
    return np.tanh(x)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Khởi tạo trọng số và độ chệch
        self.Wf=nn.Parameter(torch.randn(self.hidden_size, self.hidden_size + self.input_size))
        self.bf= nn.Parameter(torch.zeros((self.hidden_size, 1))) 
        self.Wi= nn.Parameter(torch.randn(self.hidden_size, self.hidden_size + self.input_size))
        self.bi=nn.Parameter(torch.zeros((self.hidden_size, 1)))
        self.Wo= nn.Parameter(torch.randn(self.hidden_size, self.hidden_size + self.input_size))
        self.bo= nn.Parameter(torch.zeros((self.hidden_size, 1)))
        self.Wc = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size + self.input_size))
        self.bc = nn.Parameter(torch.zeros((self.hidden_size, 1)))

        # Trạng thái ẩn và trạng thái tế bào
        self.ht_prev = torch.zeros((hidden_size, 1))
        self.ct_prev = torch.zeros((hidden_size, 1))
              
    def lstm_cell(self, xt):
        # Phương thức tính toán đầu ra cho mỗi điểm thời gian
        concatenated_input = np.concatenate((self.ht_prev, xt), axis=0)

        ft = sigmoid(np.dot(self.Wf, concatenated_input) + self.bf)
        it = sigmoid(np.dot(self.Wi, concatenated_input) + self.bi)
        ot = sigmoid(np.dot(self.Wo, concatenated_input) + self.bo)
        cct = tanh(np.dot(self.Wc, concatenated_input) + self.bc)
        
        ct = ft * self.ct_prev + it * cct
        ht = ot * tanh(ct)
        
        # Lưu giữ giá trị cần thiết để sử dụng trong lan truyền ngược
        self.cache['ft'] = ft
        self.cache['it'] = it
        self.cache['ot'] = ot
        self.cache['cct'] = cct
        self.cache['ct'] = ct
        self.cache['ht'] = ht
        self.cache['concatenated_input'] = concatenated_input

        return ht, ct


        
    def forward(self,input_sequence): 
    
        output_sequence = []

        for xt in input_sequence:
            ht, ct = self.lstm_cell(xt)
            output_sequence.append(ht)

        return output_sequence