from Neural import LSTM
import torch.nn.functional as F
from torch import nn

class RNNSeq2seq(nn.Module):
    
    def __init__(self, input_size, hidden_size):
        super(RNNSeq2seq, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.encoder = LSTM(self.input_size, self.hidden_size)
        self.decoder = LSTM(self.hidden_size,5)
    def forward(self, input_sequence):
        # Lan truyền thuận qua chuỗi đầu vào cho bộ encoder

        lstm_output_sequence = self.encoder.forward(input_sequence)
        
        # Lấy trạng thái ẩn cuối cùng làm context vector
        context_vector = lstm_output_sequence[-1]
        #decoder lấy đầu vào là context vecto
        result=self.decoder.forward(context_vector)
        return result

         
