import torch.nn as nn

# This class is the same as the one in the lab notebook, only the RNN was substituted with the LSTM

class LM_LSTM_NO_DROP(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, n_layers=1):
        super(LM_LSTM_NO_DROP, self).__init__()
        # Token ids to vectors, we will better see this in the next lab
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # RNN --> LSTM
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index
        # Linear layer to project the hidden layer to our output space
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        lstm_out, _  = self.lstm(emb)
        output = self.output(lstm_out).permute(0,2,1)
        return output


# In this class instead, the first one was taken and it was added 2 dropout layers 
# One after the embedding and one after the output of the lstm
class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1, emb_dropout=0.1, n_layers=3):
        super(LM_LSTM, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # Embedding dropout layer
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        # Output dropout layer before the last linear layer
        self.out_dropout = nn.Dropout(out_dropout)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        # Apply embedding dropout
        emb = self.emb_dropout(emb)
        lstm_out, _ = self.lstm(emb)
        # Apply dropout before the last linear layer
        lstm_out = self.out_dropout(lstm_out)
        output = self.output(lstm_out).permute(0, 2, 1)
        return output
