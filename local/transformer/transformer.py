'''
This file contains the implemantation of data training for a given setting of the task.
It outputs the test accuracy at the end.
This is modified from steps/pytorchnn/model.py. (fs2776)
The original file contains the encoder and decoder. This model has an encoder, and the decoder is just a linear layer.
'''
import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout_pe, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout_pe)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, ninp, nout, nembed, nhead, nhid, nlayers, dropout=0.3, dropout_pe=0.1, 
                 activation="relu"):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except ImportError:
            raise ImportError('TransformerEncoder module does not exist in '
                              'PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nembed, dropout_pe)
        encoder_layers = TransformerEncoderLayer(nembed, nhead, nhid, dropout,
                                                 activation, batch_first = True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Linear(ninp, nembed)
        self.ninp = ninp
        self.decoder = nn.Linear(nembed, nout)
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
                mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        self.src_mask = None
        
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)

        # get only the [CLS] node
        output = output[:,0,:]
        output = self.decoder(output)
        
        # return only the [CLS] node
        return output
