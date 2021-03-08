import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, drop_prob=0.5):
        super(DecoderRNN, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.drop_prob = drop_prob
        
        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.lstm = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=self.drop_prob, bias=True, batch_first=True)
        self.dropout = nn.Dropout(self.drop_prob)
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)
    
    def forward(self, features, captions):
        
        embedding = self.embed(captions[:,:-1])
        embedding = torch.cat((features.unsqueeze(1), embedding), 1)
        x, _ = self.lstm(embedding)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        result = []
        for i in range(max_len):
            
            out, states = self.lstm(inputs, states)
            output = self.fc(out)
            pred = torch.argmax(output, dim=-1)
            result.append(pred[0,0].item())
            inputs = self.embed(pred)

        return result
    

class DecoderRNN_cell(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, drop_prob=0.5):
        super(DecoderRNN_cell, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.drop_prob = drop_prob
        
        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.lstm_cell = nn.LSTMCell(input_size=self.embed_size, hidden_size=self.hidden_size, bias=True)
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)
        self.dropout = nn.Dropout(self.drop_prob)
    
    def forward(self, features, captions):
        
        batch_size = features.size(0)
        num_features = features.size(1)
        seq_length = captions.size(1)
        device = features.get_device()
        
        hidden_state = torch.zeros((batch_size, self.hidden_size)).to(device)
        cell_state = torch.zeros((batch_size, self.hidden_size)).to(device)
        preds = torch.zeros(batch_size, seq_length, self.vocab_size).to(device)
        
        print(captions.shape, captions.type())
        
        embedding = self.embed(captions)
        result = []
        
        for s in range(seq_length):
            if s == 0:
                hidden_state, cell_state = self.lstm_cell(features, (hidden_state, cell_state))
            else:
                hidden_state, cell_state = self.lstm_cell(embedding[:, s, :], (hidden_state, cell_state))
            
            output = self.fc(self.dropout(hidden_state))
            preds[:, s, :] = output
        
        return preds

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        result = []
        batch_size = inputs.size(0)
        device = inputs.get_device()
        
        hidden_state = torch.zeros((batch_size, self.hidden_size)).to(device)
        cell_state = torch.zeros((batch_size, self.hidden_size)).to(device)
        preds = torch.zeros(batch_size, max_len, self.vocab_size).to(device)
        captions = torch.zeros(batch_size, max_len).long().to(device)

        for s in range(max_len):
            if s == 0:
                hidden_state, cell_state = self.lstm_cell(inputs, (hidden_state, cell_state))
            else:
                hidden_state, cell_state = self.lstm_cell(embedding[:, s, :], (hidden_state, cell_state))
            
            output = self.fc(self.dropout(hidden_state))
            preds[:, s, :] = output
            
            word = torch.argmax(output[0]).item()
            captions[0, s] = word
            embedding = self.embed(captions)
        
        return captions[0].tolist()