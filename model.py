import math
import torch
import torch.nn as nn
import torch.optim
import torchmetrics
from transformers import RobertaModel, AutoModelForCausalLM
from torch.nn import init

class RoBERTa_Classify(nn.Module):
    
    def __init__(self, roberta_name, classnum, dropout_rate = 0.1):

        super(RoBERTa_Classify, self).__init__()
        self.roberta = RobertaModel.from_pretrained(roberta_name)
        self.classnum = classnum

        for param in self.roberta.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(dropout_rate)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.outdense = nn.Linear(1024, classnum) 
        self.relu = nn.ReLU()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.INFO = roberta_name + " + AvgPooling"

    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        
        token_input = input[0]
        token_input = token_input.to(self.device)
        
        mask_input = input[1]
        mask_input = mask_input.to(self.device)
        
        embedding = self.roberta(token_input, attention_mask = mask_input, encoder_hidden_states=False)['last_hidden_state']
        embedding = torch.transpose(embedding, 1, 2)
        embedding = self.pool(embedding).view(embedding.shape[0], embedding.shape[1])
        final_feature = self.dropout(embedding)
        final_feature = self.relu(final_feature)
        output = self.outdense(final_feature)
        return output

    def compile(self, optimizer, loss):

#        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr = lr, weight_decay = weight_decay)
#        self.loss_func = nn.CrossEntropyLoss(label_smoothing = label_smoothing)
        self.optimizer = optimizer
        self.loss_func = loss
        self.metric_func = torchmetrics.Accuracy(
            task='multiclass', 
            threshold=1/self.classnum, 
            num_classes=self.classnum, 
            average="micro",
            top_k = 1
        ).to(self.device)
        self.F1 = torchmetrics.F1Score(
            task='multiclass', 
            threshold=1/self.classnum, 
            num_classes=self.classnum, 
            average="macro",
            top_k = 1
        ).to(self.device)
        self.metric_name = "accuracy"

        
class LLaMa_Classify(nn.Module):
    
    def __init__(self, llama_name, classnum, dropout_rate = 0.1):

        super(LLaMa_Classify, self).__init__()
        self.llama = AutoModelForCausalLM.from_pretrained(llama_name, token = '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')
        self.classnum = classnum

        for param in self.llama.parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(dropout_rate)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.outdense = nn.Linear(4096, classnum) 
        self.relu = nn.ReLU()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        
        token_input = input[0]
        token_input = token_input.to(self.device)
        
        mask_input = input[1]
        mask_input = mask_input.to(self.device)
        
        embedding = self.llama(token_input, attention_mask = mask_input, output_hidden_states=True)['hidden_states'][-1]
        embedding = torch.transpose(embedding, 1, 2)
        embedding = self.pool(embedding).view(embedding.shape[0], embedding.shape[1])
        final_feature = self.dropout(embedding)
        final_feature = self.relu(final_feature)
        output = self.outdense(final_feature)
        return output

    def compile(self, optimizer, loss):

#        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr = lr, weight_decay = weight_decay)
#        self.loss_func = nn.CrossEntropyLoss(label_smoothing = label_smoothing)
        self.optimizer = optimizer
        self.loss_func = loss
        self.metric_func = torchmetrics.Accuracy(
            task='multiclass', 
            threshold=1/self.classnum, 
            num_classes=self.classnum, 
            average="micro",
            top_k = 1
        ).to(self.device)
        self.F1 = torchmetrics.F1Score(
            task='multiclass', 
            threshold=1/self.classnum, 
            num_classes=self.classnum, 
            average="macro",
            top_k = 1
        ).to(self.device)
        self.metric_name = "accuracy"
