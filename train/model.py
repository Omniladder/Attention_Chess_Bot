"""
    Model File Handle Model and Interface
"""

import typing
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


import chess
import pandas as pd
import numpy as np
from tqdm import tqdm

from torch.utils.data import random_split, DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from dataset import DataHandler


#TODO: Construct an Attention Layer
class InitBlock(nn.Module):
    def __init__(self, model_width: int, dropout_rate: int = .3):
        super(InitBlock, self).__init__()
        self.input_size = 837
        self.activ = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(self.input_size, model_width)
        self.layer_norm = nn.LayerNorm(model_width)
        self.model_width = model_width
        self.initializer = nn.init.xavier_normal_(self.linear.weight)
        
        
    
    def forward(self, inputs):

        if not isinstance(inputs, torch.Tensor):
            raise ValueError(f"Invalid Type Final Layer {type(inputs)} expected {type(torch.Tensor)}")
        #inputs = torch.squeeze(inputs)
        if inputs.size(1) != self.input_size:
            raise ValueError(f"inputs Board Tensor Improper Size: \n Received Size: {inputs.size(0)} \n Expected Size: {self.input_size}")
        
        embedding = self.linear(inputs)
        embedding = self.layer_norm(embedding)
        embedding = self.activ(embedding)
        embedding = self.dropout(embedding)

        return embedding
    

    #TODO: Construct an Attention Layer
class FinalBlock(nn.Module):
    def __init__(self, model_width: int):
        super(FinalBlock, self).__init__()
        self.output_size = 3
        self.linear = nn.Linear(model_width, self.output_size)
        self.layer_norm = nn.LayerNorm(model_width)
        self.model_width = model_width
        self.initializer = nn.init.xavier_normal_(self.linear.weight)
    
    
    def forward(self, inputs):

        if not isinstance(inputs, torch.Tensor):
            raise ValueError(f"Invalid Type Final Layer {type(inputs)} expected {type(torch.Tensor)}")

        if inputs.size(1) != self.model_width:
            raise ValueError(f"Final Tensor Improper Size: \n Received Size: {inputs.size(0)} \n Expected Size: {self.model_widthi}")
        
        embedding = self.linear(inputs)
        #print("Embedding " + str(embedding))
        embedding = F.softmax(embedding, dim=1)

        
        return embedding

class HiddenBlock(nn.Module):
    def __init__(self, model_width: int, dropout_rate: float = .3):
        super(HiddenBlock, self).__init__()
        self.model_width = model_width
        self.dropout_prob = dropout_rate
        self.linear = nn.Linear(model_width, model_width)
        self.layer_norm = nn.LayerNorm(model_width)
        self.activ = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.initializer = nn.init.xavier_normal_(self.linear.weight)

    def forward(self, inputs):
        if not isinstance(inputs, torch.Tensor):
            raise ValueError(f"Invalid Type Final Layer {type(inputs)} expected {type(torch.Tensor)}")

        if inputs.size(1) != self.model_width:
            raise ValueError(f"Layer Tensor Improper Size: \n Received Size: {inputs.size(0)} \n Expected Size: {self.model_width}")
        
        embedding = self.linear(inputs)
        embedding = self.layer_norm(embedding)
        embedding = self.activ(embedding)
        embedding = torch.add(embedding, inputs) # Applies Residual Connection
        embedding = self.dropout(embedding)

        return embedding

class ChessArch(nn.Module):
    """
        Underlying Chess Eval Model Architecture
    """
    def __init__(self, model_width: int, model_depth:int, dropout_rate: float = .3):
        super(ChessArch, self).__init__()
        self.model_width = model_width
        self.model_depth = model_depth
        self.data_handler = DataHandler()
        self.init_layer = InitBlock(model_width, dropout_rate)
        self.hidden_layers = nn.ModuleList()
        self.final_layer = FinalBlock(model_width)
        

        for _ in range(model_depth):
            self.hidden_layers.append(HiddenBlock(model_width, dropout_rate))

    def forward(self, inputs):
        if not isinstance(inputs, torch.Tensor):
            raise ValueError(f"Invalid Type Final Layer {type(inputs)} expected {type(torch.Tensor)}")
            
        embedding = self.init_layer(inputs)

        for layer in self.hidden_layers:
            embedding = layer(embedding)
        
        embedding = self.final_layer(embedding)

        return embedding


class ChessModel():
    """
        Chess Eval Engine Interface
    """
    def __init__(self, lr: float = .001, model_width: int = 120, model_depth: int = 5, dropout_rate: float = .3):
        self.model = ChessArch(model_width=model_width, model_depth=model_depth, dropout_rate=dropout_rate)
        self.learning_rate = lr
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.handler = DataHandler()
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = StepLR(self.optim, step_size=10, gamma=0.1)
        

        if torch.cuda.device_count() > 0:
            self.gpu = torch.device(0)
        else:
            self.gpu = torch.device("cpu")

        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = True


        self.model.to(self.gpu)

    def test_model(self):
        """
            Simply Tests Created to Make sure no Errors Occur
        """
        print("::Model Testing::")
        start_board = chess.Board()
        input_embedding = self.handler.board_to_tensor(start_board)

        input_embedding = torch.unsqueeze(input_embedding, dim=0)

        output_embedding = self.model(input_embedding)

        assert output_embedding.shape[1] == 3

        #print(f"Input Embedding: {input_embedding}")
        print(f"Output Embedding: {output_embedding}")
        print("\n\nModel Passed Test\n\n")
        
    def train(self, tensorset_path: str = "tensorset.pt", train_test_split: float = .8,  num_epochs: int = 10, batch_size: int = 32, model_name: str = "mlp"):

        dataset = torch.load(tensorset_path, weights_only=False)


        train_size = (int(len(dataset) * train_test_split))
        val_size = len(dataset) - train_size

        unused_size = len(dataset) - train_size - val_size

        train_dataset, dev_dataset, no_used = random_split(dataset, [train_size, val_size, unused_size])

        save_path = f"../models/{model_name}.pth"

        graph_loss_path = f"../results/graphs/{model_name}_loss.png"

        graph_acc_path = f"../results/graphs/{model_name}_acc.png"

        data_path = f"../results/data/{model_name}.csv"
#        train_dataset.to(self.gpu)
 #       dev_dataset.to(self.gpu)
        
        # Create DataLoaders from the combined datasets
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

        best_loss = np.inf

        train_loss_series = []
        dev_loss_series = []
        train_acc_series = []
        dev_acc_series = []

        for _ in tqdm(range(num_epochs), desc="Epochs Passed"):
            
            for batch in tqdm(train_dataloader, desc="Training Progress", leave=False):
                #print(batch[0].shape)
                inp, target = batch

                inp = inp.to(self.gpu)
                target = target.to(self.gpu)


                '''
                inp = inp.squeeze(0)
                target = target.squeeze(0)
                '''

                out = self.model(inp)
                loss = self.criterion(out, target)

                loss.backward()
                self.optim.step()
                self.optim.zero_grad()


            self.scheduler.step()
            train_loss, train_acc = self.__evaluate(train_dataloader)
            dev_loss, dev_acc = self.__evaluate(dev_dataloader)

            train_loss_series.append(train_loss)
            dev_loss_series.append(dev_loss)

            train_acc_series.append(train_acc)
            dev_acc_series.append(dev_acc)

            print(f"Average Train Loss: {train_loss} Average Dev Loss: {dev_loss}")

            

            if(dev_loss < best_loss):
                best_loss = dev_loss
                print("New Best Model ::Saving::")
                torch.save(self.model.state_dict(), save_path)
            print()
        plt.plot([i for i in range(len(train_loss_series))], train_loss_series, label='Training Loss', color='blue')
        plt.plot([i for i in range(len(dev_loss_series))], dev_loss_series, label='Dev Loss', color='red')
        plt.legend()

        plt.savefig(graph_loss_path)
        plt.clf()

        plt.plot([i for i in range(len(train_acc_series))], train_acc_series, label='Training Accuracy', color='blue')
        plt.plot([i for i in range(len(dev_loss_series))], dev_loss_series, label='Dev Accuracy', color='red')
        plt.legend()

        plt.savefig(graph_acc_path)

        training_data = {
            "train accuracies": train_acc_series,
            "dev accuracies": dev_acc_series,
            "train loss": train_loss_series,
            "dev loss": dev_loss_series
        }

        training_data = pd.DataFrame(training_data)

        training_data.to_csv(data_path)

        return train_loss_series, dev_loss_series, train_acc_series, dev_acc_series

    def __evaluate(self, dataloader: DataLoader) -> float:
        
        self.model.eval()

        loss = 0
        count = 0
        num_correct = 0
        with torch.no_grad():
            for _, batch in tqdm(enumerate(dataloader)):
                inp, target = batch
                inp = inp.to(self.gpu)
                target = target.to(self.gpu)
                log_probs = self.model(inp)
#                target = target.squeeze()
                loss += self.criterion(log_probs, target).item() * log_probs.size(0)
               # print("Max Arg : " + str(torch.argmax(target, dim=1)) + " , " + str(torch.argmax(log_probs, dim=1)))
                num_correct += (torch.argmax(target) == torch.argmax(log_probs, dim=1)).sum().item()
                count += target.size(0)

        
        return loss / count, num_correct / count

    def predict(self, game_state: chess.Board) -> List[float]:
        encoding = self.handler.board_to_tensor(game_state)
        encoding = torch.unsqueeze(encoding, dim=0)
        encoding = encoding.to(self.gpu)
        
        return self.model(encoding).tolist()[0]



#ChessModel(model_width = 3, model_depth=2).train(num_epochs=5, batch_size=128)
model = ChessModel(model_width = 3, model_depth=2)
#model.train()
#model.test_model()
print(model.predict(chess.Board()))

'''
    TODO: List
    Construct Loss Graphs
    Construct Accuracy Graphs
    Change file location to just take a name of model
    Increase Graph Visual Appeal
    Load Models from File
    Move outside train code to train
    Allow Model to Predict Board Posistions
    Playable Chessbot
'''
