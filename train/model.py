from dataset import DataHandler

import torch
import torch.nn as nn
import torch.nn.functional as F

import typing
import chess

"""
    GeLU activation Function

"""


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
        
    
    def forward(self, inputs):

        if not isinstance(inputs, torch.Tensor):
            raise ValueError(f"Invalid Type Final Layer {type(inputs)} expected {type(torch.Tensor)}")
        
        if inputs.size(0) != self.input_size:
            raise ValueError(f"inputs Board Tensor Improper Size: \n Received Size: {inputs.size(0)} \n Expected Size: {self.input_size}")
        
        embedding = self.linear(inputs)
        embedding = self.activ(embedding)
        embedding = self.layer_norm(embedding)
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
    
    
    def forward(self, inputs):

        if not isinstance(inputs, torch.Tensor):
            raise ValueError(f"Invalid Type Final Layer {type(inputs)} expected {type(torch.Tensor)}")

        if inputs.size(0) != self.model_width:
            raise ValueError(f"Final Tensor Improper Size: \n Received Size: {inputs.size(0)} \n Expected Size: {self.input_size}")
        
        embedding = self.linear(inputs)
        embedding = F.log_softmax(embedding)

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
    
    def forward(self, inputs):
        if not isinstance(inputs, torch.Tensor):
            raise ValueError(f"Invalid Type Final Layer {type(inputs)} expected {type(torch.Tensor)}")

        if inputs.size(0) != self.model_width:
            raise ValueError(f"Layer Tensor Improper Size: \n Received Size: {inputs.size(0)} \n Expected Size: {self.input_size}")
        
        embedding = self.linear(inputs)
        embedding = self.activ(embedding)
        embedding = embedding + inputs # Applies Residual Connection
        embedding = self.layer_norm(embedding)
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
    def __init__(self, model_width: int = 1000, model_depth: int = 200, dropout_rate: float = .3):
        self.model = ChessArch(model_width=model_width, model_depth=model_depth, dropout_rate=dropout_rate)
        self.optim = torch.optim.Adam(self.model.parameters())
        self.handler = DataHandler()

    def test_model(self):
        """
            Simply Tests Created to Make sure no Errors Occur
        """
        print("::Model Testing::")
        start_board = chess.Board()
        input_embedding = self.handler.board_to_tensor(start_board)

        output_embedding = self.model(input_embedding)

        assert output_embedding.shape[0] == 3

        #print(f"Input Embedding: {input_embedding}")
        print(f"Output Embedding: {output_embedding}")
        print("\n\nModel Passed Test\n\n")
        

        
"""
match type(inputs):
            case chess.Board:
                embedding = self.data_handler.board_to_tensor(inputs)
            case torch.Tensor:
                embedding = inputs
            case _:
                raise ValueError(f"Invalid Inputted Type Please Give {type(chess.Board)} or type {torch.Tensor}")
"""

ChessModel().test_model()