"""
    Code relating to data handling such as making tensors and working with datasets
"""

import os

import pandas as pd
import chess
import torch
from typing import Union, List, Tuple

from tqdm import tqdm

from torch.utils.data import DataLoader, TensorDataset



"""
    ::TENSOR STRUCTURE::

    64 Spaces
    6 Pieces
    2 Colors
    King Side Castling (White & Black)
    Queen Side Castling (White & Black)
    64 spaces for en passent
    Player Move

    PM ,King Castle (W), Queen Castle (W), King Castle (B), Queen Castle (B), Enpassent Location 


    69 Index Offset // Player Move castling White and Black followed by En passent
    64 Spaces for each Piece 768 total spaces
    Total 837
"""




class DataHandler:
    """
    Class designed for handling data and dataset
    """

    def __winner_to_tensor(self, winner: str) -> torch.Tensor:
        """
            Simply takes the winner by string and gives back number
        """

        tensor = torch.zeros(3)

        match winner:
            case "white":
                tensor[0] = 1
                return tensor
            case "draw":
                tensor[1] = 1
                return tensor
            case "black":
                tensor[2] = 1
                return tensor
            case _:
                raise ValueError(f"Unknown Winning Player: {winner}")

    def get_lichess_dataset(
        self, dataset_path: str = "../data/Small_chess_data.csv", min_elo: int = 1000 
    ) -> pd.DataFrame:
        """
        Function that gets and cleans Lichess Dataset for Usage
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Could Not Find File {dataset_path}")

        dataset = pd.read_csv(dataset_path)

        dataset = dataset[(dataset['WhiteElo' >= min_elo]) & (dataset['BlackElo' >= min_elo])]

        dataset = dataset.filter(["moves", "winner"])

        dataset["winner"] = dataset["winner"].apply(self.__winner_to_tensor)

        return dataset
    
    def get_gm_dataset(
        self, dataset_path: str = "../data/GM_games.csv" 
    ) -> pd.DataFrame:
        """
        Function that gets a GM Dataset for Usage
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Could Not Find File {dataset_path}")

        dataset = pd.read_csv(dataset_path)

        dataset["winner"] = dataset["winner"].apply(self.__winner_to_tensor)

        return dataset

    def __anot_to_tensor(self, game_moves: str) -> List[torch.Tensor]:
        """
        Function to turn chess asymptomatic notation into list FEN Strings
        """
        move_str_array = game_moves.strip().split(" ")
        
        game_state = chess.Board()

        predf = []

        for move in move_str_array:
            game_state.push_san(move)
            predf.append(self.board_to_tensor(game_state))

        return predf

    def dataset_to_tensorset(self, game_dataset: pd.DataFrame) -> TensorDataset:
        """
        Turns the Lichess dataframe into a Fen Based one with all Boards and wins accordingly for easy integration
        """

        df = [[], []]
        for data_index in tqdm(range(len(game_dataset)), desc="Converting Dataset to Tensorset"):
            try:
                new_posistions = self.__anot_to_tensor(
                    game_dataset["moves"][data_index]
                )
                # df._append(new_posistions, ignore_index = True) #Using Illegal functions because the real ones are slow TF
                df[0].extend(new_posistions)
                df[1].extend([game_dataset["winner"][data_index]] * len(new_posistions))
            except:
                continue



        return TensorDataset(torch.stack(df[0]), torch.stack(df[1]))

    def board_to_tensor(self, game_board: chess.Board) -> torch.Tensor:
        """
        Takes a Python Chess Board and Convert to a One Hot encoded Vector Tensor
        """

        tensor = torch.zeros(837)

        tensor[0] = game_board.turn == chess.WHITE
        tensor[1] = game_board.has_kingside_castling_rights(chess.WHITE)
        tensor[2] = game_board.has_queenside_castling_rights(chess.WHITE)
        tensor[3] = game_board.has_kingside_castling_rights(chess.BLACK)
        tensor[4] = game_board.has_queenside_castling_rights(chess.BLACK)

        if game_board.has_legal_en_passant():
            tensor[5 + int(game_board.ep_square)] = 1

        for square, piece in game_board.piece_map().items():
            tensor[
                69
                + int(square)
                + 64 * (int(piece.piece_type) - 1)
                + int(piece.piece_type == chess.BLACK) * 453
            ] = 1

        return tensor

    def average_tensors(self, tensorset: TensorDataset) -> TensorDataset:

        inputs_data, output_data = tensorset.tensors

        x_to_sum = {}
        x_to_count = {}

        for inp, out in zip(inputs_data, output_data):
            key = tuple(inp.tolist())
            if key in x_to_sum:
                x_to_sum[key] += out
                x_to_count[key] += 1
            else:
                x_to_sum[key] = out.clone()
                x_to_count[key] = 1
        
        new_x = []
        new_y = []

        for key in x_to_sum:
            new_x.append(torch.tensor(key, dtype=inputs_data.dtype))
            new_y.append(x_to_sum[key] / x_to_count[key])
        
        return TensorDataset(torch.stack(new_x), torch.stack(new_y))

            