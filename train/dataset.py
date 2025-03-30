import pandas as pd
from typing import List
from enum import Enum
import os
import chess
import torch


class DataHandler:
    """
    Class designed for handling data and dataset
    """

    def __winner_to_tensor(self, winner: str) -> torch.Tensor:
        match winner:
            case "white":
                return torch.Tensor([1,0,0])
            case "draw":
                return torch.Tensor([0,1,0])
            case "black":
                return torch.Tensor([0,0,1])
            case _:
                raise ValueError(f"Unknown Winning Player: {winner}")
        return -1

    def get_dataset(
        self, dataset_path: str = "../data/Small_chess_data.csv"
    ) -> pd.DataFrame:
        """
        Function that gets and cleans Lichess Dataset for Usage
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Could Not Find File {dataset_path}")

        dataset = pd.read_csv(dataset_path)
        dataset = dataset.filter(["moves", "winner"])

        #dataset["winner"] = dataset["winner"].apply(self.__winner_to_tensor)

        return dataset

    def __anot_to_fen(self, game_moves: str, winner: torch.Tensor) -> pd.DataFrame:
        """
        Function to turn chess asymptomatic notation into list FEN Strings
        """

        move_str_array = game_moves.split(" ")
        game_state = chess.Board()

        predf = {"gameState": [], "winner": [winner] * len(move_str_array)}

        for move in move_str_array:
            game_state.push_san(move)
            predf["gameState"].append(game_state.fen())

        return pd.DataFrame(predf)

    def dataset_to_tensor(self, game_dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Turns the Lichess dataframe into a Fen Based one with all Boards and wins accordingly for easy integration
        """

        df = pd.DataFrame({"gameState": [], "winner": []})
        for data_index in range(len(game_dataset)):
            new_posistions = self.__anot_to_fen(
                game_dataset["moves"][data_index], self.__winner_to_tensor(game_dataset["winner"][data_index])
            )
            # df._append(new_posistions, ignore_index = True) #Using Illegal functions because the real ones are slow TF
            df.loc[len(df)] = new_posistions.iloc[0]
            if data_index % 100 == 0:
                print(f"Reached Index {data_index}")

        return df

    def board_to_tensor(self, game_board: chess.Board) -> torch.Tensor:
        """
        Takes a Python Chess Board and Convert to a One Hot encoded Vector Tensor
        """

        tensor = torch.zeros(837)

        tensor[0] = int(game_board.turn == chess.WHITE)
        tensor[1] = int(game_board.has_kingside_castling_rights(chess.WHITE))
        tensor[2] = int(game_board.has_queenside_castling_rights(chess.WHITE))
        tensor[3] = int(game_board.has_kingside_castling_rights(chess.BLACK))
        tensor[4] = int(game_board.has_queenside_castling_rights(chess.BLACK))

        if game_board.has_legal_en_passant():
            tensor[5 + int(game_board.en_passant)] = 1

        for square, piece in game_board.piece_map().items():
            tensor[
                69
                + int(square)
                + 64 * (int(piece.piece_type) - 1)
                + int(piece.piece_type == chess.BLACK) * 453
            ] = 1

        return tensor


'''
handler = DataHandler()
df = handler.get_dataset()
df = handler.dataset_to_tensor(df)
df.to_csv("../data/Small_fen_data.csv")
'''

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
    Total 837 Bytes
"""
