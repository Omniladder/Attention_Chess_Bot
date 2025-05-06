"""
    Enhanced Chess Model with improved architecture
"""

import os
from collections import defaultdict
from typing import Dict, List, Tuple, Union, Optional
from io import StringIO

import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import chess.pgn
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from torch.utils.data import random_split, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

from dataset import DataHandler

class InitBlock(nn.Module):
    def __init__(self, model_width: int, dropout_rate: float = 0.3):
        super().__init__()
        self.input_size = 837
        self.model_width = model_width

        # 1) Pre-LayerNorm
        self.input_norm = nn.LayerNorm(self.input_size)

        # 2) MLP layers
        self.linear1 = nn.Linear(self.input_size, model_width * 2)
        self.linear2 = nn.Linear(model_width * 2, model_width)

        # 3) Skip-projection
        self.skip_proj = nn.Linear(self.input_size, model_width)

        self.activ = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

        # 4) Initialization
        nn.init.kaiming_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_normal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)
        nn.init.kaiming_normal_(self.skip_proj.weight)
        nn.init.zeros_(self.skip_proj.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not isinstance(inputs, torch.Tensor):
            raise ValueError(f"Expected Tensor but got {type(inputs)}")
        if inputs.size(1) != self.input_size:
            raise ValueError(
                f"Expected input size {self.input_size}, got {inputs.size(1)}"
            )

        # Pre-normalize
        y = self.input_norm(inputs)

        # MLP expansion → projection
        y = self.linear1(y)
        y = self.activ(y)
        y = self.dropout(y)
        y = self.linear2(y)

        # Skip + dropout
        res = self.skip_proj(inputs)
        out = res + y
        out = self.dropout(out)
        return out


class FeedForwardBlock(nn.Module):
    def __init__(self, model_width: int, expansion_factor: int = 4, dropout_rate: float = 0.3):
        super(FeedForwardBlock, self).__init__()
        self.model_width = model_width
        self.hidden_dim = model_width * expansion_factor
        
        self.linear1 = nn.Linear(model_width, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, model_width)
        
        self.activ = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(model_width)
        
        # Initialize weights
        nn.init.kaiming_normal_(self.linear1.weight)
        nn.init.kaiming_normal_(self.linear2.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        # Feed-forward network
        x = self.linear1(x)
        x = self.activ(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        # Residual connection and layer normalization
        x = x + residual
        
        return x


class EnhancedHiddenBlock(nn.Module):
    def __init__(self, model_width: int, num_heads: int = 4, dropout_rate: float = 0.3):
        super(EnhancedHiddenBlock, self).__init__()
        self.model_width = model_width
        self.attention = nn.MultiheadAttention(model_width, num_heads, dropout=dropout_rate)
        self.layer_norm = nn.LayerNorm(model_width)
        self.dropout = nn.Dropout(dropout_rate)
        self.feed_forward = FeedForwardBlock(model_width, dropout_rate=dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layer_norm(x)
        # Apply attention mechanism
        attn_out, _ = self.attention(y, y, y)
        attn_out = self.dropout(attn_out)

        x = x + attn_out
        y = self.layer_norm(x)
        # Apply feed-forward network
        ff_out = self.feed_forward(y)
        x = x + self.dropout(ff_out)

        return x


class FinalBlock(nn.Module):
    def __init__(self, model_width: int, dropout_rate: float = 0.3):
        super().__init__()
        self.output_size = 3

        # 1) Pre-LayerNorm
        self.input_norm = nn.LayerNorm(model_width)

        # 2) Two-layer MLP head
        self.linear1 = nn.Linear(model_width, model_width // 2)
        self.activ   = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(model_width // 2, self.output_size)

        # 3) Initialize weights + zero biases
        nn.init.kaiming_normal_(self.linear1.weight, nonlinearity='relu')
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_normal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # sanity checks
        if x.dim() != 2 or x.size(1) != self.input_norm.normalized_shape[0]:
            raise ValueError(f"Expected input shape [batch, {self.input_norm.normalized_shape[0]}], got {tuple(x.shape)}")

        # Pre-norm → MLP → dropout → logits
        x = self.input_norm(x)
        x = self.linear1(x)
        x = self.activ(x)
        x = self.dropout(x)
        logits = self.linear2(x)
        embedding = F.log_softmax(logits, dim=1)
        
        return embedding


class EnhancedChessArch(nn.Module):
    """
    Enhanced Chess Evaluation Model Architecture with attention mechanism
    """
    def __init__(self, model_width: int, model_depth: int, num_heads: int = 4, dropout_rate: float = 0.3):
        super(EnhancedChessArch, self).__init__()
        self.model_width = model_width
        self.model_depth = model_depth
        self.data_handler = DataHandler()
        
        # Initial embedding
        self.init_layer = InitBlock(model_width, dropout_rate)
        
        # Hidden layers with enhanced blocks
        self.hidden_layers = nn.ModuleList()
        for _ in range(model_depth):
            self.hidden_layers.append(EnhancedHiddenBlock(model_width, num_heads, dropout_rate))
        
        # Output layer
        self.final_layer = FinalBlock(model_width, dropout_rate)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not isinstance(inputs, torch.Tensor):
            raise ValueError(f"Invalid Type Final Layer {type(inputs)} expected {type(torch.Tensor)}")
        
        # Initial embedding
        embedding = self.init_layer(inputs)
        
        # Add batch dimension for attention
        if len(embedding.shape) == 2:
            embedding = embedding.unsqueeze(1)
        
        # Process through hidden layers
        for layer in self.hidden_layers:
            embedding = layer(embedding)
        
        # Remove added dimension
        if len(embedding.shape) == 3:
            embedding = embedding.squeeze(1)
        
        # Final projection
        output = self.final_layer(embedding)
        
        return output


class EnhancedChessModel:
    """
    Enhanced Chess Evaluation Engine Interface with position averaging
    """
    def __init__(
        self, 
        lr: float = 0.001,  
        model_width: int = 256, 
        model_depth: int = 8, 
        num_heads: int = 4,
        dropout_rate: float = 0.2,
        weight_decay: float = 1e-5
    ):
        self.model = EnhancedChessArch(
            model_width=model_width, 
            model_depth=model_depth, 
            num_heads=num_heads,
            dropout_rate=dropout_rate
        )
        self.learning_rate = lr
        self.optim = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=weight_decay
        )
        self.handler = DataHandler()
        self.criterion = nn.KLDivLoss(reduction='batchmean') #Got better results with this
        self.scheduler = ReduceLROnPlateau(
            self.optim, 
            mode='min', 
            factor=0.5, 
            patience=3, 
            # verbose=True
        )
        
        # Device setup
        if torch.cuda.device_count() > 0:
            self.device = torch.device("cuda:0")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")
        
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = True
        
        self.model.to(self.device)
    
    def test_model(self):
        """
        Test the model to ensure it works with the expected input/output shapes
        """
        print("Enhanced Model Testing")
        start_board = chess.Board()
        input_embedding = self.handler.board_to_tensor(start_board)
        
        input_embedding = torch.unsqueeze(input_embedding, dim=0).to(self.device)
        
        with torch.no_grad():
            output_embedding = self.model(input_embedding)
        
        assert output_embedding.shape[1] == 3, f"Expected output of shape [batch_size, 3], got {output_embedding.shape}"
        
        print(f"Output Embedding: {output_embedding}")
        print("\n\nEnhanced Model Passed Test\n\n")
    
    def build_dataset(self, dataset_path: str) -> TensorDataset:
        """
        Build a dataset with position averaging from raw GM games
        """
        # Get the GM dataset
        print("Loading GM dataset...")
        gm_dataset = self.handler.get_gm_dataset(dataset_path)
        
        # Convert to tensor dataset
        print("Converting to tensor dataset...")
        tensorset = self.handler.dataset_to_tensorset(gm_dataset)
        
        return tensorset
    
    def save_dataset(self, dataset: TensorDataset, save_path: str) -> None:
        """Save the dataset"""
        torch.save(dataset, save_path)
        print(f"Dataset saved to {save_path}")
    
    def load_dataset(self, load_path: str) -> TensorDataset:
        """Load the dataset"""
        dataset = torch.load(load_path)
        print(f"Dataset loaded from {load_path}")
        return dataset
    
    def train(
        self, 
        dataset_path: str = "../data/GM_games.csv",
        tensorset_path: Optional[str] = None,
        save_tensorset_path: str = "../data/gm_tensorset.pt",
        train_test_split_ratio: float = 0.8,
        train_val_split_ratio: float = 0.8,
        num_epochs: int = 30, 
        batch_size: int = 256, 
        model_name: str = "enhanced_chess_model",
        early_stopping_patience: int = 5
    ):
        """
        Train the enhanced model
        """
        if tensorset_path and os.path.exists(tensorset_path):
            print(f"Loading existing tensorset from {tensorset_path}")
            dataset = torch.load(tensorset_path)
            #This is a tuple [0] is board [1] is winner
        else:
            print("Building dataset from GM games...")
            # Create dataset
            dataset = self.build_dataset(dataset_path)
            
            # Save the processed dataset for future use
            print(f"Saving tensorset to {save_tensorset_path}")
            torch.save(dataset, save_tensorset_path)
        
        dataset = TensorDataset(dataset[0], dataset[1])
        # Split dataset into train, validation, and test sets
        train_size = int(len(dataset) * train_test_split_ratio * train_val_split_ratio)
        val_size = int(len(dataset) * train_test_split_ratio) - train_size
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset , [train_size, val_size, test_size]
        )
        

        print(f"Train size: {train_size}, Validation size: {val_size}, Test size: {test_size}")
        
        save_path = f"../models/{model_name}.pth"
        best_model_path = f"../models/{model_name}_best.pth"
        graph_loss_path = f"../results/graphs/{model_name}_loss.png"
        graph_acc_path = f"../results/graphs/{model_name}_acc.png"
        data_path = f"../results/data/{model_name}.csv"
        




        # Create data loaders
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        
        best_val_loss = float('inf')
        early_stopping_counter = 0
        
        train_loss_series = []
        val_loss_series = []
        train_acc_series = []
        val_acc_series = []
        
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in tqdm(range(num_epochs), desc="Epochs"):
            # Training phase
            self.model.train()
            train_loss = 0
            train_acc = 0
            train_samples = 0

            

            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Backward and optimize
                self.optim.zero_grad()
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optim.step()
                
                # Calculate accuracy
                preds = torch.argmax(outputs, dim=1)
                target_classes = torch.argmax(targets, dim=1)
                correct = (preds == target_classes).sum().item()
                
                # Accumulate statistics
                train_loss += loss.item() * inputs.size(0)
                train_acc += correct
                train_samples += inputs.size(0)
            
            # Calculate average training statistics
            avg_train_loss = train_loss / train_samples
            avg_train_acc = train_acc / train_samples
            
            # Validation phase
            val_loss, val_acc = self._evaluate(val_dataloader)
            
            # Update learning rate scheduler
            self.scheduler.step(val_loss)
            
            # Record statistics
            train_loss_series.append(avg_train_loss)
            val_loss_series.append(val_loss)
            train_acc_series.append(avg_train_acc)
            val_acc_series.append(val_acc)
            
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Check for best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"  New best model! Saving to {best_model_path}")
                torch.save(self.model.state_dict(), best_model_path)
                early_stopping_counter = 0
            else:
                print("No Improvement Not Saving")
                ''' #Removing Early Stop I dont want it to stop
                early_stopping_counter += 1
                print(f"  No improvement. Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")
                
                if early_stopping_counter >= early_stopping_patience:
                    print("Early stopping triggered!")
                    break
                '''
        
        # Save final model
        torch.save(self.model.state_dict(), save_path)
        print(f"Final model saved to {save_path}")
        
        # Load best model for final evaluation
        self.model.load_state_dict(torch.load(best_model_path))
        
        # Final evaluation on test set
        test_loss, test_acc = self._evaluate(test_dataloader)
        print(f"Final Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        
        # Plot training curves
        self._plot_training_curves(
            train_loss_series, val_loss_series, 
            train_acc_series, val_acc_series,
            graph_loss_path, graph_acc_path
        )
        
        # Save training data
        self._save_training_data(
            train_loss_series, val_loss_series,
            train_acc_series, val_acc_series,
            test_loss, test_acc, data_path
        )
        
        return train_loss_series, val_loss_series, train_acc_series, val_acc_series, test_loss, test_acc
    
    def _evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate the model on a given dataset
        """
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Calculate accuracy
                preds = torch.argmax(outputs, dim=1)
                target_classes = torch.argmax(targets, dim=1)
                correct += (preds == target_classes).sum().item()
                
                # Accumulate statistics
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
        
        avg_loss = total_loss / total_samples
        accuracy = correct / total_samples
        
        return avg_loss, accuracy
    
    def _plot_training_curves(
        self, 
        train_loss: List[float], 
        val_loss: List[float],
        train_acc: List[float], 
        val_acc: List[float],
        loss_path: str, 
        acc_path: str
    ) -> None:
        """
        Plot and save training curves
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(loss_path), exist_ok=True)
        
        # Plot loss curves
        plt.figure(figsize=(10, 6))
        epochs = list(range(1, len(train_loss) + 1))
        
        plt.plot(epochs, train_loss, 'b-', label='Training Loss')
        plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
        
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(loss_path)
        plt.close()
        
        # Plot accuracy curves
        plt.figure(figsize=(10, 6))
        
        plt.plot(epochs, train_acc, 'b-', label='Training Accuracy')
        plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
        
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(acc_path)
        plt.close()
    
    def _save_training_data(
        self, 
        train_loss: List[float], 
        val_loss: List[float],
        train_acc: List[float], 
        val_acc: List[float],
        test_loss: float,
        test_acc: float,
        data_path: str
    ) -> None:
        """
        Save training data to CSV
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        
        # Create DataFrame
        epochs = list(range(1, len(train_loss) + 1))
        data = {
            'epoch': epochs,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc
        }
        
        df = pd.DataFrame(data)
        
        # Add test metrics
        df_test = pd.DataFrame({
            'metric': ['test_loss', 'test_acc'],
            'value': [test_loss, test_acc]
        })
        
        # Save to CSV
        df.to_csv(data_path, index=False)
        df_test.to_csv(data_path.replace('.csv', '_test.csv'), index=False)
    
    def predict(self, game_state: chess.Board) -> List[float]:
        """
        Predict the outcome of a given chess position
        """
        
        encoding = self.handler.board_to_tensor(game_state)
        encoding = torch.unsqueeze(encoding, dim=0).to(self.device)
        
        with torch.no_grad():
            prediction = np.exp(self.model(encoding).cpu()).tolist()[0]
        
        return prediction
    
    def load_model(self, model_path: str) -> None:
        """
        Load a trained model from file
        """
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded from {model_path}")
    
    def game_analysis(self, pgn_string: str) -> List[Dict[str, Union[str, List[float]]]]:
        """
        Analyze a chess game from PGN notation
        """
        game = chess.pgn.read_game(StringIO(pgn_string))
        board = game.board()
        
        analysis = []
        
        # Add initial position
        analysis.append({
            'move': 'Initial',
            'fen': board.fen(),
            'evaluation': self.predict(board)
        })
        
        # Process each move
        for move in game.mainline_moves():
            board.push(move)
            
            analysis.append({
                'move': board.san(move),
                'fen': board.fen(),
                'evaluation': self.predict(board)
            })
        
        return analysis


if __name__ == "__main__":
    # Configuration
    MODEL_WIDTH = 512       # Increased from original model
    MODEL_DEPTH = 16       # Increased from original model
    NUM_HEADS = 4           # New parameter for attention mechanism
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 256
    NUM_EPOCHS = 50
    DROPOUT_RATE = 0.05
    
    # Create and train enhanced model
    print("Initializing enhanced chess model...")
    model = EnhancedChessModel(
        lr=LEARNING_RATE,
        model_width=MODEL_WIDTH,
        model_depth=MODEL_DEPTH,
        num_heads=NUM_HEADS,
        dropout_rate=DROPOUT_RATE
    )
    
    # Test the model architecture
    model.test_model()
    
    # Training options
    TRAIN_NEW = True
    USE_EXISTING_TENSORSET = True
    
    if TRAIN_NEW:
        print("\nStarting training process...")
        tensorset_path = "../data/tensorset.pt" if USE_EXISTING_TENSORSET else None
        
        model.train(
            dataset_path="../data/GM_games.csv",
            tensorset_path=tensorset_path,
            save_tensorset_path="../data/tensorset.pt",
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            model_name="enhanced_chess_model"
        )
    else:
        # Load pre-trained model
        model.load_model("../models/enhanced_chess_model_best.pth")
    
    # Test the model on a few positions
    print("\nTesting model predictions on sample positions...")
    
    # Starting position
    start_board = chess.Board()
    print("Starting position prediction:", model.predict(start_board))
    
    # After e4
    e4_board = chess.Board()
    e4_board.push_san("e4")
    print("After e4 prediction:", model.predict(e4_board))
    
    # After e4, e5
    e5_board = chess.Board()
    e5_board.push_san("e4")
    e5_board.push_san("e5")
    print("After e4, e5 prediction:", model.predict(e5_board))
    
    print("\nEnhanced chess model training and evaluation complete!")