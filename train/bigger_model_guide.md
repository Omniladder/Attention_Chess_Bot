# Enhanced Chess Evaluation Model Guide

This guide explains how the enhanced chess evaluation model works and provides instructions for training and using it.

## How the Model Works

The enhanced chess model is designed to predict the outcome of a chess game based on a given position. It takes a chess position as input and outputs three probabilities:

1. White win probability
2. Draw probability
3. Black win probability

### Architecture Overview

The model uses a transformer-inspired architecture with the following components:

1. **Input Representation**: Chess positions are converted to 837-dimensional tensors using one-hot encoding for pieces, castling rights, etc.

2. **Initial Block**: Processes the input tensor through two linear layers with expansion, normalization, and activation.

3. **Enhanced Hidden Blocks**: A series of blocks that each contain:
   - **Self-Attention Mechanism**: Allows the model to focus on important parts of the board
   - **Feed-Forward Network**: Processes the attention outputs with expansion and projection

4. **Final Block**: Projects the processed features to three output probabilities using two linear layers.

5. **Position Averaging**: Maintains a database of positions and their averaged outcomes across multiple games.

### Key Improvements Over Baseline Model

- **Increased Model Size**: Width increased from 3 to 256 neurons, depth from 2 to 8 layers
- **Attention Mechanism**: Added multi-headed self-attention to capture board relationships
- **Position Averaging**: Combines outcomes when same position appears multiple times
- **Better Training**: Learning rate scheduling, early stopping, gradient clipping
- **Enhanced Initialization**: Better weight initialization for improved learning

## Walkthrough: How to Use the Model

### Prerequisites

Make sure you have the following dependencies installed:

```bash
pip install torch pandas numpy matplotlib tqdm chess
```

You also need to have the following files in your project structure:
- `dataset.py` (handles data processing)
- `model.py` (the original model, for reference)
- `bigger_model.py` (the enhanced model)

### Directory Structure

Ensure your project has the following directory structure:

```
project/
├── data/
│   ├── GM_games.csv       # Grand Master games dataset
│   └── gm_tensorset.pt    # (Optional) Pre-processed tensor dataset
├── models/                # Directory for saving trained models
├── results/
│   ├── data/              # Training statistics
│   └── graphs/            # Training curves
├── dataset.py
├── model.py
└── bigger_model.py
```

### Training the Model

1. **Prepare your data**:
   - Ensure your GM dataset is in the correct location (`../data/GM_games.csv`)
   - The CSV should have columns for moves and winner

2. **Configure the model parameters**:
   - Open `bigger_model.py` and adjust these parameters at the bottom of the file:
     ```python
     MODEL_WIDTH = 256      # Width of the model
     MODEL_DEPTH = 8        # Number of hidden layers
     NUM_HEADS = 4          # Number of attention heads
     LEARNING_RATE = 0.001  # Learning rate
     BATCH_SIZE = 256       # Batch size for training
     NUM_EPOCHS = 30        # Number of training epochs
     DROPOUT_RATE = 0.2     # Dropout rate for regularization
     ```

3. **Run the training**:
   ```bash
   python bigger_model.py
   ```
   
   If you want to use a pre-processed dataset (faster):
   - Set `USE_EXISTING_TENSORSET = True`
   - Make sure the tensorset is at `../data/gm_tensorset.pt`

   If you want to load a pre-trained model instead of training:
   - Set `TRAIN_NEW = False`
   - Ensure the model file exists at `../models/enhanced_chess_model_best.pth`

### Using the Trained Model

The trained model can be used to:

1. **Evaluate a single position**:
   ```python
   # Initialize the model and load weights
   model = EnhancedChessModel()
   model.load_model("../models/enhanced_chess_model_best.pth")
   
   # Create a chess position
   board = chess.Board()
   board.push_san("e4")  # Make a move
   
   # Get prediction
   prediction = model.predict(board)
   print(f"Position evaluation: {prediction}")
   # Output: [white_win_prob, draw_prob, black_win_prob]
   ```

2. **Analyze an entire game**:
   ```python
   # Load a PGN game
   pgn_string = """
   [Event "Example Game"]
   [Site "?"]
   [Date "2023.01.01"]
   [Round "1"]
   [White "Player A"]
   [Black "Player B"]
   [Result "1-0"]
   
   1. e4 e5 2. Nf3 Nc6 3. Bb5 a6
   """
   
   # Analyze the game
   analysis = model.game_analysis(pgn_string)
   
   # Print analysis
   for move_analysis in analysis:
       print(f"Move: {move_analysis['move']}")
       print(f"Evaluation: {move_analysis['evaluation']}")
       print()
   ```

### Understanding the Output

The model outputs three probabilities:
- Index 0: Probability of white winning
- Index 1: Probability of a draw
- Index 2: Probability of black winning

Example output: `[0.65, 0.25, 0.10]` means:
- 65% chance white wins
- 25% chance of draw
- 10% chance black wins

### Using Position Averaging

The model can use position averaging to improve predictions:

```python
# With averaging (default)
prediction = model.predict(board, use_averaging=True)

# Without averaging (use raw model output)
prediction = model.predict(board, use_averaging=False)
```

Position averaging combines statistics from all occurrences of the same position in the training data. This creates more consistent evaluations by aggregating outcomes across multiple games.

## Customizing the Model

### Changing the Architecture

To modify the model's architecture:

1. Adjust the dimensions:
   ```python
   model = EnhancedChessModel(
       model_width=512,     # Increase width
       model_depth=12,      # Increase depth
       num_heads=8          # Increase attention heads
   )
   ```

2. Modify dropout for regularization:
   ```python
   model = EnhancedChessModel(dropout_rate=0.3)
   ```

3. Adjust learning parameters:
   ```python
   model = EnhancedChessModel(
       lr=0.0005,            # Slower learning rate
       weight_decay=1e-5     # Stronger regularization
   )
   ```

### Training on Different Data

To train on a different dataset:

1. Make sure your dataset has the same format as the GM dataset
2. Adjust the path in the training call:
   ```python
   model.train(
       dataset_path="path/to/your/dataset.csv",
       # other parameters...
   )
   ```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce `BATCH_SIZE`
   - Reduce `MODEL_WIDTH` or `MODEL_DEPTH`

2. **Slow Training**:
   - Use a preprocessed tensorset (`USE_EXISTING_TENSORSET = True`)
   - Check if CUDA is being utilized

3. **Poor Performance**:
   - Try increasing `NUM_EPOCHS`
   - Adjust `LEARNING_RATE`
   - Check your dataset quality

### Tips for Better Results

1. **More Data**: Use larger GM datasets for better generalization
2. **Position Diversity**: Ensure dataset has diverse positions
3. **Quality Games**: Higher-rated games tend to provide better training signals
4. **Balanced Outcomes**: Try to have a balanced distribution of white wins, draws, and black wins
