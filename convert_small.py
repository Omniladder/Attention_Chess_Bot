import torch
from train.bigger_model import EnhancedChessArch

# Assuming you have a model class defined as `YourModel`
model = EnhancedChessArch(
            model_width=512, 
            model_depth=8, 
            num_heads=4,
            dropout_rate=.2
        ) # Instantiate the model

# Load the model state_dict from the .pt file
model.load_state_dict(torch.load('./models/enhanced_chess_model_best.pth'), strict=False)

# Print all the model parameters (weights and biases)
for name, param in model.named_parameters():
    print(f"Layer: {name}, Shape: {param.shape}, Weights: {param.data}")