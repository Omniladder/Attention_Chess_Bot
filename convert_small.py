from train.dataset import DataHandler
import torch

handler = DataHandler()

dataset = handler.get_gm_dataset(dataset_path="./data/GM_games_small.csv")

#dataset = dataset[:10000]

tensorset = handler.dataset_to_tensorset(dataset)

tensorset = handler.average_tensors(tensorset)

torch.save(tensorset.tensors, 'tensorset.pt')
    