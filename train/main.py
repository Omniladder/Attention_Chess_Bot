from dataset import DataHandler
from model import ChessEvalModel

handler = DataHandler()
df = handler.get_tensorset()

'''`
df = handler.get_lichess_dataset("../data/Small_chess_data.csv")
df = handler.dataset_to_tensorset(df)
print(df)

df.to_csv("../data/Small_tensor_data.csv")
'''


