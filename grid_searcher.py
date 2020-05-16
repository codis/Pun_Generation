from grid_variables import grid_params
import random
from pungen import Pungen

class GridSearcher:
    def __init__(self, grid_params):
        self.grid_params = grid_params

    def get_model_config(self):
        lstm = random.choice(self.grid_params['lstm'])
        merge_layer = random.choice(self.grid_params['merge_layer'])
        dense_size = random.choice(self.grid_params['dense']['size'])
        dense_dropout = random.choice(self.grid_params['dense']['dropout'])
        dense_act = random.choice(self.grid_params['dense']['act'])
        optimizer = random.choice(self.grid_params['optimizer'])
        learning_rate = random.choice(self.grid_params['lr'])

        grid_dict = {
            'lstm':lstm,
            'dense': {
                'size':dense_size,
                'dropout': dense_dropout,
                'act': dense_act
            },
            'merge_layer': merge_layer,
            'optimizer': optimizer,
            'lr': learning_rate
        }
        return grid_dict

    def start_grid_search(self, num_models, bs, emb_dim):
        pungen = Pungen(filepath='all.txt', batch_size=bs, max_len=50,
                        emb_dim=emb_dim, max_words=40000, split=0.15)

        for i in range(num_models):
            model_params = gs.get_model_config()
            pungen.create_model(model_params=model_params)
            pungen.train(epochs=20)

if __name__ == '__main__':
    gs = GridSearcher(grid_params)
    gs.start_grid_search(num_models=50, bs=1024, emb_dim=300)