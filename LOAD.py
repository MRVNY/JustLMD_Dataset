from GLOBAL import *
from LMD_Dataset import LMD_Dataset

from multiprocessing import freeze_support
from torch.utils.data import DataLoader
import random
    
if __name__ == '__main__':
    freeze_support()
    
    dataset = LMD_Dataset('./')
    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=1)

    dataiter = iter(dataloader)
    data = next(dataiter)
    
    print(data['lyrics'].size(), data['music'].size(), data['dance'].size())
    
    seq = random.choice(list(dataset.indexing.values()))
    dataset.visualize(seq)
    # dataset.export(seq)
