from torch.utils.data import Dataset

#Lyrics_Music_Dance_Dataset
class LMD_Dataset(Dataset): 
    def __init__ (self, dict, indexing):
        self.LAD_Dict = dict
        self.indexing = indexing
        
    def __getitem__(self,index):
        key = self.indexing[str(index)]
        item = self.LAD_Dict[key]
        return item#['lyrics'], item['music'], item['dance']
    
    def __len__ (self):
        return len(self.indexing.keys())