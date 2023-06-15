from GLOBAL import *
from LMD_Dataset import LMD_Dataset

from multiprocessing import freeze_support
from torch.utils.data import DataLoader
import librosa
import random

from transformers import BertTokenizer, BertModel

def load_music(audio_path, start):
    audio = librosa.core.load(audio_path, sr=sr, offset=start, duration=sequenceLength)[0]
    
    # Extract features
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, hop_length=601, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_db_norm = (mel_spec_db - np.mean(mel_spec_db)) / np.std(mel_spec_db)
    audio_features = torch.from_numpy(mel_spec_db_norm).T
    return audio_features

def load_dance(full_dance, timestamp):
    dance = []
    start = toSeconds(timestamp)*fps
    
    for offset in range(sequenceLength*fps):
        stamp = str(int(start + offset)).zfill(6)
        dance.append(full_dance[stamp]['annots'][0]['poses'][0])
    
    dance = torch.from_numpy(np.array(dance))
    return dance

def load_lyrics(lyrics, tokenizer, model):
    tokens = tokenizer.encode_plus(lyrics, add_special_tokens=True, return_tensors='pt')
    outputs = model(**tokens)
    # get the cls token
    lyrics_embeddings = outputs[0][:,0,:]
    lyrics_embeddings = outputs.last_hidden_state[0].detach()
    
    lyrics_embeddings = torch.nn.functional.pad(lyrics_embeddings, pad=(0,0,0,lyrics_padding - lyrics_embeddings.size(0)), mode='constant', value=0)
    return lyrics_embeddings

def init_dataset (songs_collection):
    # tokenizer for lyrics
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    LMD_Dict = {}
    indexing = {}
    index = 0
    
    for year_dir in songs_collection:
        for song in os.listdir(year_dir):
            print(song)
            song_path = year_dir + song
            
            if song[0] in ['.','_'] or not os.path.isdir(song_path):
                continue
            
            # Load sliced lyrics as timestamps for cutting
            sliced = json.load(open(song_path + '/sliced.json', 'r'))            
            start = list(sliced.keys())[0]
            todo = list(sliced.keys())
            del todo[-1] # To avoid the last sequence being shorter than 6s
            
            full_dance = json.load(open('%s/smplfull.json'%song_path, 'r'))

            for timestamp in todo:
                # trimed_timestamps = timestamp - start
                trimed_timestamp = toTimestamp(toSeconds(timestamp)-toSeconds(start))
                seconds = toSeconds(timestamp)
                tag = str(int(seconds))
                
                # LAD Dict
                data = {'lyrics':load_lyrics(sliced[timestamp], tokenizer, model), \
                    'music':load_music('%s/audio.wav'%song_path, seconds), \
                    'dance':load_dance(full_dance, trimed_timestamp)}
                
                LMD_Dict[song+"_"+tag] = data
                indexing[index] = song+"_"+tag
                index += 1

    with open("indexing.json", "w", encoding="utf-8") as json_file:
        json.dump(indexing, json_file, ensure_ascii=False, indent=4)
        
    return LMD_Dict
    
if __name__ == '__main__':
    freeze_support()
    
    if not os.path.exists('JD20-22_LMD_Dict.pth'):
        LMD_Dict = init_dataset(songs_collection)
        torch.save(LMD_Dict, 'JD20-22_LMD_Dict.pth')
    else:
        LMD_Dict = torch.load('JD20-22_LMD_Dict.pth')
    
    indexing = json.load(open("indexing.json", 'r'))
    
    dataset = LMD_Dataset(LMD_Dict, indexing)

    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=1)

    dataiter = iter(dataloader)
    data = next(dataiter)
    
    print(data['lyrics'].size(), data['music'].size(), data['dance'].size())
    
    seq = random.choice(list(dataset.indexing.values()))
    dataset.visualize(seq)
