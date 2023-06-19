from GLOBAL import *

from torch.utils.data import Dataset

from nosmpl.smpl_onnx import SMPLOnnxRuntime
from nosmpl.vis.vis_o3d import Open3DVisualizer

import numpy as np
import torch
import os

from transformers import BertTokenizer, BertModel
import librosa

#Lyrics_Music_Dance_Dataset
class LMD_Dataset(Dataset): 
    def __init__ (self, data_dir):
        if(os.path.exists(data_dir+'LMD_Dict.pth') and os.path.exists(data_dir+'indexing.json')):
            self.LMD_Dict = torch.load('LMD_Dict.pth')
            self.indexing = json.load(open("indexing.json", 'r'))
        else: 
            print("Creating new dataset")
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
                        data = {'lyrics':self.load_lyrics(sliced[timestamp], tokenizer, model), \
                            'music':self.load_music('%s/audio.wav'%song_path, seconds), \
                            'dance':self.load_dance(full_dance, trimed_timestamp)}
                        
                        LMD_Dict[song+"_"+tag] = data
                        indexing[index] = song+"_"+tag
                        index += 1

            with open("indexing.json", "w", encoding="utf-8") as json_file:
                json.dump(indexing, json_file, ensure_ascii=False, indent=4)
                
            torch.save(LMD_Dict, "LMD_Dict.pt")

    def load_music(self, audio_path, start):
        audio = librosa.core.load(audio_path, sr=sr, offset=start, duration=sequenceLength)[0]
        
        # Extract features
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, hop_length=601, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_db_norm = (mel_spec_db - np.mean(mel_spec_db)) / np.std(mel_spec_db)
        audio_features = torch.from_numpy(mel_spec_db_norm).T
        return audio_features

    def load_dance(self, full_dance, timestamp):
        dance = []
        start = toSeconds(timestamp)*fps
        
        for offset in range(sequenceLength*fps):
            stamp = str(int(start + offset)).zfill(6)
            dance.append(full_dance[stamp]['annots'][0]['poses'][0])
        
        dance = torch.from_numpy(np.array(dance))
        return dance

    def load_lyrics(self, lyrics, tokenizer, model):
        tokens = tokenizer.encode_plus(lyrics, add_special_tokens=True, return_tensors='pt')
        outputs = model(**tokens)
        # get the cls token
        lyrics_embeddings = outputs[0][:,0,:]
        lyrics_embeddings = outputs.last_hidden_state[0].detach()
        
        lyrics_embeddings = torch.nn.functional.pad(lyrics_embeddings, pad=(0,0,0,lyrics_padding - lyrics_embeddings.size(0)), mode='constant', value=0)
        return lyrics_embeddings

    def __getitem__(self,index):
        key = self.indexing[str(index)]
        item = self.LMD_Dict[key]
        return item#['lyrics'], item['music'], item['dance']
    
    def __len__ (self):
        return len(self.indexing.keys())
    
    def visualize(self,seq_name):
        seq = self.LMD_Dict[seq_name]
        
        if seq != None:
            poses = seq['dance']
        else:
            print("Sequence does not exist")
            return

        smpl = SMPLOnnxRuntime()
        o3d_vis = Open3DVisualizer(fps=30, enable_axis=False)

        poses = poses.reshape(180,24,3)
        poses = torch.index_select(poses, dim=1, index=torch.arange(0, poses.size(1)-1))
        poses = poses.detach().numpy().astype(np.float32)

        global_orient = [[[0,0,0]]]
        trans = [0,0,0]

        for body in poses:
            data = smpl.forward(body[None], global_orient)

            [vertices, joints, faces] = data
            vertices = vertices[0].squeeze()
            joints = joints[0].squeeze()
            faces = faces.astype(np.int32)

            o3d_vis.update(vertices, faces, trans, R_along_axis=[0, 0, 0], waitKey=1)

        o3d_vis.release()
        
    def export(self, seq_name):
        seq = self.LMD_Dict[seq_name]
        
        if seq != None:
            poses = seq['dance']
        else:
            print("Sequence does not exist")
            return

        save_dir = 'Previews/'+seq_name
        smpl = SMPLOnnxRuntime()
        o3d_vis = Open3DVisualizer(fps=30, save_img_folder=save_dir, enable_axis=True)

        poses = poses.reshape(180,24,3)
        poses = torch.index_select(poses, dim=1, index=torch.arange(0, poses.size(1)-1))
        poses = poses.detach().numpy().astype(np.float32)

        global_orient = [[[0,0,0]]]
        trans = [0,0,0]

        for body in poses:
            data = smpl.forward(body[None], global_orient)

            [vertices, joints, faces] = data
            vertices = vertices[0].squeeze()
            joints = joints[0].squeeze()
            faces = faces.astype(np.int32)

            o3d_vis.update(vertices, faces, trans, R_along_axis=[0, 0, 0], waitKey=1)

        o3d_vis.release()
        
        # Load audio and lyrics
        [song, tag] = seq_name.split("_")
        for year_dir in songs_collection:
            if os.path.isdir(year_dir + "/" + song):
                song_path = year_dir + "/" + song
                slicecd = json.load(open(song_path + "/sliced.json", "r"))
                for timestamp in slicecd:
                    if str(int(toSeconds(timestamp))) == tag:
                        lyrics = slicecd[timestamp]
                        os.system('ffmpeg -i %s/audio.wav -ss "%s" -t 00:06 -c copy %s/audio.wav'%(song_path, timestamp, save_dir))
        
        # Merge Video audio and lyrics
        os.system('ffmpeg -framerate 30 -i ' + save_dir + '/temp_%04d.png -c:v libx264 -r 30 -pix_fmt yuv420p ' + save_dir + '/mesh.mp4')
        os.system('ffmpeg -i %s/mesh.mp4 -i %s/audio.wav -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 %s/tmp.mp4'%(save_dir, save_dir, save_dir))
        os.system("ffmpeg -i %s/tmp.mp4 -vf \"drawtext=fontfile=Roboto-Regular.ttf:text='%s':fontsize=30:x=(w-tw)/2:y=h-th-10:fontcolor=black\" -codec:a copy %s/%s.mp4"%(save_dir, lyrics, save_dir, seq_name))
        
        os.system('rm %s/*.png'%save_dir)
        os.system('rm %s/tmp.mp4'%save_dir)
        os.system('rm %s/mesh.mp4'%save_dir)
        os.system('rm %s/audio.wav'%save_dir)