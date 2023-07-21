# JustLMD Dataset

## Shape of each modality:
```
Lyrics: 180 x 128
Music: 180 x 128
Dance: 180 x 72
```

## To build the dataset
Be sure to install all the dependencies in `requirements.txt` first.
```
pip install -r requirements.txt
```

Execute in terminal as follows:
```shell
python LOAD.py
```
After the first time being executed, `LMD_Dict.pth` and `indenxing.json` will be generated.

## To use the dataset
Pass the directory where `LMD_Dict.pth` and `indenxing.json`, was well as the list of directories of raw data to `LMD_Dataset` when initializing the class, if the directoy doesn't exist, the class will build the dataset automatically.
```python
dataset = LMD_Dataset('./', ['./Songs_2020/', './Songs_2021/', './Songs_2022/'])
```

## To change the embeddings of lyrics and audio
In `load_lyrics` and `load_audio` in the `LMD_Dataset` class, customize the tokenizer and the audio feature extractor as you want.
The default tokenizer is bert and the default audio feature extractor is librosa.

## To visualize a sequence
To quickly visualize the animations of a sequence without audio or lyrics
```python
dataset.visualize(sequence_name)
```

## To export the video of a query
To view the video of the animation with audio and lyrics
```python
dataset.export(sequence_name)
```

## To export as .bvh
To convert the smpl animation of a sequence to .bvh and export"
```python 
dataset.toBvh(sequence_name)
```
```