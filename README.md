# JustLMD Dataset

## Shape of each modality:
```
Lyrics: 180 x 768
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
After the first time being executed, `JD20-22_LMD_Dict.pth` and `indenxing.json` will be generated.

## To use the dataset
Load the .pth and .json and pass them into the `LMD_Dataset` class as follows:
```python
LMD_Dict = torch.load('JD20-22_LMD_Dict.pth')
indexing = json.load(open("indexing.json", 'r'))

LMD_Dataset(LMD_Dict, indexing)
```

## To change the embeddings of lyrics and audio
In `load_lyrics` and `load_audio` of `LOAD.py`, customize the tokenizer and the audio feature extractor as you want.
The default tokenizer is bert and the default audio feature extractor is librosa.

## To visualize a sequence
```python
dataset.visualize(sequence_name)
```