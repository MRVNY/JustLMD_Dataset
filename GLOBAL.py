import os
import json
import numpy as np
import torch
import datetime

path = './'

all_years = ['2020', '2021', '2022']
songs_collection = [path + 'Songs_2020/', path + 'Songs_2021/', path + 'Songs_2022/']

# lyrics embedding param
lyrics_padding = 180

# audio param
fps = 30
sr = 18000
sequenceLength = 6

# utils
def toSeconds(time_stamp):
    minutes, seconds = map(float, time_stamp.split(':'))
    return datetime.timedelta(minutes=minutes, seconds=seconds).total_seconds()

def toTimestamp(seconds): #format muniute:second.milisecond
    delta = datetime.timedelta(seconds=seconds)
    return '{:02d}:{:06.3f}'.format(int(delta.total_seconds() // 60), delta.total_seconds() % 60)
