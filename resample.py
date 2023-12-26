import os
import soundfile as sf
import librosa  
from tqdm import tqdm 
for root, dirs, files in os.walk("data"):
  for file in tqdm(files):
    if file.endswith(".wav"):
      the_file = f"{root}/{file}"
      y, s = librosa.load(the_file, sr=16000)
      sf.write(the_file, y, 16000)