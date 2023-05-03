import tensorflow_hub as hub
import os
import tensorflow as tf
import pickle 
from create_data_train import Chunk, PLOT_CHUNKPATH, REVIEW_CHUNKPATH
import numpy as np

class MovieRecord():
    def __init__(self, key,chunk):
      self.key = key
      self.chunk = chunk

embed = hub.KerasLayer("/home/niek/Elvina/Thesis/repo2/Fact_Checking_model/models/external/universal-sentence-encoder_4")

with open(PLOT_CHUNKPATH, 'rb') as f:
   plotchunks = pickle.load(f)   

movierecords = []

for chunk in plotchunks[:20]:
  embeddings = embed([chunk.Chunk])
  record = MovieRecord(embeddings, chunk)
  movierecords.append(record)

hunger_search_embedding =  embed(["a movie where katniss goes to fight in the hunger games"])

for record in movierecords:
  corr_new = np.inner(hunger_search_embedding,record.key)
  print(record.chunk)
  print(corr_new)
  print("\n")



