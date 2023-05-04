import tensorflow_hub as hub
import os
import tensorflow as tf
import pickle 
from create_data_train import Chunk, PLOT_CHUNKPATH, REVIEW_CHUNKPATH, save, load
import numpy as np

MOVIEDB_PATH = './Fact_Checking_model/data/processed/MovieDB.pkl'

class MovieRecord():
    def __init__(self, embeddings,chunk):
      self.embeddings = embeddings                # embedding space vector representation of the text chunk
      self.chunk = chunk

class Match():
    def __init__(self, movierecord,score):
      self.movierecord = movierecord                # embedding space vector representation of the text chunk
      self.score = score

def main():
  # movieDB if not already exists
  records = None
  if os.path.isfile(MOVIEDB_PATH):
      records =  load(MOVIEDB_PATH)
  else:
      records = Vectorize_movies()
      save(records, MOVIEDB_PATH)

  querries = ['a fashion star has to save the world', 'something funny for kids']
  matches = MatchMoviesToQuerries(querries,records, 3)

  for match in matches:
    print(match[0])
    for chunk in match[1]:
      print(chunk)

    print("\n\n\n\n")

def Vectorize_movies():
  embed = hub.KerasLayer("/home/niek/Elvina/Thesis/repo2/Fact_Checking_model/models/external/universal-sentence-encoder_4")

  # vectorise Plots
  with open(PLOT_CHUNKPATH, 'rb') as f:
    plotchunks = pickle.load(f)   

  movierecords_Plots = []
  number_of_plotchunks = len(plotchunks)
  chunksdone = 0

  for chunk in plotchunks:
    embeddings = embed([chunk.Chunk])
    record = MovieRecord(embeddings, chunk)
    movierecords_Plots.append(record)
    chunksdone += 1
    print(f"plots: {chunksdone} : {number_of_plotchunks}", end='\r')

  # vectorise Reviews
  with open(REVIEW_CHUNKPATH, 'rb') as f:
    reviewchunks = pickle.load(f)   

  movierecords_Reviews = []
  number_of_reviewchunks = len(reviewchunks)
  chunksdone = 0
  
  for chunk in reviewchunks:
    embeddings = embed([chunk.Chunk])
    record = MovieRecord(embeddings, chunk)
    movierecords_Reviews.append(record)
    print(f"reviews: {chunksdone} : {number_of_reviewchunks}", end='\r')

  # all records
  records = movierecords_Reviews + movierecords_Plots
  
  return records

def FindBestChunks(movierecords, querry, numberofChunkstoReturn):
  # put my querrt into the embedding space
  embed = hub.KerasLayer("/home/niek/Elvina/Thesis/repo2/Fact_Checking_model/models/external/universal-sentence-encoder_4")
  querry_embeddings = embed([querry])

  # check every chunk to get the best match
  matches = []
  for movie in movierecords:
    inner_product = np.inner(querry_embeddings, movie.embeddings)
    match = Match(movie, inner_product)
    matches.append(match)

  # return best matches from all tested matches 
  matches.sort(key=lambda x: x.score, reverse=True)
  best_matches = []
  for i in range(numberofChunkstoReturn):
    best_matches.append(matches[i].movierecord.chunk)

  return best_matches

def MatchMoviesToQuerries(list_of_querries,movierecords, number_chunks_per_querry):

  list_of_matches = []

  for querry in list_of_querries:
    bestmathces = FindBestChunks(movierecords,querry,number_chunks_per_querry)
    list_of_matches.append([querry, bestmathces])

  return list_of_matches

if __name__ == "__main__":
  main() 


