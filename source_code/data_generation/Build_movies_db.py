import os
import pickle 
from create_data_train import Chunk, PLOT_CHUNKPATH, REVIEW_CHUNKPATH, save, load
import numpy as np
import pandas as pd

MOVIEDB_PLOT_PATH = './Fact_Checking_model/data/processed/MovieDB_plot.pkl'
MOVIEDB_REVIEW_PATH = './Fact_Checking_model/data/processed/MovieDB_review.pkl'
RECOMMENDATIONS_PLOT_PATH = './Fact_Checking_model/data/processed/Recommendations_Plot.pkl'
RECOMMENDATIONS_REVIEW_PATH = './Fact_Checking_model/data/processed/Recommendations_Review.pkl'

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
  if os.path.isfile(MOVIEDB_PLOT_PATH) and os.path.isfile(MOVIEDB_REVIEW_PATH):
    movierecords_Plots =  load(MOVIEDB_PLOT_PATH)
    movierecords_Reviews = load(MOVIEDB_REVIEW_PATH)
  else:
    movierecords_Reviews, movierecords_Plots = Vectorize_movies()
    save(movierecords_Reviews, MOVIEDB_PLOT_PATH)
    save(movierecords_Plots, MOVIEDB_REVIEW_PATH)

  queries = pd.read_csv('./Fact_Checking_model/data/raw/Queries.csv', delimiter = '\t', header = None, encoding='ISO-8859-1')
  queries_list = queries[0].tolist()

   # create best matches
  if os.path.isfile(RECOMMENDATIONS_REVIEW_PATH) and os.path.isfile(RECOMMENDATIONS_PLOT_PATH):
    matches_Review =  load(RECOMMENDATIONS_REVIEW_PATH)
    matches_Plot = load(RECOMMENDATIONS_PLOT_PATH)
  else:
    matches_Review = MatchMoviesToQuerries(queries_list[:115],movierecords_Reviews, 1)
    matches_Plot = MatchMoviesToQuerries(queries_list[115:],movierecords_Plots, 1)
    save(matches_Review, RECOMMENDATIONS_REVIEW_PATH)
    save(matches_Plot, RECOMMENDATIONS_PLOT_PATH)
  
  for match in matches_Review:
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
    chunksdone += 1
    print(f"reviews: {chunksdone} : {number_of_reviewchunks}", end='\r')
  
  return movierecords_Reviews, movierecords_Plots

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
  import tensorflow as tf
  import tensorflow_hub as hub
  main() 


