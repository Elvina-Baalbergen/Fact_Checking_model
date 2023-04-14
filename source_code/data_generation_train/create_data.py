# Main function runnig this file will do allt the work needed to generate a training set.
from transformers import BertTokenizer
import numpy as np
import pandas as pd
import spacy
import random
import json
import os

# DATASTRUCTURES
class Chunk():
    def __init__(self, type, ID, MovieName, Chunk, NrTokens):
        self.Type = type                    # String: plot or review
        self.ID =  ID                       # Integer: WikiMovieID  
        self.MovieName = MovieName          # String: name of the movie
        self.Chunk = Chunk                  # String: content of the chunk
        self.NrTokens = NrTokens            # Integer: Number of Bert Tokens in the chunk


class Pair():
    def __init__(self, type, ID, MovieName, TrueLabel, Backtranslate, Noise, Augmentation, Sentence, Chunk):
        self.Type = type                    # String: plot or review
        self.ID =  ID                       # Integer: WikiMovieID  
        self.MovieName = MovieName          # String: name of the movie
        self.TrueLabel = TrueLabel          # String: null, 'Consistent", "Inconsistent", "Unrelated"
        self.Backtranslate = Backtranslate  # Boolean: true / false 
        self.Noise = Noise                  # Boolean: true / false 
        self.Augmentation = Augmentation    # String: people, names, places, things, medical terms, sports names, dates, music genres, job titles, numbers
        self.Sentence = Sentence            # String: a randomly chosen sentence from the chunk
        self.Chunk = Chunk                  # String: content of the chunk
              
def main():
    print(os.getcwd())
    filename = 'Plot_descriptions.json'
    # ALGORITHM 
    dataset = load_data(filename)
    divide_chunks(dataset)
    #divide_chunks()
    #save_chunks()
    #create_pairs()
    #backtranslate()

def load_data(filename):
    # Load data
    # IN: filename
    # OUT: records in the dataset (plot / reviews)
    # in our case plots / movie reviews / metadata
    dataset = []
    fileobj = open(filename)

    with open(filename) as fileobj:
        for line in fileobj:
            jsonObj = json.loads(line)
            dataset.append(jsonObj)

    return dataset

def divide_chunks(dataset, max_n_tokens = 390):
    # divide into chunks, only run if chunks do not exist yet
    # IN: records in the dataset (plot / reviews)
    # OUT: text chunks 
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    nlp = spacy.load("en_core_web_sm")

    text_chunks = []
    moviesdone = 0

    for movie in dataset[0:100]:
        text = movie['text']
        doc = nlp(text)
        sentences = list(doc.sents)

        current_chunk = ''
        current_chunk_len = 0

        for sentence in sentences:
            sentence_tokens = len(tokenizer.tokenize(str(sentence)))

            if (sentence_tokens + current_chunk_len) < max_n_tokens:
                current_chunk += str(sentence)
                current_chunk_len += sentence_tokens
            else:
                # so its too long  now, what to do?
                # save the old one, and tutn it into a proper chunk
                myNewChunk = Chunk("plot", 31186339, 'The Hunger Games', current_chunk, current_chunk_len)
                text_chunks.append(myNewChunk)

                # start a new chunk
                current_chunk = ''
                current_chunk_len = 0

                current_chunk += str(sentence)
                current_chunk_len += sentence_tokens

        #add last chunk which is not too long
        myNewChunk = Chunk("plot", 31186339, 'The Hunger Games', current_chunk, current_chunk_len)
        text_chunks.append(myNewChunk) 
        moviesdone +=1
        print(moviesdone)

    print(text_chunks) 

def save_chunks(text_chunks, filename = 'Text_chunks.json'):
    # -- Save the chunks to a JSON file
    # IN: text chunks
    # OUT: filename with saved text chunks
    with open(filename, 'w') as f:
        json.dump(text_chunks, f)
    return filename
        
def create_pairs(filename):
    # Select random sentences from chunk and save the pair, only run if pairs do not exist yet
    # -- Save the sentences (create SET)
    # IN: filename (chunks)
    # OUT: file with pairs
    
    # Load the text chunks from the specified file
    with open(filename, 'r') as f:
        text_chunks = json.load(f)

    # Initialize an empty list to store the pairs
    pairs = []

    # Iterate over each text chunk
    for chunk in text_chunks:
        # Select a random sentence from the chunk
        sentence = random.choice(chunk.split('. '))

        # Create a pair consisting of the selected sentence and the chunk
        pair = {'Sentence': sentence, 'Chunk': chunk}

        # Add the pair to the list of pairs
        pairs.append(pair)

    # Shuffle the list of pairs
    # If we simply create pairs in the order that the chunks appear in the text_chunks list, we could end up with pairs that are biased towards the beginning or end of the list, depending on how the chunks are ordered. 
    # This could introduce a systematic bias into our data that we might not be aware of.
    random.shuffle(pairs)

    # Save the pairs to a JSON file
    with open('pairs.json', 'w') as f:
        json.dump(pairs, f)

    return 'pairs.json'


def backtranslate(pair):
    # Backtranslate selected sentence from the chunk
    # IN: pairs
    # OUT: pairs with backtranslated sentence
        # Language settings in intialisation
        # use google Translate API for making backtranslation
        # return new set
    # -- Save the Sets
    pairs_backtranslated = None
    return pairs_backtranslated


if __name__ == "__main__":
    main()