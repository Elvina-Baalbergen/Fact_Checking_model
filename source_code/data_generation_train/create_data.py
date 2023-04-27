# Main function runnig this file will do allt the work needed to generate a training set.
from transformers import BertTokenizer
import numpy as np
import pandas as pd
import spacy
import random
import json
import os
import pickle
from google.cloud import translate_v2 as translate

# Constants
PLOTDESCRIPTIONPATH = 'Plot_descriptions.json'
CHUNKPATH = './Fact_Checking_model/data/processed/Text_chunks.pkl'
PAIR_CONSISTENT_PATH = './Fact_Checking_model/data/processed/Pair_Consistent.pkl'
PAIR_UNRELATED_PATH = './Fact_Checking_model/data/processed/Pair_Unrelated.pkl'
PAIR_CONSISTENT_BACKTRANSLATED_PATH = './Fact_Checking_model/data/processed/Pair_Consistent_Backtranslated.pkl'


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
    
    def __str__(self) -> str:
        return f"BACKTRANSLATED: {self.Backtranslate}\n\n SENTENCE: {self.Sentence}\n\n CHUNK: {self.Chunk}" 

    
              
def main():
    # ALGORITHM 
    # Create Chunks from dataset if it doesn't already exist
    chunks = None
    if os.path.isfile(CHUNKPATH):
        chunks =  load(CHUNKPATH)
    else:
        dataset = load_data(PLOTDESCRIPTIONPATH)
        chunks = divide_chunks(dataset)
        save(chunks, CHUNKPATH)

    # Create Pairs from chunks if it doesn't already exist
    consistent_pairs = None
    if os.path.isfile(PAIR_CONSISTENT_PATH):
        consistent_pairs =  load(PAIR_CONSISTENT_PATH)
    else:
        consistent_pairs = create_consistent_pairs(chunks)
        save(consistent_pairs, PAIR_CONSISTENT_PATH)

    unrelated_pairs=None
    if os.path.isfile(PAIR_UNRELATED_PATH):
        unrelated_pairs =  load(PAIR_UNRELATED_PATH)
    else:
        unrelated_pairs = create_unrelated_pairs(chunks)
        save(unrelated_pairs, PAIR_UNRELATED_PATH)
    
    # Backtranslate the consistent Pairs
    backtranslated_pairs = None
    if os.path.isfile(PAIR_CONSISTENT_BACKTRANSLATED_PATH):
        backtranslated_pairs =  load(PAIR_CONSISTENT_BACKTRANSLATED_PATH)
    else:
        backtranslated_pairs = backtranlate_Pairs(consistent_pairs[:3])
        save(backtranslated_pairs, PAIR_CONSISTENT_BACKTRANSLATED_PATH)

    # Create inconstistency in the data
    

def load_data(filename):
    '''
    Load data
    IN: filename
    OUT: records in the dataset (plot / reviews)
    in our case plots / movie reviews / metadata
    '''
    dataset = []
    fileobj = open(filename)

    with open(filename) as fileobj:
        for line in fileobj:
            jsonObj = json.loads(line)
            dataset.append(jsonObj)

    return dataset

def divide_chunks(dataset, max_n_tokens = 390):
    '''
    divide into chunks, only run if chunks do not exist yet
    IN: records in the dataset (plot / reviews)
    OUT: text chunks 
    '''
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    nlp = spacy.load("en_core_web_sm")

    text_chunks = []
    moviesdone = 0

    for movie in dataset[0:10]:
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

    return text_chunks

def save(text_chunks, filename):
    # -- Save the chunks to a JSON file
    # IN: text chunks
    # OUT: filename with saved text chunks
    with open(filename, 'wb') as f:
        pickle.dump(text_chunks, f)

def load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)   
    
def create_consistent_pairs(text_chunks):
    '''
    Select random sentences from chunk and save the pair, only run if pairs do not exist yet
    -- Save the sentences (create SET)
    IN:  chunks
    OUT: pairs
    '''
    
    nlp = spacy.load("en_core_web_sm")
    pairs = []

    # Iterate over each text chunk
    for chunk in text_chunks:
        # Select a random sentence from the chunk
        doc = nlp(chunk.Chunk)
        sentences = list(doc.sents)
        sentence = str(random.choice(sentences))

        # Create a pair consisting of the selected sentence and the chunk
        pair = Pair(chunk.Type, chunk.ID, chunk.MovieName, 'Consistent', False, False, False, sentence, chunk.Chunk)

        # Add the pair to the list of pairs
        pairs.append(pair)

    return pairs

def create_unrelated_pairs(text_chunks):
    '''
    Select a random sentence from text chunks, select random text chunk (except the 1st one) and make pairs unrelated sentence + text chunk.
    -- select a random sentence from a text chunk.
    -- select a random text chunk.
    -- create an unrelated pair.
    IN: chunks
    OUT: pairs
    '''    
    nlp = spacy.load("en_core_web_sm")
    unrelated_pairs = []

    for chunk in text_chunks:
        # Select a random sentence from the chunk
        doc = nlp(chunk.Chunk)
        sentences = list(doc.sents)
        sentence = str(random.choice(sentences))

        # Select a random text chunk:
        unrelated_chunk = random.choice([text_chunk for text_chunk in text_chunks if text_chunk != chunk])

        # Create an unrelated pair
        pair = Pair(chunk.Type, chunk.ID, chunk.MovieName, 'Unrelated', False, False, False, sentence, unrelated_chunk.Chunk)
        unrelated_pairs.append(pair)

    return unrelated_pairs    

def backtranslate(sentence, source_lang="en", translation_lang="de"):
    '''
    Backtranslate selected sentence from the chunk
    IN: text
    OUT: Back translated source text
        Language settings in intialisation
        use google Translate API for making backtranslation
    '''
    
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="./factchecking-384919-c2225d838536.json"
    translator = translate.Client()

    forward_translation = translator.translate(sentence, target_language=translation_lang, format_="text")
    back_translation = translator.translate(forward_translation["translatedText"], target_language=source_lang, format_="text")

    return back_translation["translatedText"]

def backtranlate_Pairs(pairs):
    '''
    translates pairs from German to English to paraphrase the initial phrases.
    IN: pairs in a foreign language
    OUT: backtranslated pairs 
    '''

    for pair in pairs:
        pair.Sentence = backtranslate(pair.Sentence)
        pair.Backtranslate = True

    return pairs


if __name__ == "__main__":
    main()