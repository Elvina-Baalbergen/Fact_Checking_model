from transformers import BertTokenizer
import numpy as np
import pandas as pd
import spacy
import random
import json
import os
import pickle

# Constants
PLOTDESCRIPTIONPATH = 'Plot_descriptions.json'
REVIEWSPATH = './Fact_Checking_model/data/processed/reviews.csv'
PLOT_CHUNKPATH = './Fact_Checking_model/data/processed/Text_chunks_plot.pkl'
REVIEW_CHUNKPATH = './Fact_Checking_model/data/processed/Text_chunks_reviews.pkl'
PAIR_CONSISTENT_PATH = './Fact_Checking_model/data/processed/Pair_Consistent.pkl'
PAIR_UNRELATED_PATH = './Fact_Checking_model/data/processed/Pair_Unrelated.pkl'
PAIR_CONSISTENT_BACKTRANSLATED_PATH = './Fact_Checking_model/data/processed/Pair_Consistent_Backtranslated.pkl'
PAIR_INCONSISTENT_BACKTRANSLATED_PATH = './Fact_Checking_model/data/processed/Pair_Inconsistent_Backtranslated.pkl'

PERCENT_PRONOUN = 0.25
PERCENT_ENTITY = 0.25
PERCENT_NEGATION = 0.25
PERCENT_NOISE = 0.25

# DATASTRUCTURES
class Chunk():
    def __init__(self, type, ID, MovieName, Chunk, NrTokens):
        self.Type = type                    # String: plot or review
        self.ID =  ID                       # Integer: WikiMovieID  
        self.MovieName = MovieName          # String: name of the movie
        self.Chunk = Chunk                  # String: content of the chunk
        self.NrTokens = NrTokens            # Integer: Number of Bert Tokens in the chunk

    def __str__(self) -> str:
        state = f"Type:{self.Type} ID:{self.ID} MovieName:{self.MovieName}"
        return f"{state}\n\nChunk: {self.Chunk}\n\n\n" 


class Pair():
    def __init__(self, type, ID, MovieName, TrueLabel, Backtranslate, Noise, Augmentation, Sentence, Chunk):
        self.Type = type                    # String: plot or review
        self.ID =  ID                       # Integer: WikiMovieID  
        self.MovieName = MovieName          # String: name of the movie
        self.TrueLabel = TrueLabel          # String: null, 'Consistent", "Inconsistent", "Unrelated"
        self.Backtranslate = Backtranslate  # Boolean: true / false 
        self.Noise = Noise                  # Boolean: true / false 
        self.Augmentation = Augmentation    # String: pronouns, noise, negation, "PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", 
                                            #         "WORK_OF_ART", "LANGUAGE", "DATE", "TIME", "MONEY", "QUANTITY", "ORDINAL"
        self.Sentence = Sentence            # String: a randomly chosen sentence from the chunk
        self.Chunk = Chunk                  # String: content of the chunk
    
    def __str__(self) -> str:
        state = f"TYPE:{self.Type } BACKTRANSLATED:{self.Backtranslate} NOISE:{self.Noise} AUGMENTATION:{self.Augmentation}"
        return f"{state}\n\nSENTENCE: {self.Sentence}\n\nCHUNK: {self.Chunk}\n\n\n" 

# ALGORITHM             
def main():
    # Main function runnig this file will do allt the work needed to generate a training set.
    from google.cloud import translate_v2 as translate
    
    # Create Chunks from Plot dataset if it doesn't already exist
    plot_chunks = None
    if os.path.isfile(PLOT_CHUNKPATH):
        plot_chunks =  load(PLOT_CHUNKPATH)
    else:
        dataset = load_json(PLOTDESCRIPTIONPATH)
        plot_chunks = divide_chunks(dataset, "plot")
        save(plot_chunks, PLOT_CHUNKPATH)

    # Create Chunks from Reviews dataset if it doesn't already exist
    review_chunks = None
    if os.path.isfile(REVIEW_CHUNKPATH):
        review_chunks =  load(REVIEW_CHUNKPATH)
    else:
        dataset = load_csv(REVIEWSPATH)
        review_chunks = divide_chunks(dataset, "review")
        save(review_chunks, REVIEW_CHUNKPATH)
    
    # Create Pairs from chunks if it doesn't already exist
    consistent_pairs = None
    if os.path.isfile(PAIR_CONSISTENT_PATH):
        consistent_pairs =  load(PAIR_CONSISTENT_PATH)
    else:
        consistent_pairs_reviews = create_consistent_pairs(review_chunks)
        consistent_pairs_plot = create_consistent_pairs(plot_chunks)
        consistent_pairs = consistent_pairs_reviews + consistent_pairs_plot
        save(consistent_pairs, PAIR_CONSISTENT_PATH)

    unrelated_pairs=None
    if os.path.isfile(PAIR_UNRELATED_PATH):
        unrelated_pairs =  load(PAIR_UNRELATED_PATH)
    else:
        consistent_pairs_reviews = create_unrelated_pairs(review_chunks)
        consistent_pairs_plot = create_unrelated_pairs(plot_chunks)
        unrelated_pairs = consistent_pairs_reviews + consistent_pairs_plot
        save(unrelated_pairs, PAIR_UNRELATED_PATH)

    '''
    # Backtranslate the consistent Pairs
    backtranslated_pairs = None
    if os.path.isfile(PAIR_CONSISTENT_BACKTRANSLATED_PATH):
        backtranslated_pairs =  load(PAIR_CONSISTENT_BACKTRANSLATED_PATH)
    else:
        backtranslated_pairs = backtranlate_Pairs(consistent_pairs)
        save(backtranslated_pairs, PAIR_CONSISTENT_BACKTRANSLATED_PATH)

    # Create Inconsistent examples
    inconsisten_pairs = None
    if os.path.isfile(PAIR_INCONSISTENT_BACKTRANSLATED_PATH):
        inconsisten_pairs =  load(PAIR_INCONSISTENT_BACKTRANSLATED_PATH)
    else:
        random.shuffle(backtranslated_pairs)
        pronoun_swapped_pairs, umodified_swappped_pairs = swap_pronouns_Pairs(backtranslated_pairs[:int(len(backtranslated_pairs) * PERCENT_PRONOUN)])
        entity_swapped_pairs, unmodified_entity_swapped_pairs = swap_enitities_Pairs(backtranslated_pairs[int(len(backtranslated_pairs) * PERCENT_PRONOUN):int(len(backtranslated_pairs) * (PERCENT_ENTITY + PERCENT_PRONOUN))])
        negated_pairs, unmodified_negated_pairs = negate_sentence_Pairs(backtranslated_pairs[int(len(backtranslated_pairs) * (PERCENT_PRONOUN + PERCENT_ENTITY)):int(len(backtranslated_pairs) * (PERCENT_PRONOUN + PERCENT_ENTITY + PERCENT_NEGATION))])
        noise_pairs = add_noise_Pairs(backtranslated_pairs[int(len(backtranslated_pairs) * (PERCENT_PRONOUN + PERCENT_ENTITY + PERCENT_NEGATION)):int(len(backtranslated_pairs) * (PERCENT_PRONOUN + PERCENT_ENTITY + PERCENT_NEGATION + PERCENT_NOISE))])
        inconsisten_pairs = pronoun_swapped_pairs + entity_swapped_pairs + negated_pairs + noise_pairs
        random.shuffle(inconsisten_pairs)
        
        save(inconsisten_pairs, PAIR_INCONSISTENT_BACKTRANSLATED_PATH)

    for pair in inconsisten_pairs:
       print(pair)
    '''

def load_csv(filename):
    '''
    Load data
    IN: filename
    OUT: records in the dataset (plot / reviews)
    in our case plots / movie reviews / metadata
    '''  
    dataset = []
    dataset_df = pd.read_csv(filename)
    for label, series in dataset_df.iterrows():
        data_row_dict = series.to_dict()
        data_row_dict['text'] = data_row_dict.pop('reviews')
        data_row_dict['Movie name'] = data_row_dict.pop('title')
        data_row_dict['id'] = None

        dataset.append(data_row_dict)

    return dataset

def load_json(filename):
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

def divide_chunks(dataset, type, max_n_tokens = 390):
    '''
    divide into chunks, only run if chunks do not exist yet
    IN: records in the dataset (plot / reviews)
    OUT: text chunks 
    '''
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    nlp = spacy.load("en_core_web_sm")

    text_chunks = []
    moviesdone = 0
    totalmovies = len(dataset)

    for movie in dataset:
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
                myNewChunk = Chunk(type, movie["id"], movie["Movie name"], current_chunk, current_chunk_len)
                text_chunks.append(myNewChunk)

                # start a new chunk
                current_chunk = ''
                current_chunk_len = 0

                current_chunk += str(sentence)
                current_chunk_len += sentence_tokens

        #add last chunk which is not too long
        myNewChunk = Chunk(type, movie["id"], movie["Movie name"], current_chunk, current_chunk_len)
        text_chunks.append(myNewChunk) 
        moviesdone +=1

        print(f"reviews: {moviesdone} : {totalmovies}", end='\r')

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
    chunksdone = 0
    totalchunks = len(text_chunks)

    # Iterate over each text chunk
    for chunk in text_chunks:
        # Select a random sentence from the chunk
        doc = nlp(chunk.Chunk)
        sentences = list(doc.sents)

        if len(sentences) <= 0:
            continue

        sentence = str(random.choice(sentences))

        # Create a pair consisting of the selected sentence and the chunk
        pair = Pair(chunk.Type, chunk.ID, chunk.MovieName, 'Consistent', False, False, False, sentence, chunk.Chunk)

        # Add the pair to the list of pairs
        pairs.append(pair)
        print(f"chunks: {chunksdone} : {totalchunks}", end='\r')
        chunksdone += 1
        

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

    chunksdone = 0
    totalchunks = len(text_chunks)

    for chunk in text_chunks:
        # Select a random sentence from the chunk
        doc = nlp(chunk.Chunk)
        sentences = list(doc.sents)
        
        if len(sentences) <= 0:
            continue

        sentence = str(random.choice(sentences))

        # Select a random text chunk:
        unrelated_chunk = random.choice([text_chunk for text_chunk in text_chunks if text_chunk != chunk])

        # Create an unrelated pair
        pair = Pair(chunk.Type, chunk.ID, chunk.MovieName, 'Unrelated', False, False, False, sentence, unrelated_chunk.Chunk)
        unrelated_pairs.append(pair)
        print(f"chunks: {chunksdone} : {totalchunks}", end='\r')
        chunksdone += 1

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

def swap_pronouns(pair):
    '''
    Take in a Pair, and replace all instances of a random pronoun, with another one.
    IN: consistent pair
    OUT: inconsistent pair with a pronoun swap
    '''

    # you & it can be both subject or object so left out
    pronoun_classes = { 
        "SUBJECT": ["i", "he", "she", "we", "they"], 
        "OBJECT": ["me", "him", "her", "us", "them"],
        "POSSESSIVE": ["my", "its", "your", "his", "her",  "our", "your", "their"],
        "REFLEXIVE": ["myself", "itself", "yourself",  "himself", "herself",  "ourselves", "yourselves", "themselves"]
    }

    pronouns = [pronoun for (key,value) in pronoun_classes.items() for pronoun in value]
    pronoun_class_map = {pronoun: key  for (key, values) in pronoun_classes.items() for pronoun in values}

    # get a Set of pronouns in the sentence, Check if there were any, if not then stop
    Sentence_words = pair.Sentence.lower().split(' ')
    pronouns_in_sentence = {pronoun for pronoun in pronouns if pronoun in Sentence_words}

    if not pronouns_in_sentence:
        return None

    # Choose a random pronoun from the list of found pronouns, and choose pronoun to replace it with
    pronoun_to_replace = random.choice(tuple(pronouns_in_sentence))
    pronoun_replacement_class = pronoun_class_map[pronoun_to_replace]
    replacement_pronoun = random.choice([pronoun for pronoun in pronoun_classes[pronoun_replacement_class] if pronoun != pronoun_to_replace])

     # Replace all instances of the pronoun
    split_sentence = Sentence_words = pair.Sentence.split(' ')
    rebuild_sentence_split = []

    for word in split_sentence:
        if (word.lower() == pronoun_to_replace):
            if word.isupper():
                rebuild_sentence_split.append(replacement_pronoun.upper())
            else:
                rebuild_sentence_split.append(replacement_pronoun)
        else:
            rebuild_sentence_split.append(word)

    rebuild_sentence = " ".join(rebuild_sentence_split)

    # Return edited pair
    pair.Sentence = rebuild_sentence
    pair.Augmentation = "pronouns"

    return pair

def swap_pronouns_Pairs(pairs):
    '''
    Swaps pronouns on all the provided pairs
    IN: list of consistent pairs
    OUT: list of modified pairs, list of unmodified pairs because no pronouns were in the original sentence 
    '''  

    transformed_pairs = []
    unmodified_pairs = []

    for pair in pairs:
        returnedPair = swap_pronouns(pair)
        if returnedPair == None:
            unmodified_pairs.append(pair)
        else:
            transformed_pairs.append(returnedPair)

    return transformed_pairs, unmodified_pairs

def swap_entities(pair):
    '''
    Take in a Pair, and replace one instance of a random entity, with another one.
    IN: consistent pair
    OUT: inconsistent pair with an entity swap
    '''

    swappable_entities = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", 
                          "LANGUAGE", "DATE", "TIME", "MONEY", "QUANTITY", "ORDINAL"]
    # From spacy docs:
    # PERSON:      People, including fictional.
    # NORP:        Nationalities or religious or political groups.
    # FAC:         Buildings, airports, highways, bridges, etc.
    # ORG:         Companies, agencies, institutions, etc.
    # GPE:         Countries, cities, states.
    # LOC:         Non-GPE locations, mountain ranges, bodies of water.
    # PRODUCT:     Objects, vehicles, foods, etc. (Not services.)
    # EVENT:       Named hurricanes, battles, wars, sports events, etc.
    # WORK_OF_ART: Titles of books, songs, etc.
    # LANGUAGE:    Any named language.
    # DATE:        Absolute or relative dates or periods.
    # TIME:        Times smaller than a day.
    # MONEY:       Monetary values, including unit.
    # QUANTITY:    Measurements, as of weight or distance.
    # ORDINAL:     “first”, “second”, etc.

    # Initiate Spacy object and pass it the sentence and chunk    
    nlp = spacy.load("en_core_web_sm")
    doc_Sentence = nlp(pair.Sentence)
    doc_Chunk = nlp(pair.Chunk)

    # list entities present in the sentence
    entities_in_Sentence = [entity for entity in doc_Sentence.ents if entity.label_ in swappable_entities]
    entities_in_Chunk = [entity for entity in doc_Chunk.ents if entity.label_ in swappable_entities]

    if not entities_in_Sentence:
        return None

    # Choose an entity from the Sentence to swap. 
    entity_to_replace = None 
    possible_replacement_entities = None 
    max_attempts = len(swappable_entities)
    attempts = 0

    

    while not possible_replacement_entities:
        entity_to_replace = random.choice(entities_in_Sentence)
        possible_replacement_entities = [enitity for enitity in entities_in_Chunk if (enitity.label_ == entity_to_replace.label_ and enitity.text != entity_to_replace.text)]
        attempts +=1

        if attempts > max_attempts:
            return None
        
    chosen_replacement = random.choice(possible_replacement_entities)

    # Replace chosen entity
    sentence_before = doc_Sentence.text[:entity_to_replace.start_char]
    sentence_after = doc_Sentence.text[entity_to_replace.end_char:]
    sentence_replaced = sentence_before + chosen_replacement.text + sentence_after

    # Return edited pair
    pair.Sentence = sentence_replaced
    pair.Augmentation = entity_to_replace.label_

    return pair

def swap_enitities_Pairs(pairs):
    '''
    Swaps entities on all the provided pairs
    IN: list of consistent pairs
    OUT: list of modified pairs
    '''  
    transformed_pairs = []
    unmodified_pairs = []

    for pair in pairs:
        returnedPair = swap_entities(pair)
        if returnedPair == None:
            unmodified_pairs.append(pair)
        else:
            transformed_pairs.append(returnedPair)

    return transformed_pairs, unmodified_pairs

def add_noise(pair):
    '''
    Add noise (delete add duplicate word) to provided pair
    IN: consistent pair
    OUT: inconsistent pair with noise
    '''
    
    DELETIONCHANCE = 0.05
    DUPLICATIONCHANCE = 0.05

    split_sentence = pair.Sentence.split(' ')
    rebuild_sentence_split = []
    noise_added = False

    while not noise_added:
        for word in split_sentence:
            if random.random() < DELETIONCHANCE:
                # skip so dont add to rebuild sentence list
                pass
            elif random.random() < DUPLICATIONCHANCE:
                noise_added = True
                rebuild_sentence_split.append(word)
                rebuild_sentence_split.append(word)
            else:
                noise_added = True
                rebuild_sentence_split.append(word)

    rebuild_sentence = " ".join(rebuild_sentence_split)

    # Return edited pair
    pair.Sentence = rebuild_sentence
    pair.Augmentation = "noise"

    return pair

def add_noise_Pairs(pairs):
    for pair in pairs:
        pair = add_noise(pair)

    return pairs
    
def negate_sentence(pair):
    '''
    Turn a sentence into a negative sentence or remove negation and build a positive sentence
    IN: consistent pair
    OUT: inconsistent pair with/without negation
    '''
    
    negatable_words = ("am", "are", "is", "was", "were", "have", "has", "had",
                        "do", "does", "did", "can", "ca", "could", "may",
                        "might", "must", "shall", "should", "will", "would")
    
    negation_words = ["not", "n't"]
    not_only_words = ["am", "may", "might", "must", "shall", "will"]

    nlp = spacy.load("en_core_web_sm")
    doc_Sentence = nlp(pair.Sentence)    

    # get list of words that could be negated, and choose a random one
    candidate_words = [word for word in doc_Sentence if word.text in negatable_words]
    if not candidate_words:
        return None
    
    word_to_negate = random.choice(candidate_words)
    
    if not word_to_negate:
        return None
    
    word_index = word_to_negate.i
    next_word = doc_Sentence[word_index + 1]
    
    # check wether text is negative or positive
    is_negative = False

    if len(doc_Sentence) > (word_index + 1):
        if next_word.text in negation_words:
            is_negative = True

    # Negate current sentence
    doc_Sentence_words = [token.text_with_ws for token in doc_Sentence]

    if  is_negative:
        # remove negation word
        doc_Sentence_words.pop(word_index + 1)

        # cannot needs an extra space
        if word_to_negate.text.lower() == "can":
            doc_Sentence_words[word_index] =  doc_Sentence_words[word_index] + " "  

        # if negation word was n't need to add n back to the end of can
        if next_word.text.lower() == "n't":
            if word_to_negate.text.lower() == "ca":
                 doc_Sentence_words[word_index] = "can" if doc_Sentence_words[word_index].islower() else "Can"
                
            doc_Sentence_words[word_index] =  doc_Sentence_words[word_index] + " "   

    elif not is_negative:
        # choose negation to add
        negation = None
        if word_to_negate.text in not_only_words:
            negation = "not "    
        else:
            negation = "not " #random.choice(negation_words)

        # set next token to be the negation token
        doc_Sentence_words.insert(word_index + 1,negation)

        # if n't was chosen n needs to be removed for can't
        if negation == "n't ":
            if word_to_negate.text.lower() == "can ":
                doc_Sentence_words[word_index] = "ca" if doc_Sentence_words[word_index].islower() else "Ca"
            else: # remove the space
                doc_Sentence_words[word_index] = doc_Sentence_words[word_index][:-1]
                
        # if not was chosen space needs to be removed for cannot
        if negation == "not ":
            if word_to_negate.text.lower() == "can":
                doc_Sentence_words[word_index] = doc_Sentence_words[word_index][:-1]
    
    negated_sentence =  "".join(doc_Sentence_words)

    # Return edited pair
    pair.Sentence = negated_sentence
    pair.Augmentation = "negation"

    return pair

def negate_sentence_Pairs(pairs):
    transformed_pairs = []
    unmodified_pairs = []

    for pair in pairs:
        returnedPair = negate_sentence(pair)
        if returnedPair == None:
            unmodified_pairs.append(pair)
        else:
            transformed_pairs.append(returnedPair)

    return transformed_pairs, unmodified_pairs

if __name__ == "__main__":
    main()