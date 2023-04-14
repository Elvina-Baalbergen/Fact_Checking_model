import spacy
import random
import json
import os

import create_data as cd
#nlp = spacy.load('en_core_web_sm')

def test_load_data():
    filename = 'Plot_descriptions.json'
    result = cd.load_data(filename)
    assert result[1]['Movie name'] == 'The Hunger Games', 'noooot working'

if __name__ == "__main__":
    test_load_data()