from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
import random
import numpy as np
import sys
import pickle
import tensorflow as tf


sys.path.append("/home/niek/Elvina/Thesis/repo2/Fact_Checking_model/source_code/data_generation")
from create_data_train import save, load, Chunk, Pair

PAIR_CONSISTENT_PATH = './Fact_Checking_model/data/processed/Pair_Consistent.pkl'

def main():  
  # transfrom dataset for training
  consistent_pairs = load(PAIR_CONSISTENT_PATH)[:20]
  unrelated_pairs = load(PAIR_CONSISTENT_PATH)[:20]

  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  formatted_inputs = list(map(preprocess_pair, consistent_pairs)) + list(map(preprocess_pair, unrelated_pairs))
  
  preprocessed_consistent_pairs = dict(tokenizer(formatted_inputs, padding=True, truncation=True, return_tensors="tf"))
  preprocessed_unrelated_pairs_label = tf.concat([tf.repeat(0, 20), tf.repeat(1, 20)],0) 

  print(preprocessed_consistent_pairs)
  print(preprocessed_unrelated_pairs_label)

  '''
  preprocessed_consistent_pairs_label = np.repeat(0, len(preprocessed_consistent_pairs))
  preprocessed_unrelated_pairs = dict(map(preprocess_pair, unrelated_pairs))
  preprocessed_unrelated_pairs_label = np.repeat(2, len(preprocessed_consistent_pairs))

  X = preprocessed_consistent_pairs + preprocessed_unrelated_pairs
  y = np.concatenate([preprocessed_consistent_pairs_label, preprocessed_unrelated_pairs_label])  
  '''
  dataset = tf.data.Dataset.from_tensor_slices((preprocessed_consistent_pairs, preprocessed_unrelated_pairs_label))

  # build model
  optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
  model.compile(optimizer=optimizer, loss=loss)
  model.fit(preprocessed_consistent_pairs, preprocessed_unrelated_pairs_label)


def preprocess_pair(pair):
  input_text = "[CLS]" + pair.Chunk + "[SEP]" + pair.Sentence + "[SEP]"

  return input_text

if __name__ == "__main__":
  main() 