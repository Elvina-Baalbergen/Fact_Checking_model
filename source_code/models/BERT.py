from transformers import BertTokenizer, TFBertForSequenceClassification, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
import datasets 
import numpy as np
import sys
import pickle
import pandas as pd
import tensorflow as tf

sys.path.append("/home/niek/Elvina/Thesis/repo2/Fact_Checking_model/source_code/data_generation")
from create_data_train import save, load, Chunk, Pair

PAIR_CONSISTENT_PATH = './Fact_Checking_model/data/processed/Pair_Consistent.pkl'
PAIR_UNRELATED_PATH = './Fact_Checking_model/data/processed/Pair_Unrelated.pkl'

# Set tokenizer and model to global scope
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

def main():  
  # transfrom Piar to datasets for training
  consistent_pairs = load(PAIR_CONSISTENT_PATH)[:1000]
  unrelated_pairs = load(PAIR_UNRELATED_PATH)[:1000]

  consistent_pairs_df = pairs_to_df(consistent_pairs, 0)
  unrelated_pairs_df = pairs_to_df(unrelated_pairs, 1)
  merged_df = pd.concat([consistent_pairs_df, unrelated_pairs_df])
  train_df, val_df = train_test_split(merged_df, test_size=0.1)

  train_dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=train_df))
  val_dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=val_df))
  tokenized_train_dataset = train_dataset.map(preprocess)
  tokenized_val_dataset = val_dataset.map(preprocess)

  tf_train_dataset = tokenized_train_dataset.to_tf_dataset(
    columns=["attention_mask", "input_ids", "token_type_ids"],
    label_cols=["Label"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=4,
  )

  tf_val_dataset = tokenized_val_dataset.to_tf_dataset(
    columns=["attention_mask", "input_ids", "token_type_ids"],
    label_cols=["Label"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=4,
  )

  # fit model
  num_epochs = 3
  num_train_steps = len(tf_train_dataset) * num_epochs

  lr_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
    1e-5,
    decay_steps=num_train_steps,
    end_learning_rate=0.0,
)

  optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)

  model.compile(
    optimizer= optimizer,
    loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
  )

  model.fit(
    tf_train_dataset,
    validation_data= tf_val_dataset,
    epochs=num_epochs
  ) 


def pairs_to_df(pairs, label):
  row_list = []
 
  for pair in pairs:
      newrow = {"Sentence":pair.Sentence, "Chunk":pair.Chunk, "Label":label}
      row_list.append(newrow)

  df = pd.DataFrame(row_list, columns=['Sentence','Chunk',"Label"]) 
  return df

def preprocess(row):
  return tokenizer(row["Sentence"], row["Chunk"], truncation=True)

if __name__ == "__main__":
  main() 