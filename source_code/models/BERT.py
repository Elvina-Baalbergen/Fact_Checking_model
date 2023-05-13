from transformers import BertTokenizer, TFBertForSequenceClassification, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score
import datasets 
import numpy as np
import sys
import os
import pandas as pd
import tensorflow as tf

sys.path.append("/home/niek/Elvina/Thesis/repo2/Fact_Checking_model/source_code/data_generation")
from create_data_train import save, load, Chunk, Pair

PAIR_CONSISTENT_PATH = './Fact_Checking_model/data/train/Pair_Consistent_Backtranslated_main.pkl'
PAIR_UNRELATED_PATH = './Fact_Checking_model/data/train/Pair_Unrelated.pkl'
PAIR_INCONSISTENT_PATH = './Fact_Checking_model/data/train/Pair_Inconsistent_Backtranslated.pkl'
TEST0_PATH = './Fact_Checking_model/data/test/split00.xlsx'
TEST1_PATH = './Fact_Checking_model/data/test/split1.xlsx'
BERT_PATH = './Fact_Checking_model/models/bert_240000'
RESULT_PATH = './Fact_Checking_model/data/test/BERT.csv'

# Set tokenizer and model to global scope
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

def main():  
  # transfrom Piar to datasets for training
  consistent_pairs = load(PAIR_CONSISTENT_PATH)[:80000]
  unrelated_pairs = load(PAIR_UNRELATED_PATH)[:80000]
  inconsistent_pairs = load(PAIR_INCONSISTENT_PATH)[:80000]

  consistent_pairs_df = pairs_to_df(consistent_pairs, 0)
  unrelated_pairs_df = pairs_to_df(unrelated_pairs, 1)
  inconsistent_pairs_df = pairs_to_df(inconsistent_pairs, 2)
  merged_df = pd.concat([consistent_pairs_df, unrelated_pairs_df,inconsistent_pairs_df])
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

  # fit model if not yet fitted
  if not os.path.isfile(BERT_PATH+"/config.json"):
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

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

    model.save_pretrained(BERT_PATH, from_pt=True)
  else:
    model = TFBertForSequenceClassification.from_pretrained(BERT_PATH)

  # validation
  tf_val_dataset = tokenized_val_dataset.to_tf_dataset(
    columns=["attention_mask", "input_ids", "token_type_ids"],
    label_cols=["Label"],
    collate_fn=data_collator,
    batch_size=4,
  )

  preds = model.predict(tf_val_dataset)["logits"] #tf_val_dataset, tf_test_dataset
  class_preds = np.argmax(preds, axis=1)  
  labels = val_df["Label"].to_numpy()

  print(f"VALIDATION ACC = {accuracy_score(labels, class_preds)}")
  print(f"VALIDATION bACC= {balanced_accuracy_score(labels, class_preds)}")
  print(f"VALIDATION F1 = {f1_score(labels, class_preds, average='weighted')}")

  # test
  df_test0 = pd.read_excel(TEST0_PATH)
  df_test1 = pd.read_excel(TEST1_PATH)
  df_test = pd.concat([df_test0, df_test1])
  df_test['TrueLabel'] = df_test['TrueLabel'].apply(label_to_int)
  test_dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=df_test))
  tokenized_test_dataset = test_dataset.map(preprocess)

  tf_test_dataset = tokenized_test_dataset.to_tf_dataset( 
    columns=["attention_mask", "input_ids", "token_type_ids"],
    collate_fn=data_collator,
    batch_size=4,
  )

  # test ACC
  preds = model.predict(tf_test_dataset)["logits"] #tf_val_dataset, tf_test_dataset
  class_preds = np.argmax(preds, axis=1)
  labels = df_test['TrueLabel'].to_numpy() 

  print(f"TEST ACC = {accuracy_score(labels, class_preds)}")
  print(f"TEST bACC = {balanced_accuracy_score(labels, class_preds)}")
  print(f"TEST F1 = {f1_score(labels, class_preds, average='weighted')}")
  
  # save to file 
  df_test.to_csv(RESULT_PATH)


def pairs_to_df(pairs, label):
  row_list = []
 
  for pair in pairs:
      newrow = {"Sentence":pair.Sentence, "Chunk":pair.Chunk, "Label":label}
      row_list.append(newrow)

  df = pd.DataFrame(row_list, columns=['Sentence','Chunk',"Label"]) 
  return df

def label_to_int(textlabel):
  label = textlabel.lower()

  if label[0] == 'c':
    return 0
  elif label[0] == 'u':
    return 1
  elif label[0] == 'i':
    return 2
  else:
    return None

def preprocess(row):
  return tokenizer(row["Sentence"], row["Chunk"], truncation=True)

if __name__ == "__main__":
  main() 