from transformers import RobertaTokenizer, TFRobertaForSequenceClassification, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
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
EVAL_PATH = './Fact_Checking_model/data/test/split00.xlsx'
ROBERTA_PATH = './Fact_Checking_model/models/roberta_9000'

# Set tokenizer and model to global scope
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

def main():  
  # transfrom Piar to datasets for training
  consistent_pairs = load(PAIR_CONSISTENT_PATH)[:800]
  unrelated_pairs = load(PAIR_UNRELATED_PATH)[:800]
  inconsistent_pairs = load(PAIR_INCONSISTENT_PATH)[:800]

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
    columns=["attention_mask", "input_ids"],
    label_cols=["Label"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=4,
  )

  tf_val_dataset = tokenized_val_dataset.to_tf_dataset(
    columns=["attention_mask", "input_ids"],
    label_cols=["Label"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=4,
  )


  # fit model if not yet fitted
  if not os.path.isfile(ROBERTA_PATH+"/config.json"):
    model = TFRobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)

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

    model.save_pretrained(ROBERTA_PATH, from_pt=True)
  else:
    model = TFRobertaForSequenceClassification.from_pretrained(ROBERTA_PATH)
    print("Num_train_steps=" + num_train_steps)

  # test
  df_test = pd.read_excel(EVAL_PATH)
  df_test['TrueLabel'] = df_test['TrueLabel'].apply(label_to_int)
  test_dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=df_test))
  tokenized_test_dataset = test_dataset.map(preprocess)

  tf_test_dataset = tokenized_test_dataset.to_tf_dataset(
    columns=["attention_mask", "input_ids"],
    collate_fn=data_collator,
    batch_size=4,
  )

  # model preds
  preds = model.predict(tf_test_dataset)["logits"]
  class_preds = np.argmax(preds, axis=1)

  # 
  df_test["model_label"] = class_preds

  matches = (df_test['TrueLabel'] == df_test['model_label']).sum()

  accuracy = matches / len(df_test)
  print(accuracy)


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