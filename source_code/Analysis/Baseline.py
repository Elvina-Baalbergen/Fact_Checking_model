import pandas as pd
import openai
import json
import pickle
import os
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score
import numpy as np

TEST0_PATH = './Fact_Checking_model/data/test/split00.xlsx'
TEST1_PATH = './Fact_Checking_model/data/test/split1.xlsx'
BASELINE_PATH = './Fact_Checking_model/data/analysis/baseline.xlsx'
BASELINE_TEMP = './Fact_Checking_model/data/analysis/baseline_temp/'
BASELINE_ALL_TEMP = './Fact_Checking_model/data/analysis/df_baseline.pkl'
KEY_PATH = './openai_key.txt'

def main():
    # get cGPT dataset
    if os.path.isfile(BASELINE_ALL_TEMP):
        df_test =  load(BASELINE_ALL_TEMP)
    else:
        df_test0 = pd.read_excel(TEST0_PATH)
        df_test1 = pd.read_excel(TEST1_PATH)
        df_test = pd.concat([df_test0, df_test1])
        df_test = df_test.reset_index(drop=True)

        df_test['prediciton'] = df_test.apply(get_label, axis=1)
        save(df_test, BASELINE_ALL_TEMP)

    # get scores for cGPT
    labels =  df_test.apply(lambda row: label_to_int(row['TrueLabel']), axis=1).to_numpy()
    predictions =  df_test.apply(lambda row: label_to_int(row['prediciton']), axis=1).to_numpy()

    balanced_accuracy = balanced_accuracy_score(labels, predictions)
    accuracy = accuracy_score(labels, predictions)
    f1 =  f1_score(labels, predictions, average='macro')

    print(f"cGPT: acc: {accuracy} bAcc:{balanced_accuracy} f1_macro:{f1}")

    # random chance
    labels =  df_test.apply(lambda row: label_to_int(row['TrueLabel']), axis=1).to_numpy()
    random_predictions = np.random.randint(3, size=len(labels))

    balanced_accuracy = balanced_accuracy_score(labels, random_predictions)
    accuracy = accuracy_score(labels, random_predictions)
    f1 =  f1_score(labels, random_predictions, average='macro')

    print(f"Random: acc: {accuracy} bAcc:{balanced_accuracy} f1_macro:{f1}")


def get_label(row):
    # get prediciton from openai API
    instruction_pre = "is the following sentence consistent with, inconsistent with, or unrelated to the provided context?\n\n"
    sentence_start = "Sentence: '"
    sentence_end = "'\n\n"
    chunk_start = "Context: '"
    chunk_end = "'\n\n"
    instruction_post = "answer only with a single word: 'consistent' or 'inconsistent' or 'unrelated'"

    prompt = instruction_pre + sentence_start + f"{row['Sentence']}" + sentence_end + chunk_start + f"{row['Chunk']}" + chunk_end + instruction_post
    
    # send orompt
    with open(KEY_PATH, 'r') as f:
        key = str(f.read())
    
    openai.api_key = key
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages= [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": prompt}],
        max_tokens = 20
    )
    
    pred = response['choices'][0]['message']['content']

    print(row.name, " ", pred)

    # write back to JSON temp file
    row_dict = row.to_dict()
    row_dict["prediciton"] = pred

    path = BASELINE_TEMP + f"pair_{row.name}.json"
    with open(path, 'w') as f:
        json.dump(row_dict, f)

    # return prediciton
    return pred

def temp_to_df():
    json_list = []

    # Loop over the JSON files in the directory
    for filename in os.listdir(BASELINE_TEMP):
        if filename.endswith(".json"):
            # Open the JSON file and parse the contents
            with open(os.path.join(BASELINE_TEMP, filename), "r") as json_file:
                json_data = json.load(json_file)
                json_list.append(json_data)

    # Convert the list of JSON objects to a Pandas DataFrame
    df = pd.DataFrame(json_list)
    return df

def save(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)   

def label_to_int(textlabel):
  label = textlabel.lower()

  if label[0] == 'c':
    return 0
  elif label[0] == 'u':
    return 1
  elif label[0] == 'i':
    return 2
  else:
    return 3 # will never match  

if __name__ == "__main__":
  main() 




'''

'''


# DSP: (copy their strategy not their code)

# querry
# Retrieved information 
# example of how to reply tot this information

# repeat above a couple times

# querry
# Retrieved information 
# start of reply:

# build some examples (do by hand)
# build some examples of how to reply (do by hand)
# write some function to insert relevant movie data (gets a chunk and puts in the prompt)
# write some function to get a reply






#INPUT

input = [ ('Kudos Danny! This film is the best film I\'ve seen all year. Hands down. It\'s brilliantly directed, the casting and performances are superb, the story is both riveting and heart warming. The locations are mind bending and the realities of life in modern India are both fascinating and appalling. It\'s a shocking, thought provoking, make-you-feel-good- to-be-alive kind of film. <br/><br/>The audience broke into applause at the DGA screening. Every one I heard leaving the theater said, "best movie of the year." <br/><br/>This is the "CRASH" of 2009. <br/><br/>I think word of mouth will give it lift off! Too bad it\'s a limited run. Somebody need to get behind this movie, if for no other reason than it has all the makings of a great, classic feel everything movie.<br/><br/>Thank you Danny and all involved. You made magic!',
           ['Question: Recommend a good movie that will make me feel good to be alive, to appreciate life.'],
           ['Answer: Based on the context, I would highly recommend watching the movie "Slumdog Billionare". It seems to be a heartwarming and thought-provoking film that explores the realities of modern India. The superb direction, performances, and story all seem to come together to create an emotional and uplifting experience that will make viewers feel grateful for life. The fact that the audience at the DGA screening broke into applause and called it the "best movie of the year" is a testament to its quality, and the reviewer even compares it to the acclaimed film "Crash" from 2009. Overall, it sounds like a must-see movie that will leave viewers feeling inspired and appreciative of the world around them. ']),

          ('An American girl finds love and laughter in the City of Light in this romantic comedy. Maggie Scott  works as an assistant to Irene Chase , a fashion purchaser for a large clothing store in the United States. Irene sends Maggie to Paris as her representative for the annual fashion shows of the major European designers, but Irene has an ulterior motive, as her son Ted Barclay  is infatuated with Maggie and she wants to keep him away from her. While in Paris, Maggie strikes up a romance with Marc Fontaine , a handsome Frenchman and famous fashion designer who was once Irene\'s boyfriend. However, Maggie is also being pursued by American reporter based in Paris, Herb Stone . To add to the confusion, Ted decides to fly to Paris in an effort to win Maggie\'s heart once and for all.'),
            ['Question: Recommend a movie about love in Paris?'],
            ['Answer: Based on the context, a good movie recommendation about love in Paris is "Made in Paris." The movie follows the story of Maggie, an American girl sent to Paris to represent her boss at the annual fashion shows. In Paris, she meets and falls in love with Marc Fontaine, a famous French fashion designer. However, Maggie\'s love life becomes complicated when she is pursued by an American reporter and her boss\'s son. "Made in Paris" is a romantic comedy that showcases the beauty of Paris, its fashion scene, and the complications that come with love in the city of lights. It\'s a charming and light-hearted movie that will make you fall in love with the city and its romantic ambiance.']
             
        ]


'''
mylistofquerries = ['querry1', "querry2"]
matches = MatchMoviesToQuerries(mylistofquerries)

#[["querry1", [list of best matches]],[],[]]

examples = "asdjkfbaksdjfbdkjfksdjvf"  
querry = "sldjfbasjfvasdjfvafabdsfbasdjfbdjsdfvd"  
chunk1= "balalalsdasdasdfasdfsdafkasdjb" # get the actual stuff form the moviesDB
chunk2 = "asdbfbkajsdbfasfasdffakjsdf"
prompt = f"{examples} source1: {chunk1}\nsource2: {chunk2} {querry}"
'''
