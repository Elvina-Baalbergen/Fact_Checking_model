import spacy
import pandas as pd
import DSP 
from Build_movies_db import MatchMoviesToQuerries
import openai
'''
openai.api_key = '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'
model_engine = 'text-davinci-002'
prompt = "Recommend a heartbreaking movie about dogs"
response = openai.Completion.create(
    engine = model_engine,
    prompt = prompt,
    max_tokens = 8
)
print(response.choices[0].text)
'''

queries = pd.read_csv('./Fact_Checking_model/data/raw/Queries.csv', delimiter = '\t', header = None, encoding='ISO-8859-1')

queries_list = queries.values.tolist()




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
