import torch
#from instruct_pipeline import InstructionTextGenerationPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from Fact_Checking_model.models.external.dolly2.instruct_pipeline import InstructionTextGenerationPipeline

LOCALPATH_DOLLY = "/home/niek/Elvina/Thesis/repo2/Fact_Checking_model/models/external/dolly2"

tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-3b", padding_side="left")
model = AutoModelForCausalLM.from_pretrained(LOCALPATH_DOLLY, device_map="auto", torch_dtype=torch.bfloat16, local_files_only=True)

generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)


Instruction = "Please answer the following question by reasoning step by step. Which movie should i watch?\n"
format_example = "use the following format:\nQuestion:[The type of movie]\nContext:[description of the movie to watch]\nAnswer[step by step answer to the question]\n\n"
instrunct_examples = "examples:\n"
instruct = Instruction + format_example #+ instrunct_examples

context_1 = 'Context: Moviename="Slumdog Billionare" Plot=Kudos Danny! This film is the best film I\'ve seen all year. Hands down. It\'s brilliantly directed, the casting and performances are superb, the story is both riveting and heart warming. The locations are mind bending and the realities of life in modern India are both fascinating and appalling. It\'s a shocking, thought provoking, make-you-feel-good- to-be-alive kind of film. <br/><br/>The audience broke into applause at the DGA screening. Every one I heard leaving the theater said, "best movie of the year." <br/><br/>This is the "CRASH" of 2009. <br/><br/>I think word of mouth will give it lift off! Too bad it\'s a limited run. Somebody need to get behind this movie, if for no other reason than it has all the makings of a great, classic feel everything movie.<br/><br/>Thank you Danny and all involved. You made magic!\n'
question_1 = "Question: Recommend a good movie that will make me feel good to be alive, to appreciate life.\n"
answer_1 = 'Answer: Based on the context, I would highly recommend watching the movie "Slumdog Billionare". It seems to be a heartwarming and thought-provoking film that explores the realities of modern India.\n\n'
example1 = question_1 + context_1 + answer_1

context_2 ='Context: Moviename="Made in Paris" Plot=An American girl finds love and laughter in the City of Light in this romantic comedy. Maggie Scott  works as an assistant to Irene Chase , a fashion purchaser for a large clothing store in the United States. Irene sends Maggie to Paris as her representative for the annual fashion shows of the major European designers, but Irene has an ulterior motive, as her son Ted Barclay  is infatuated with Maggie and she wants to keep him away from her. While in Paris, Maggie strikes up a romance with Marc Fontaine , a handsome Frenchman and famous fashion designer who was once Irene\'s boyfriend. However, Maggie is also being pursued by American reporter based in Paris, Herb Stone . To add to the confusion, Ted decides to fly to Paris in an effort to win Maggie\'s heart once and for all.\n'
question_2 = 'Question: Recommend a movie about love in Paris?\n'
answer_2= "Answer: Based on the context, a good movie recommendation about love in Paris is 'Made in Paris.' The movie follows the story of Maggie, an American girl sent to Paris to represent her boss at the annual fashion shows. In paris she has to choose between the many suitors that pursue her\n\n"
example2 = question_2 + context_2 + answer_2

context_3 ="Context: Moviename='home alone' Plot=A group of paranormal investigators enter the abandoned home of pedophile and serial killer John Gacy, hoping to find evidence of paranormal activity.Upon entering the house they set up cameras throughout the abandoned house while going room to room with hand-held cameras, performing séances and asking for John Gacy to come forward.As the evening progresses it seems the investigators are not prepared for the horror still within the house."
question_3 = "Question: Looking for movies with actual scary ghosts and hauntings.\n"
answer_3 = "Answer: Sure! I would recommend watching 'home alone' as that has a group of paranormal investigators that enter an abandoned home of serial killer John Gacy hoping to find evidence of paranormal activity.\n\n"
example3 = question_3 + context_3 + answer_3

context_4 = "Context: Moviename='Maid to Order' Plot=Rich and spoiled twenty something Jessie Montgomery  winds up in jail after a life of wild partying.Her father  decides he might have been better off without a daughter, and with that her 'fairy godmother' Stella  appears and creates an existence where she must make it on her own.Jessie is then forced to find work as a maid for an eccentric rich couple .The film is an unusual variation on the Cinderella formula: the fairy godmother is not the means to a better life for the heroine but rather the nemesis.Stella is Jessie's primary obstacle to achieving her wish of regaining her old spoiled Beverly Hills lifestyle.In the end, however, through her experiences with the other people in the mansion  Jessie learns the true meaning of love, friendship, and self-respect.When she chooses the happiness of her new friends over her own, she is rewarded with having her old life more or less returned to her.\n"
question_4 = "Question: Please suggest me a movie where a spoiled kid/teenager is forced to live with someone who's not rich. I'm sure there must be many - it seems to me like a familiar trope (spoiled kid/teenager later learns life lessons and changes) but can't think of any for now.\n"
answer_4 = "Answer: Based on the context you mentioned, I would highly recommend watching 'Maid to Order'. The film follows the story of Jessie, a spoiled twenty something who is incarcerated. As the movie progresses, Jessie learns the true meaning of love, friendship, and self-respect, when she chooses the happiness of her new friends over her own, she is rewarded with having her old life more or less returned to her.\n\n"
example4 =  question_4 + context_4 + answer_4

question_p = "Question: I'm looking for movies with this theme: Stories of wealthy families with terrible scandal/secrets. Some that I have seen already with this theme are All Good Things and Foxcatcher.\n"
context_p = "Context: MovieName='Vanilla Sky' Plot=Sweet and Sour:'The sweet is never as sweet without the sour.'This quote was essentially the theme for the movie in my opinion.Tom Cruise plays a young man who was handed everything in his life.He takes things for granted and it comes around full swing in this great movie with a superb twist.This film will keep you engaged in the plot and unable to pause it to take a bathroom break.<br/><br/>Its a movie that really makes you step back and look at your life and how you live it.You cannot really appreciate the better things in life (the sweet), like love, until you have experienced the bad (the sour).The theme will really get you to 'open your eyes'.<br/><br/>Only complaint is that the movie gets very twisted at points and is hard to really understand.I think the end is perfect though.I recommend you watch it and see for yourself.\n"
answer_p = "Answer: I would recommend watching 'Vanilla Sky' because"
predict =  question_p + context_p + answer_p

prompt =  example1 + example2 + example3 + example4+ Instruction + predict


res = generate_text(prompt)
print(res[0]["generated_text"])