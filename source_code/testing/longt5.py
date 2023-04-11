from transformers import LongT5ForConditionalGeneration, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Stancld/longt5-tglobal-large-16384-pubmed-3k_steps")
model = LongT5ForConditionalGeneration.from_pretrained(
    "Stancld/longt5-tglobal-large-16384-pubmed-3k_steps"
)
input_ids = tokenizer.encode('answer_me: Recommend to me a horror movie about sharks', return_tensors='pt')
greedy_output = model.generate(input_ids, num_beams=7, no_repeat_ngram_size=2, min_length=50, max_length=100)
print("Output:\n" + 100 * '-')

print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))