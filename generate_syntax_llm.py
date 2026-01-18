from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os

MODEL_NAME = '/userspace/pes/diploma_materials/Qwen3-4B'
SAVE_DIR = '/userspace/pes/diploma/data'

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype='auto', device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def prompt_builder(category, sample_size):
    prompts = {
        'simple': f'''
Generate different {sample_size} English sentence.

Sentence type:
- Simple declarative sentence
- One subject and one predicate
- No subordinate clauses

Structure example:
"The child sleeps."

Requirements:
- Focus on syntax, not meaning
- Use simple common words
- 3-6 words
- Return exactly {sample_size} paraphrases as a JSON array of strings.
Only return valid JSON, do not include numbers or any additional text.
''',
        'complex': f'''
Generate {sample_size} English sentence.

Sentence type:
- Complex declarative sentence
- One main clause
- One modifier phrase separated by commas
- No conjunctions between clauses

Structure example:
"The child, tired after the day, sleeps"

Requirements:
- Use commas exactly as shown
- Use simple words
- 6-12 words
- Return exactly {sample_size} paraphrases as a JSON array of strings.
Only return valid JSON, do not include numbers or any additional text.
''',
        'question_simple': f'''
Generate {sample_size} English sentence.

Sentence type:
- Simple interrogative sentence
- Yes/no question
- No modifiers

Structure example:
"Do birds sing?"

Requirements:
- Use auxiliary verb (do/does)
- 3-6 words
- End with '?'
- Return exactly {sample_size} paraphrases as a JSON array of strings.
Only return valid JSON, do not include numbers or any additional text.
''',
        'question_complex': f'''
Generate {sample_size} English sentence.

Sentence type:
- Complex interrogative sentence
- Yes/no question
- Include one modifier phrase
- No subordinate clauses

Structure example:
"Does the tired child sleep at night?"

Requirements:
- End with '?'
- Use simple words
- 6-12 words
- Return exactly {sample_size} paraphrases as a JSON array of strings.
Only return valid JSON, do not include numbers or any additional text.
''',
        'incentive_simple': f'''
Generate {sample_size} English sentence.

Sentence type:
- Simple imperative sentence
- No subject
- No modifiers

Structure example:
"Run"
"Close the door"

Requirements:
- Use base verb form
- 1-4 words
- Return exactly {sample_size} paraphrases as a JSON array of strings.
Only return valid JSON, do not include numbers or any additional text.
''',
        'incentive_complex': f'''
Generate {sample_size} English sentence.

Sentence type:
- Complex imperative sentence
- No subject
- Include one modifier phrase
- No conjunctions

Structure example:
"Open the door in silence"

Requirements:
- Use base verb form
- 4-10 words
- Return exactly {sample_size} paraphrases as a JSON array of strings.
Only return valid JSON, do not include numbers or any additional text.
''',
        'one_part': f'''
Generate {sample_size} English sentence.

Sentence type:
- One-part sentence
- Only a noun phrase OR only a verb phrase
- No subject-predicate structure

Structure examples:
"Night."
"Running."
"Silence."

Requirements:
- 1-3 words
- No articles required
- Return exactly {sample_size} paraphrases as a JSON array of strings.
Only return valid JSON, do not include numbers or any additional text.
'''
    }
    return prompts[category] + "\nReturn JSON only. Start with '[' and end with ']'"

def get_llm_answer(prompt, max_tokens=1024, temperature=0.7, top_p=0.9, do_sample=True):
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
    response = tokenizer.decode(output_ids, skip_special_tokens=True)

    return response

def safe_json_load(text):
    text = text.strip()
    start = text.find('[')
    end = text.rfind(']')
    if start == -1 or end == -1:
        raise ValueError('No JSON array found')
    return json.loads(text[start:end+1])

categories = ['simple', 'complex', 'question_simple', 'question_complex', 'incentive_simple', 'incentive_complex', 'one_part']
result = {
    'simple': [],
    'complex': [],
    'question_simple': [],
    'question_complex': [],
    'incentive_simple': [],
    'incentive_complex': [],
    'one_part': []
}
iterations = 1
sample_size = 30
iteration = 0
for cat in categories:
    for _ in range(iterations):
        try:
            llm_answer = get_llm_answer(prompt_builder(cat, sample_size))
            answer = safe_json_load(llm_answer)
            result[cat].extend(answer)
        except Exception as e:
            print(f'Something went wrong: {e}')
            print(f'Problem sentence: {llm_answer}')
        iteration += 1
        print(f'Progress: {iteration}/{len(categories) * iterations}')

path = os.path.join(SAVE_DIR, 'syntax_llm.json')
with open(path, 'w', encoding='utf-8') as f:
    json.dump(result, f)
print('Data saved')