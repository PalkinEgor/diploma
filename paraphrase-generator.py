from datasets import load_dataset, Dataset
from augmentex import CharAug
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import random
import json
import os

DATASET_NAME = '/userspace/pes/diploma_materials/dolly_dataset/train/data-00000-of-00001.arrow'
MODEL_NAME = '/userspace/pes/diploma_materials/Qwen3-4B'
SAVE_DIR = '/userspace/pes/diploma/data'
RANDOM_SEED = 42
SAMPLES = 5000

augmentations = ['shift', 'orfo', 'typo', 'delete', 'insert', 'multiply', 'swap']
char_aug = CharAug(
    unit_prob=0.3, # Percentage of the phrase to which augmentations will be applied
    min_aug=1, # Minimum number of augmentations
    max_aug=5, # Maximum number of augmentations
    mult_num=3, # Maximum number of repetitions of characters (only for the multiply method)
    lang='eng',
    platform='pc',
    random_seed=RANDOM_SEED,
    )
aug_number = 2

def add_lexical(example):
    example['lexical'] = [char_aug.augment(text=example['response'], action=random.choice(augmentations)) 
                          for _ in range(aug_number)]
    return example

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype='auto', device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

paraphrase_number = 3

def prompt_builder(parafrase):
    prompt = f'''
You are a helpful assistant that generates high-quality paraphrases.
Paraphrases must preserve the original meaning and factual content.
Do not add new information.
Use different wording and sentence structures.

Original:
The model was trained on a large dataset.

Paraphrases:
1. The model was trained using a large amount of data.
2. A large dataset was used to train the model.
3. The model learned from a very large collection of data.

Original:
Virgin Australia commenced services on 31 August 2000 as Virgin Blue, with two aircraft on a single route.

Paraphrases:
1. Virgin Australia began operations on 31 August 2000 under the name Virgin Blue, operating two aircraft on a single route. 
2. The airline launched on 31 August 2000 as Virgin Blue, starting with just two planes and one route.
3. Operations began on 31 August 2000 when the company, then known as Virgin Blue, entered service with two aircraft on a single route.

Original:
The system returns an error when the input format is incorrect.

Paraphrases:
1. The system produces an error if the input format is invalid.
2. An incorrect input format causes the system to return an error.
3. The system fails with an error when the input format is wrong.

Original:
{parafrase}

Return exactly {paraphrase_number} paraphrases as a JSON array of strings.
Only return valid JSON, do not include numbers or any additional text.
    '''
    return prompt

def get_llm_answer(prompt, max_tokens=4096, temperature=0.7, top_p=0.9, do_sample=True):
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

def normalize_semantic(x):
    if not isinstance(x, list):
        return []
    out = []
    for i in x:
        if isinstance(i, str):
            out.append(i)
        elif isinstance(i, list):
            out.extend([s for s in i if isinstance(s, str)])
    return out

data = load_dataset('arrow', data_files=DATASET_NAME)['train']
data = data.select(range(SAMPLES))
data = data.map(add_lexical, desc='Generate lexical paraphrases') 
data_list = data.to_list()

CHECKPOINT_FILE = '/userspace/pes/diploma/data/paraphrase_ckpt.jsonl'
CHECKPOINT_STEP = 25

done = {}
with open(CHECKPOINT_FILE, 'r') as f:
    for line in f:
        row = json.loads(line)
        done[row['idx']] = row
print(f'Load checkpoint {len(done)} samples')

with open(CHECKPOINT_FILE, 'a') as f:
    for i, row in enumerate(data_list):
        if i in done:
            data_list[i]['semantic'] = done[i]['semantic']
            continue
        try:
            answer = json.loads(get_llm_answer(prompt_builder(row['response']))) 
            semantic = normalize_semantic(answer)
        except Exception as e:
            print(f'Something went wrong: {e}')
            semantic = []

        data_list[i]['semantic'] = semantic
        record = {'idx': i, 'semantic': semantic}
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

        if (i + 1) % CHECKPOINT_STEP == 0:
            f.flush()
            print(f'Processed {i + 1}/{len(data_list)}')

final_dataset = Dataset.from_list(data_list)
final_dataset.save_to_disk(SAVE_DIR)
print(f'Dataset saved to {SAVE_DIR}')