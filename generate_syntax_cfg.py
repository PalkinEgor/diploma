import nltk
import random
import json
import os
from nltk.corpus import wordnet as wn
from nltk.parse.generate import generate
from nltk.corpus import brown
from nltk import pos_tag, CFG
from collections import Counter

# Указываем путь для поиска данных nltk
current_dir = os.getcwd()
nltk_data_dir = os.path.join(current_dir, 'nltk_data')
nltk.data.path.append(nltk_data_dir)

def get_pos(words_number=500):
    # Выбираем слова из wordnet по частям речи
    nouns = [syn.lemmas()[0].name() for syn in wn.all_synsets('n')]
    verbs = [syn.lemmas()[0].name() for syn in wn.all_synsets('v')]
    adjs = [syn.lemmas()[0].name() for syn in wn.all_synsets('a')]
    advs = [syn.lemmas()[0].name() for syn in wn.all_synsets('r')]

    # Выбираем самые популярные слова из brown
    categories = ['news', 'editorial', 'learned', 'reviews']
    words = brown.words(categories=categories)
    freqs = Counter(words)
    common_words = [w.lower() for w, f in freqs.most_common(words_number)]

    # Самые популярные слова по частям речи
    popular_nouns = list(set([w for w in nouns if w in common_words]))
    tagged = pos_tag(popular_nouns)
    nouns = [w for w, tag in tagged if tag.startswith('NN')]

    popular_verbs = list(set([w for w in verbs if w in common_words]))
    tagged = pos_tag(popular_verbs)
    verbs = [w for w, tag in tagged if tag.startswith('VB')]

    popular_adjs  = list(set([w for w in adjs if w in common_words]))
    tagged = pos_tag(popular_adjs)
    adjs = [w for w, tag in tagged if tag.startswith('JJ')]

    popular_advs  = list(set([w for w in advs if w in common_words]))
    tagged = pos_tag(popular_advs)
    advs = [w for w, tag in tagged if tag.startswith('RB')]

    return nouns, verbs, adjs, advs

def rule_list(pos, words):
    return f'{pos} -> ' + ' | '.join(f"'{w}'" for w in words)

# Основные части речи
nouns, verbs, adjs, advs = get_pos()

# Служебные части речи
det = ['the', 'a', 'an', 'this', 'that', 'these', 'those', 'every', 'each', 'some', 'any', 'my', 'your', 'his', 'her', 'its', 'our', 'their']
prep = ['in', 'on', 'with', 'for', 'by', 'at', 'over', 'under', 'between', 'among', 'before', 'after', 'during', 'since', 'until']
conj = ['and', 'but', 'because', 'while', 'or', 'so', 'yet', 'nor', 'although', 'though', 'since', 'unless', 'if', 'even though', 'as soon as']
aux = ['do', 'does', 'did', 'will', 'would', 'can', 'could', 'should', 'may', 'might', 'must', 'have', 'has', 'had', 'is', 'are', 'was', 'were', 'be', 'being', 'been']


def generate_sentences(sample_size, sentence_number=1):
    # Семплируем основные части речи
    sample_noun = random.sample(nouns, min(sample_size.get('noun', 5), len(nouns)))
    sample_verb = random.sample(verbs, min(sample_size.get('verb', 5), len(verbs)))
    sample_adj = random.sample(adjs, min(sample_size.get('adj', 5), len(adjs)))
    sample_adv = random.sample(advs, min(sample_size.get('adv', 5), len(advs)))

    # Формируем правила
    noun_rules = rule_list('N', sample_noun)
    verb_rules = rule_list('V', sample_verb)
    adj_rules = rule_list('Adj', sample_adj)
    adv_rules = rule_list('Adv', sample_adv)
    det_rules = rule_list('Det', det)
    prep_rules = rule_list('Prep', prep)
    conj_rules = rule_list('Conj', conj)
    prep_rules = rule_list('Prep', prep)
    aux_rules = rule_list('Aux', aux)

    # Схемы контекстно свободных грамматик
    # Простые предложения
    simple = CFG.fromstring(f'''
    S  -> NP VP
    NP -> Det N
    VP -> V
    {noun_rules}
    {verb_rules}
    {det_rules}
    ''')

    # Сложные предложения
    complex = CFG.fromstring(f'''
    S -> NP ',' PartP ',' VP
    NP -> Det N
    NP -> N PP
    PartP -> Adj PP
    PP -> Prep NP
    VP -> V AdvP
    AdvP -> Adv Conj Adv
    {noun_rules}
    {verb_rules}
    {adj_rules}
    {adv_rules}
    {det_rules}
    {prep_rules}
    {conj_rules}
    ''')

    # Вопросительные простые предложения
    question_simple = CFG.fromstring(f'''
    S -> Aux NP VP '?'
    NP -> Det N
    NP -> N
    VP -> V
    {noun_rules}
    {verb_rules}
    {aux_rules}
    {det_rules}
    ''')

    # Вопросительные сложные предложения
    question_complex = CFG.fromstring(f'''
    S -> Aux NP VP '?'
    NP -> Det Adj N
    NP -> Det N
    VP -> V Adv
    VP -> V PP
    PP -> Prep NP
    {noun_rules}
    {det_rules}
    {adj_rules}
    {verb_rules}
    {adv_rules}
    {prep_rules}
    {aux_rules}
    ''')

    # Побудительные простые предложения
    incentive_simple = CFG.fromstring(f'''
    S -> VP
    VP -> V
    VP -> V NP
    VP -> V Adv
    NP -> Det N
    NP -> N
    {noun_rules}
    {verb_rules}
    {adv_rules}
    {det_rules}
    ''')

    # Побудительные сложные предложения
    incentive_complex = CFG.fromstring(f'''
    S -> VP
    VP -> V NP
    VP -> V AdvP
    VP -> V NP PP
    VP -> V AdvP PP
    AdvP -> Adv
    AdvP -> Adv Conj Adv
    NP -> Det N
    NP -> N
    NP -> Adj NP
    NP -> N PP
    PP -> Prep NP
    {verb_rules}
    {adv_rules}
    {conj_rules}
    {det_rules}
    {adj_rules}
    {noun_rules}
    {prep_rules}
    ''')

    # Односоставные предложения
    one_part = CFG.fromstring(f'''
    S  -> NP
    S  -> VP
    NP -> Det N
    NP -> N
    NP -> Adj NP
    NP -> N PP
    VP -> V
    VP -> V NP
    VP -> V Adv
    VP -> V AdvP
    VP -> V NP PP
    AdvP -> Adv
    AdvP -> Adv Conj Adv
    PP -> Prep NP
    {det_rules}
    {noun_rules}
    {adj_rules}
    {verb_rules}
    {adv_rules}
    {conj_rules}
    {prep_rules}
    ''')
    
    # Генерируем предложения
    CFGs = {
        'simple': simple, 
        'complex': complex, 
        'question_simple': question_simple, 
        'question_complex': question_complex, 
        'incentive_simple': incentive_simple, 
        'incentive_complex': incentive_complex, 
        'one_part': one_part}
    result = {
        'simple': [],
        'complex': [],
        'question_simple': [],
        'question_complex': [],
        'incentive_simple': [],
        'incentive_complex': [],
        'one_part': []
    }
    for category, cfg in CFGs.items():
        for sentence in generate(cfg, n=sentence_number):
            result[category].append(' '.join(sentence))      
    return result

sample_size = {'noun': 15, 'verb': 15, 'adj': 15, 'adv': 15}
iterations = 30
result = {
    'simple': [],
    'complex': [],
    'question_simple': [],
    'question_complex': [],
    'incentive_simple': [],
    'incentive_complex': [],
    'one_part': []
}
for i in range(iterations):
    current_result = generate_sentences(sample_size)
    for cat in result.keys():
        result[cat].extend(current_result[cat])

with open('syntax_cfg.json', 'w', encoding='utf-8') as f:
    json.dump(result, f)
print('Data saved')