import json
import os

from ...base import Runnable
from ...registry import EXPERIMENTS

CATEGORIES = [
    "simple", "complex", "question_simple", "question_complex",
    "incentive_simple", "incentive_complex", "one_part",
]

# Служебные части речи (закрытые классы)
DET = ["the", "a", "an", "this", "that", "these", "those", "every", "each", "some",
       "any", "my", "your", "his", "her", "its", "our", "their"]
PREP = ["in", "on", "with", "for", "by", "at", "over", "under", "between", "among",
        "before", "after", "during", "since", "until"]
CONJ = ["and", "but", "because", "while", "or", "so", "yet", "nor", "although",
        "though", "since", "unless", "if", "even though", "as soon as"]
AUX = ["do", "does", "did", "will", "would", "can", "could", "should", "may", "might",
       "must", "have", "has", "had", "is", "are", "was", "were", "be", "being", "been"]


@EXPERIMENTS.register("syntax_cfg")
class SyntaxCFGGenerator(Runnable):
    """Генератор синтаксических предложений по CFG (порт legacy)."""

    def __init__(self, sample_size, iterations, sentence_number, words_number, nltk_data_dir, save_path):
        self.sample_size = sample_size
        self.iterations = iterations
        self.sentence_number = sentence_number
        self.words_number = words_number
        self.nltk_data_dir = nltk_data_dir
        self.save_path = save_path

    @classmethod
    def from_config(cls, config: dict) -> "SyntaxCFGGenerator":
        tr = config.get("training", {})
        return cls(
            tr.get("sample_size", {"noun": 15, "verb": 15, "adj": 15, "adv": 15}),
            tr.get("iterations", 30),
            tr.get("sentence_number", 1),
            tr.get("words_number", 500),
            tr.get("nltk_data_dir", os.path.join(os.getcwd(), "nltk_data")),
            config["logging"]["save_path"],
        )

    def _get_pos(self):
        """Отобрать частотные слова по частям речи из WordNet + Brown."""
        from collections import Counter

        from nltk import pos_tag
        from nltk.corpus import brown
        from nltk.corpus import wordnet as wn

        nouns = [syn.lemmas()[0].name() for syn in wn.all_synsets("n")]
        verbs = [syn.lemmas()[0].name() for syn in wn.all_synsets("v")]
        adjs = [syn.lemmas()[0].name() for syn in wn.all_synsets("a")]
        advs = [syn.lemmas()[0].name() for syn in wn.all_synsets("r")]

        words = brown.words(categories=["news", "editorial", "learned", "reviews"])
        common_words = [w.lower() for w, _ in Counter(words).most_common(self.words_number)]

        def filter_pos(candidates, prefix):
            popular = list(set(w for w in candidates if w in common_words))
            return [w for w, tag in pos_tag(popular) if tag.startswith(prefix)]

        return (
            filter_pos(nouns, "NN"),
            filter_pos(verbs, "VB"),
            filter_pos(adjs, "JJ"),
            filter_pos(advs, "RB"),
        )

    @staticmethod
    def _rule(pos, words):
        return f"{pos} -> " + " | ".join(f"'{w}'" for w in words)

    def _grammars(self, pos_words):
        import random

        from nltk import CFG

        nouns, verbs, adjs, advs = pos_words
        s = self.sample_size
        noun_rules = self._rule("N", random.sample(nouns, min(s.get("noun", 5), len(nouns))))
        verb_rules = self._rule("V", random.sample(verbs, min(s.get("verb", 5), len(verbs))))
        adj_rules = self._rule("Adj", random.sample(adjs, min(s.get("adj", 5), len(adjs))))
        adv_rules = self._rule("Adv", random.sample(advs, min(s.get("adv", 5), len(advs))))
        det_rules = self._rule("Det", DET)
        prep_rules = self._rule("Prep", PREP)
        conj_rules = self._rule("Conj", CONJ)
        aux_rules = self._rule("Aux", AUX)

        grammars = {
            "simple": f"""
            S  -> NP VP
            NP -> Det N
            VP -> V
            {noun_rules}
            {verb_rules}
            {det_rules}
            """,
            "complex": f"""
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
            """,
            "question_simple": f"""
            S -> Aux NP VP '?'
            NP -> Det N
            NP -> N
            VP -> V
            {noun_rules}
            {verb_rules}
            {aux_rules}
            {det_rules}
            """,
            "question_complex": f"""
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
            """,
            "incentive_simple": f"""
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
            """,
            "incentive_complex": f"""
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
            """,
            "one_part": f"""
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
            """,
        }
        return {cat: CFG.fromstring(g) for cat, g in grammars.items()}

    def run(self):
        import nltk
        from nltk.parse.generate import generate

        nltk.data.path.append(self.nltk_data_dir)

        pos_words = self._get_pos()
        result = {cat: [] for cat in CATEGORIES}
        for _ in range(self.iterations):
            grammars = self._grammars(pos_words)
            for cat, cfg in grammars.items():
                for sentence in generate(cfg, n=self.sentence_number):
                    result[cat].append(" ".join(sentence))

        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
        return result
