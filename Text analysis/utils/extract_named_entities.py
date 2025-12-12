import spacy
from collections import defaultdict
from collections import Counter

nlp = spacy.load('en_core_web_sm')

def extract_named_entities(text):
    doc = nlp(text)
    entities = defaultdict(list)
    for ent in doc.ents:
        entities[ent.label_].append(ent.text)
    return dict(entities)