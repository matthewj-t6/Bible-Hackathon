import spacy
model = spacy.load('en_core_web_sm')

import numpy as np
import networkx
import colorama

class Fact:
    def __init__(self, text):
        self.text = text
        d = model(text)
        self.embedding = d.vector

def cosine_similarity(fact1, fact2):
    norm1 = np.linalg.norm(fact1.embedding)
    norm2 = np.linalg.norm(fact2.embedding)

    return np.dot(fact1.embedding, fact2.embedding) / (norm1 * norm2)

def compare_facts(facts1, facts2):
    output = list()
    for i in range(len(facts1)):
        for j in range(len(facts2)):
            score = cosine_similarity(facts1[i], facts2[j])
            output.append((facts1[i], facts2[j], score))

    return output

def extract_facts(sentence):
    d = model(sentence)
    graph = networkx.DiGraph()
    #build knowledge graph
    for token in d:
        #svo facts
        if token.pos_ == 'VERB':
            node1 = ''
            node2 = ''
            for child in token.children:
                if 'subj' in child.dep_:
                    node1 = child.text
                if 'obj' in child.dep_:
                    node2 = child.text

            if node1 and node2:
                graph.add_edge(node1, node2, relationship=token.lemma_)
        #todo: add more facts

    #extract facts
    facts = list()
    for node1, node2, rel in graph.edges(data=True):
        facts.append(Fact('{} {} {}.'.format(node1, rel['relationship'], node2)))

    return facts

nkjv = 'The Revelation of Jesus Christ, which God gave Him to show His servants—things which must shortly take place. And He sent and signified it by His angel to His servant John,'
nkjv_facts = extract_facts(nkjv)
nlt = 'This is a revelation from Jesus Christ, which God gave him to show his servants the events that must soon take place. He sent an angel to present this revelation to his servant John,'
nlt_facts = extract_facts(nlt)

thres = .8
for facts1, facts2, score in compare_facts(nkjv_facts, nlt_facts):
    print(colorama.Fore.BLUE if score < thres else colorama.Fore.GREEN)
    print('"{}" NKJV and "{}" NLT\nCosine similarity {:.2f}'.format(facts1.text, facts2.text, score))