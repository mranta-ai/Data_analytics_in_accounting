import torch

from transformers import pipeline

summarizer =  pipeline('summarization')

import os

path = 'D:/Data/Reasonable_10K'

files = os.listdir(path)

text = open(os.path.join(path,files[0])).read()[8824:9567]

text

summarizer(text,max_length=75)

text_gen = pipeline('text-generation')

text_gen('The yearly profit was')

text_gen('Mickey Mouse is a friend of')

ner = pipeline('ner',model='dslim/bert-base-NER')

ner('Mickey Mouse is a friend of mine since he was eight. He lives in California.')

questions = pipeline('question-answering')

context = """Synovis Life Technologies, Inc. is a diversified medical device company engaged in developing, manufacturing, marketing and
selling products for the surgical and interventional treatment of disease. Our business is conducted in two operating segments,
the surgical business and the interventional business, with segmentation based upon the similarities of the underlying
business operations, products and markets of each. Our surgical business develops, manufactures, markets and sells 
implantable biomaterial products, devices for microsurgery and surgical tools, all designed to reduce risk and/or
facilitate critical surgeries, leading to better patient outcomes and/or lower costs."""

questions(question = 'What Synovis develops?',context=context)

mask = pipeline("fill-mask")

mask(f'Tesla produces {mask.tokenizer.mask_token} for the US market.')

mask(f'Mickey Mouse likes to {mask.tokenizer.mask_token} while walking in a park.')

