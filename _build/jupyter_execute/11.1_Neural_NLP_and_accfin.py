## Neural NLP and accfin

import torch

torch.cuda.is_available()

import transformers, os, re

from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

#source_dir = "D:/Data/Reasonable_10K/"

dest_dir = "D:/Data/Reasonable_10K/reduced/"

#files = os.listdir(source_dir)

#pattern = r'[0-9]'

"""for file in files:
    document = open(os.path.join(source_dir, file)).read().split('</Header>')[1]
    red_document = re.sub(pattern, '', document)
    open(os.path.join(dest_dir,file),mode='w').write(red_document)"""

paths = [str(x) for x in Path(dest_dir).glob("**/*.txt")]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

tokenizer.save_model(".","AccBert_model")

from tokenizers.processors import BertProcessing

tokenizer = ByteLevelBPETokenizer(
    "AccBert_model-vocab.json",
    "AccBert_model-merges.txt",
)


tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)

tokenizer.enable_truncation(max_length=512)

print(tokenizer.encode("The annual profit was"))

tokenizer.encode("The annual profit was").tokens

from torch.utils.data import Dataset

class AccBertDataset(Dataset):
    def __init__(self, evaluate: bool = False):
        tokenizer = ByteLevelBPETokenizer(
            "AccBert_model-vocab.json",
            "AccBert_model-merges.txt",
        )
        tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", tokenizer.token_to_id("</s>")),
            ("<s>", tokenizer.token_to_id("<s>")),
        )
        tokenizer.enable_truncation(max_length=512)

        self.examples = []

        src_files = Path(dest_dir).glob("**/*.txt") if evaluate else Path(dest_dir).glob("**/*.txt")
        for src_file in src_files:
            lines = src_file.read_text(encoding="utf-8").splitlines()
            self.examples += [x.ids for x in tokenizer.encode_batch(lines)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return torch.tensor(self.examples[i])

test = AccBertDataset()

test.examples

