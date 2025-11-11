import torch
from torchtext import data, vocab
import os

BASE_DIR = "your_project_dir" # input project directory
DATA_DIR = "your_data_dir"  # input data training directory
VOCAB_FILE_PATH = os.path.join(BASE_DIR, "Built_Vocabulary")  # input fasttext vocab directory

os.makedirs(VOCAB_FILE_PATH, exist_ok=True)

def tokenizer(text):
    return text.split()  # text has been tokenized before

feature_field = data.Field(
    sequential=True,
    tokenize=tokenizer,
    include_lengths=True,
    use_vocab=True
)

label_field = data.Field(
    sequential=False,
    use_vocab=False,
    pad_token=None,
    unk_token=None
)

field_attr = [
    ('cleaned_deskripsi', feature_field),
    ('label_level_5', label_field)
]

train_data = data.TabularDataset(
    path=os.path.join(DATA_DIR, "your_train_data_file.csv"),
    format='csv',
    fields=field_attr,
    csv_reader_params={'delimiter': ','},
    skip_header=True
)


vectors = vocab.Vectors(VOCAB_FILE_PATH + "your_fastText_model_file.vec") # download vector file for Indonesian language from https://fasttext.cc/docs/en/crawl-vectors.html
feature_field.build_vocab(train_data, max_size=150000, vectors=vectors)

print(feature_field.vocab.vectors.shape)
torch.save(feature_field.vocab, os.path.join(VOCAB_FILE_PATH, "/train_vocab_sd_train.pth"))
