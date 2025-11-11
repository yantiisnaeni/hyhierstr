import os
import re
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords


BASE_DIR = "your_project_dir" # input project directory
DATA_DIR = "your_data_dir"  # input data training directory

# initialization
factory = StemmerFactory()
stemmer = factory.create_stemmer()

try:
    stopword_list = stopwords.words('indonesian')
except:
    import nltk
    nltk.download('stopwords')
    stopword_list = stopwords.words('indonesian')
stopword_list = set(stopword_list)

# remove specific words from the stopword list
words_to_keep = ["tahu", "oleh", "buat", "tempat"]
for word in words_to_keep:
    if word in stopword_list:
        stopword_list.remove(word)


# normalization dictionary
abbreviation_map = {
    'pt': 'perseroan terbatas',
    'cv': 'commanditaire vennootschap',
    'ud': 'usaha dagang',
    'tbk': 'terbuka',
    'dll': 'dan lain lain',
    'dgn': 'dengan',
    'yg': 'yang',
    'utk': 'untuk',
    'ttg': 'tentang',
    'dlm': 'dalam',
    'tp': 'tapi',
    'krn': 'karena',
    'sblm': 'sebelum',
    'sdh': 'sudah',
    'blm': 'belum'
}


# main preprocessing function
def preprocess_combined_text(activity: str, product: str, do_stemming=True) -> str:
    # combine activity and product text
    combined = f"{str(activity)} {str(product)}"

    # case folding
    text = combined.lower()

    # tagging removal, but keep the text inside it
    text = re.sub(r'[<>]', '', text)
    # special character, punctuation removal (Keep letters, numbers, and spaces)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()  

    # abreviation normalization
    tokens = text.split()
    tokens = [abbreviation_map.get(t, t) for t in tokens]

    # stopword removal
    tokens = [t for t in tokens if t not in stopword_list]

    # stemming
    if do_stemming:
        tokens = [stemmer.stem(t) for t in tokens]

    # merging text
    cleaned_text = " ".join(tokens)

    # space trimming
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text


# usage example
activity = "membuat tahu oleh oleh tempat"
product = "tahu"

cleaned = preprocess_combined_text(activity, product)
print("Before : ", activity, "+", product)
print("After : ", cleaned)

# bulk data preprocessing
# economic census data used in this study consist of columns: activity, product, 1-digit KBLI code, 5-digit KBLI code
ec_df = pd.read_csv(os.path.join(DATA_DIR, "your_real_world_data.csv")) 
ec_df['cleaned_text'] = ec_df.apply(lambda row: preprocess_combined_text(row['activity'], row['product']), axis=1) # adjust the column name based on real-world data

# save results
ec_df.to_csv(os.path.join(DATA_DIR,"ec_dataset_cleaned.csv"), index=False)