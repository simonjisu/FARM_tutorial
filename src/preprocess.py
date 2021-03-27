import os
import re
import emoji
import pandas as pd
from pathlib import Path
from soynlp.normalizer import repeat_normalize

def read_data(path:str, header=None):
    return pd.read_csv(path, sep='\t', header=header)

def clean(x):
    emojis = ''.join(emoji.UNICODE_EMOJI.keys())
    pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')
    url_pattern = re.compile(
        r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')
    
    x = pattern.sub(' ', x)
    x = url_pattern.sub('', x)
    x = x.strip()
    x = repeat_normalize(x, num_repeats=2)
    return x

def preprocess_dataframe(df:pd.DataFrame):
    r"""
    Changed the code
    source from: https://colab.research.google.com/drive/1IPkZo1Wd-DghIOK6gJpcb0Dv4_Gv2kXB
    """

    label_dict = {0:"bad", 1:"good"}
    df['document'] = df['document'].apply(lambda x: clean(str(x)))
    df['label'] = df['label'].apply(label_dict.get)
    return df

if __name__ == "__main__":
    df_train = preprocess_dataframe(read_data("./nsmc/ratings_train.txt", header=0))
    df_test = preprocess_dataframe(read_data("./nsmc/ratings_test.txt", header=0))
    df_train.loc[:, ["label", "document"]].to_csv("./nsmc/train.tsv", sep="\t", index=False)
    df_test.loc[:, ["label", "document"]].to_csv("./nsmc/test.tsv", sep="\t", index=False)
    print("Done!")