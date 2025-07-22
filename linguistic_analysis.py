import re
import string

from gensim.utils import simple_preprocess
from nltk import WordNetLemmatizer
import pandas as pd

import nltk
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


# Merge both train and test set for analysis
df_train = pd.read_csv('./data/mental_labeled_train.csv').dropna()
df_test = pd.read_csv('./data/mental_labeled_test.csv').dropna()
df = pd.concat([df_train, df_test])

label_wise_df = {
    label: group for label, group in df.groupby('Label')
}


def preprocess(row):
    text = str(row['Title'] + " " + row['Content']).lower()

    # Remove special characters and digits
    text = re.sub(r'\W+', ' ', text)
    # Tokenize and remove stop words
    tokens = [word for word in simple_preprocess(text) if word not in stop_words]

    # Remove punctuation and special characters
    table = str.maketrans('', '', string.punctuation)
    tokens = [token.translate(table) for token in tokens]

    return [WordNetLemmatizer().lemmatize(token) for token in tokens]


for label, group in label_wise_df.items():
    # ------------ Total Hashtag ----------------
    total_hashtag = sum(
        str(row.loc['Title'] + " " + row['Content']).count('#')
        for _, row in group.iterrows()
    )
    print(f'Avg. HashTag count of {label} labeled post: {total_hashtag / len(group)}')

    # ----------- Posts with URL ----------------
    posts_with_url = sum(
        1 if 'http' in str(row.loc['Title'] + " " + row['Content']).lower() else 0
        for _, row in group.iterrows()
    )
    print(f'Avg. URL count of {label} labeled post: {posts_with_url / len(group)}')

    # ----------- Posts Length ------------------
    total_posts_length = sum(
        len(str(row['Content']).lower())
        for _, row in group.iterrows()
    )
    print(f'Avg. character count of {label} labeled post: {total_posts_length/len(group)}')

    # ------------ Tokens Count --------------
    docs = [preprocess(row) for _, row in group.iterrows()]
    total_tokens = sum(len(doc) for doc in docs)
    print(f'Avg. Token count of {label} labeled post: {total_tokens / len(group)}')

    # ----------- Parts-of-Speech -----------------------
    # Define the POS tags for verbs, nouns, pronouns, and adjectives
    verb_tags = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
    noun_tags = {'NN', 'NNS', 'NNP', 'NNPS'}
    pronoun_tags = {'PRP', 'PRP$', 'WP', 'WP$'}
    adjective_tags = {'JJ', 'JJR', 'JJS'}

    pos_tag_count = {}
    for _, row in group.iterrows():
        tokens = nltk.word_tokenize(row["Content"])
        pos_tags = nltk.pos_tag(tokens)
        for _, pos in pos_tags:
            if pos in verb_tags:
                pos_tag_count["verb"] = pos_tag_count.get("verb", 0) + 1
            elif pos in noun_tags:
                pos_tag_count["noun"] = pos_tag_count.get("noun", 0) + 1
            elif pos in pronoun_tags:
                pos_tag_count["pronoun"] = pos_tag_count.get("pronoun", 0) + 1
            elif pos in adjective_tags:
                pos_tag_count["adjective"] = pos_tag_count.get("adjective", 0) + 1

    print(f"{label} POS tag count: ", pos_tag_count)
    print(f"Avg. {label} pos tag count: ", [(k, v / len(group)) for k, v in pos_tag_count.items()])
