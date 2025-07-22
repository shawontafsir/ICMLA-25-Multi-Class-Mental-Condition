# ================= 0. Imports & Setup ==================
import re, string, math, json, itertools
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import textstat

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import silhouette_score, f1_score, pairwise_distances
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.manifold import TSNE
from umap import UMAP
from scipy.stats import kruskal
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
sid = SentimentIntensityAnalyzer()

# ============== 1. Load Data ===========================
df = pd.concat([
    pd.read_csv('./data/mental_labeled_train.csv').dropna(),
    pd.read_csv('./data/mental_labeled_test.csv').dropna()
])
labels = sorted(df['Label'].unique())

# ============== 2. NRC Emotion Lexicon (mini) =========
# small helper: returns contingency of 8 basic emotions
NRC_EMO = defaultdict(set)  # word -> {emo1, emo2}
# Load your own NRC file; here we fake a tiny demo subset:
demo_words = {
    "fear": ["panic", "scared", "trembling"],
    "anger": ["hate", "anger", "furious"],
    "joy": ["happy", "joy", "delight"],
    "sadness": ["sad", "hopeless", "unhappy"]
}
for emo, words in demo_words.items():
    for w in words: NRC_EMO[w].add(emo)
EMO_CATS = sorted({e for s in NRC_EMO.values() for e in s})


# ============== 3. Feature Extraction ==================
def extract_features(text):
    tokens = nltk.word_tokenize(text.lower())
    words = [t for t in tokens if t.isalpha()]
    N = len(words) or 1  # avoids zero-division
    pos = nltk.pos_tag(words)

    # surface / lexical
    f = {
        'len_tokens': N, 'type_token': len(set(words)) / N,
        'avg_wordlen': sum(map(len, words)) / N,
        'first_pron': sum(1 for w, _ in pos if w == 'i') / N,
        'verb_ratio': sum(1 for _, tag in pos if tag.startswith('VB')) / N
    }
    # pronouns / tense
    past = sum(1 for _, tag in pos if tag == 'VBD')
    pres = sum(1 for _, tag in pos if tag in {'VB', 'VBP', 'VBZ'})
    f['past_present_ratio'] = past / (pres + 1e-6)

    # negations
    f['neg_words'] = sum(1 for w in words if w in {'no', 'not', 'never'}) / N

    # sentiment (VADER)
    vs = sid.polarity_scores(text)
    f['sent_pos'] = vs['pos']
    f['sent_neg'] = vs['neg']
    f['sent_comp'] = vs['compound']

    # readability
    try:
        f['fk_grade'] = textstat.flesch_kincaid_grade(text)
    except:
        f['fk_grade'] = 0.0

    # emotion lexicon counts
    emo_counts = Counter()
    for w in words:
        for emo in NRC_EMO.get(w, []):
            emo_counts[emo] += 1
    for emo in EMO_CATS:
        f[f'emo_{emo}'] = emo_counts[emo] / N

    # punctuation
    f['qmark_density'] = text.count('?') / (N + 1e-6)
    return f


tqdm.pandas(desc="Extracting feats")
feat_df = df['Content'].progress_apply(extract_features).apply(pd.Series).fillna(0.0)
X_ling = feat_df.values
y = df['Label'].values

# ============== 4. Statistical Tests ==================
print("Kruskal–Wallis across classes:")
signif = []
for col in feat_df.columns:
    groups = [feat_df[y == lab][col] for lab in labels]
    H, p = kruskal(*groups)
    if p < 0.001: signif.append(col)
    print(f"{col:20s} H={H}  p={p}")
print("\nFeatures with p < 0.001:", signif)


# ------------ 5. Cosine-Similarity Heat-map (TF-IDF) ------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

# 1) Collect one “document” per class by concatenating all posts
class_docs = [
    " ".join(df[df["Label"] == lab]["Content"].astype(str).tolist())
    for lab in labels
]

# 2) Fit a TF-IDF on the concatenated class docs
tfidf = TfidfVectorizer(ngram_range=(1, 2),           # unigrams + bigrams
                        min_df=5,                     # ignore very rare n-grams
                        stop_words="english",
                        max_features=20_000)
X = tfidf.fit_transform(class_docs)                   # shape = (n_classes, n_terms)
X = normalize(X)                                      # L2-normalise row vectors

# 3) Compute class-to-class cosine similarity, then convert to dissimilarity
cos_sim   = cosine_similarity(X)                      # 1 == identical
cos_dist  = 1 - cos_sim                               # 0 == identical; 1 == orthogonal

# 4) Visualise
plt.figure(figsize=(7,6))
sns.heatmap(cos_dist,
            xticklabels=labels,
            yticklabels=labels,
            annot=True, fmt=".2f",
            cmap="mako_r")                            # darker = more distant
plt.title("Pairwise TF-IDF Cosine Dissimilarity (Unigrams+Bigrams)")
plt.tight_layout()
plt.show()



# ============== 5. Divergence Heat-map ================
def unigram_dist(series):
    counts = Counter()
    for txt in series:
        counts.update(re.findall(r'\b\w+\b', txt.lower()))
    total = sum(counts.values()) or 1
    return {w: c / total for w, c in counts.items()}


dists = {lab: unigram_dist(df[df["Label"] == lab]["Content"]) for lab in labels}
js_mat = np.zeros((len(labels), len(labels)))
for i, a in enumerate(labels):
    for j, b in enumerate(labels):
        vocab = set(dists[a]) | set(dists[b])
        p = np.array([dists[a].get(w, 1e-12) for w in vocab])
        q = np.array([dists[b].get(w, 1e-12) for w in vocab])
        js_mat[i, j] = jensenshannon(p, q, base=2)

plt.figure(figsize=(7, 6))
sns.heatmap(js_mat, xticklabels=labels, yticklabels=labels, annot=True, cmap='viridis')
plt.title("Pairwise Jensen–Shannon Divergence (Unigrams)")
plt.tight_layout()
plt.show()


# ============== 6. Dimensionality Plots ==============
def scatter(coords, title):
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=y, palette='tab10',
                    s=12, alpha=0.7, legend=False)
    sil = silhouette_score(X_ling, y)
    plt.title(f"{title}  (silhouette={sil:.2f})")
    plt.tight_layout()
    plt.show()


tsne = TSNE(n_components=2, random_state=0, perplexity=40).fit_transform(X_ling)
scatter(tsne, "t-SNE on Linguistic Features")

umap = UMAP(n_neighbors=40, min_dist=0.1, metric='euclidean', random_state=0).fit_transform(X_ling)
scatter(umap, "UMAP on Linguistic Features")

# ============== 7. Baseline Logistic Regression =======
pipeline = make_pipeline(StandardScaler(with_mean=False),
                         LogisticRegression(max_iter=4000, solver='saga',
                                            penalty='l2', multi_class='multinomial'))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
f1 = cross_val_score(pipeline, X_ling, y, cv=cv, scoring='f1_macro')
print(f"Scaled Logistic Regression macro-F1 = {f1.mean():.3f}")

# ============== 8. Hybrid TF-IDF + Linguistics ========
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000,
                             min_df=5, stop_words='english')
X_tfidf = vectorizer.fit_transform(df["Content"])
from scipy.sparse import hstack, csr_matrix

X_full = hstack([X_tfidf, csr_matrix(X_ling)])

pipe_full = make_pipeline(StandardScaler(with_mean=False, copy=False),
                          LogisticRegression(max_iter=3000, solver='saga',
                                             penalty='l2', multi_class='multinomial'))
f1_full = cross_val_score(pipe_full, X_full, y, cv=cv, scoring='f1_macro')
print(f"Hybrid TF-IDF + Ling macro-F1 = {f1_full.mean():.3f}")
