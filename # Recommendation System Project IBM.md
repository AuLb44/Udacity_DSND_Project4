# Recommendation System Project: IBM Community — Readable Guide

This markdown document is a human-readable, well-formatted version of the completed recommendations notebook.  
It contains descriptions, the required functions and example code snippets to implement the project tasks, and guidance on evaluation and next steps.

Use this file as the primary README-style guide for the project. If you want the runnable notebook (.ipynb) version, I already prepared that and can re-attempt a push to your repo when you're ready.

---

## Table of Contents

- Introduction
- Rubric checklist
- Getting started (dependencies & dataset)
- Part I — Exploratory Data Analysis (EDA)
- Part II — Rank-Based Recommendations
- Part III — User-User Collaborative Filtering
- Part IV — Content-Based Recommendations (TF-IDF + Clustering)
- Part V — Matrix Factorization (SVD)
- Evaluation & Discussion
- Tips to make the project stand out
- Appendix: Required function names & key variables
- How to add this file to GitHub

---

## Introduction

This project builds multiple recommendation systems on IBM Watson Studio interaction data:
- Simple popularity-based ranking (baseline / cold-start).
- User-user collaborative filtering.
- Content‑based recommendations using TF‑IDF + LSA + KMeans.
- Matrix factorization (SVD) to find latent similarities.

The original notebook tests rely on specific function names and variable names. This guide describes the functions and includes example implementations.

---

## Rubric checklist (what to satisfy)

- Code passes tests and uses required function/variable names.
- Good docstrings and modular functions.
- Part I & II: compute correct EDA values and top-article functions.
- Part III: build user-item matrix, find similar users, produce recommendations, and improve consistency.
- Part IV: TF-IDF → LSA → KMeans content clusters; create content recommendations.
- Part V: SVD, pick number of features, find similar articles via latent space.
- Provide evaluation strategy and tradeoffs.

---

## Getting started

Dependencies (commonly available via pip):
- pandas, numpy
- matplotlib
- scikit-learn

Dataset path expected (relative to notebook):
- `data/user-item-interactions.csv`

Example import block:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
```

Load data:
```python
df = pd.read_csv(
    'data/user-item-interactions.csv',
    dtype={'article_id': int, 'title': str, 'email': str}
)
```

---

## Part I — Exploratory Data Analysis (EDA)

Goals:
- Identify missing values and fix them (fill `email` NaNs with `"unknown_user"`).
- Calculate interaction counts and descriptive stats.
- Produce `median_val`, `max_views_by_user`, `user_article_interactions`, `unique_articles`, `total_articles`, `unique_users`, `most_viewed_article_id`, `max_views`.

Example code:
```python
# Fill missing emails
df['email'] = df['email'].fillna('unknown_user')

# Per-user interaction counts
user_interaction_counts = df.groupby('email').size()
median_val = int(user_interaction_counts.median())
max_views_by_user = int(user_interaction_counts.max())
user_article_interactions = int(df.shape[0])
unique_articles = int(df['article_id'].nunique())
total_articles = unique_articles  # if no separate article list
unique_users = int(df['email'].nunique())

# Most viewed article info
article_counts = df['article_id'].value_counts()
most_viewed = int(article_counts.idxmax())
max_views = int(article_counts.max())
most_viewed_article_id = f"{most_viewed}.0"  # some tests expect this string format
```

Mapping email to user_id (required by many tests):
```python
def email_mapper(df=df):
    coded_dict = {email: num for num, email in enumerate(df['email'].unique(), start=1)}
    return [coded_dict[val] for val in df['email']]

df['user_id'] = email_mapper(df)
del df['email']
```

Visualizations:
- Histogram of per-user counts (use log-scale if skewed).
- Bar plot of article popularity.

---

## Part II — Rank-Based Recommendations

Goal: Recommend the top-n most popular articles (by number of interactions).

Required functions:
- `get_top_articles(n, df=df)` → returns list of top-n article titles.
- `get_top_article_ids(n, df=df)` → returns list of top-n article ids (integers).

Example implementations:
```python
def get_top_articles(n, df=df):
    top = df.groupby('title').size().sort_values(ascending=False).head(n)
    return list(top.index)

def get_top_article_ids(n, df=df):
    top = df.groupby('article_id').size().sort_values(ascending=False).head(n)
    return list(top.index.astype(int))
```

Usage:
```python
top_10_titles = get_top_articles(10)
top_10_ids = get_top_article_ids(10)
```

This method is ideal for new users (cold start) because it requires no user history.

---

## Part III — User-User Collaborative Filtering

Goal: Build user-item matrix and make recommendations based on similarity between users.

1) Build user-item matrix:
- Function: `create_user_item_matrix(df, fill_value=0)`
- Output: `user_item` DataFrame where rows are `user_id`, columns are `article_id`, and values are 1 if user interacted with article else 0.

Example:
```python
def create_user_item_matrix(df, fill_value=0):
    user_item = df.groupby(['user_id', 'article_id']).size().unstack(fill_value=0)
    user_item = (user_item > 0).astype(int)
    user_item.columns = user_item.columns.astype(int)
    return user_item

user_item = create_user_item_matrix(df)
```

2) Find similar users:
- Function: `find_similar_users(user_id, user_item=user_item, include_similarity=False)`
- Use cosine similarity on user vectors; return ordered user ids (highest similarity first). Exclude the user itself.

3) Helper functions:
- `get_article_names(article_ids, df=df)` → map article ids → titles (order preserved).
- `get_ranked_article_unique_counts(article_ids, user_item=user_item)` → return list of `[article_id, num_users]`, sorted desc by number of users.
- `get_user_articles(user_id, user_item=user_item)` → return `(article_ids, article_names)` seen by the user.

4) Basic user-user recommendations:
- `user_user_recs(user_id, m=10)` → iterate through most similar users and collect articles they saw that the target user hasn't, until m recommendations are collected.

Example of `get_article_names`:
```python
def get_article_names(article_ids, df=df):
    df_unique = df.drop_duplicates(subset=['article_id','title'])[['article_id','title']]
    id_to_title = dict(zip(df_unique['article_id'], df_unique['title']))
    return [id_to_title.get(int(a), '') for a in article_ids]
```

5) Improved consistency:
- `get_top_sorted_users(user_id, user_item=user_item)` → return DataFrame `neighbor_id`, `similarity`, `num_interactions` sorted by similarity desc, then num_interactions desc.
- `user_user_recs_part2(user_id, m=10)` → iterate neighbors in that order and choose neighbor articles sorted by global popularity; return `recs` and `rec_names`. This ensures deterministic choices when ties occur.

Deliverable variables (for tests):
- `user1_most_sim`, `user2_6th_sim`, `user131_10th_sim`
- `new_user_recs` — list of top 10 article ids to recommend to a brand-new user (should be the top 10 popular ids).

---

## Part IV — Content-Based Recommendations

Goal: Use article titles to build clusters of similar articles and recommend articles from the same cluster.

Steps:
1) Build unique article DataFrame:
```python
df_unique_articles = df[['article_id','title']].drop_duplicates().reset_index(drop=True)
```

2) Vectorize titles (TF-IDF):
```python
vectorizer = TfidfVectorizer(
    max_df=0.75,
    min_df=5,
    stop_words='english',
    max_features=200
)
X_tfidf = vectorizer.fit_transform(df_unique_articles['title'])
```

3) Dimensionality reduction (LSA):
```python
lsa = make_pipeline(TruncatedSVD(n_components=50), Normalizer(copy=False))
X_lsa = lsa.fit_transform(X_tfidf)
explained_variance = lsa.steps[0][1].explained_variance_ratio_.sum()
```

4) KMeans clustering:
```python
kmeans = KMeans(n_clusters=50, max_iter=50, n_init=5, random_state=42).fit(X_lsa)
article_cluster_map = dict(zip(df_unique_articles['article_id'], kmeans.labels_))
df['title_cluster'] = df['article_id'].map(article_cluster_map)
```

5) Content-based functions:
- `get_similar_articles(article_id, df=df)` → return list of article_ids in same cluster (excluding input).
- `make_content_recs(article_id, n, df=df)` → return top n similar article ids, ranked by popularity. Also return article titles.

Example:
```python
def get_similar_articles(article_id, df=df):
    if article_id not in article_cluster_map:
        return []
    cluster = article_cluster_map[article_id]
    return [int(a) for a, c in article_cluster_map.items() if c == cluster and a != article_id]

def make_content_recs(article_id, n, df=df):
    similar = get_similar_articles(article_id, df)
    ranked = get_ranked_article_unique_counts(similar, user_item)
    top_n = [int(a) for a, _ in ranked[:n]]
    return top_n, get_article_names(top_n, df=df)
```

Notes & improvements:
- Titles are short and noisy. Additional textual data (article body/abstract/tags) would considerably improve results.
- Consider using sentence embeddings (e.g., SBERT) for better semantic similarity.

---

## Part V — Matrix Factorization (SVD)

Goal: Use SVD on the user-item matrix to learn latent features and compute item similarities in that space.

1) Compute SVD:
```python
svd = TruncatedSVD(n_components=user_item.shape[1], n_iter=5, random_state=42)
u = svd.fit_transform(user_item)
v = svd.components_
s = svd.singular_values_
```

2) Choose number of latent features:
- Experiment with different values of `k` (e.g., 10..200) and evaluate reconstruction using accuracy, precision, recall on binary matrix reconstruction:
```python
u_new, vt_new = u[:, :k], v[:k, :]
user_item_est = abs(np.around(np.dot(u_new, vt_new))).astype(int)
user_item_est = np.clip(user_item_est, 0, 1)
# compute metrics: accuracy_score, precision_score, recall_score
```
- Choose `k` balancing performance, overfitting, and computational cost. The rubric suggests plotting metrics vs. k and picking a region of diminishing returns.

3) Similar items in latent space:
- `get_svd_similar_article_ids(article_id, vt, user_item=user_item, include_similarity=False)`:
  - Compute cosine similarity between columns of `vt.T` (each column corresponds to an article vector).
  - Return sorted article ids (exclude the article itself). Optionally include similarities.

Example:
```python
from sklearn.metrics.pairwise import cosine_similarity

def get_svd_similar_article_ids(article_id, vt, user_item=user_item, include_similarity=False):
    article_idx = list(user_item.columns).index(article_id)
    cos_sim = cosine_similarity(vt.T)
    sim_scores = list(enumerate(cos_sim[article_idx]))
    idx_article_pairs = [(user_item.columns[i], score) for i, score in sim_scores]
    sorted_pairs = sorted(idx_article_pairs, key=lambda x: x[1], reverse=True)
    sorted_pairs = [p for p in sorted_pairs if p[0] != article_id]
    if include_similarity:
        return sorted_pairs
    return [int(p[0]) for p in sorted_pairs]
```

4) Create reduced vt (for example, k=200):
```python
k = min(200, v.shape[0])
vt_new = v[:k, :]
rec_articles = get_svd_similar_article_ids(4, vt_new)[:10]
```

---

## Evaluation & Discussion

Offline evaluation:
- Use train/test or time-based splits.
- Metrics: precision@k, recall@k, F1@k, MAP@k, NDCG@k.
- Use negative sampling if needed for ranking tasks.

Online evaluation:
- A/B testing measuring CTR, conversions, session duration, retention.

Tradeoffs between methods:
- Rank-based: simple, robust for cold-start users, non-personalized.
- User-user CF: personalized, intuitive, but struggles with sparsity and scalability.
- Content-based: handles new items, depends on quality of item metadata.
- Matrix factorization: captures latent structure and often gives better personalized results but needs periodic retraining and careful selection of hyperparameters.

Recommendations for cold-start:
- New user: ask onboarding questions or recommend popular items (rank-based).
- New item: use content-based methods.

---

## Tips to make the project stand out

- Wrap code into a Python class (fit/recommend/update) and add tests.
- Create a Flask app with endpoints:
  - GET /recommend/user/<user_id>
  - GET /recommend/article/<article_id>
- Use richer item features (tags, body, author) and pretrained embeddings (SBERT).
- Package the engine to be pip-installable.

---

## Appendix — Quick reference of required function names & key variables

- EDA variables:
  - median_val, max_views_by_user, user_article_interactions, unique_articles, total_articles, unique_users, most_viewed_article_id, max_views
- Rank-based:
  - get_top_articles(n, df=df)
  - get_top_article_ids(n, df=df)
- Collaborative filtering:
  - create_user_item_matrix(df, fill_value=0)
  - find_similar_users(user_id, user_item=user_item, include_similarity=False)
  - get_article_names(article_ids, df=df)
  - get_ranked_article_unique_counts(article_ids, user_item=user_item)
  - get_user_articles(user_id, user_item=user_item)
  - user_user_recs(user_id, m=10)
  - get_top_sorted_users(user_id, user_item=user_item)
  - user_user_recs_part2(user_id, m=10)
  - user1_most_sim, user2_6th_sim, user131_10th_sim
  - new_user_recs (list of article ids)
- Content-based:
  - df_unique_articles
  - X_tfidf, X_lsa, explained_variance
  - article_cluster_map
  - get_similar_articles(article_id, df=df)
  - make_content_recs(article_id, n, df=df)
- Matrix factorization:
  - u, s, v (from SVD), vt_new (reduced vt)
  - get_svd_similar_article_ids(article_id, vt, user_item=user_item, include_similarity=False)
