import pandas as pd
from sklearn.model_selection import train_test_split

disorder_files = [
    'adhd', 'anxiety', 'bipolar', 'cptsd', 'depression', 'schizophrenia'
]

control_files = [
    'bicycletouring', 'confidence', 'politics', 'sports', 'travel', 'geopolitics'
]

df_disorder_list = [pd.read_csv(f'./data/{disorder_file}_posts.csv') for disorder_file in disorder_files]
df_control_list = [pd.read_csv(f'./data/{control_file}_posts.csv') for control_file in control_files]

# Make the control group distinct from mental conditions regarding author commonality
df_disorder = pd.concat(df_disorder_list, ignore_index=True)
df_control = pd.concat(df_control_list, ignore_index=True)
common_author = set(df_disorder.merge(df_control, on='Author', how='inner')['Author'])
df_control_unique_list = [_df[_df['Author'].isin(common_author) == False] for _df in df_control_list]

# Make each mental condition distinct regarding user commonality
df_disorder_unique_list = []
for i, _df_disorder in enumerate(df_disorder_list):
    _df_merged = pd.concat([*df_disorder_list[:i], *df_disorder_list[i+1:], df_control], ignore_index=True)
    _common_author = set(_df_merged.merge(_df_disorder, on='Author', how='inner')['Author'])
    df_disorder_unique_list.append(_df_disorder[_df_disorder['Author'].isin(_common_author) == False])

df_labeled_list = [_df.sample(15000).assign(Label=disorder_files[_i]) for _i, _df in enumerate(df_disorder_unique_list)] + [pd.concat(df_control_unique_list, ignore_index=True).sample(15000).assign(Label="Control")]
df_labeled = pd.concat(df_labeled_list, ignore_index=True).sample(frac=1)
df_train, df_test = train_test_split(df_labeled, stratify=df_labeled["Label"], test_size=0.2, random_state=42)
df_train.to_csv('./data/mental_labeled_train.csv', index=False)
df_test.to_csv('./data/mental_labeled_test.csv', index=False)
