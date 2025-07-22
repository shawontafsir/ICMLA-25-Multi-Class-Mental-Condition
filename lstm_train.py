import os
import random
import re
import string
import time
from collections import Counter
from itertools import chain

import pandas as pd
import torch
import torch.nn as nn
from nltk import word_tokenize, WordNetLemmatizer
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer

from lstm import LSTM

import nltk

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

from nltk.corpus import wordnet, stopwords


# df = pd.read_csv('./documents/labeled_suicidewatch_posts_pushshiftapi.csv')
# df = df.sort_values(by='Created_UTC', ascending=False)
# df = df[df['Content'].str.len() >= 1]
# df1 = df[df['Sentiment_Label'].isin(['positive'])].iloc[:5000]
# df2 = df[df['Sentiment_Label'].isin(['negative'])].iloc[:5000]
# df = pd.concat([df1, df2])
#
# # Encode target labels using LabelEncoder
# label_encoder = LabelEncoder()
# df['label_encoded'] = label_encoder.fit_transform(df['Sentiment_Label'])
#
# # Split documents into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(
#     df['Content'], df['label_encoded'],
#     test_size=0.15, random_state=42
# )


# Define documents augmentation functions
def synonym_replacement(sentence, n=1):
    words = sentence.split()
    new_words = words.copy()
    for _ in range(n):
        random_word = random.choice(words)
        synonyms = wordnet.synsets(random_word)
        if synonyms:
            synonym = random.choice(synonyms).lemma_names()[0]
            new_words = [synonym if word == random_word else word for word in new_words]
    return ' '.join(new_words)


def create_train_loader(_tokenizer, _x_train, _y_train, _batch_size=16):
    _x_train_tokens = _tokenizer(_x_train.tolist(), padding=True, truncation=True, return_tensors='pt')
    _x_train_input_ids = _x_train_tokens.input_ids
    # _x_train_attention_mask = _x_train_tokens.attention_mask
    _y_train_tensor = torch.tensor(_y_train.values, dtype=torch.long)

    # Define train dataset and documents loader
    _train_dataset = TensorDataset(_x_train_input_ids, _y_train_tensor)
    _train_loader = DataLoader(_train_dataset, batch_size=_batch_size, shuffle=True)

    return _train_loader


def create_validation_loader(_tokenizer, _x_validation, _y_validation, _batch_size=16):
    _x_validation_tokens = _tokenizer(_x_validation.tolist(), padding=True, truncation=True, return_tensors='pt')
    _x_validation_input_ids = _x_validation_tokens.input_ids
    # _x_validation_attention_mask = _x_validation_tokens.attention_mask
    _y_validation_tensor = torch.tensor(_y_validation.values, dtype=torch.long)

    # Define train dataset and documents loader
    _validation_dataset = TensorDataset(_x_validation_input_ids, _y_validation_tensor)
    _validation_loader = DataLoader(_validation_dataset, batch_size=_batch_size, shuffle=False)

    return _validation_loader, _y_validation_tensor


def create_test_loader(_tokenizer, _x_test, _y_test, _batch_size=16):
    _x_test_tokens = _tokenizer(_x_test.tolist(), padding=True, truncation=True, return_tensors='pt')
    _x_test_input_ids = _x_test_tokens.input_ids
    # _x_test_attention_mask = _x_test_tokens.attention_mask
    _y_test_tensor = torch.tensor(_y_test.values, dtype=torch.long)

    # Define test dataset and documents loader
    _test_dataset = TensorDataset(_x_test_input_ids)
    _test_loader = DataLoader(_test_dataset, batch_size=_batch_size, shuffle=False)

    return _test_loader, _y_test_tensor


def train_with_cross_validation(_tokenizer, _model_params, _config, features, labels, _k_folds):
    # Define k-fold cross-validation
    skf = StratifiedKFold(n_splits=_k_folds, shuffle=True, random_state=42)

    # Perform k-fold cross-validation
    for fold, (train_idx, valid_idx) in enumerate(skf.split(features, labels), 1):
        print(f'Fold {fold}/{_k_folds}')

        # Split documents into train and validation sets
        _X_train, _X_validation = features.iloc[train_idx], features.iloc[valid_idx]
        _y_train, _y_validation = labels.iloc[train_idx], labels.iloc[valid_idx]

        train_loader = create_train_loader(_tokenizer, _X_train, _y_train, _batch_size=16)
        validation_loader, y_validation_tensor = create_validation_loader(_tokenizer, _X_validation, _y_validation)

        _model = LSTM(**_model_params)
        train(_model, _config, train_loader, validation_loader, y_validation_tensor, fold)


def train_with_non_transformer_embedding_with_cross_validation(_model_params, _config, _features_tensor, _labels_tensor, _k_folds):
    # Create a stratified K-fold splitter
    skf = StratifiedKFold(n_splits=_k_folds, shuffle=True, random_state=42)

    # Perform k-fold cross-validation
    for _fold, (train_idx, val_idx) in enumerate(skf.split(_features_tensor, _labels_tensor), 1):
        _train_dataset = TensorDataset(_features_tensor[train_idx], _labels_tensor[train_idx])
        _val_dataset = TensorDataset(_features_tensor[val_idx], _labels_tensor[val_idx])

        _train_loader = DataLoader(_train_dataset, batch_size=16)
        _val_loader = DataLoader(_val_dataset, batch_size=16)

        _model = LSTM(**_model_params)
        train(_model, _config, _train_loader, _val_loader, _labels_tensor[val_idx], _fold)


def train(_model: LSTM, _config, _train_loader, _validation_loader, _y_validation_tensor, _fold=None):
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    lr = _config["lr"]
    weight_decay = _config["weight_decay"]
    epochs = _config["epochs"]

    # Define optimizer
    optimizer = torch.optim.AdamW(_model.parameters(), lr=lr, weight_decay=weight_decay)
    _scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.1, verbose=True
    )
    criterion = nn.CrossEntropyLoss()
    _model = _model.to(device)

    # LSTM with BERT Training loop
    scaler = torch.cuda.amp.GradScaler()
    train_losses, valid_losses, min_valid_loss = list(), list(), float("infinity")

    # Set patience for early stopping
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        tqdm_train_loader = tqdm(
            _train_loader, desc=f"{f'Fold {_fold}: ' if _fold else ''}Epoch {epoch + 1}/{epochs}"
        )
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss = 0

        # Training
        _model.train()
        for batch in tqdm_train_loader:
            torch.cuda.empty_cache()
            optimizer.zero_grad()

            input_ids, attention_mask, labels = batch[0].to(device), None, batch[1].to(device)


            outputs = _model(input_ids)
            loss = criterion(outputs, labels)

            # backward propagation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            tqdm_train_loader.set_postfix(train_loss=loss.item(), lr=current_lr)

        train_losses.append(train_loss / len(tqdm_train_loader))

        # Validation
        tqdm_valid_loader = tqdm(
            _validation_loader,
            desc=f"Evaluating validation dataset of {len(_validation_loader.dataset)} instances"
        )
        predicted_labels = torch.IntTensor([]).to(device)
        valid_loss = 0

        _model.eval()
        with torch.no_grad():
            for batch in tqdm_valid_loader:
                input_ids, attention_mask, labels = batch[0].to(device), None, batch[1].to(device)

                logits = _model(input_ids)
                loss = criterion(logits, labels)
                valid_loss += loss.item()
                predicted_labels = torch.cat((predicted_labels, torch.argmax(logits, dim=1)), 0)

            valid_losses.append(valid_loss / len(tqdm_valid_loader))
            _scheduler.step(valid_losses[-1])

        # Calculate accuracy
        accuracy = accuracy_score(_y_validation_tensor.cpu().numpy(), predicted_labels.cpu().numpy())
        print(f"Accuracy: {accuracy}")

        f1 = f1_score(_y_validation_tensor.cpu().numpy(), predicted_labels.cpu().numpy(), average='macro')
        print(f"F1 score: {f1}")

        precision = precision_score(_y_validation_tensor.cpu().numpy(), predicted_labels.cpu().numpy(), average='macro')
        print(f"Precision score: {precision}")

        recall = recall_score(_y_validation_tensor.cpu().numpy(), predicted_labels.cpu().numpy(), average='macro')
        print(f"Recall score: {recall}")
        print(f"Train and validation losses: {train_losses[-1]}, {valid_losses[-1]}")

        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            patience_counter = 0

            checkpoint = {
                "state_dict": _model.state_dict(),
                "optimizer": optimizer.state_dict()
            }

            save_checkpoint(
                checkpoint,
                filename=f"{_model.class_name}_{f'{_fold}_' if _fold else ''}checkpoint.tar"
            )
        else:
            patience_counter += 1

        # Early stopping condition
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print(f"{f'Fold {_fold}: ' if _fold else ''}Train losses per epoch: {train_losses}")
    print(f"{f'Fold {_fold}: ' if _fold else ''}Valid losses per epoch: {valid_losses}")


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(_model, filepath):
    print("=> Loading checkpoint")
    checkpoint = torch.load(filepath)
    _model.load_state_dict(checkpoint["state_dict"])


def evaluate_non_transformer_embedding_with_cross_validation(_model: LSTM, _test_loader, _y_test_tensor, _k_folds):
    _predictions = []
    for fold in range(1, _k_folds + 1):
        _predictions.append(evaluate(_model, _test_loader, _y_test_tensor, fold))

    # Take majority vote for classification
    _weighted_predicted_labels = []
    for j in range(len(_predictions[0])):
        # A list containing predicted labels of each model for a particular instance indexed at j
        _labels = [_predictions[i][j] for i in range(len(_predictions))]

        # Take the label having maximum occurrence in each model fold evaluation
        _weighted_predicted_labels.append(max(_labels, key=_labels.count))

    # Calculate accuracy
    accuracy = accuracy_score(_y_test_tensor.cpu().tolist(), _weighted_predicted_labels)
    print(f"Cross Validation Accuracy: {accuracy}")

    # Calculate F1-score
    f1 = f1_score(_y_test_tensor.cpu().tolist(), _weighted_predicted_labels)
    print(f"Cross Validation F1-score: {f1}")

    precision = precision_score(_y_test_tensor.cpu().tolist(), _weighted_predicted_labels)
    print(f"Cross Validation Precision: {precision}")

    recall = recall_score(_y_test_tensor.cpu().tolist(), _weighted_predicted_labels)
    print(f"Cross Validation Recall: {recall}")

    return _weighted_predicted_labels


def evaluate(_model: LSTM, _test_loader, _y_test_tensor, fold=None):
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _model = _model.to(device)
    load_checkpoint(_model, f"{_model.class_name}_{f'{fold}_' if fold else ''}checkpoint.tar")

    # Evaluation
    tqdm_test_loader = tqdm(
        _test_loader,
        desc=f"{f'Fold {fold}: ' if fold else ''}Evaluating test dataset of {len(_test_loader.dataset)} instances"
    )
    predicted_labels = torch.IntTensor([]).to(device)

    _model.eval()
    with torch.no_grad():
        for batch in tqdm_test_loader:
            torch.cuda.empty_cache()
            input_ids, attention_mask = batch[0].to(device), None
            logits = _model(input_ids)
            predicted_labels = torch.cat((predicted_labels, torch.argmax(logits, dim=1)), 0)

    # Calculate accuracy
    accuracy = accuracy_score(_y_test_tensor.cpu().numpy(), predicted_labels.cpu().numpy())
    print(f"{f'Fold {fold}: ' if fold else ''}Accuracy: {accuracy}")
    cm = confusion_matrix(_y_test_tensor.cpu().numpy(), predicted_labels.cpu().numpy())
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    print(per_class_accuracy)

    # Calculate F1-score
    f1 = f1_score(_y_test_tensor.cpu().numpy(), predicted_labels.cpu().numpy(), average='macro')
    print(f"{f'Fold {fold}: ' if fold else ''}F1-score: {f1}")
    f1 = f1_score(_y_test_tensor.cpu().numpy(), predicted_labels.cpu().numpy(), average=None)
    print(f"{f'Fold {fold}: ' if fold else ''}F1-score: {f1}")

    precision = precision_score(_y_test_tensor.cpu().numpy(), predicted_labels.cpu().numpy(), average='macro')
    print(f"Precision: {precision}")
    precision = precision_score(_y_test_tensor.cpu().numpy(), predicted_labels.cpu().numpy(), average=None)
    print(f"Precision: {precision}")

    recall = recall_score(_y_test_tensor.cpu().numpy(), predicted_labels.cpu().numpy(), average='macro')
    print(f"Recall: {recall}")
    recall = recall_score(_y_test_tensor.cpu().numpy(), predicted_labels.cpu().numpy(), average=None)
    print(f"Recall: {recall}")

    # # Inverse transform labels to original classes
    # predicted_labels_original = label_encoder.inverse_transform(predicted_labels.cpu().numpy())
    # print("Predicted labels:", predicted_labels_original)

    return predicted_labels.tolist()

def evaluate_with_cross_validation(_tokenizer, _model_params, _x_test, _y_test, _k_folds):
    _test_loader, _y_test_tensor = create_test_loader(_tokenizer, _x_test, _y_test)
    _model = LSTM(**_model_params)

    predictions = []
    for fold in range(1, _k_folds + 1):
        predictions.append(evaluate(_model, _test_loader, _y_test_tensor, fold))

    # Take majority vote for classification
    weighted_predicted_labels = []
    for j in range(len(predictions[0])):
        # A list containing predicted labels of each model for a particular instance indexed at j
        labels = [predictions[i][j] for i in range(len(predictions))]

        # Take the label having maximum occurrence in each model fold evaluation
        weighted_predicted_labels.append(max(labels, key=labels.count))

    # Calculate accuracy
    accuracy = accuracy_score(_y_test_tensor.cpu().tolist(), weighted_predicted_labels)
    print(f"Cross Validation Accuracy: {accuracy}")
    cm = confusion_matrix(_y_test_tensor.cpu().tolist(), weighted_predicted_labels)
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    print(per_class_accuracy)

    # Calculate F1-score
    f1 = f1_score(_y_test_tensor.cpu().tolist(), weighted_predicted_labels, average='macro')
    print(f"Cross Validation F1-score: {f1}")
    f1 = f1_score(_y_test_tensor.cpu().tolist(), weighted_predicted_labels, average=None)
    print(f"Cross Validation F1-score: {f1}")

    precision = precision_score(_y_test_tensor.cpu().tolist(), weighted_predicted_labels, average='macro')
    print(f"Cross Validation Precision: {precision}")
    precision = precision_score(_y_test_tensor.cpu().tolist(), weighted_predicted_labels, average=None)
    print(f"Cross Validation Precision: {precision}")

    recall = recall_score(_y_test_tensor.cpu().tolist(), weighted_predicted_labels, average='macro')
    print(f"Cross Validation Recall: {recall}")
    recall = recall_score(_y_test_tensor.cpu().tolist(), weighted_predicted_labels, average=None)
    print(f"Cross Validation Recall: {recall}")

    return weighted_predicted_labels


def preprocess_text(text, remove_stopwords=True):
    """Cleans and tokenizes text for Word2Vec/GloVe"""
    # Convert to lowercase
    text = text.lower()

    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove punctuation & special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces

    # Tokenize words
    tokens = word_tokenize(text)

    # Remove stopwords (optional)
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]

    # Remove punctuation and special characters
    table = str.maketrans('', '', string.punctuation)
    tokens = [token.translate(table) for token in tokens]

    return [WordNetLemmatizer().lemmatize(token) for token in tokens]


def build_or_update_vocab(sentences, existing_vocab=None, min_freq=1):
    """
    Build a new vocabulary or update an existing one from a list of sentences.

    Args:
        sentences (list of str): List of sentences to process.
        existing_vocab (dict, optional): Existing vocabulary to update. Defaults to None.
        min_freq (int): Minimum frequency of words to be included.

    Returns:
        dict: Updated vocabulary mapping words to indices.
    """
    # Preprocess all sentences to get lemmatized tokens
    tokens = [preprocess_text(sentence) for sentence in sentences]

    # Count word frequencies
    word_counts = Counter(chain(*tokens))

    if existing_vocab is None:
        existing_vocab = {}  # Create new vocabulary if not provided

    # Get current vocabulary size (excluding '<PAD>' if it exists)
    next_index = len(existing_vocab) - int('<PAD>' in existing_vocab)

    # Add new words that meet min_freq
    for word, count in word_counts.items():
        if word not in existing_vocab and count >= min_freq:
            existing_vocab[word] = next_index
            next_index += 1

    # Ensure '<PAD>' is the last index
    existing_vocab['<PAD>'] = next_index

    return existing_vocab


def encode_sentences(_sentences, _vocab):
    encoded_sentences = []
    for _sentence in _sentences:
        # Preprocess a sentence to get lemmatized tokens
        tokens = preprocess_text(_sentence)
        encoded_sentence = [_vocab.get(word, _vocab['<PAD>']) for word in tokens]
        encoded_sentences.append(encoded_sentence)
    return encoded_sentences


if __name__ == "__main__":
    label_encoder = LabelEncoder()
    df_train = pd.read_csv("./data/mental_labeled_train.csv").sample(frac=1, random_state=42).reset_index(drop=True)
    df_train["Text"] = df_train.apply(lambda row: row['Title'] + ". " + row['Content'], axis=1)
    df_train['label_encoded'] = label_encoder.fit_transform(df_train['Label'])
    X_train, y_train = df_train['Text'], df_train['label_encoded']

    df_test = pd.read_csv("./data/mental_labeled_test.csv").sample(frac=1, random_state=42).reset_index(drop=True)
    df_test["Text"] = df_test.apply(lambda row: row['Title'] + ". " + row['Content'], axis=1)
    df_test['label_encoded'] = label_encoder.transform(df_test['Label'])
    X_test, y_test = df_test['Text'], df_test['label_encoded']

    # Split train documents into train and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, stratify=y_train, test_size=0.20, random_state=42
    )

    model_params = {
        'hidden_size': 128,  # Size of LSTM hidden state
        'num_classes': len(label_encoder.classes_)  # Number of output classes
    }
    config = {
        "lr": 1e-6,
        "weight_decay": 1e-2,
        "epochs": 300
    }
    k_folds = 5

    # Tokenize text using BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    print("BiLSTM with Attention:")
    tic = time.perf_counter()
    model_params.update(dict(is_bidirectional=True, has_attention=True))
    train_with_cross_validation(
        tokenizer, model_params, config, pd.concat([X_train, X_valid]), pd.concat([y_train, y_valid]), _k_folds=k_folds
    )
    print(label_encoder.classes_)
    evaluate_with_cross_validation(tokenizer, model_params, X_test, y_test, k_folds)
    toc = time.perf_counter()
    print(f"Trained BiLSTM with Attention model in {toc - tic:0.4f} seconds")
    #
    # print("BiLSTM:")
    # tic = time.perf_counter()
    # model_params.update(dict(is_bidirectional=True, has_attention=False))
    # train_with_cross_validation(
    #     tokenizer, model_params, config, pd.concat([X_train, X_valid]), pd.concat([y_train, y_valid]), _k_folds=k_folds
    # )
    # print(label_encoder.classes_)
    # evaluate_with_cross_validation(tokenizer, model_params, X_test, y_test, k_folds)
    # toc = time.perf_counter()
    # print(f"Trained BiLSTM model in {toc - tic:0.4f} seconds")
    #
    # print("LSTM with Attention:")
    # tic = time.perf_counter()
    # model_params.update(dict(is_bidirectional=False, has_attention=True))
    # train_with_cross_validation(
    #     tokenizer, model_params, config, pd.concat([X_train, X_valid]), pd.concat([y_train, y_valid]), _k_folds=k_folds
    # )
    # print(label_encoder.classes_)
    # evaluate_with_cross_validation(tokenizer, model_params, X_test, y_test, k_folds)
    # toc = time.perf_counter()
    # print(f"Trained LSTM with Attention model in {toc - tic:0.4f} seconds")

    # print("LSTM:")
    # tic = time.perf_counter()
    # model_params.update(dict(is_bidirectional=False, has_attention=False))
    # train_with_cross_validation(
    #     tokenizer, model_params, config, pd.concat([X_train, X_valid]), pd.concat([y_train, y_valid]), _k_folds=k_folds
    # )
    # print(label_encoder.classes_)
    # evaluate_with_cross_validation(tokenizer, model_params, X_test, y_test, k_folds)
    # toc = time.perf_counter()
    # print(f"Trained LSTM model in {toc - tic:0.4f} seconds")

    # Train dataset
    train_vocab = build_or_update_vocab(pd.concat([X_train, X_valid])[0])
    encoded_sentences = encode_sentences(pd.concat([X_train, X_valid]), train_vocab)
    max_len = max(len(sentence) for sentence in encoded_sentences)
    padded_sentences = [sentence + [train_vocab['<PAD>']] * (max_len - len(sentence)) for sentence in encoded_sentences]
    sentences_tensor = torch.tensor(padded_sentences, dtype=torch.long)
    labels_tensor = torch.tensor(pd.concat([y_train, y_valid]).values, dtype=torch.long)

    # Perform stratified split
    train_indices, validation_indices, _, _ = train_test_split(
        range(len(sentences_tensor)), labels_tensor, test_size=0.2, stratify=labels_tensor, random_state=42
    )

    train_dataset = TensorDataset(sentences_tensor[train_indices], labels_tensor[train_indices])
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Validation dataset
    # encoded_sentences = encode_sentences(X_valid, vocab)
    # max_len = max(len(sentence) for sentence in encoded_sentences)
    # padded_sentences = [sentence + [vocab['<PAD>']] * (max_len - len(sentence)) for sentence in encoded_sentences]
    # sentences_tensor = torch.tensor(padded_sentences, dtype=torch.long)
    # labels_tensor = torch.tensor(y_valid.values, dtype=torch.long)

    validation_dataset = TensorDataset(sentences_tensor[validation_indices], labels_tensor[validation_indices])
    validation_loader = DataLoader(validation_dataset, batch_size=16, shuffle=False)
    y_validation_tensor = labels_tensor[validation_indices]

    # Test dataset
    test_vocab = build_or_update_vocab(X_test, train_vocab)
    encoded_sentences = encode_sentences(X_test, test_vocab)
    max_len = max(len(sentence) for sentence in encoded_sentences)
    padded_sentences = [sentence + [test_vocab['<PAD>']] * (max_len - len(sentence)) for sentence in encoded_sentences]
    sentences_tensor = torch.tensor(padded_sentences, dtype=torch.long)
    labels_tensor = torch.tensor(y_test.values, dtype=torch.long)

    test_dataset = TensorDataset(sentences_tensor, labels_tensor)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)


    # glove_path = "embeddings/glove.6B.300d.txt"
    #
    # print("BiLSTMWithAttentionWithGlove:")
    # model_params.update(
    #     dict(
    #         embedding_type='glove', pretrained_embedding_path=glove_path, vocab=train_vocab, embedding_dim=300,
    #         is_bidirectional=True, has_attention=True
    #     )
    # )
    # model = LSTM(**model_params)
    # tic = time.perf_counter()
    # train(model, config, train_loader, validation_loader, y_validation_tensor)
    # # train_with_non_transformer_embedding_with_cross_validation(
    # #     model_params, config, torch.cat([train_dataset.tensors[0], validation_dataset.tensors[0]]),
    # #     torch.cat([train_dataset.tensors[1], validation_dataset.tensors[1]], dim=0), 5
    # # )
    # toc = time.perf_counter()
    # print(f"Trained {model.class_name} model in {toc - tic:0.4f} seconds")
    # evaluate(model, test_loader, y_test_tensor)
    # # evaluate_non_transformer_embedding_with_cross_validation(model, test_loader, y_test_tensor, 5)
    #
    #
    # print("BiLSTMWithGlove:")
    # model_params.update(
    #     dict(
    #         embedding_type='glove', pretrained_embedding_path=glove_path, vocab=train_vocab, embedding_dim=300,
    #         is_bidirectional=True, has_attention=False
    #     )
    # )
    # model = LSTM(**model_params)
    # tic = time.perf_counter()
    # train(model, config, train_loader, validation_loader, y_validation_tensor)
    # # train_with_non_transformer_embedding_with_cross_validation(
    # #     model_params, config, torch.cat([train_dataset.tensors[0], validation_dataset.tensors[0]]),
    # #     torch.cat([train_dataset.tensors[1], validation_dataset.tensors[1]], dim=0), 5
    # # )
    # toc = time.perf_counter()
    # print(f"Trained {model.class_name} model in {toc - tic:0.4f} seconds")
    # evaluate(model, test_loader, y_test_tensor)
    # # evaluate_non_transformer_embedding_with_cross_validation(model, test_loader, y_test_tensor, 5)
    #
    #
    # print("LSTMWithAttentionWithGlove:")
    # model_params.update(
    #     dict(
    #         embedding_type='glove', pretrained_embedding_path=glove_path, vocab=train_vocab, embedding_dim=300,
    #         is_bidirectional=False, has_attention=True
    #     )
    # )
    # model = LSTM(**model_params)
    # tic = time.perf_counter()
    # train(model, config, train_loader, validation_loader, y_validation_tensor)
    # # train_with_non_transformer_embedding_with_cross_validation(
    # #     model_params, config, torch.cat([train_dataset.tensors[0], validation_dataset.tensors[0]]),
    # #     torch.cat([train_dataset.tensors[1], validation_dataset.tensors[1]], dim=0), k_folds
    # # )
    # toc = time.perf_counter()
    # print(f"Trained {model.class_name} model in {toc - tic:0.4f} seconds")
    # evaluate(model, test_loader, y_test_tensor)
    # # evaluate_non_transformer_embedding_with_cross_validation(model, test_loader, y_test_tensor, k_folds)
    #
    #
    # print("LSTMWithGlove:")
    # model_params.update(
    #     dict(
    #         embedding_type='glove', pretrained_embedding_path=glove_path, vocab=train_vocab, embedding_dim=300,
    #         is_bidirectional=False, has_attention=False
    #     )
    # )
    # model = LSTM(**model_params)
    # tic = time.perf_counter()
    # train(model, config, train_loader, validation_loader, y_validation_tensor)
    # toc = time.perf_counter()
    # print(f"Trained {model.class_name} model in {toc - tic:0.4f} seconds")
    # evaluate(model, test_loader, y_test_tensor)

    # Train with Word2Vec
    word2vec_path = "embeddings/GoogleNews-vectors-negative300.bin"

    print("BiLSTMWithAttentionWithWord2Vec:")
    model_params.update(
        dict(
            embedding_type='word2vec', pretrained_embedding_path=word2vec_path, vocab=train_vocab, embedding_dim=300,
            is_bidirectional=True, has_attention=True
        )
    )
    model = LSTM(**model_params)
    tic = time.perf_counter()
    train(model, config, train_loader, validation_loader, y_validation_tensor)
    # train_with_non_transformer_embedding_with_cross_validation(
    #     model_params, config, torch.cat([train_dataset.tensors[0], validation_dataset.tensors[0]]),
    #     torch.cat([train_dataset.tensors[1], validation_dataset.tensors[1]], dim=0), k_folds
    # )
    toc = time.perf_counter()
    print(f"Trained {model.class_name} model in {toc - tic:0.4f} seconds")
    evaluate(model, test_loader, y_test_tensor)
    # evaluate_non_transformer_embedding_with_cross_validation(model, test_loader, y_test_tensor, k_folds)


    print("BiLSTMWithWord2Vec:")
    model_params.update(
        dict(
            embedding_type='word2vec', pretrained_embedding_path=word2vec_path, vocab=train_vocab, embedding_dim=300,
            is_bidirectional=True, has_attention=False
        )
    )
    model = LSTM(**model_params)
    tic = time.perf_counter()
    train(model, config, train_loader, validation_loader, y_validation_tensor)
    toc = time.perf_counter()
    print(f"Trained {model.class_name} model in {toc - tic:0.4f} seconds")
    evaluate(model, test_loader, y_test_tensor)

    print("LSTMWithAttentionWithWord2Vec:")
    model_params.update(
        dict(
            embedding_type='word2vec', pretrained_embedding_path=word2vec_path, vocab=train_vocab, embedding_dim=300,
            is_bidirectional=False, has_attention=True
        )
    )
    model = LSTM(**model_params)
    tic = time.perf_counter()
    train(model, config, train_loader, validation_loader, y_validation_tensor)
    # train_with_non_transformer_embedding_with_cross_validation(
    #     model_params, config, torch.cat([train_dataset.tensors[0], validation_dataset.tensors[0]]),
    #     torch.cat([train_dataset.tensors[1], validation_dataset.tensors[1]], dim=0), k_folds
    # )
    toc = time.perf_counter()
    print(f"Trained {model.class_name} model in {toc - tic:0.4f} seconds")
    evaluate(model, test_loader, y_test_tensor)
    # evaluate_non_transformer_embedding_with_cross_validation(model, test_loader, y_test_tensor, k_folds)

    print("LSTMWithWord2Vec:")
    model_params.update(
        dict(
            embedding_type='word2vec', pretrained_embedding_path=word2vec_path, vocab=train_vocab, embedding_dim=300,
            is_bidirectional=False, has_attention=False
        )
    )
    model = LSTM(**model_params)
    tic = time.perf_counter()
    train(model, config, train_loader, validation_loader, y_validation_tensor)
    toc = time.perf_counter()
    print(f"Trained {model.class_name} model in {toc - tic:0.4f} seconds")
    evaluate(model, test_loader, y_test_tensor)
