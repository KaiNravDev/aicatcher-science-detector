
import pandas as pd
import numpy as np
import string
import spacy
import nltk
import language_tool_python
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Conv1D, GlobalMaxPooling1D, Concatenate, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# === Setup ===
nltk.download("stopwords")
nltk.download("vader_lexicon")
spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")
tool = language_tool_python.LanguageTool("en-US")
stop_words = set(stopwords.words("english"))
sid = SentimentIntensityAnalyzer()

discourse_markers = {"however", "moreover", "furthermore", "therefore", "thus", "additionally", "meanwhile"}

# === Feature extraction ===
def extract_features(text):
    doc = nlp(text)
    sentences = list(doc.sents)
    words = [token.text for token in doc if token.is_alpha]

    avg_sentence_length = sum(len(sent.text.split()) for sent in sentences) / len(sentences) if sentences else 0
    avg_word_length = sum(len(w) for w in words) / len(words) if words else 0
    unique_word_ratio = len(set(words)) / len(words) if words else 0
    grammar_errors = len(tool.check(text))
    discourse_count = sum(text.lower().count(marker) for marker in discourse_markers)

    pos_counts = {"NOUN": 0, "PRON": 0, "VERB": 0, "ADV": 0, "ADJ": 0}
    for token in doc:
        if token.pos_ in pos_counts:
            pos_counts[token.pos_] += 1
    total_tokens = len(doc)
    pos_ratios = {k: v / total_tokens if total_tokens else 0 for k, v in pos_counts.items()}

    punctuation_ratio = sum(1 for char in text if char in string.punctuation) / len(text.split()) if text else 0
    stopword_ratio = sum(1 for w in words if w.lower() in stop_words) / len(words) if words else 0
    sentiment_words = sum(1 for w in words if sid.polarity_scores(w)["compound"] != 0)
    sentiment_ratio = sentiment_words / len(words) if words else 0

    return {
        "avg_sentence_length": avg_sentence_length,
        "avg_word_length": avg_word_length,
        "unique_word_ratio": unique_word_ratio,
        "grammar_errors": grammar_errors,
        "discourse_markers": discourse_count,
        **pos_ratios,
        "punctuation_ratio": punctuation_ratio,
        "stopword_ratio": stopword_ratio,
        "sentiment_word_ratio": sentiment_ratio,
    }

# === Load dataset ===
df = pd.read_csv("AIGTxt_DataSet.csv")
df.rename(columns={"Domain": "text"}, inplace=True)

# Extract features
features = df["text"].apply(extract_features).apply(pd.Series)
df = pd.concat([df, features], axis=1)

# === Tokenization ===
MAX_NUM_WORDS = 3000
MAX_SEQUENCE_LENGTH = 200
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(df["text"])
sequences = tokenizer.texts_to_sequences(df["text"])
padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

# === MLP features
mlp_features = df[[
    'avg_sentence_length', 'avg_word_length', 'unique_word_ratio',
    'grammar_errors', 'discourse_markers', 'NOUN', 'PRON', 'VERB',
    'ADV', 'ADJ', 'punctuation_ratio', 'stopword_ratio', 'sentiment_word_ratio'
]].values

# === Label creation
def get_label(row):
    if isinstance(row["Human-generated text"], str) and len(row["Human-generated text"].strip()) > 0:
        return "Human"
    elif isinstance(row["ChatGPT-generated text"], str) and len(row["ChatGPT-generated text"].strip()) > 0:
        return "ChatGPT"
    elif isinstance(row["Mixed text"], str) and len(row["Mixed text"].strip()) > 0:
        return "Mixed"
    else:
        return np.nan

df["label"] = df.apply(get_label, axis=1)
label_map = {"Human": 0, "ChatGPT": 1, "Mixed": 2}
labels = df["label"].map(label_map)
valid_mask = labels.notna()
labels = labels[valid_mask].astype(int)
mlp_features = mlp_features[valid_mask]
padded_sequences = padded_sequences[valid_mask]

# === Train/test split
X_mlp_train, X_mlp_test, X_seq_train, X_seq_test, y_train, y_test = train_test_split(
    mlp_features, padded_sequences, labels, test_size=0.2, random_state=42)

# === AI-Catcher architecture
mlp_input = Input(shape=(13,), name="mlp_input")
x_mlp = Dense(128, activation='relu')(mlp_input)
x_mlp = Dense(64, activation='relu')(x_mlp)

cnn_input = Input(shape=(MAX_SEQUENCE_LENGTH,), name="cnn_input")
x_cnn = Embedding(MAX_NUM_WORDS, 100)(cnn_input)
x_cnn = SpatialDropout1D(0.2)(x_cnn)
x_cnn = Conv1D(128, 5, activation='relu')(x_cnn)
x_cnn = GlobalMaxPooling1D()(x_cnn)

merged = Concatenate()([x_mlp, x_cnn])
x = Dense(64, activation='relu')(merged)
output = Dense(3, activation='softmax')(x)

model = Model(inputs=[mlp_input, cnn_input], outputs=output)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# === Train
model.fit([X_mlp_train, X_seq_train], y_train, validation_split=0.1, epochs=20, batch_size=32)

# === Evaluate
loss, accuracy = model.evaluate([X_mlp_test, X_seq_test], y_test)
print(f"✅ Test accuracy: {accuracy:.4f}")
