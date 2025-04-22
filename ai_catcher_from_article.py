import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout

# Завантаження датасету
df = pd.read_csv("AIGTxt_DataSet.csv")

# Формування датафрейму з текстами та мітками
human_df = pd.DataFrame({'text': df['Human-generated text'], 'label': 0})
chatgpt_df = pd.DataFrame({'text': df['ChatGPT-generated text'], 'label': 1})
mixed_df = pd.DataFrame({'text': df['Mixed text'], 'label': 2})
full_df = pd.concat([human_df, chatgpt_df, mixed_df], ignore_index=True).sample(frac=1.0, random_state=42)

# Параметри
vocab_size = 10000
max_len = 200
embedding_dim = 100

# Токенізація
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(full_df['text'])
sequences = tokenizer.texts_to_sequences(full_df['text'])
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

# One-hot encoding міток
labels = to_categorical(full_df['label'], num_classes=3)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# Побудова моделі
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
    Conv1D(128, 5, activation='relu'),
    GlobalMaxPooling1D(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Навчання
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# Оцінка
loss, accuracy = model.evaluate(X_test, y_test)
print("Test accuracy:", accuracy)
