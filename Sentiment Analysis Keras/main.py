from tensorflow.keras.preprocessing import text_dataset_from_directory
from tensorflow.strings import regex_replace
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout


def prepareData(dir):
    data = text_dataset_from_directory(dir)
    return data.map(
        lambda text, label: (regex_replace(text, '<br />', ' '), label),
    )


# Assumes you're in the root level of the dataset directory.
# If you aren't, you'll need to change the relative paths here.

train_data = prepareData('./dataset/train')
test_data = prepareData('./dataset/test')

for text_batch, label_batch in train_data.take(1):
    print(text_batch.numpy()[0])
    print(label_batch.numpy()[0])  # 0 = negative, 1 = positive

model = Sequential()

# ----- 1. INPUT
# We need this to use the TextVectorization layer next.
model.add(Input(shape=(1,), dtype="string"))

# ----- 2. TEXT VECTORIZATION
# This layer processes the input string and turns it into a sequence of
# max_len integers, each of which maps to a certain token.
max_tokens = 1000
max_len = 100
vectorize_layer = TextVectorization(
    # Max vocab size. Any words outside of the max_tokens most common ones
    # will be treated the same way: as "out of vocabulary" (OOV) tokens.
    max_tokens=max_tokens,
    # Output integer indices, one per string token
    output_mode="int",
    # Always pad or truncate to exactly this many tokens
    output_sequence_length=max_len,
)

# Call adapt(), which fits the TextVectorization layer to our text dataset.
# This is when the max_tokens most common words (i.e. the vocabulary) are selected.
train_texts = train_data.map(lambda text, label: text)
vectorize_layer.adapt(train_texts)

model.add(vectorize_layer)

# ----- 3. EMBEDDING
# This layer turns each integer (representing a token) from the previous layer
# an embedding. Note that we're using max_tokens + 1 here, since there's an
# out-of-vocabulary (OOV) token that gets added to the vocab.
model.add(Embedding(max_tokens + 1, 128))

# ----- 4. RECURRENT LAYER
# model.add(LSTM(64, return_sequences=True))
model.add(LSTM(64, dropout=0.25, recurrent_dropout=0.25))


# ----- 5. DENSE HIDDEN LAYER
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
# ----- 6. OUTPUT
model.add(Dense(1, activation="sigmoid"))

# Compile and train the model.
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.summary()
model.fit(train_data, epochs=10)
model.save_weights('rnn')
# model.load_weights('rnn')

# Try the model on our test dataset.
model.evaluate(test_data)

# Should print a very high score like 0.98.
print(model.predict([
    "i loved it! highly recommend it to anyone and everyone looking for a great movie to watch.",
]))

# Should print a very low score like 0.01.
print(model.predict([
    "this was awful! i hated it so much, nobody should watch this. the acting was terrible, the music was terrible, overall it was just bad.",
]))
