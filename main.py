# Step 1: Import libraries and load the model
import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.models import load_model

# Step 2: Load the IMDB word index
max_features = 10000
max_len = 500

word_index = imdb.get_word_index()

# Load your already-trained model
model = load_model("simple_rnn_imdb.h5")


# Step 3: Preprocess user input
def preprocess_text(text):
    words = text_to_word_sequence(text)
    encoded_review = [1]  # Start-of-review token

    for word in words:
        word_id = word_index.get(word)

        if word_id is None:
            encoded_review.append(2)
        else:
            word_id += 3

            if word_id < max_features:
                encoded_review.append(word_id)
            else:
                encoded_review.append(2)

    return sequence.pad_sequences(
        [encoded_review],
        maxlen=max_len,
        padding="pre",
        truncating="pre",
        value=0
    )


# Step 4: Predict sentiment
def predict_sentiment(text):
    processed_text = preprocess_text(text)

    prediction = float(
        model.predict(processed_text, verbose=0)[0][0]
    )

    sentiment = "Positive" if prediction >= 0.5 else "Negative"

    return sentiment, prediction


# Step 5: Streamlit interface
st.title("IMDB Movie Review Sentiment Analysis")

st.write(
    "Enter a movie review to classify it as positive or negative."
)

user_input = st.text_area(
    "Movie Review",
    placeholder="Example: This movie was good. The plot twist was amazing."
)

if st.button("Classify"):
    if not user_input.strip():
        st.warning("Please enter a movie review.")
    else:
        sentiment, score = predict_sentiment(user_input)

        st.subheader(f"Sentiment: {sentiment}")

        st.write(f"Predicted probability of positive sentiment: {score:.4f}")

        st.progress(min(max(score, 0.0), 1.0))

        st.caption(
            "This is a model prediction, so it may not always be correct."
        )