'''
Book Recommendation System
Author: K Mohammad
'''

import pickle
from pathlib import Path

import numpy as np
import streamlit as st


# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .block-container {
            max-width: 1200px;
            padding-top: 2.5rem;
            padding-bottom: 3rem;
        }

        .hero {
            text-align: center;
            padding: 1.5rem 0 1.75rem 0;
        }

        .hero h1 {
            font-size: 2.6rem;
            margin-bottom: 0.45rem;
            letter-spacing: -0.03em;
        }

        .hero p {
            font-size: 1.05rem;
            opacity: 0.72;
            margin: 0 auto;
            max-width: 680px;
        }

        .section-label {
            font-weight: 700;
            font-size: 1.05rem;
            margin-bottom: 0.35rem;
        }

        .helper-text {
            opacity: 0.65;
            font-size: 0.9rem;
            margin-bottom: 0.75rem;
        }

        div[data-testid="stImage"] img {
            height: 280px;
            width: 100%;
            object-fit: cover;
            border-radius: 10px;
        }

        div[data-testid="stButton"] > button {
            width: 100%;
            min-height: 46px;
            font-weight: 700;
            border-radius: 10px;
        }

        .how-it-works {
            text-align: center;
            padding: 1.5rem 0 0.5rem 0;
        }

        .how-it-works p {
            opacity: 0.7;
            max-width: 760px;
            margin: 0.5rem auto 0 auto;
        }

        .footer {
            text-align: center;
            opacity: 0.55;
            font-size: 0.82rem;
            margin-top: 2.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Load model artifacts once per app process
# -----------------------------------------------------------------------------
ARTIFACTS = Path("artifacts")


@st.cache_resource(show_spinner=False)
def load_artifacts():
    required_files = [
        "model.pkl",
        "user_ids.pkl",
        "final_rating.pkl",
        "book_pivot.pkl",
    ]

    missing = [name for name in required_files if not (ARTIFACTS / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model artifacts: " + ", ".join(missing)
        )

    with open(ARTIFACTS / "model.pkl", "rb") as file:
        model = pickle.load(file)
    with open(ARTIFACTS / "user_ids.pkl", "rb") as file:
        book_names = pickle.load(file)
    with open(ARTIFACTS / "final_rating.pkl", "rb") as file:
        final_rating = pickle.load(file)
    with open(ARTIFACTS / "book_pivot.pkl", "rb") as file:
        book_pivot = pickle.load(file)

    return model, book_names, final_rating, book_pivot


try:
    model, book_names, final_rating, book_pivot = load_artifacts()
except Exception as exc:
    st.error("The recommendation model could not be loaded.")
    st.caption(f"Technical detail: {exc}")
    st.stop()


# -----------------------------------------------------------------------------
# Recommendation logic
# -----------------------------------------------------------------------------
def fetch_posters(suggestions):
    """Return cover URLs for the books returned by the nearest-neighbor model."""
    selected_book_names = [book_pivot.columns[index] for index in suggestions[0]]
    poster_urls = []

    for book_name in selected_book_names:
        matches = np.where(final_rating["user_id"] == book_name)[0]
        if len(matches):
            poster_urls.append(final_rating.iloc[matches[0]]["image_url"])
        else:
            poster_urls.append(None)

    return poster_urls


def recommend_books(book_name):
    """Find books nearest to the selected book using the trained model."""
    matches = np.where(book_pivot.columns == book_name)[0]
    if len(matches) == 0:
        raise ValueError("The selected book is not available in the recommendation model.")

    book_id = matches[0]
    n_neighbors = min(6, len(book_pivot))

    _, suggestions = model.kneighbors(
        book_pivot.iloc[book_id, :].values.reshape(1, -1),
        n_neighbors=n_neighbors,
    )

    recommended_books = [book_pivot.index[index] for index in suggestions[0]]
    poster_urls = fetch_posters(suggestions)

    # Remove the selected book from the displayed results.
    results = [
        (title, poster)
        for title, poster in zip(recommended_books, poster_urls)
        if title != book_name
    ]

    return results[:5]


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>📚 Book Recommendation System</h1>
        <p>Discover your next great read with machine learning-powered recommendations.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container():
    st.markdown('<div class="section-label">Choose a book you like</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="helper-text">Select a book and we\'ll find five similar reads for you.</div>',
        unsafe_allow_html=True,
    )

    selected_book = st.selectbox(
        "Book selection",
        book_names,
        label_visibility="collapsed",
    )

    if st.button("🔍 Recommend Books", type="primary"):
        with st.spinner("Finding books you may enjoy..."):
            try:
                recommendations = recommend_books(selected_book)
            except Exception as exc:
                st.error("Sorry, we couldn't generate recommendations for this book.")
                st.caption(f"Technical detail: {exc}")
                st.stop()

        if recommendations:
            st.markdown("### ✨ Recommended for you")
            st.caption(f"Because you selected **{selected_book}**")

            columns = st.columns(5, gap="medium")
            for column, (title, poster_url) in zip(columns, recommendations):
                with column:
                    with st.container(border=True):
                        if poster_url:
                            st.image(poster_url, use_column_width=True)
                        else:
                            st.info("Cover unavailable")
                        st.markdown(f"**{title}**")
        else:
            st.info("We couldn't find similar books for this selection.")

st.markdown(
    """
    <div class="how-it-works">
        <h3>⚙️ How it works</h3>
        <p>
            The app uses collaborative filtering and a Nearest Neighbors model
            to identify books with similar reader-interaction patterns.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="footer">Built with Python · Streamlit · scikit-learn</div>',
    unsafe_allow_html=True,
)
