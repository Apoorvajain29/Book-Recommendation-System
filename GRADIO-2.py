from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np


from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain.embeddings.base import Embeddings

import gradio as gr
from gradio.themes import Monochrome

from sentence_transformers import SentenceTransformer

class HuggingFaceEmbeddings(Embeddings):
    def __init__(self, model_id="sentence-transformers/LaBSE"):
        print(f"Downloading or loading the embedding model: {model_id}. This may take a while.")
        self.model = SentenceTransformer(model_id)


    def embed_documents(self, texts):

        embeddings = self.model.encode(texts, convert_to_numpy=False)
        return [embedding.tolist() for embedding in embeddings]

    def embed_query(self, text):
        embedding = self.model.encode(text, convert_to_numpy=False)
        return embedding.tolist()

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
assert HUGGINGFACE_TOKEN, "Please set HUGGINGFACE_TOKEN in your .env file"

books = pd.read_csv('books_with_emotions.csv')
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

documents = []
with open('tagged_description.txt', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        first_space = line.find(' ')
        if first_space == -1: continue
        isbn = line[:first_space].strip('"')
        description = line[first_space + 1:].strip()
        documents.append(Document(page_content=description, metadata={"isbn13": isbn}))


persist_directory = "./chroma_db"
huggingface_embeddings = HuggingFaceEmbeddings()

if os.path.exists(persist_directory):
    # Load the existing database
    print("Loading existing Chroma vector store...")
    db_books = Chroma(
        persist_directory=persist_directory,
        embedding_function=huggingface_embeddings
    )
    print("Chroma vector store loaded.")
else:

    print("Building and saving new Chroma vector store from Hugging Face model. This may take a while.")
    db_books = Chroma.from_documents(
        documents,
        embedding=huggingface_embeddings,
        persist_directory=persist_directory
    )
    print("Vector store built and saved.")

print("Number of documents in vector store:", len(db_books.get()['documents']))

def retrieve_semantic_recommendation(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16
) -> pd.DataFrame:

    recs = db_books.similarity_search_with_score(query, k=initial_top_k)
    books_list = []


    for doc, score in recs:

        isbn = doc.metadata.get("isbn13")
        if isbn:
            books_list.append(int(isbn))


    books["isbn13"] = pd.to_numeric(books["isbn13"], errors='coerce')


    book_recs = books[books["isbn13"].isin(books_list)]

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category]

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    print(book_recs)
    return book_recs.head(final_top_k)


def recommend_books(
        query: str,
        category: str,
        tone: str
):

    if not query.strip():

        return []

    recommendations = retrieve_semantic_recommendation(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        captions = f"{row['title']} by {authors_str} : {truncated_description}"
        results.append((row["large_thumbnail"], captions))

    return results


categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=Monochrome()) as dashboard:
    gr.Markdown("# Semantic book recommender")
    with gr.Row():
        user_query = gr.Textbox(label="Please enter a description of a book:",
                                placeholder="e.g. A story about forgiveness")
        category_dropdown = gr.Dropdown(choices=categories, label="Select a category:", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Select an emotional tone:", value="All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommended books", columns=8, rows=2)

    submit_button.click(fn=recommend_books,
                        inputs=[user_query, category_dropdown, tone_dropdown],
                        outputs=output)

if __name__ == "__main__":
    dashboard.launch()