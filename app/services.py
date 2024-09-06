import PyPDF2.errors
import requests
from bs4 import BeautifulSoup
import PyPDF2
from io import BytesIO
import uuid
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import re

# Load the model once and reuse it 
bert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# In-memory storage for simplicity.
content_store = {}

def sentence_to_vec(sentence, model):
    words = sentence.split()
    word_vecs = []
    
    for word in words:
        if word in model:
            word_vecs.append(model[word])
    
    if len(word_vecs) == 0:  # if no word vectors were found
        return np.zeros(model.vector_size)
    
    return np.mean(word_vecs, axis=0)

def process_url(url: str) -> str:
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    text = soup.get_text()
    cleaned_text = ' '.join(text.split())  # Simple cleaning
    
    chat_id = str(uuid.uuid4())
    content_store[chat_id] = cleaned_text
    
    return chat_id

def process_pdf(file) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(file.file.read()))
        text = ""
        for page in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page].extract_text()
        
        # More robust cleaning
        cleaned_text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)  # Remove punctuation
        cleaned_text = cleaned_text.strip()  # Remove leading/trailing whitespace
        
        chat_id = str(uuid.uuid4())
        content_store[chat_id] = cleaned_text
        
        return chat_id
    except PyPDF2.errors.PdfReadError:
        raise ValueError("Invalid or corrupted PDF file")
    except Exception as e:
        raise ValueError(f"Error processing PDF: {str(e)}")


def chat_with_content(chat_id: str, question: str) -> str:
    try:
        if chat_id not in content_store:
            raise ValueError("Invalid chat_id")
    
        content = content_store[chat_id]
        content_sentences = content.split('.')
        content_sentences = [sentence.strip() for sentence in content_sentences if sentence.strip()]
        
        # Generate embeddings for all sentences using BERT
        content_embeddings = bert_model.encode(content_sentences)
        
        # Generate embedding for the user's query
        question_embedding = bert_model.encode([question])[0]
        
        # Calculate cosine similarity between query and each content sentence
        sentence_similarities = cosine_similarity([question_embedding], content_embeddings)[0]
        
        # Get the top 3 most relevant sentences
        top_sentence_indices = sentence_similarities.argsort()[-3:][::-1]
        
        # Combine top sentences into a response
        response = '. '.join([content_sentences[i] for i in top_sentence_indices])
        
        if not response:
            raise ValueError("Could not generate a relevant response")
        
        return response

    except ValueError as ve:
        # Handle specific known errors
        return f"Error: {str(ve)}"
    except Exception as e:
        # Handle unexpected errors
        return f"An unexpected error occurred: {str(e)}"