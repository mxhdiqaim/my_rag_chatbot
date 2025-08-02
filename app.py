import os
from dotenv import load_dotenv
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from flask import Flask, request, jsonify

# --- Configuration ---
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")

# --- The Knowledge Base (Documents) ---
documents = [
    "The capital of France is Paris. Paris is known for the Eiffel Tower.",
    "The largest ocean is the Pacific Ocean.",
    "A dog is a common pet animal known for its loyalty.",
    "Cats are independent pets often found sleeping.",
    "Mount Everest is the highest mountain in the world, located in the Himalayas."
]

# --- Initialize the Vector Store ---
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(documents)

# --- Initialize the Language Model ---
llm = ChatGroq(temperature=0.7, groq_api_key=GROQ_API_KEY, model_name="llama3-8b-8192")

# --- RAG Logic Functions ---
def retrieve_documents(query, top_n=2):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, doc_vectors).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    retrieved_docs = [documents[i] for i in top_indices]
    return " ".join(retrieved_docs)

# This dictionary will store conversation memory per session/user
session_memories = {}

def get_conversation_chain(session_id):
    if session_id not in session_memories:
        session_memories[session_id] = ConversationBufferMemory()
    return ConversationChain(
        llm=llm,
        memory=session_memories[session_id],
        verbose=False
    )

# --- Flask App Setup ---
app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_query = data.get('query')
    session_id = data.get('session_id', 'default_session') # Use session_id for multi-user memory

    if not user_query:
        return jsonify({"error": "Query not provided"}), 400

    # Retrieve relevant info
    context_info = retrieve_documents(user_query)

    # Get conversation chain for the session
    conversation = get_conversation_chain(session_id)

    # Prepare the full prompt with context
    full_prompt = f"Using the following context, answer the user's question. If the answer is not in the context, say you don't know.\n\nContext: {context_info}\n\nUser: {user_query}"

    # Get response from AI
    response = conversation.predict(input=full_prompt)

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) # Runs the web server on port 5000