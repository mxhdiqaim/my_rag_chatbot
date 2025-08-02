import os
from dotenv import load_dotenv
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq # Specific for Groq with LangChain

# --- Configuration ---
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")

# --- Your Knowledge Base (Documents) ---
documents = [
    "The capital of France is Paris. Paris is known for the Eiffel Tower.",
    "The largest ocean is the Pacific Ocean.",
    "A dog is a common pet animal known for its loyalty.",
    "Cats are independent pets often found sleeping.",
    "Mount Everest is the highest mountain in the world, located in the Himalayas."
]

# --- Initialize the Vector Store (TF-IDF for simplicity, as in video) ---
# This prepares the "super library" for quick searching.
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(documents)

# --- Initialize the Language Model (Groq) ---
# This is the "AI Brain" ready to think and generate text.
llm = ChatGroq(temperature=0.7, groq_api_key=GROQ_API_KEY, model_name="llama3-8b-8192") # I choose "llama3-8b-8192" or "mixtral-8x7b-32768"

# --- Build the RAG Agent Logic ---
# This combines the AI brain, the super library, and the memory.

# Function to retrieve relevant documents based on a query
def retrieve_documents(query, top_n=2):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, doc_vectors).flatten()
    # Get indices of top 2 most similar documents
    top_indices = similarities.argsort()[-top_n:][::-1]
    retrieved_docs = [documents[i] for i in top_indices]
    return " ".join(retrieved_docs) # Combine them into a single string

# Set up conversation memory for the chatbot
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# --- Main Chatbot Interaction Loop ---
print("Welcome to your custom RAG chatbot! Type 'quit' to exit.")
while True:
    user_query = input("\nYou: ")
    if user_query.lower() == 'quit':
        print("Chatbot: Goodbye!")
        break

    # Retrieve relevant info from your documents
    context_info = retrieve_documents(user_query)

    # Combine user query with retrieved context
    # Integrate the context directly into the prompt for the conversation chain
    # LangChain's ConversationChain handles the memory automatically
    full_prompt = f"Using the following context, answer the user's question. If the answer is not in the context, say you don't know.\n\nContext: {context_info}\n\nUser: {user_query}"

    # Get response from the AI
    response = conversation.predict(input=full_prompt)
    print(f"Chatbot: {response}")