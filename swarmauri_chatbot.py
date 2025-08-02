import os
from dotenv import load_dotenv

from swarmauri_standard.documents.Document import Document
from swarmauri_standard.vector_stores.TfidfVectorStore import TfidfVectorStore
from swarmauri_standard.llms.GroqModel import GroqModel
from swarmauri_standard.conversations.Conversation import Conversation
from swarmauri_standard.messages.SystemMessage import SystemMessage

# # from swarmauri.documents.concrete.Document import Document
# # from swarmauri.vector_stores.concrete import TfidfVectorStore
# # from swarmauri.llms.concrete import GroqModel
# # from swarmauri.conversations.concrete import MaxSystemContextConversation, Conversation
# # from swarmauri.messages.concrete import SystemMessage, HumanMessage

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    print("API key is not set. Please set the GROQ_API_KEY environment variable in your .env file.")
    exit()
print(f"API key loaded: {'True' if API_KEY else 'False'}")

vector_store = TfidfVectorStore()

documents = [
    Document(content="The capital of France is Paris. Paris is known for the Eiffel Tower."),
    Document(content="The largest ocean is the Pacific Ocean."),
    Document(content="A dog is a common pet animal known for its loyalty."),
    Document(content="Cats are independent pets often found sleeping."),
    Document(content="Mount Everest is the highest mountain in the world, located in the Himalayas.")
]

vector_store.add_documents(documents)
print(f"{len(vector_store.documents)} documents loaded into the vector store.")

llm = GroqModel(api_key=API_KEY, name="llama3-8b-8192")
print(f"Language model initialized with model: {llm.name}")

# Main Chatbot Interaction Loop
print("\nWelcome to your Swarmauri RAG chatbot! Type 'quit' to exit.")
while True:
    user_query = input("\nYou: ")
    if user_query.lower() == 'quit':
        print("Chatbot: Goodbye!")
        break

    # Step 1 Retrieve relevant info from your documents
    retrieved_docs = vector_store.retrieve(query=user_query, top_k=2)
    context_info = " ".join([doc.content for doc in retrieved_docs])

    # Step 1 Combine user query with retrieved context into a prompt
    rag_conversation = Conversation()
    prompt_template = f"""
    Using ONLY the following context, answer the user's question.
    If the answer is not in the context, say you don't know.

    Context: {context_info}

    User Question: {user_query}
    """
    system_message = SystemMessage(content=prompt_template)
    rag_conversation.add_message(system_message)

    # Step 1 Get response from the AI
    prediction = llm.predict(conversation=rag_conversation)
    response = prediction.get_last().content

    print(f"Chatbot: {response}")