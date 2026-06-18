import os
from google import genai
from google.genai import types
import chromadb
from dotenv import load_dotenv

load_dotenv()
client = genai.Client()

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="Autism_knowledge_base")


def get_embedding(text):
    """Generates a 768-dimension vector for search queries using Gemini."""
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return response.embeddings[0].values

def retrieve_relevant_chunks(query, num_results=3):
    """Searches ChromaDB for the most contextually relevant chunks matching the query."""
    query_vector = get_embedding(query)
    
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=num_results
    )

    documents = results['documents'][0] if results['documents'] else []
    metadatas = results['metadatas'][0] if results['metadatas'] else []
    
    return documents, metadatas



def generate_rag_response(user_question, chat_history=None):

    if chat_history is None:
        chat_history=[]

    context_chunks, metadata_list = retrieve_relevant_chunks(user_question, num_results=3)

    context_str = ""
    for i, (doc, meta) in enumerate(zip(context_chunks, metadata_list)):
        context_str += f"\n[Source Chunk {i+1} | Pillar: {meta.get('pillar')}, Topic: {meta.get('topic')}]\n{doc}\n"

    #system instruction
    system_instruction = (
        "You are an empathetic, highly knowledgeable AI assistant specializing in autism spectrum conditions, "
        "providing guidance to parents, caregivers, and researchers.\n\n"
        "ADAPTIVE TONE & RESPONSIVENESS DIRECTION:\n"
        "- Assess the emotional state of the user before answering.\n"
        "- IF THE USER SEEMS SAD, OVERWHELMED, STRESSED, OR ANXIOUS: You must prioritize emotional validation first. "
        "Open your response with a deeply compassionate, warm, and comforting sentence. Let them know they are heard, "
        "and that their feelings are completely valid before gently transitioning into helpful information. Avoid sounding clinical or robotic.\n"
        "- IF THE USER SENDS A PLAIN, FACTUAL, OR STRIPPED QUERY: Deliver a highly structured, standard, clear, and direct clinical response "
        "without excessive emotional preambles, focusing strictly on efficiency and accurate data delivery.\n\n"
        "CRITICAL RAG RULES:\n"
        "1. Base your answer PRIMARILY on the provided 'Retrieved Clinical Knowledge' below.\n"
        "2. If the retrieved context doesn't contain the answer, use your general clinical knowledge but state clearly "
        "that it wasn't explicitly found in the primary research files.\n"
        "3. Never guess or hallucinate diagnostic criteria. Keep advice grounded and safe."
    
    )

    user_prompt = f"### Retrieved Clinical Knowledge:\n{context_str}\n\n"

    #CALLING chat history
    if chat_history:
        user_prompt += "### Recent Conversation History:\n"
        for turn in chat_history:
            role_label = "Parent" if turn['role'] == 'user' else "AI Companion"
            user_prompt += f"{role_label}: {turn['text']}\n"
        user_prompt += "\n"
        
    user_prompt += f"### Current Parent Question:\n{user_question}"

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.3, # Keeps responses focused and accurate to your research data
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_prompt,
        config=config
    )
    
    return response.text

if __name__ == "__main__":
    # Quick interactive test loop directly inside your terminal!
    print("\n🤖 Autism RAG Core Retriever Ready. Type 'exit' to stop.")
    print("-" * 50)
    
    # Mock short-term session memory list
    session_history = []

    while True:
        query = input("\nAsk a question: ").strip()
        if query.lower() in ['exit', 'quit']:
            break
        if not query:
            continue
            
        print("\nThinking and searching database...")
        try:
            answer = generate_rag_response(query, chat_history=session_history)
            print(f"\nResponse:\n{answer}")
            
            # Append to history so it retains context for the next turn in this terminal session
            session_history.append({"role": "user", "text": query})
            session_history.append({"role": "model", "text": answer})
            
        except Exception as e:
            print(f"An error occurred: {e}")