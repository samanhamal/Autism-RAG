import os
import chromadb
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = genai.Client()

def parse_markdown_file(file_path):
    
    if not os.path.exists(file_path):
        print(f"Error:file not found {file_path}")
        return[]
    
    with open(file_path,'r',encoding='utf-8')as f:
        content = f.read()

    raw_chunks = content.split("---")
    parsed_chunks  = []

    for chunk in raw_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        
        lines = chunk.split("\n")
        pillar = "General"
        topic = "Unknown"
        cleaned_lines =  []

        for line in lines:
            clean_line = line.strip()
            if clean_line.startswith("\\"):
                clean_line = clean_line[1:].strip()
                
            # Parse our target metadata tags
            if clean_line.lower().startswith("# pillar:"):
                pillar = clean_line[9:].strip()
            elif clean_line.lower().startswith("# topic:"):
                topic = clean_line[8:].strip()
            else:
                # Keep the non-metadata lines as the core text chunk
                cleaned_lines.append(clean_line)

        reconstructed_text = "\n".join(cleaned_lines).strip()

        if reconstructed_text:
            parsed_chunks.append({
                "text": reconstructed_text,
                "metadata": {"pillar": pillar, "topic": topic}
            })
            
    return parsed_chunks


def get_embeddings(text):
    response = client.models.embed_content(
        model = 'gemini-embedding-001',
        contents = text

    )
    return response.embeddings[0].values

def build_vector_database(file_path):

    chunks = parse_markdown_file(file_path)

    if not chunks:
        print("no valid chunks found!")
        return
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name = "Autism_knowledge_base")

    for i, item in enumerate(chunks):
        vector = get_embeddings(item["text"])
        
        # Insert everything securely into your local database instance
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[vector],
            documents=[item["text"]],
            metadatas=[item["metadata"]]
        )
        
    print(f"🎉 Success! Vector database built completely with {collection.count()} items.")

if __name__ == "__main__":
    # Target your clean file path
    target_file = "./knowledge_base/KB_file_02.md"
    build_vector_database(target_file)