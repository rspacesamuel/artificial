
# **This is a notebook to test RAG and LLM eval by sequencing LLMs**

from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.config import Settings
from pypdf import PdfReader
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
import gradio as gr

load_dotenv(override=True)

# Initialize the LLM client
llm_client = OpenAI()
# Initialize the vector store
chroma_client = chromadb.PersistentClient(
    path="/Users/rajsamuel/projects/agents/1_foundations/chroma_db"
)
chroma_collection = chroma_client.get_or_create_collection(name="resume_collection")

# Function to extract text from a PDF file  
def get_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to split the text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

# Extract text from the pdf files
linkedin_text = get_pdf_text("/Users/rajsamuel/projects/agents/1_foundations/me/Raj Samuel linkedin.pdf")
resume_text = get_pdf_text("/Users/rajsamuel/projects/agents/1_foundations/me/Raj Samuel.pdf")
dtld_experience_text = get_pdf_text("/Users/rajsamuel/projects/agents/1_foundations/me/Raj Samuel summary.pdf")

# Index the documents
context_documents = [
    {"id": "linkedin", "text": linkedin_text},
    {"id": "resume", "text": resume_text},
    {"id": "dtld_experience", "text": dtld_experience_text}
]

# Embed the documents and add them to the vector store
for doc in context_documents:
    context_chunks = get_text_chunks(doc["text"])
    context_embeddings_client = llm_client.embeddings.create(
        input=context_chunks, 
        model="text-embedding-3-small"
        )
    
    #gather data for vector store
    #first get embeddings per chunk into separate lists. context_embeddings_client.data[0].embedding is the 1st chunk
    context_embeddings = [data.embedding for data in context_embeddings_client.data]
    #then get id and metadata per chunk into separate lists as well. metadata is a list of dictionaries,
    #such that there is one dictionary per chunk
    context_ids = [f"{doc['id']}_{i}" for i in range(len(context_chunks))]
    context_metadatas = [{"source": doc["id"], "chunk": i} for i in range(len(context_chunks))]
    
    #add data to vector store
    chroma_collection.add(
        ids=context_ids,
        embeddings=context_embeddings,
        documents=context_chunks,
        metadatas=context_metadatas
    )


#query_text = "What is a quick summary of Raj's experience?"

#process user's message in chat window and return response using RAG+LLM
def chat_processor(message, history):
    #embed user's message
    query_embeddings_client = llm_client.embeddings.create(
        input=message, 
        model="text-embedding-3-small"
    )
    query_embeddings = query_embeddings_client.data[0].embedding

    #retrieve relevant context using embeddings of user's message
    results = chroma_collection.query(
        query_embeddings=query_embeddings,
        n_results=1
    )
    context = results["documents"][0][0]

    #main prompt for LLM
    system_prompt = f"You're acting as Raj Samuel. The user is asking questions to you as an interviewer\
    to understand your background and fit for a potential job. You need to exhibit leadership qualities,\
    technical acumen, and resourcefulness.\
    For each question, you might be given 'additional context' about your qualifications.\
    You can use that context to answer the question.\
    You should be concise.\
    If you don't have enough information to answer the question, try to be creative and make it realistic.\
    But when you do that, add that word 'Alright' at the beginning of your response.\
    Don't use the word 'Alright' if the answer is known to you or found in the 'additional context'.\
    \
    Additional context:\
    {context}"

    #combine main system prompt, chat history, and user's current message
    llm_input = [{"role": "system", "content": system_prompt}]
    #parse chat history from gradio into OpenAI format
    for user, bot in history:
        llm_input.append({"role": "user", "content": user})
        llm_input.append({"role": "assistant", "content": bot})
    #add user's current message
    llm_input.append({"role": "user", "content": message})

    print(f"Message being sent to 4o: {llm_input}")
    #call LLM
    response = llm_client.chat.completions.create(model="gpt-4o-mini", messages=llm_input)
    print(f"result from 4o: {response.choices[0].message.content}")
    #return LLM's response to the chat window
    return response.choices[0].message.content

#launch chat interface
gr.ChatInterface(chat_processor).launch()




