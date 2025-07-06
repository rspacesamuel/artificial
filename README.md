# artificial
interview-chat.py
Uses OpenAI API for a RAG solution.
Context Chunking is implemented using Langchain.
ChromaDB for vector DB, and OpenAI's small text embedding model is used.
A chat interface is provided using gradio.
RAG Context comes from 3 files detailing the experience of a job applicant.
The chatbot acts as the candidate, where a user can interview (ask questions) about the candidate.

This is not professional grade code. Eg secret management, environment management etc. aren't properly handled.
