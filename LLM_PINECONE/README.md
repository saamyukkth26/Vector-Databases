## Building an llm langchain project using pinecone vector databases

## Vector Search database used - pinecone

## step 1 : create a virtual environment
_code : python3 -m venv .venv  \
to start the virtual environment  \
in mac : source .venv/bin/activate  \
in windows - .venv\Scripts\activate_

## step 2 : create a documents folder, where we save the document that we are going to work on
_documents : budget_speech.pdf_

## step 3: creating the requirement.txt file
_including all the packages inside this file to be installed  \ 
code : pip install -r requirements.txt_

## step 4 :  creating an .env file 
_to set the openai api key_

## step 5 : necessary imports for the project
_import os  \
import time  \
from dotenv import load_dotenv  \
from langchain_community.document_loaders import PyPDFDirectoryLoader (to load the document from a specific directory)  \
from langchain_text_splitters import RecursiveCharacterTextSplitter (which splits the document recursively into chunks)  \
from langchain_community.embeddings import HuggingFaceEmbeddings (creating embeddings for the chunks of the documents)  \
from langchain_pinecone import PineconeVectorStore (pinecone vector to store the embeddings into the vector store)  \
from pinecone import Pinecone, ServerlessSpec (for specifying the index_name, cloud, and model of the pinecone used)_  \

## step 6 : setting the api keys

_# Load environment variables_  \
_load_dotenv()_

_# Configure with explicit API key - don't rely on environment variables
PINECONE_API_KEY = ""  \
INDEX_NAME = ""  \
MODEL_NAME = ""_

## step 7 : function to read the pdf documents

## step 8 : function to chunk the documents

## step 9 : Initialize Pinecone

## step 10 : Upload documents to Pinecone using direct method

## step 11 : Query function

## step 12 : Main execution






