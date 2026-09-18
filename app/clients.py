from openai import OpenAI
from pinecone import Pinecone

from app.config import settings

openai_client = OpenAI(api_key=settings.openai_api_key)
pinecone_client = Pinecone(api_key=settings.pinecone_api_key)
pinecone_index = pinecone_client.Index(settings.pinecone_index_name)
