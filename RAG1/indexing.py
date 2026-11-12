from .file_reader import extract_paragraphs
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model = 'text-embedding-3-large')
vectorstore = InMemoryVectorStore(embeddings)

chucks = extract_paragraphs('pdfs/Penguins_ACL.pdf')
vectorstore.add_texts(chucks)

print(f'Chunks {len(chucks)} was created...\n')

