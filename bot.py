from fastapi import FastAPI, HTTPException
import os
from datetime import datetime
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import VectorStoreRetrieverMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
import faiss
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv
from langchain.output_parsers import PydanticOutputParser
from typing import List
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
import json

load_dotenv()

# Initialize OpenAI
embeddings = OpenAIEmbeddings()
openAI = OpenAI(temperature=0)

# Initializing and Loading vectors
embedding_size = 1536  # Dimensions of the OpenAIEmbeddings
index = faiss.IndexFlatL2(embedding_size)

# first time
if os.path.exists(f"./store/index.faiss"):
    vectorstore = FAISS.load_local(f"./store", embeddings)
else:
    vectorstore = FAISS(embeddings.embed_query, index,
                        InMemoryDocstore({}), {})
retriever = vectorstore.as_retriever()
memory = VectorStoreRetrieverMemory(retriever=retriever)


class example(BaseModel):
    input: str
    output: str


class question(BaseModel):
    prompt: str


# Prompt Template
template = """
Marv is a assistant for an Accounting Software company, which will instruct users how to use the platform and will tell people about the platform.
Marv will answer how software works to logged in users with step by step instuctions, and for normal visitor, Marv will tell about the platform, it's pricing and all.

This is the sample questions and answers.
{context}

You: {input}
Marv: 
"""

prompt = PromptTemplate(
    input_variables=["input", "context"], template=template,
)

# Chain initialization
chain = load_qa_chain(
    openAI,
    chain_type="stuff",
    prompt=prompt
)

# text-splitter initialization
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)


def train(fileName: str):
    json_file = open(fileName)
    train_examples = json.load(json_file)
    text = ""
    for example in train_examples:
        text += f"""
        Human: {example["prompt"]}
        Marv: {example["completion"]}

        """

    vectorstore.add_texts(text_splitter.split_text(text))
    vectorstore.save_local("./store")


# training


app = FastAPI()


@app.post("/train")
async def train():
    train("dataset.json")


@app.post("/get-answer")
async def get_answer(question: question):
    examples = vectorstore.similarity_search(question.prompt)
    return chain.run(
        input_documents=examples, input=question)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(host="0.0.0.0", port=8080, app=app)
