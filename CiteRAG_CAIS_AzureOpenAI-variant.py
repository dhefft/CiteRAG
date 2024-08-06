# -*- coding: utf-8 -*-
"""
Created on Tue May  7 13:37:53 2024

@authors: dhefft (Daniel Hefft; daniel.hefft@iml.fraunhofer.de), ngrosse (Nick Große; nick.grosse@tu-dortmund.de)
"""

"""
CiteRAG is a scientific exploration tool that supports Systematic Literature Reviews
Find our paper at: DOI (tbd)
©Fraunhofer IML
License: LICENSE-CC-BY-NC-ND-4.0
"""
#add pdf path automization + result in csv

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from typing import List
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from operator import itemgetter
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain.chains import create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
import os
import pandas as pd
from dotenv import load_dotenv

### Initialize ###
### SET UP .env according to (https://learn.microsoft.com/en-us/azure/ai-services/openai/quickstart?tabs=command-line%2Cpython-new&pivots=programming-language-python)
load_dotenv()
### SET YOUR PATH TO YOUR FOLDER CONTAINING YOUR PDFs HERE ###
directory="<YOUR PATH GOES HERE>"
#set llm
llm=AzureChatOpenAI(model="gpt-4o-2024-05-13") ###NOTE: gpt-3.5-turbo-1106 would be cheaper!!! (GPT-3.5: 1$/1M tokens in & 2$/1M tokens out; GPT-4o:5$/1M tokens in & 15$/1M tokens out)
#set splitter for chunking
splitter=CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
#Set chat history storage
store = {}
#Set Pandas Dataframe for storing questions and results (answers and quotations) ###YOU CAN ADD MORE COLUMNS FOR ADDITIONAL QUESTIONS###
df_results=pd.DataFrame(columns=['File','Query 1','Answer 1','Quotes 1','Query 2','Answer 2','Quotes 2','Query 3','Answer 3','Quotes 3','Query 4','Answer 4','Quotes 4','Query 5','Answer 5','Quotes 5','Query 6','Answer 6','Quotes 6'])

###Get Paths to PDFs###
#get paths
def list_pdf_files(directory):
    # List to hold all PDF file names
    pdfs=[]
    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pdf'):
                #Add Path and file
                pdfs.append(f"{directory}/{file}")
    return pdfs

#tagging metadata (content vs Literature atm)
def tagging_docs(docs):
    for doc in range(0,len(docs)):
        if "doi:" in docs[doc].page_content:
            docs[doc].metadata["tag"]="annex"
        elif "doi:" not in docs[doc].page_content:
            docs[doc].metadata["tag"]="content"
    return docs

#define vectorstore and retriever
def build_retriever(docs):
    #Vectorestore initialization from specific documents
    vectorstore=FAISS.from_documents(documents=docs, embedding=AzureOpenAIEmbeddings(model="text-embedding-ada-002-2", openai_api_version="2023-12-01-preview")) #expensive model: text-embedding-3-large
    #Set parameter for retriever
    fetch_k=round(len(docs)/2)
    k=round(len(docs)/4)
    #Define retriever
    retriever=vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': k, 'fetch_k': fetch_k, 'filter': {'tag':'content'}})
    return retriever

#Prompts for contextualizing and answering
def build_prompts(retriever):
    ### Prompt Contextualize question ###
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    ### Prompt Answer question ###
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You're a helpful AI assistant. Given a user question and some pdf article snippets, answer the user question. If none of the articles answer the question, just say you don't know.\n\nHere are the pdf articles:{context}",
            ),
            ("human", "{question}"),
        ]
    )   
    
    return history_aware_retriever, prompt

#Session ID management for chat history
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

#formatting function
def format_docs_with_id(docs: List[Document]) -> str:
    formatted = [
        f"Source ID: {i}\nArticle Title: {doc.metadata['source']}\nArticle Snippet: {doc.page_content}"
        for i, doc in enumerate(docs)
    ]
    return "\n\n" + "\n\n".join(formatted)

#tool creation for citation (as classes)
class Citation(BaseModel):
    source_id: int = Field(
        ...,
        description="The integer ID of SPECIFIC sources which justify the answer.",
    )
    quote: str = Field(
        ...,
        description="The VERBATIM quotes from the specified sources that justify the answer.",
    )

class quoted_answer(BaseModel):
    """Answer the user question based only on the given sources, and cite the sources used."""

    answer: str = Field(
        ...,
        description="The answer to the user question, which is based only on the given sources.",
    )
    citations: List[Citation] = Field(
        ..., description="Citations from the given sources that justify the answer."
    )

###Program###
# Get the list of PDF files
pdfs=list_pdf_files(directory)

for i in range(0,len(pdfs)):
    #Load, chunk and index the pdfs.
    loader=PyPDFLoader(pdfs[i])
    docs=loader.load_and_split(text_splitter=splitter)
    docs=tagging_docs(docs)
    #set retrievers
    retriever=build_retriever(docs)
    history_aware_retriever, prompt=build_prompts(retriever)
    #set parser
    output_parser=JsonOutputKeyToolsParser(key_name="quoted_answer", first_tool_only=True)
    #bind tool to llm (could also bind functions!)
    llm_with_tool=llm.bind_tools([quoted_answer], tool_choice="quoted_answer")
    #specify chain parameters and init chain
    format1=itemgetter("docs") | RunnableLambda(format_docs_with_id)
    answer=prompt | llm_with_tool | output_parser
    chain=(
        RunnableParallel(question=RunnablePassthrough(), docs=history_aware_retriever)
        .assign(context=format1)
        .assign(quoted_answer=answer)
        .pick(["quoted_answer", "docs"])
    )
    ###SET QUESTIONS###
    query1="Question 1 (e.g. 'Is this paper related to a specific topic?')"
    query2="Question 2"
    query3="Question 3"
    query4="Question 4"
    query5="Question 5"
    query6="Question 6"
    ###GET RESULTS###
    result1=chain.invoke({"input": query1}, config={"configurable": {"session_id": f"Paper{i}"}})
    result2=chain.invoke({"input": query2}, config={"configurable": {"session_id": f"Paper{i}"}})
    result3=chain.invoke({"input": query3}, config={"configurable": {"session_id": f"Paper{i}"}})
    result4=chain.invoke({"input": query4}, config={"configurable": {"session_id": f"Paper{i}"}})
    result5=chain.invoke({"input": query5}, config={"configurable": {"session_id": f"Paper{i}"}})
    result6=chain.invoke({"input": query6}, config={"configurable": {"session_id": f"Paper{i}"}})    
    
    ### TRANSFER RESULTS ###
    #No.1
    df_results.loc[i,'File']=pdfs[i]
    df_results.loc[i,'Query 1']=query1
    df_results.loc[i,'Answer 1']=result1["quoted_answer"]["answer"]
    for j in range(0,len(result1["quoted_answer"]["citations"][:])):
        result1["quoted_answer"]["citations"][j]["metadata"]=result1["docs"][result1["quoted_answer"]["citations"][j]["source_id"]].metadata
    df_results.loc[i,'Quotes 1']=result1["quoted_answer"]["citations"]
    #No.2
    df_results.loc[i,'Query 2']=query2
    df_results.loc[i,'Answer 2']=result2["quoted_answer"]["answer"]
    for k in range(0,len(result2["quoted_answer"]["citations"][:])):
        result2["quoted_answer"]["citations"][k]["metadata"]=result2["docs"][result2["quoted_answer"]["citations"][k]["source_id"]].metadata
    df_results.loc[i,'Quotes 2']=result2["quoted_answer"]["citations"]
    #No.3
    df_results.loc[i,'Query 3']=query3
    df_results.loc[i,'Answer 3']=result3["quoted_answer"]["answer"]
    for l in range(0,len(result3["quoted_answer"]["citations"][:])):
        result3["quoted_answer"]["citations"][l]["metadata"]=result3["docs"][result3["quoted_answer"]["citations"][l]["source_id"]].metadata
    df_results.loc[i,'Quotes 3']=result3["quoted_answer"]["citations"]
    #No.4
    df_results.loc[i,'Query 4']=query4
    df_results.loc[i,'Answer 4']=result4["quoted_answer"]["answer"]
    for l in range(0,len(result4["quoted_answer"]["citations"][:])):
        result4["quoted_answer"]["citations"][l]["metadata"]=result4["docs"][result4["quoted_answer"]["citations"][l]["source_id"]].metadata
    df_results.loc[i,'Quotes 4']=result4["quoted_answer"]["citations"]
    #No.5
    df_results.loc[i,'Query 5']=query5
    df_results.loc[i,'Answer 5']=result5["quoted_answer"]["answer"]
    for l in range(0,len(result5["quoted_answer"]["citations"][:])):
        result5["quoted_answer"]["citations"][l]["metadata"]=result5["docs"][result5["quoted_answer"]["citations"][l]["source_id"]].metadata
    df_results.loc[i,'Quotes 5']=result5["quoted_answer"]["citations"]
    #No.6
    df_results.loc[i,'Query 6']=query6
    df_results.loc[i,'Answer 6']=result6["quoted_answer"]["answer"]
    for l in range(0,len(result6["quoted_answer"]["citations"][:])):
        result6["quoted_answer"]["citations"][l]["metadata"]=result6["docs"][result6["quoted_answer"]["citations"][l]["source_id"]].metadata
    df_results.loc[i,'Quotes 6']=result6["quoted_answer"]["citations"]
    print(i) ###DEBUG###

### SAVE RESULTS ###
df_results.to_excel(f"{directory}/01_Results_CiteRAG_export.xlsx")