import datetime
import chromadb
import traceback

import csv
from datetime import datetime

from chromadb.utils import embedding_functions

from model_configurations import get_model_configuration

gpt_emb_version = 'text-embedding-ada-002'
gpt_emb_config = get_model_configuration(gpt_emb_version)

dbpath = "./"
csvFile = "COA_OpenData.csv"
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key = gpt_emb_config['api_key'],
    api_base = gpt_emb_config['api_base'],
    api_type = gpt_emb_config['openai_type'],
    api_version = gpt_emb_config['api_version'],
    deployment_id = gpt_emb_config['deployment_name']
)

import sqlite3

def GetTables(db_file = 'chroma.sqlite3'):

    conn = sqlite3.connect(db_file)

    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cur.fetchall()

    # 對每個表格執行查詢並打印其內容
    for table in tables:
        table_name = table[0]
        print(f"Table: {table_name}")
        cur.execute(f"SELECT * FROM {table_name}")
        results = cur.fetchall()
        for row in results:
            print(row)
        print("\n")  # 分隔各個表格的輸出



def generate_hw01(question):
    with open(csvFile, mode="r", encoding="utf-8") as file:
        lines = csv.reader(file)
        
        documents = []
        
        metadatas = []
        
        ids = []
        
        for i, line in enumerate(lines):
            if i==0:
                continue
            documents.append(line[5])
            date_obj = datetime.strptime(line[9], '%Y-%m-%d')
            date_timestamp = str(round(date_obj.timestamp()))            
            if line[2] == "農場":
                line[2] = "旅遊"
            metadatas.append({
            "file_name": csvFile,
            "name": line[1],
            "type": line[2],
            "address": line[3],
            "tel": line[4],
            "city": line[7],
            "town": line[8],
            "date": date_timestamp
            })
            ids.append(line[0])
            
    chroma_client = chromadb.PersistentClient(path=dbpath)
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    ) 
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,            
    )
    
    return collection    
def generate_hw02(question, city, store_type, start_date, end_date):
    pass
    
def generate_hw03(question, store_name, new_store_name, city, store_type):
    pass
    
def demo(question):
    chroma_client = chromadb.PersistentClient(path=dbpath)

    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )
    
    return collection
