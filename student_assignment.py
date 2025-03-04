import datetime
import chromadb
import traceback

import csv
import datetime

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


def timestampTrans(time):
    if isinstance(time, datetime.datetime):
        date_timestamp = int(round(time.timestamp()))
    else:
        date_obj = datetime.datetime.strptime(time, '%Y-%m-%d')
        date_timestamp = int(round(date_obj.timestamp()))   
    return date_timestamp
def generate_hw01(question):
    chroma_client = chromadb.PersistentClient(path=dbpath)
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    ) 
    if collection.count() == 0:
        with open(csvFile, mode="r", encoding="utf-8") as file:
            lines = csv.reader(file)
            
            documents = []
            
            metadatas = []
            
            ids = []
            
            for i, line in enumerate(lines):
                if i==0:
                    continue
                documents.append(line[5])
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
                "date": timestampTrans(line[9])
                })
                ids.append(line[0])
                

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,            
        )
    
    return collection    
def generate_hw02(question, city, store_type, start_date, end_date):
    start_date = timestampTrans(start_date)
    end_date = timestampTrans(end_date)
    
    chroma_client = chromadb.PersistentClient(path=dbpath)
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )
    
    city_filter = [{"city": {"$eq": t}} for t in city] if city else []

    filters = []

    if city_filter:
        filters.append({"$or": city_filter})

    if store_type:
        filters.append({"type": {"$eq": store_type}})

    if start_date:
        filters.append({"date": {"$gte": start_date}})

    if end_date:
        filters.append({"date": {"$lte": end_date}})
    
    filterMetadatas = {"$and": filters} if filters else {}

    # print("Filter Metadatas:", filterMetadatas)

    result = collection.query(
        query_texts=[question],
        where=filterMetadatas,
        include=["metadatas", "distances"],
        n_results=10
    )
    # print(result)
    metadatas = result.get("metadatas", [])
    distances = result.get("distances", [])

    filtered_results = []
    for metadata_list, distance_list in zip(metadatas, distances):
        for metadata, distance in zip(metadata_list, distance_list):
            if distance < 0.2 and "name" in metadata:
                filtered_results.append((metadata, distance))

    filtered_results.sort(key=lambda x: x[1])

    names = [metadata["name"] for metadata, distance in filtered_results if "name" in metadata]

    return names
    
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
