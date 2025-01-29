from flask import Flask, request, jsonify
import spacy
import pandas as pd
import json
import fitz  # PyMuPDF for PDFs
from bs4 import BeautifulSoup
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

app = Flask(__name__)

nlp = spacy.load("en_core_web_sm")

model = SentenceTransformer("all-MiniLM-L6-v2")

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "put_your_user"
NEO4J_PASSWORD = "put_your_password"
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

embedding_dim = 384  # Model output dimension
index = faiss.IndexFlatL2(embedding_dim)
entity_embeddings = {}

def load_text_from_csv(file_path):
    df = pd.read_csv(file_path)
    return " ".join(df.iloc[:, 0].astype(str))  # Assume text is in the first column

def load_text_from_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return " ".join(data.values()) if isinstance(data, dict) else " ".join(str(d) for d in data)

def load_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    return " ".join(page.get_text() for page in doc)

def load_text_from_html(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
    return soup.get_text()

def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def extract_relations(text):
    doc = nlp(text)
    relations = []
    for ent in doc.ents:
        for token in ent.head.children:
            if token.dep_ in ("prep", "agent", "dobj"):
                relations.append((ent.text, token.text, ent.head.text))
    return relations

def store_knowledge_graph(entities, relations):
    with driver.session() as session:
        for entity, label in entities:
            session.run("MERGE (e:Entity {name: $name, label: $label})", name=entity, label=label)
            embedding = model.encode(entity, convert_to_numpy=True)
            entity_embeddings[entity] = embedding
            index.add(np.array([embedding]))
        for entity1, relation, entity2 in relations:
            session.run(
                "MATCH (e1:Entity {name: $entity1}), (e2:Entity {name: $entity2}) "
                "MERGE (e1)-[:RELATION {type: $relation}]->(e2)",
                entity1=entity1, entity2=entity2, relation=relation
            )

def process_text(file_path, file_type):
    loaders = {
        "csv": load_text_from_csv,
        "json": load_text_from_json,
        "pdf": load_text_from_pdf,
        "html": load_text_from_html
    }
    
    if file_type not in loaders:
        raise ValueError("Unsupported file type")
    
    raw_text = loaders[file_type](file_path)
    clean_text = preprocess_text(raw_text)
    entities = extract_entities(clean_text)
    relations = extract_relations(clean_text)
    store_knowledge_graph(entities, relations)
    
    return {
        "entities": entities,
        "relations": relations
    }

@app.route('/process', methods=['POST'])
def process():
    file_path = request.json.get('file_path')
    file_type = request.json.get('file_type')
    
    if not file_path or not file_type:
        return jsonify({"error": "file_path and file_type are required"}), 400
    
    try:
        results = process_text(file_path, file_type)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/search', methods=['POST'])
def search():
    query = request.json.get('query')
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    query_embedding = model.encode(query, convert_to_numpy=True)
    D, I = index.search(np.array([query_embedding]), k=5)  # Get top-5 closest entities
    results = [list(entity_embeddings.keys())[i] for i in I[0] if i < len(entity_embeddings)]
    
    return jsonify({"similar_entities": results})

if __name__ == "__main__":
    app.run(debug=True)
