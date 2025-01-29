# CurotecProject

## Overview
This project processes large volumes of unstructured text data, extracts meaningful entities and relationships, and stores them in a **Neo4j-based Knowledge Graph**. It also supports **semantic search** using **FAISS** and **vector embeddings**.

## Features
- **Text Ingestion**: Supports CSV, JSON, PDF, and HTML files.
- **Natural Language Processing (NLP)**: Uses **spaCy** for Named Entity Recognition (NER).
- **Graph Database**: Stores extracted entities and relationships in **Neo4j**.
- **Semantic Search**: Utilizes **FAISS** for fast similarity searches.
- **Flask API**: Provides endpoints for processing text and searching the knowledge graph.
- **Web Interface**: Simple UI for file uploads and data processing.

##  Installation

### **Clone the Repository**
```bash
git clone https://github.com/josuedantas/CurotecProject.git
cd CurotecProjec

## Create a Virtual Environment

python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate      # Windows

## Install Dependencies
pip install -r requirements.txt

## Set Up Neo4j
Put your user and password.

## Run the Backend

python main_project.py

# Frontend

Open the index.html file in your browser



