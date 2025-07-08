Absolutely! Here’s a professional, well-structured `README.md` for your project, reflecting the two tasks you have completed.

---

# 📊 Intelligent Complaint Analysis for Financial Services

**RAG-powered chatbot to turn customer feedback into actionable insights**

---

## 🚀 Overview

This project builds an intelligent, Retrieval-Augmented Generation (RAG) powered chatbot for **CrediTrust Financial**, a digital finance company operating across East Africa. The chatbot transforms raw, unstructured customer complaint data into strategic insights. It allows internal teams—like Product Managers, Compliance, and Support—to query customer pain points in natural language and receive evidence-backed, concise answers instantly.

---

## 🎯 Business Objectives

- **Reduce time to detect major complaint trends** from days to minutes.
- **Empower non-technical teams** to independently explore customer feedback without data analysts.
- **Shift from reactive to proactive problem solving**, leveraging real-time customer insights.

---

## 💡 Key Features

✅ Supports natural language questions like:

> _“Why are customers unhappy with BNPL?”_

✅ Uses **semantic search** (via FAISS / ChromaDB) over complaint narratives.
✅ Answers questions by grounding LLM responses in retrieved complaints.
✅ Supports filtering and comparative analysis across multiple products:

- Credit Cards
- Personal Loans
- Buy Now, Pay Later (BNPL)
- Savings Accounts
- Money Transfers

---

## 📂 Project Structure

```
project-root/
│
├── data/
│   └── filtered_complaints.csv       # Cleaned dataset after Task 1
│
├── vector_store/
│   └── faiss_index/                  # Vector DB from Task 2
│
├── notebooks/
│   └── 1.0-eda.ipynb       # EDA & preprocessing
│
├── src/
│   ├── chunking_and_embedding.py         # Text chunking, embedding, indexing
│   └── utils.py                      # Utility functions
│
├── reports/
│   └── analysis_report.md            # Summary of EDA findings & design choices
│
├── README.md                         # This file
└── requirements.txt                  # Python dependencies
```

---

## ✅ Completed Tasks

### 📝 Task 1: Exploratory Data Analysis & Preprocessing

- Loaded CFPB consumer complaints dataset.
- Performed EDA to understand:

  - Complaint distribution across products.
  - Narrative length distributions.
  - Share of complaints with missing narratives.

- Filtered dataset to include only:

  - Five specified products (Credit Cards, Personal Loans, BNPL, Savings Accounts, Money Transfers).
  - Complaints that have non-empty narratives.

- Cleaned narratives:

  - Lowercased text.
  - Removed special characters and boilerplate phrases.

- Saved filtered dataset to:
  ➡️ `data/filtered_complaints.csv`.

### 🧩 Task 2: Chunking, Embedding & Vector Store Indexing

- Implemented a text chunking strategy:

  - Used `LangChain`'s `RecursiveCharacterTextSplitter`.
  - Final parameters: `chunk_size=500`, `chunk_overlap=50`.

- Chose `sentence-transformers/all-MiniLM-L6-v2` for embeddings, balancing speed & semantic quality.
- Embedded all text chunks & stored in a FAISS index along with metadata (complaint ID, product).
- Persisted the vector store to:
  ➡️ `vector_store/faiss_index/`.

---

## ⚙️ How to Run

### 🔧 Environment Setup

```bash
# Clone the repo
git clone https://github.com/dagmaros27/Intelligent-Complaint-Analysis
cd Intelligent-Complaint-Analysis

# Set up a Python virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

### 🚀 Running the Pipeline

1. **EDA & Preprocessing**

   ```bash
   jupyter notebook notebooks/1.0-eda.ipynb
   ```

2. **Chunking, Embedding & Indexing**

   ```bash
   python src/chunking_embedding.py
   ```

---

## 📝 Key Learnings So Far

- Handling real-world complaint data is messy—narrative lengths vary hugely, and many complaints are vague.
- Short narratives dominate, but long ones often contain richer context—highlighting why chunking was critical.
- `all-MiniLM-L6-v2` proved an excellent starting point for balancing semantic relevance with performance on modest hardware.

---

## 📌 Next Steps

- Build the **retriever-LLM pipeline** to generate grounded answers.
- Develop a simple **streamlit-based UI** to allow natural language querying.
- Run user simulations to test insights across all five product categories.

---
