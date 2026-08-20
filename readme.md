# Message Processing Pipeline (KaStack Assignment) - L1 & L2

This repository contains an end-to-end NLP pipeline built to classify, extract, and sanitize incoming messages while strictly adhering to data privacy constraints. **No raw data is sent to external APIs.**

## 🚀 L2 Upgrades

### 1. Priority & Action Engine
* **How it works:** Extended the existing pipeline to determine the urgency of grouped tasks. By scanning for keywords like "urgent", "ASAP", or "tomorrow" in the grouped messages, the engine assigns Critical, High, Medium, or Low priority dynamically.
* **Details:** Outputs Priority, Reason, Signals used, and Confidence score without hallucinating details.

### 2. Related-Message Grouping (Vector Semantic Search)
* **How it works:** Instead of relying on keyword matching, this utilizes `sentence-transformers` (`all-MiniLM-L6-v2`) to create high-dimensional embeddings of every message.
* **Details:** Messages with a cosine similarity > 0.6 are grouped together. Chronological ordering is preserved, allowing the system to update statuses correctly (e.g., if a later message says "completed", the entire task group is marked as completed).

### 3. Semantic Assistant & Privacy Routing
* **How it works:** A chat interface for answering user queries using processed local data.
* **Privacy Routing Firewall:**
  - **Blocked:** Uses regex to block queries containing sensitive PII (passwords, OTPs).
  - **Confirmation Required:** Blocks bulk extraction attempts (e.g., "analyze all messages") pending user confirmation.
  - **Processed Locally:** Uses cosine similarity between the query embedding and the aggregated group embeddings to fetch the most relevant group and construct a factual answer with supporting message IDs.

### 4. Benchmarking
* **How it works:** The execution block is wrapped in `time.time()` to measure exact performance. The time taken for standard NLP (Zero-Shot + spaCy) is compared directly against the vector grouping (Sentence-Transformers) time, displayed right after processing.

## 🧠 L1 Methodology (Preserved)

### 1. Sensitive Information Detection & Masking
* **How it works:** Regex masking for common PII before model ingestion. High-risk items trigger a "do_not_store" recommendation.

### 2. Message Classification
* **How it works:** Local Zero-Shot Classification via Hugging Face `transformers` using `facebook/bart-large-mnli`.

### 3. Tasks and Events Extraction
* **How it works:** Local `spaCy` Named Entity Recognition (NER) model (`en_core_web_sm`) for strict entity extraction.

## ⚠️ Assumptions and Limitations
* **Limitation (Speed & Memory):** Loading three models (`bart-large-mnli`, `spacy`, `all-MiniLM-L6-v2`) requires significant memory. Optimization via mean embeddings for groups is done to speed up the Semantic Assistant search.
* **Limitation:** Context logic is still somewhat heuristic despite vector representations; further threshold tuning might be needed for perfect grouping in noisy datasets.

## 🤖 AI-Tool Usage Declaration
During the development of this L2 upgrade, an AI assistant was used for scaffolding the Streamlit UI, constructing cosine-similarity matrix logic, and drafting this markdown file. All algorithmic choices, particularly the privacy router and local embedding constraints, were independently verified to ensure strict adherence to assignment requirements.
