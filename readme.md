# Message Processing Pipeline (KaStack Assignment)

This repository contains an end-to-end NLP pipeline built to classify, extract, and sanitize incoming messages while strictly adhering to data privacy constraints. **No raw data is sent to external APIs.**

## 🧠 Methodology

### 1. Sensitive Information Detection & Masking (Data Privacy First)
* **How it works:** Before any machine learning models process the text, the message is scanned using Python's `re` (Regex) module. 
* **Details:** I configured strict patterns for common PII (One-Time Passwords, Credit Cards, Email Addresses, and Phone Numbers). When detected, the actual value is replaced with a masked tag (e.g., `[ONE_TIME_PASSWORD_MASKED]`). High-risk items trigger a "do_not_store" recommendation.

### 2. Message Classification
* **How it works:** I utilize a local Zero-Shot Classification pipeline via Hugging Face `transformers` using the `facebook/bart-large-mnli` model. 
* **Details:** The *masked* text is fed into the local model alongside the 6 target categories. The model returns the most probable category and a confidence score. This approach requires no fine-tuning and runs entirely locally.

### 3. Tasks and Events Extraction
* **How it works:** If a message is classified as "Action Required" or "Meeting or Event", it is passed to a local `spaCy` Named Entity Recognition (NER) model (`en_core_web_sm`).
* **Details:** The model searches for specific entity tags: `DATE`, `TIME`, and `PERSON`. If the NLP model does not confidently find an entity, the field is strictly left as `null` to avoid AI hallucinations. Priority is determined by rule-based keyword matching (e.g., "urgent", "ASAP").

## ⚠️ Assumptions and Limitations
* **Limitation (Speed):** Because the Zero-Shot classifier runs locally on CPU hardware (via Streamlit Community Cloud), processing all 900 rows takes several minutes. 
* **Assumption (Missing Data):** Following the strict instructions to "not guess missing information," if `spaCy` fails to extract a date or time, it is marked as `null` rather than using context clues to guess.
* **Limitation (Contextual Sarcasm/Slang):** The zero-shot model may occasionally misclassify highly nuanced, sarcastic, or poorly spelled messages.

## 🤖 AI-Tool Usage Declaration
During the development of this project, an AI assistant (Claude/ChatGPT/Gemini) was used as a pair-programming partner to help structure the Streamlit UI, refine regex patterns for PII detection, and draft boilerplate markdown. However, all architectural decisions, compliance with assignment constraints (local-only models), and code comprehension were independently verified and owned by me.
