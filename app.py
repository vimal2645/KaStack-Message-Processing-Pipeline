import streamlit as st
import pandas as pd
import re
import json
import spacy
from transformers import pipeline

# --- CACHING MODELS ---
@st.cache_resource
def load_models():
    # Local Zero-Shot Classifier
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    # Local NLP model for Named Entity Recognition (NER)
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        import os
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    return classifier, nlp

classifier, nlp = load_models()

# --- HELPER FUNCTIONS ---
# Pre-compile regex patterns for performance
PATTERNS = {
    "password": re.compile(r'(?i)(?:password|pwd|pass)\s*[:=]\s*([A-Za-z0-9@#$%^&+=!]{6,})'),
    "one_time_password": re.compile(r'\b\d{4,6}\b'),
    "bank_details": re.compile(r'\b(?:\d[ -]*?){13,16}\b'),
    "authentication_token": re.compile(r'\b(?:eyJ[A-Za-z0-9-_=]+\.[A-Za-z0-9-_=]+\.?[A-Za-z0-9-_.+/=]*)\b'),
    "personal_id": re.compile(r'\b(?!000|666)[0-8][0-9]{2}-(?!00)[0-9]{2}-(?!0000)[0-9]{4}\b'),
    "contact_details": re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}|\b\d{10}\b')
}

def process_sensitive_info(text, msg_id):
    findings = []
    masked_text = str(text)
    
    for sen_type, pattern in PATTERNS.items():
        if re.search(pattern, masked_text):
            masked_text = re.sub(pattern, f"[{sen_type.upper()}_MASKED]", masked_text)
            risk = "high" if sen_type in ["bank_details", "one_time_password", "authentication_token", "password", "personal_id"] else "medium"
            action = "do_not_store" if risk == "high" else "ask_for_confirmation"
            
            findings.append({
                "message_id": msg_id,
                "sensitivity_type": sen_type,
                "risk": risk,
                "masked_text": masked_text,
                "recommended_action": action
            })
            
    return masked_text, findings

def classify_message(text, msg_id):
    if not text or not str(text).strip():
        return {
            "message_id": msg_id,
            "category": "Unknown",
            "confidence": 0.0,
            "reason": "Empty message."
        }

    labels = [
        "Action Required", "Meeting or Event", "Personal Information", 
        "General Information", "Promotional", "Sensitive Information"
    ]
    result = classifier(text, candidate_labels=labels)
    top_label = result['labels'][0]
    confidence = round(result['scores'][0], 2)
    
    # Simple rule-based reasoning
    reason = f"The message was classified as '{top_label}' with {confidence} confidence based on zero-shot semantic analysis of its context."
    
    return {
        "message_id": msg_id,
        "category": top_label,
        "confidence": confidence,
        "reason": reason
    }

def extract_entities(text, msg_id, category):
    if category not in ["Action Required", "Meeting or Event"]:
        return None
        
    doc = nlp(text)
    extracted = {
        "item_id": f"TASK_{msg_id.split('_')[-1] if '_' in msg_id else msg_id}",
        "type": "event" if category == "Meeting or Event" else "task",
        "title": text[:30] + "..." if len(text) > 30 else text, 
        "description": text,
        "deadline": None,
        "time": None,
        "person": None,
        "priority": "high" if any(word in text.lower() for word in ["urgent", "asap", "immediately"]) else "medium",
        "source_message_id": msg_id
    }
    
    for ent in doc.ents:
        if ent.label_ == "DATE" and not extracted["deadline"]:
            extracted["deadline"] = ent.text
        elif ent.label_ == "TIME" and not extracted["time"]:
            extracted["time"] = ent.text
        elif ent.label_ == "PERSON" and not extracted["person"]:
            extracted["person"] = ent.text
            
    return extracted

# --- UI APP ---
st.set_page_config(layout="wide", page_title="KaStack AI Intern Project")
st.title("HR Team - Message Processing Pipeline")
st.markdown("Processed locally via `transformers` and `spacy` to ensure strict data privacy.")

col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Upload Message CSV (900 messages)", type="csv")
with col2:
    mandatory_file = st.file_uploader("Upload Mandatory IDs CSV/TXT (Optional)", type=["csv", "txt"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(f"Loaded {len(df)} messages.")
    
    mandatory_ids = []
    if mandatory_file:
        try:
            mandatory_df = pd.read_csv(mandatory_file)
            if 'message_id' in mandatory_df.columns:
                mandatory_ids = mandatory_df['message_id'].astype(str).str.strip().tolist()
            else:
                mandatory_ids = mandatory_df.iloc[:, 0].astype(str).str.strip().tolist()
            st.success(f"Loaded {len(mandatory_ids)} mandatory IDs.")
        except:
            st.warning("Could not parse mandatory IDs file. Ensure it is a simple list.")

    if st.button("Run Full Pipeline"):
        classifications, extractions, sensitive_alerts = [], [], []
        
        progress_bar = st.progress(0)
        total_rows = len(df)
        
        for index, row in df.iterrows():
            msg_id = str(row.get('message_id', row.get('Message ID', f'MSG_{index}'))).strip()
            raw_text = str(row.get('message', row.get('Message', '')))
            
            # 1. Mask
            masked_text, pii_data = process_sensitive_info(raw_text, msg_id)
            if pii_data:
                sensitive_alerts.extend(pii_data)
            
            # 2. Classify (pass masked text for safety)
            class_data = classify_message(masked_text, msg_id)
            classifications.append(class_data)
            
            # 3. Extract
            extract_data = extract_entities(masked_text, msg_id, class_data['category'])
            if extract_data:
                extractions.append(extract_data)
                
            # Update Progress
            if index % 10 == 0:
                progress_bar.progress(min((index + 1) / total_rows, 1.0))
                
        progress_bar.progress(1.0)
        st.success("Pipeline Execution Complete!")
        
        st.session_state['classifications'] = classifications
        st.session_state['extractions'] = extractions
        st.session_state['sensitive_alerts'] = sensitive_alerts
        st.session_state['pipeline_run'] = True

    if st.session_state.get('pipeline_run'):
        classifications = st.session_state['classifications']
        extractions = st.session_state['extractions']
        sensitive_alerts = st.session_state['sensitive_alerts']

        # --- GENERATE DOWNLOAD FILES ---
        st.subheader("Download Generated JSON Output Files")
        d_col1, d_col2, d_col3 = st.columns(3)
        with d_col1:
            st.download_button("Download Classifications", data=json.dumps(classifications, indent=2), file_name="classifications.json", mime="application/json")
        with d_col2:
            st.download_button("Download Extractions", data=json.dumps(extractions, indent=2), file_name="extractions.json", mime="application/json")
        with d_col3:
            st.download_button("Download Sensitive Info", data=json.dumps(sensitive_alerts, indent=2), file_name="sensitive_info.json", mime="application/json")

        # --- VIEW TABS ---
        st.subheader("Data Viewer")
        tab1, tab2, tab3 = st.tabs(["Classifications", "Task Extractions", "Sensitive Data Alerts"])
        
        with tab1:
            if mandatory_ids:
                st.write("**Mandatory IDs View:**")
                st.json([c for c in classifications if c['message_id'] in mandatory_ids])
            else:
                st.json(classifications[:10])
        with tab2:
            st.json(extractions[:10])
        with tab3:
            st.json(sensitive_alerts[:10])