import streamlit as st
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel
import os

load_dotenv()

# Get the token
hf_token = os.getenv("HUGGING_FACE_TOKEN")

# --- 1. Define Label Mapping (Hardcoded for portability) ---
id2label = {
    0: "Bank account or service",
    1: "Checking or savings account",
    2: "Consumer Loan",
    3: "Credit card",
    4: "Credit card or prepaid card",
    5: "Credit reporting",
    6: "Credit reporting or other personal consumer reports",
    7: "Credit reporting, credit repair services, or other personal consumer reports",
    8: "Debt collection",
    9: "Debt or credit management",
    10: "Money transfer, virtual currency, or money service",
    11: "Money transfers",
    12: "Mortgage",
    13: "Other financial service",
    14: "Payday loan",
    15: "Payday loan, title loan, or personal loan",
    16: "Payday loan, title loan, personal loan, or advance loan",
    17: "Prepaid card",
    18: "Student loan",
    19: "Vehicle loan or lease"
}

label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)

# --- 2. Page Config ---
st.set_page_config(page_title="Ticket Triage AI", page_icon="🎫")
st.title("Intelligent Ticket Triage")
st.markdown("Enter a customer complaint below to automatically route it to the correct department.")

# --- 3. Load Model (Cached) ---
@st.cache_resource
def load_model():
    # Base Model ID
    base_model_id = "meta-llama/Meta-Llama-3.1-8B"
    
    # Path to saved adapter
    adapter_path = "./model" 

    # Quantization Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Load Base Model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    
    # Load Adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

# Display spinner while loading
with st.spinner("Loading Llama 3 AI Model... (this may take a minute)"):
    try:
        model, tokenizer = load_model()
        st.success("Model Loaded Successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# --- 4. User Input & Prediction ---
user_text = st.text_area("Customer Complaint:", height=150, placeholder="Example: I tried to pay my bill but the website crashed...")

if st.button("Classify Ticket"):
    if not user_text:
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            # Tokenize
            inputs = tokenizer(user_text, return_tensors="pt", truncation=True, max_length=512).to("cuda")
            
            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Get Probabilities
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class_id = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][predicted_class_id].item()
            
            # Map ID to Label
            predicted_label = id2label.get(predicted_class_id, "Unknown")
            
            # --- Display Results ---
            st.markdown("---")
            st.subheader("Classification Result")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Department", predicted_label)
            with col2:
                # Color code confidence
                color = "green" if confidence > 0.8 else "orange" if confidence > 0.5 else "red"
                st.markdown(f"**Confidence Score:** :{color}[{confidence:.1%}]")
            
            # Optional: Show full distribution
            with st.expander("See all confidence scores"):
                # Create a dict of label: score
                scores = {id2label[i]: probs[0][i].item() for i in range(num_labels)}
                # Sort by score descending
                sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
                st.write(sorted_scores)