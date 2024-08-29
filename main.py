import re
import json
import plotly.graph_objects as go
import pandas as pd
from transformers import pipeline
import streamlit as st

# Load a pre-trained BERT model for NER
nlp_bert = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

def extract_information_with_bert(text: str) -> dict:
    # Initialize dictionaries
    customer_requirements = {
        "Car Type": None,
        "Fuel Type": None,
        "Color": None,
        "Distance Travelled": None,
        "Make Year": None,
        "Transmission Type": None
    }
    company_policies = {
        "Free RC Transfer": False,
        "5-Day Money Back Guarantee": True,
        "Free RSA for One Year": False,
        "Return Policy": True
    }
    customer_objections = {
        "Refurbishment Quality": False,
        "Car Issues": True,
        "Price Issues": True,
        "Customer Experience Issues": False
    }

    # Extract company policies and customer objections using BERT
    doc_bert = nlp_bert(text)
    
    # Track matches in a set
    matched_policies = set()
    matched_objections = set()

    for entity in doc_bert:
        word = entity['word'].lower()

        # Check for Company Policies keywords
        if 'rc transfer' in word:
            matched_policies.add("Free RC Transfer")
        if 'money back guarantee' in word:
            matched_policies.add("5-Day Money Back Guarantee")
        if 'rsa' in word or 'roadside assistance' in word:
            matched_policies.add("Free RSA for One Year")
        if 'return policy' in word:
            matched_policies.add("Return Policy")

        # Check for Customer Objections keywords
        if 'refurbishment' in word:
            matched_objections.add("Refurbishment Quality")
        if 'car issues' in word or 'reliability' in word:
            matched_objections.add("Car Issues")
        if 'price' in word:
            matched_objections.add("Price Issues")
        if 'wait time' in word or 'salesperson behavior' in word:
            matched_objections.add("Customer Experience Issues")

    # Update company policies and objections based on matches
    for policy in matched_policies:
        company_policies[policy] = True

    for objection in matched_objections:
        customer_objections[objection] = True

    # Additional regex-based extraction for Customer Requirements
    patterns = {
        "Car Type": r'\b(suv|sedan|hatchback|mpv|wagon)\b',
        "Fuel Type": r'\b(diesel|petrol|gas|electric)\b',
        "Color": r'\b(blue|white|red|black|grey|green)\b',
        "Distance Travelled": r'\b(\d+(?:,\d+)*)\s*km\b',
        "Make Year": r'\b(20\d{2})\b',
        "Transmission Type": r'\b(manual|automatic|any)\b'
    }

    for requirement, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            customer_requirements[requirement] = match.group(0)

    return {
        "Customer Requirements": customer_requirements,
        "Company Policies Discussed": company_policies,
        "Customer Objections": customer_objections
    }

def create_pie_chart(data, title):
    labels = [k for k, v in data.items() if isinstance(v, bool) or v is not None]
    values = [1 if data[k] else 0 for k in labels]
    
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
    fig.update_layout(title_text=title)
    return fig

def main():
    st.title("Car Sales Conversation Analyzer")

    uploaded_file = st.file_uploader("Choose a conversation text file", type=["txt"])

    if uploaded_file is not None:
        # Read file
        text = uploaded_file.read().decode('utf-8')

        # Extract information using the BERT-enhanced function
        extracted_info = extract_information_with_bert(text)

        # Display the extracted information
        st.json(extracted_info)

        # Button to download JSON file
        json_data = json.dumps(extracted_info, indent=4)
        st.download_button(
            label="Download JSON file",
            data=json_data,
            file_name="extracted_information.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()
