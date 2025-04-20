import streamlit as st

# Page configuration must be the first Streamlit command
st.set_page_config(
    page_title="Advanced NLP System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    .sentiment-positive {
        color: #28a745;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #dc3545;
        font-weight: bold;
    }
    .entity-tag {
        background-color: #e9ecef;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        margin: 0.2rem;
        display: inline-block;
    }
    </style>
    """, unsafe_allow_html=True)

from nlp_system.nlp_system import AdvancedNLPSystem
import time
import pandas as pd
import plotly.express as px
from typing import Dict, List, Any
import json
import numpy as np

# Initialize the NLP system
@st.cache_resource
def get_nlp_system():
    return AdvancedNLPSystem()

def convert_numpy_types(obj: Any) -> Any:
    """Convert NumPy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    return obj

def display_sentiment(sentiment_result: Dict):
    """Display sentiment analysis results with visual indicators."""
    st.markdown("### 🎭 Sentiment Analysis")
    
    # Create a color-coded sentiment indicator
    sentiment = sentiment_result["sentiment"]
    confidence = sentiment_result["confidence"]
    
    sentiment_color = "sentiment-positive" if sentiment == "POSITIVE" else "sentiment-negative"
    st.markdown(f"""
        <div style='padding: 1rem; border-radius: 0.5rem; background-color: #f8f9fa;'>
            <span class='{sentiment_color}'>{sentiment}</span>
            <br>
            Confidence: {confidence:.2%}
        </div>
    """, unsafe_allow_html=True)

def display_entities(entities: List[Dict]):
    """Display named entities in a structured format."""
    st.markdown("### 🏷️ Named Entities")
    
    if not entities:
        st.info("No named entities found in the text.")
        return
    
    # Group entities by type
    entity_types = {}
    for entity in entities:
        entity_type = entity["entity_group"]
        if entity_type not in entity_types:
            entity_types[entity_type] = []
        entity_types[entity_type].append(entity["word"])
    
    # Display entities in expandable sections
    for entity_type, words in entity_types.items():
        with st.expander(f"{entity_type} ({len(words)} found)"):
            st.write(", ".join(words))

def display_key_phrases(key_phrases: List[str]):
    """Display key phrases in a visually appealing way."""
    st.markdown("### 🔑 Key Phrases")
    
    if not key_phrases:
        st.info("No key phrases found in the text.")
        return
    
    # Create a word cloud-like display
    cols = st.columns(3)
    for i, phrase in enumerate(key_phrases):
        cols[i % 3].markdown(f"""
        <div style='padding: 10px; margin: 5px; border-radius: 5px; background-color: #f0f2f6;'>
            {phrase}
        </div>
        """, unsafe_allow_html=True)

def display_dependencies(dependencies: List[tuple]):
    """Display dependency relationships in a structured format."""
    st.markdown("### 🔄 Dependency Parsing")
    
    if not dependencies:
        st.info("No dependency relationships found in the text.")
        return
    
    # Create a DataFrame for better visualization
    df = pd.DataFrame(dependencies, columns=["Word", "Dependency", "Head"])
    st.dataframe(df, use_container_width=True)

def main():
    # Initialize NLP system
    nlp_system = get_nlp_system()
    
    # Sidebar
    st.sidebar.title("🤖 Advanced NLP System")
    st.sidebar.markdown("---")
    
    # Analysis options
    analysis_options = st.sidebar.multiselect(
        "Select Analysis Types",
        ["Sentiment Analysis", "Named Entities", "Key Phrases", "Dependencies"],
        default=["Sentiment Analysis", "Named Entities", "Key Phrases"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About
    This application provides advanced Natural Language Processing capabilities:
    - 🎭 Sentiment Analysis
    - 🏷️ Named Entity Recognition
    - 🔑 Key Phrase Extraction
    - 🔄 Dependency Parsing
    """)
    
    # Main content
    st.title("Emmanuel.O - Advanced Text Analysis")
    
    # Text input
    text_input = st.text_area(
        "Enter text to analyze",
        height=150,
        placeholder="Enter your text here... (e.g., 'Apple Inc. is planning to open a new store in New York City next month.')"
    )
    
    # Batch input option
    use_batch = st.checkbox("Enable batch processing")
    
    if use_batch:
        uploaded_file = st.file_uploader(
            "Upload a CSV or TXT file with texts (one per line)",
            type=['csv', 'txt']
        )
    
    analyze_button = st.button("Analyze Text")
    
    if analyze_button:
        if not text_input and not (use_batch and uploaded_file):
            st.error("Please enter some text or upload a file for analysis.")
            return
        
        with st.spinner("Analyzing text..."):
            if use_batch and uploaded_file:
                # Handle batch processing
                if uploaded_file.type == "text/csv":
                    df = pd.read_csv(uploaded_file)
                    texts = df.iloc[:, 0].tolist()  # Assume first column contains texts
                else:
                    texts = uploaded_file.getvalue().decode().split('\n')
                    texts = [t.strip() for t in texts if t.strip()]
                
                results = []
                progress_bar = st.progress(0)
                
                for i, text in enumerate(texts):
                    result = nlp_system.analyze_text(text)
                    results.append(result)
                    progress_bar.progress((i + 1) / len(texts))
                
                # Display batch results
                st.subheader(f"Analysis Results ({len(texts)} texts)")
                
                for i, (text, result) in enumerate(zip(texts, results)):
                    with st.expander(f"Text {i+1}: {text[:100]}..."):
                        display_analysis_results(result, analysis_options)
            
            else:
                # Single text analysis
                result = nlp_system.analyze_text(text_input)
                display_analysis_results(result, analysis_options)
                
                # Convert results to JSON-serializable format
                serializable_results = convert_numpy_types(result)
                
                # Add a download button for the results
                st.download_button(
                    label="Download Analysis Results",
                    data=json.dumps(serializable_results, indent=2),
                    file_name="nlp_analysis_results.json",
                    mime="application/json"
                )

def display_analysis_results(result, analysis_options):
    """Display the analysis results based on selected options."""
    if "Sentiment Analysis" in analysis_options:
        display_sentiment(result["sentiment"])
    
    if "Named Entities" in analysis_options:
        display_entities(result["entities"])
    
    if "Key Phrases" in analysis_options:
        display_key_phrases(result["key_phrases"])
    
    if "Dependencies" in analysis_options:
        display_dependencies(result["dependencies"])

if __name__ == "__main__":
    main() 