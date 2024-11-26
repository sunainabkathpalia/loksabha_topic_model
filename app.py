import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from gensim.models.ldaseqmodel import LdaSeqModel
from gensim.corpora import Dictionary
import numpy as np
import pyLDAvis
import streamlit.components.v1 as components
import tempfile
import json

# Page config
st.set_page_config(
    page_title="Indian Parliamentary Bills Analysis",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("Thematic Analysis of Indian Parliamentary Bills")
st.markdown("""
This dashboard visualizes the evolution of themes in Indian parliamentary bills over time,
using Dynamic Topic Modeling to track shifting legislative priorities.
""")

# Function to load data and model
@st.cache_data
def load_data():
    bills_data = pd.read_csv("bills_data_processed.csv")
    bills_data = bills_data.sort_values(by='year', ascending=True)
    bills_data = bills_data[(bills_data['year'] != 2001) & (bills_data['year'] != 2004)].copy()
    return bills_data

@st.cache_resource
def load_model_and_corpus():
    # Load your saved DTM model
    dtm = LdaSeqModel.load("ldaseq7_model")
    
    # Recreate dictionary and corpus
    bills_data = load_data()
    texts = [text.split() for text in bills_data['cleaned_summary']]
    dictionary = Dictionary(texts)
    dictionary.filter_extremes(no_below=5, no_above=0.7)
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    return dtm, dictionary, corpus

def get_topic_evolution_data(dtm, dictionary, bills_data):
    """
    Get topic evolution data for visualization
    """
    years = [int(x) for x in sorted(bills_data['year'].unique())]
    n_topics = dtm.num_topics
    
    # Initialize results dictionary
    topic_evolution = {i: {"years": years, "terms": {}} for i in range(n_topics)}
    
    # For each topic
    for topic in range(n_topics):
        # Get all terms across all time periods
        all_terms = set()
        for t, year in enumerate(years):
            topic_terms = dict(dtm.print_topic(topic, t, top_terms=10))
            all_terms.update(topic_terms.keys())
        
        # For each term, get its probability over time
        for term in all_terms:
            probabilities = []
            for t, _ in enumerate(years):
                topic_terms = dict(dtm.print_topic(topic, t, top_terms=10))
                probabilities.append(topic_terms.get(term, 0))
            topic_evolution[topic]["terms"][term] = probabilities
    
    return topic_evolution

def create_status_distribution(bills_data):
    """
    Create a horizontal bar chart for bill status distribution
    """
    status_counts = bills_data['status'].value_counts().reset_index()
    status_counts.columns = ['Status', 'Count']
    
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=status_counts['Count'],
            y=status_counts['Status'],
            orientation='h'
        )
    )
    
    fig.update_layout(
        title="Distribution of Bill Statuses",
        xaxis_title="Number of Bills",
        yaxis_title="Status",
        height=300,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    return fig

def create_pyldavis_visualization(dtm, corpus, time_slice):
    """
    Create pyLDAvis visualization for selected time slice
    """
    # Get visualization data for selected time slice
    doc_topic, topic_term, doc_lengths, term_frequency, vocab = dtm.dtm_vis(
        time=time_slice, 
        corpus=corpus
    )
    
    # Prepare visualization
    vis_data = pyLDAvis.prepare(
        topic_term_dists=topic_term,
        doc_topic_dists=doc_topic,
        doc_lengths=doc_lengths,
        vocab=vocab,
        term_frequency=term_frequency
    )
    
    # Save visualization to temporary HTML file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as f:
        pyLDAvis.save_html(vis_data, f.name)
        return f.name

# Load data and model
bills_data = load_data()
try:
    dtm, dictionary, corpus = load_model_and_corpus()
    model_loaded = True
except:
    st.error("Error loading DTM model. Please ensure the model file exists.")
    model_loaded = False

if model_loaded:
    # Get topic evolution data
    topic_evolution = get_topic_evolution_data(dtm, dictionary, bills_data)
    
    # Create tabs for different visualizations
    tab1, tab2 = st.tabs(["Topic Evolution", "LDA Visualization"])
    
    # Define topic classifications
    topic_classifications = {
        0: "Labor and Social Welfare",
        1: "Criminal Justice and Law Enforcement",
        2: "Governance and Administrative",
        3: "Digital and Financial Regulation",
        4: "Public Service and Resource Management",
        5: "Education and Development",
        6: "Infrastructure and Utilities"
    }

    with tab1:
        # Topic evolution visualization
        st.subheader("Topic Evolution Over Time")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Get selected topic
            selected_topic = st.selectbox(
                "Select Topic to Visualize",
                options=range(dtm.num_topics),
                format_func=lambda x: f"Topic {x + 1}: {topic_classifications[x]}"
            )
        
        with col2:
            # Download button moved here
            download_data = []
            for topic in range(dtm.num_topics):
                for term in topic_evolution[topic]["terms"]:
                    for i, year in enumerate(topic_evolution[topic]["years"]):
                        download_data.append({
                            "Topic": f"{topic + 1}: {topic_classifications[topic]}",
                            "Term": term,
                            "Year": year,
                            "Probability": topic_evolution[topic]["terms"][term][i]
                        })
            
            download_df = pd.DataFrame(download_data)
            st.download_button(
                "📥 Download Topic Evolution Data",
                download_df.to_csv(index=False).encode('utf-8'),
                "topic_evolution.csv",
                "text/csv",
                key='download-csv'
            )

        # Create topic evolution plot
        fig = go.Figure()
        
        # Get years and top terms for selected topic
        years = topic_evolution[selected_topic]["years"]
        terms_data = topic_evolution[selected_topic]["terms"]
        
        # Sort terms by average probability
        top_terms = sorted(
            terms_data.keys(),
            key=lambda x: np.mean(terms_data[x]),
            reverse=True
        )[:10]
        
        # Add trace for each term
        for term in top_terms:
            fig.add_trace(
                go.Scatter(
                    x=years,
                    y=terms_data[term],
                    name=term,
                    mode='lines+markers'
                )
            )
        
        fig.update_layout(
            title=f"Evolution of Terms in {topic_classifications[selected_topic]}",
            xaxis_title="Year",
            yaxis_title="Term Probability",
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.05
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show current topic composition
        st.subheader(f"Current Composition: {topic_classifications[selected_topic]}")
        latest_time_slice = len(topic_evolution[0]["years"]) - 1
        current_terms = dict(dtm.print_topic(selected_topic, latest_time_slice, top_terms=10))
        
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                y=list(current_terms.keys()),
                x=list(current_terms.values()),
                orientation='h'
            )
        )
        
        fig.update_layout(
            xaxis_title="Probability",
            yaxis_title="Term",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # pyLDAvis visualization
        st.subheader("LDA Visualization")
        
        # Create year options
        years = sorted(bills_data['year'].unique())
        year_to_slice = {year: idx for idx, year in enumerate(years)}
        slice_to_year = {idx: year for idx, year in enumerate(years)}
        
        # Year selector using selectbox instead of slider
        selected_year = st.selectbox(
            "Select Year",
            options=years,
            format_func=lambda x: str(x)
        )
        
        # Convert selected year to time slice
        selected_time = year_to_slice[selected_year]
        
        # Create and display pyLDAvis visualization
        vis_path = create_pyldavis_visualization(dtm, corpus, selected_time)
        with open(vis_path, 'r', encoding='utf-8') as f:
            html_string = f.read()
        components.html(html_string, height=800)

    # Download section
    st.sidebar.markdown("---")
    st.sidebar.header("Download Data")
    if st.sidebar.button("Download Topic Evolution Data"):
        download_data = []
        for topic in range(dtm.num_topics):
            for term in topic_evolution[topic]["terms"]:
                for i, year in enumerate(topic_evolution[topic]["years"]):
                    download_data.append({
                        "Topic": topic + 1,
                        "Term": term,
                        "Year": year,
                        "Probability": topic_evolution[topic]["terms"][term][i]
                    })
        
        download_df = pd.DataFrame(download_data)
        st.sidebar.download_button(
            "Click to Download",
            download_df.to_csv(index=False).encode('utf-8'),
            "topic_evolution.csv",
            "text/csv",
            key='download-csv'
        )

# Footer
st.markdown("---")
st.markdown(
    """
    Data source: PRS Legislative Research's bill tracker
    """
)