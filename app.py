import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from gensim.models.ldaseqmodel import LdaSeqModel
from gensim.corpora import Dictionary
import numpy as np
import pyLDAvis
import streamlit.components.v1 as components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import textwrap
import tempfile
import json
import os
import base64

def get_img_as_base64(file_path):
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Get the base64 encoded image
icon_path = "ashoka_chakra.png"  # Make sure this matches your image file path
icon_base64 = get_img_as_base64(icon_path)

# Custom CSS to modify the spinner
spinner_css = f"""
    <style>
        .stSpinner > div {{
            background-image: url("data:image/png;base64,{icon_base64}");
            background-size: contain;
            background-repeat: no-repeat;
            background-position: center;
            width: 100px;
            height: 100px;
        }}
        .stSpinner > div > div {{
            display: none;
        }}
    </style>
"""

# Page config with custom icon
st.set_page_config(
    page_title="Indian Parliamentary Bills Analysis",
    page_icon=icon_path,  # Use the image file path directly
    layout="wide"
)

# Inject custom CSS for spinner
st.markdown(spinner_css, unsafe_allow_html=True)


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
    bills_data['year'] = bills_data['year'].astype(int)
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
        corpus=corpus,
    )
    
    # Prepare visualization
    vis_data = pyLDAvis.prepare(
        topic_term_dists=topic_term,
        doc_topic_dists=doc_topic,
        doc_lengths=doc_lengths,
        vocab=vocab,
        term_frequency=term_frequency,
        sort_topics=False
    )
    
    # Save visualization to temporary HTML file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as f:
        pyLDAvis.save_html(vis_data, f.name)
        return f.name

    
def get_top_ministries(bills_data, n=15):
    """Get the top n ministries by number of bills"""
    ministry_counts = bills_data['ministry'].value_counts()
    return ministry_counts.head(n).index.tolist()

def calculate_ministry_topic_overlap(bills_data, dtm, corpus, top_ministries):
    """Calculate topic distribution for selected ministries"""
    # Get topic distributions for all documents
    topic_distributions = np.array([dtm.doc_topics(i) for i in range(len(corpus))])
    
    # Create ministry-topic matrix
    ministry_topics = {}
    
    for ministry in top_ministries:
        ministry_bills = bills_data[bills_data['ministry'] == ministry].index
        if len(ministry_bills) > 0:
            ministry_topic_dist = topic_distributions[ministry_bills].mean(axis=0)
            ministry_topics[ministry] = ministry_topic_dist
    
    return ministry_topics

def calculate_administration_topic_overlap(bills_data, dtm, corpus):
    """Calculate topic distribution for each administration"""
    # Get topic distributions for all documents
    topic_distributions = np.array([dtm.doc_topics(i) for i in range(len(corpus))])
    
    # Create administration-topic matrix
    admin_topics = {}
    
    for admin in bills_data['administration'].unique():
        admin_bills = bills_data[bills_data['administration'] == admin].index
        if len(admin_bills) > 0:
            admin_topic_dist = topic_distributions[admin_bills].mean(axis=0)
            admin_topics[admin] = admin_topic_dist
    
    return admin_topics

# Load data and model
bills_data = load_data()
try:
    dtm, dictionary, corpus = load_model_and_corpus()
    model_loaded = True
except:
    st.error("Error loading DTM model.")
    model_loaded = False

if model_loaded:
    # Get topic evolution data
    topic_evolution = get_topic_evolution_data(dtm, dictionary, bills_data)
    
    
    # Define topic classifications
    topic_classifications = {
        0: "Labor and Employment",
        1: "Criminal Justice and Law Enforcement",
        2: "Administrative and Judicial Governance",
        3: "Financial and Digital Regulation",
        4: "Public Service & Resource Management",
        5: "Education and Research Infrastructure",
        6: "Public Utilities and Consumer Services"
    }
    topic_descriptions = {
        0: """This topic encompasses legislation related to worker rights, employment conditions, 
        labor regulations, and workplace safety. Key terms include 'worker', 'establishment', 
        'wage', 'labor', and 'employee'. Shifted from broader employment terms to more specific 
        focus on social security and worker protections in recent years""",

        1: """Focused on criminal justice penalties and law enforcement. This topic includes 
        bills about criminal procedures, penalties, law enforcement powers, and judicial processes. 
        Key terms include 'offence', 'imprisonment', 'fine', and 'penalty'. Started with emphasis 
        on property crimes and punishment, shifted toward more procedural aspects (evidence, records) 
        in recent years. Child-related offenses remain consistent throughout""",

        2: """Covers administrative governance and judicial systems, including court procedures, 
        tribunal operations, and administrative reforms. Terms like 'tribunal', 'justice', and 
        'district' reflect its focus on judicial and administrative infrastructure.""",

        3: """Encompasses financial sector regulation and digital governance. This topic has evolved 
        to include both traditional banking regulation and modern data protection concerns, as 
        shown by terms like 'bank', 'financial', 'data', and 'protection'.""",

        4: """Addresses public service management and resource allocation, including medical services, 
        salary structures, and resource management. Key terms include 'medical', 'salary', 
        'allowance', and 'mineral'.""",

        5: """Focuses on educational and research infrastructure, including universities, research 
        institutions, and land management. Key terms include 'institute', 'university', 'research', 
        and 'forest' indicate its scope. Safety regulations remain consistent throughout""",

        6: """Covers core public utilities and consumer services, including power supply, consumer 
        protection, and service delivery. Key terms include 'service', 'power', 'consumer', and 
        'electricity', reflecting its focus on public service delivery. Shifted from traditional 
        utilities to include digital services over time"""
    }

    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Topic Evolution", "LDA Visualization", 
                                           "Topic Prevalence", "Legislative Focus Analysis"])

    with tab1:
        st.markdown("""
        ## Project Overview
        This dashboard presents an analysis of thematic patterns in Indian parliamentary bills 
        from 2005 to 2024. Using Dynamic Topic Modeling (DTM), seven major legislative themes 
        over time were identified and tracked, providing insights into how India's 
        legislative priorities have evolved.
        
        #### What is Dynamic Topic Modeling (DTM)?
        Dynamic Topic Modeling (DTM) is an extension of LDA that is smarter about time. While LDA assumes that 
        topics are fixed, DTM recognizes that topics can change and evolve over time. 
        For example, consider the "healthcare" topic:
        - In 2000, the focus might be on "insurance" and "hospital access."
        - By 2024, the focus might have shifted to "telemedicine" and "AI in healthcare."

        DTM achieves this by linking topics across time. Instead of treating each time period separately, 
        the model ensures that the "healthcare" topic in 2000 is slightly different from 2001, and so on. 
        This creates a smooth evolution of themes.
        So DTM connects topics from one time period to the next, ensuring that changes are gradual and meaningful. 
        Rather than generating a topic independently at each time, the model "nudges" the topic at time \( t \) 
        to resemble its state at time \( t-1 \), allowing for gradual evolution. 
        By adding a "memory" of past topics, DTM extends LDA to reveal not just what topics exist, 
        but also how they shift over time.

        Think of LDA as a snapshot of your document collection: it provides a fixed view of the topics, 
        like a single photo. In contrast, DTM is like a time-lapse video, showing how topics grow, evolve, 
        and shift over time.

        ## Identified Topics
        The analysis revealed seven distinct legislative themes:
        """)

        # Create columns for topics
        cols = st.columns(2)
        for i, (topic_num, topic_name) in enumerate(topic_classifications.items()):
            col_idx = i % 2
            with cols[col_idx]:
                st.markdown(f"#### {i+1}. {topic_name}")
                st.markdown(topic_descriptions[topic_num])
                st.markdown("---")

        st.markdown("""
        ### What you can learn through this dashboard

        1. **Topic Evolution**: 
            - Visualizes how the language and focus within each topic have changed over time
            - Select different topics to see their evolution
            - Watch animated bar chart races showing term importance changes

        2. **LDA Visualization**: 
            - Interactive visualization of topic relationships and term distributions
            - Select different years to see how topics evolved
            - Explore term-topic associations through interactive bubbles

        3. **Topic Prevalence**: 
            - Shows how prominent each topic was in different time periods
            - Track the rise and fall of different legislative priorities
            - View detailed statistics about topic prevalence

        4. **Legislative Focus Analysis**: 
            - Examine how different ministries and administrations approached these topics
            - Compare topic distributions across government bodies
            - Visualize relationships between ministries based on their legislative focus
        """)    
    
    with tab2:
        # Topic evolution visualization
        st.subheader("Topic Evolution Over Time")
        
        st.markdown("""
            ##### How has the language used within bill topics evolved over time?
            Explore how the terminology and focus within each legislative theme has changed over the years.
            The visualization shows the shifting importance of different terms within each topic.
            """)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Get selected topic
            selected_topic = st.selectbox(
                "Select Topic to Visualize",
                options=range(dtm.num_topics),
                format_func=lambda x: f"Topic {x + 1}: {topic_classifications[x]}"
            )
        
        with col2:
            # Download button 
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

        # Display bar chart race video for selected topic
        video_path = f"plots/bar_chart_race_topic_{selected_topic + 1}.mp4"
        if os.path.exists(video_path):
            st.video(video_path)
        else:
            st.error(f"Video file not found: {video_path}")

        # Show current topic composition's presence over time
        st.subheader(f"Current Composition Over Time")
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
        
    
    with tab3:
        # pyLDAvis visualization
        st.subheader("LDA Visualization")
        st.markdown("""
        #### Interpretation

        This interactive visualization helps understand the relationships between topics and terms:

        - **Left Panel**: Shows the topics as circles. Size indicates topic prevalence.
            - Select a topic to see its most relevant terms on the right
            - Distance between circles suggests topic similarity

        - **Right Panel**: Displays term relevance within selected topic
            - Blue bars show term frequency within the topic
            - Red bars show term's overall frequency
            - λ slider adjusts term ranking criteria
        """)        
        
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
        
            
    with tab4:
        st.subheader("Topic Prevalence Over Time")
        st.markdown("""
        ##### Understanding Topic Prevalence

        This visualization shows how prominent each topic was in different time periods. 
        The graph displays the proportion of bills that primarily focused on each topic, 
        helping us understand:

        - Which legislative themes dominated in different periods
        - How legislative priorities shifted over time
        - When certain topics peaked or declined in importance

        Use the topic selector to explore different themes and their prevalence patterns.
        """)        
    
        # Topic selector
        selected_topic = st.selectbox(
        "Select Topic to Analyze",
        options=range(dtm.num_topics),
        format_func=lambda x: topic_classifications[x]
        )
    
        # Calculate yearly topic proportions for the selected topic
        document_topics = [np.argmax(dtm.doc_topics(i)) for i in range(len(corpus))]
        bills_data['dominant_topic'] = document_topics
        bills_data['topic_name'] = bills_data['dominant_topic'].map(topic_classifications)
    
        yearly_topic_counts = bills_data.groupby(['year', 'topic_name']).size().unstack(fill_value=0)
        yearly_topic_proportions = yearly_topic_counts.div(yearly_topic_counts.sum(axis=1), axis=0)
    
        # Create figure for selected topic
        fig = go.Figure()
        
        topic_name = topic_classifications[selected_topic]
    
        fig.add_trace(
            go.Scatter(
                x=yearly_topic_proportions.index,
                y=yearly_topic_proportions[topic_name],
                mode='lines+markers',
                name=topic_name,
                line=dict(color='navy', width=2),
                marker=dict(size=8)
            )
        )
    
        fig.update_layout(
            title=f'Prevalence of {topic_name} Over Time',
            xaxis_title="Year",
            yaxis_title="Proportion of Bills",
            yaxis_tickformat='.0%',
            hovermode='x unified',
            height=500
        )
    
        st.plotly_chart(fig, use_container_width=True)
    
        # Add some statistics
        avg_proportion = yearly_topic_proportions[topic_name].mean()
        max_proportion = yearly_topic_proportions[topic_name].max()
        max_year = yearly_topic_proportions[topic_name].idxmax()
        min_proportion = yearly_topic_proportions[topic_name].min()
        min_year = yearly_topic_proportions[topic_name].idxmin()
    
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Proportion", f"{avg_proportion:.1%}")
        with col2:
            st.metric("Peak", f"{max_proportion:.1%} ({max_year})")
        with col3:
            st.metric("Lowest", f"{min_proportion:.1%} ({min_year})")
            
    with tab5:
        st.subheader("Legislative Focus Analysis")
        st.markdown("""
        This analysis examines how different government entities approached legislative topics:

        - **Ministry Analysis**: Shows how different ministries focused on various themes
        - **Administration Analysis**: Shows how legislative priorities shifted across different governments

        Use the selector below to switch between these two perspectives.
        """)        

        # Add analysis type selector
        analysis_type = st.radio(
            "Select Analysis Level",
            ["Administration", "Ministry"],
            horizontal=True
        )

        if analysis_type == "Ministry":
            # Get top 15 ministries
            top_ministries = get_top_ministries(bills_data, n=15)
            filtered_bills = bills_data[bills_data['ministry'].isin(top_ministries)].copy()
            ministry_topics = calculate_ministry_topic_overlap(filtered_bills, dtm, corpus, top_ministries)

            # Create ministry heatmap
            ministry_topic_df = pd.DataFrame(ministry_topics).T
            ministry_topic_df.columns = [topic_classifications[i] for i in range(dtm.num_topics)]

            # Add and sort by bill count
            ministry_counts = bills_data['ministry'].value_counts()
            ministry_topic_df['Bill Count'] = ministry_counts[ministry_topic_df.index]
            ministry_topic_df = ministry_topic_df.sort_values('Bill Count', ascending=False)
            ministry_topic_df = ministry_topic_df.drop('Bill Count', axis=1)

            # Create wrapped labels
            wrapped_topic_labels = ['<br>'.join(textwrap.wrap(label, width=15)) 
                                  for label in ministry_topic_df.columns]

            # Display heatmap
            fig = px.imshow(ministry_topic_df,
                            labels=dict(x="Topic", y="Ministry", color="Topic Proportion"),
                            aspect="auto",
                            color_continuous_scale="Viridis_r")

            fig.update_layout(
                title="Ministry-Topic Distribution Heatmap (Top 15 Ministries)",
                height=800,
                xaxis=dict(
                    ticktext=['<br>'.join(textwrap.wrap(label, width=15)) for label in ministry_topic_df.columns],
                    tickvals=list(range(len(ministry_topic_df.columns))),
                    tickangle=0
                )
            )

            st.plotly_chart(fig, use_container_width=True)

            # Ministry network visualization
            st.subheader("Ministry Topic Similarity Network")
            ministry_similarities = cosine_similarity(ministry_topic_df)
            G = nx.Graph()

            ministries = list(ministry_topic_df.index)
            for i in range(len(ministries)):
                for j in range(i+1, len(ministries)):
                    if ministry_similarities[i,j] > 0.5:
                        G.add_edge(ministries[i], ministries[j], weight=ministry_similarities[i,j])

            np.random.seed(17)  

            # Create layout with fixed seed
            pos = nx.spring_layout(G, k=1, iterations=50, seed=17)

            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines')

            # Calculate node colors first
            node_adjacencies = []
            for node in G.nodes():
                node_adjacencies.append(len(list(G.neighbors(node))))

            node_x = []
            node_y = []
            node_text = []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)

            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=node_text,
                textposition="top center",
                marker=dict(
                    showscale=True,
                    colorscale='YlGnBu',
                    color=node_adjacencies,  # Set the color array
                    size=20,
                    colorbar=dict(
                        thickness=15,
                        title='Node Connections',
                        xanchor='center',
                        x=0.5,
                        y=-0.15,
                        orientation='h',
                        titleside='top',
                        len=0.5,
                    )
                ))


            fig_network = go.Figure(data=[edge_trace, node_trace],
                                  layout=go.Layout(
                                      showlegend=False,
                                      hovermode='closest',
                                      margin=dict(b=20,l=5,r=5,t=40),
                                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                      title=""
                                  ))

            st.plotly_chart(fig_network, use_container_width=True)

            st.markdown("""
            #### Interpretation:
            - **Heatmap**: 
                - Shows the distribution of topics across the top 15 ministries by bill count. 
                - Each cell tells us - "On average, how much did this ministry's bills focus on this topic?"
                - Core metric: Average topic proportions based on probability assignments from the DTM results
                - Darker colors indicate stronger association with a topic.
            - **Network Graph**: 
                - Each node represents a ministry
                - Connections indicate significant topic overlap between ministries
                - Node size and color intensity indicate number of connections
                - Core metric: Cosine similarity between ministries' topic distributions
                - Only connections above a similarity threshold of 0.5 are shown
            """)

        else:  # Administration analysis
            # Calculate administration-topic overlap
            admin_topics = calculate_administration_topic_overlap(bills_data, dtm, corpus)

            # Create administration heatmap
            admin_topic_df = pd.DataFrame(admin_topics).T
            admin_topic_df.columns = [topic_classifications[i] for i in range(dtm.num_topics)]

            # Add bill count and sort chronologically
            admin_counts = bills_data['administration'].value_counts()
            admin_topic_df['Bill Count'] = admin_counts[admin_topic_df.index]
            admin_topic_df = admin_topic_df.sort_index()
            admin_topic_df = admin_topic_df.drop('Bill Count', axis=1)

            # Create wrapped labels
            wrapped_topic_labels = ['\n'.join(textwrap.wrap(label, width=20)) 
                                  for label in admin_topic_df.columns]

            # Display heatmap
            fig = px.imshow(admin_topic_df.T,  # Transpose the dataframe
                            labels=dict(x="Administration", y="", color="Topic Proportion"),
                            aspect="auto",
                            color_continuous_scale="Viridis_r")
            fig.update_layout(
                title="Administration-Topic Distribution Heatmap",
                height=800,
                yaxis=dict(
                    ticktext=['\n'.join(textwrap.wrap(label, width=15)) for label in admin_topic_df.columns],
                    tickvals=list(range(len(admin_topic_df.columns))),
                    tickangle=0
                )
            )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            #### Interpretation:
            - **Heatmap**: 
                - Shows the distribution of topics across different administrations. 
                - Each cell tells us - "On average, how much did this administration's bills focus on this topic?"
                - Core metric: Average topic proportions based on probability assignments from the DTM results
                - Darker colors indicate stronger association with a topic.
            """)
                        

    # Download section
    st.sidebar.markdown("---")
    st.sidebar.header("Data")
    if st.sidebar.button("Download Indian Parliamentary Bills Data"):
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
