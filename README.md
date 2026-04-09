# 🏛️ Thematic Analysis of Indian Parliamentary Bills
 
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://loksabhatopicmodel-evkjfx7ez8ljjiyp8vlcry.streamlit.app/)
 
**Live App**: [https://loksabhatopicmodel-evkjfx7ez8ljjiyp8vlcry.streamlit.app/](https://loksabhatopicmodel-evkjfx7ez8ljjiyp8vlcry.streamlit.app/)
 
An interactive web dashboard for exploring how legislative themes in Indian parliamentary bills have evolved from 2005 to 2024, built using Dynamic Topic Modeling (DTM).
 
## Overview
 
This project analyzes 800+ parliamentary bills scraped from [PRS Legislative Research](https://prsindia.org/billtrack) to uncover shifting legislative priorities across administrations and ministries. Seven distinct themes were identified and tracked over time.
 
## Features
 
- **Topic Evolution** – Visualizes how terminology within each legislative theme has shifted year-over-year, with animated bar chart races
- **LDA Visualization** – Interactive pyLDAvis display of topic-term relationships for any selected year
- **Topic Prevalence** – Tracks the rise and fall of each theme's prominence over time
- **Legislative Focus Analysis** – Heatmaps and network graphs comparing topic distributions across ministries and administrations
 
## Identified Topics
 
| # | Topic |
|---|-------|
| 1 | Labor and Employment |
| 2 | Criminal Justice and Law Enforcement |
| 3 | Administrative and Judicial Governance |
| 4 | Financial and Digital Regulation |
| 5 | Public Service & Resource Management |
| 6 | Education and Research Infrastructure |
| 7 | Public Utilities and Consumer Services |
 
## Tools and Libraries
 
- **Frontend/App**: [Streamlit](https://streamlit.io/)
- **Topic Modeling**: Gensim `LdaSeqModel` (Dynamic Topic Modeling)
- **Visualization**: Plotly, pyLDAvis, NetworkX
- **NLP Preprocessing**: NLTK, scikit-learn
 
## Data
 
Bills data was scraped from PRS Legislative Research's bill tracker, covering 801 bills introduced between 2005 and 2024, with fields including bill name, summary, year, ministry, status, and administration.
 
## References
 
- Blei, D. M., & Lafferty, J. D. (2006). Dynamic topic models. In *Proceedings of the 23rd international conference on Machine learning* (pp. 113-120). Association for Computing Machinery. https://doi.org/10.1145/1143844.1143859
- Greene, D., & Cross, J. P. (2017). Exploring the Political Agenda of the European Parliament Using a Dynamic Topic Modeling Approach. *Political Analysis, 25*(1), 77–94.
- Kapadia, S. (2019, August 19). Evaluate Topic Models: Latent Dirichlet Allocation (LDA). *Towards Data Science*. https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0
- Lefebure, L. (2018, October 17). Exploring the UN General Debates with Dynamic Topic Models. *Towards Data Science*. https://towardsdatascience.com/exploring-the-un-general-debates-with-dynamic-topic-models-72dc0e307696
- Pavithra, & Savitha. (2024). Topic modeling for evolving textual data using LDA, HDP, NMF, BERTopic, and DTM with a focus on research papers. *Journal of Technology and Informatics, 5*(2), 53–63. https://doi.org/10.37802/joti.v5i2.618
- PRS Legislative Research. Bill Track. Retrieved October 24, 2024, from https://prsindia.org/billtrack
