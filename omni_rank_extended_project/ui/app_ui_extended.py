"""
app_ui_extended.py
------------------

This Streamlit application provides an interactive interface for the extended feed ranking
platform.  Users can upload images for environment detection, enter natural language queries
to be parsed by an LLM, browse personalised feed recommendations, and monitor policy
experiment metrics.  The UI communicates with the FastAPI backend to obtain
recommendations and log interactions.

To run the app:

```bash
streamlit run ui/app_ui_extended.py
```
"""

import streamlit as st
import requests
import json
from PIL import Image
import io
import pandas as pd
import altair as alt

API_URL = st.secrets.get('api_url', 'http://localhost:8000')

# -----------------------------------------------------------------------------
# Page configuration and theming
#
# We set a custom title and icon and specify a wide layout to maximise the
# available space.  The CSS below tweaks colours, fonts and button styles to
# provide a distinctive look and feel compared to a vanilla Streamlit app.

st.set_page_config(
    page_title="OmniRank – Multi‑Modal Feed Ranking Platform",
    page_icon="🌟",
    layout="wide"
)

st.markdown(
    """
    <style>
    /* Global page style */
    .reportview-container {
        background-color: #f5f7fa;
    }
    h1, h2, h3, h4 {
        color: #1f2937;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stButton > button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton > button:hover {
        background-color: #1e40af;
    }
    /* Card container styling */
    .post-card {
        background-color: white;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌐 OmniRank | Personalised & Conversational Feed Ranking")

st.write(
    "Harness the power of multi‑modal AI to deliver hyper‑personalised content. "
    "Upload images to infer your environment, ask natural‑language questions to refine "
    "your feed and monitor real‑time metrics for continuous optimisation."
)

with st.sidebar:
    st.header("User Settings")
    user_id = st.number_input("User ID", min_value=1, value=1, step=1)
    top_k = st.slider("Number of recommendations", min_value=5, max_value=50, value=10)
    st.markdown("---")
    st.header("Environment Inference")
    image_file = st.file_uploader("Upload an image", type=['jpg','jpeg','png'])
    if image_file:
        image = Image.open(image_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        # In production, call a vision model to infer environment
        st.write("**Environment:** living room (demo)")
    st.markdown("---")
    st.header("Query Parsing")
    user_query = st.text_input("Ask me anything")
    if user_query:
        # In production, call LLM to parse intent and filters
        st.write("**Parsed Intent:** search for tech posts (demo)")


# Use tabs to organise the interface into distinct functional areas
tab_feed, tab_query, tab_metrics = st.tabs(["🎯 Feed", "🔍 Query & Filters", "📊 Experiment Metrics"])

with tab_feed:
    st.subheader("Personalised Feed")
    st.write("Click the button below to fetch recommendations based on your current context.")
    if st.button("Get Recommendations"):
        try:
            resp = requests.get(f"{API_URL}/recommendations", params={'user_id': user_id, 'k': top_k})
            if resp.status_code == 200:
                data = resp.json()
                st.success(f"Fetched {len(data['recommended_posts'])} recommendations for user {user_id}")
                # Display posts in a responsive two‑column layout
                cols = st.columns(2)
                for idx, post_id in enumerate(data['recommended_posts']):
                    col = cols[idx % 2]
                    with col:
                        with st.container():
                            st.markdown(
                                f"<div class='post-card'>"
                                f"<h4>Post {post_id}</h4>"
                                f"<img src='https://picsum.photos/seed/{post_id}/600/300' style='width:100%;border-radius:6px;'/>"
                                f"<p>This is a placeholder image for post {post_id}. In a production system, this would display the actual post image or video.</p>"
                                "</div>",
                                unsafe_allow_html=True
                            )
                            if st.button(f"Log view for Post {post_id}", key=f"view_{post_id}"):
                                # Prompt for a timestamp input only when logging events
                                timestamp = st.date_input('Event Date', key=f"date_{post_id}")
                                log_resp = requests.post(
                                    f"{API_URL}/log", 
                                    json={
                                        'user_id': user_id,
                                        'post_id': post_id,
                                        'event_type': 'view',
                                        'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp)
                                    }
                                )
                                if log_resp.status_code == 200:
                                    st.toast("Event logged successfully", icon='✅')
                                else:
                                    st.warning("Failed to log event")
            else:
                st.error(f"Error fetching recommendations: {resp.text}")
        except Exception as e:
            st.error(f"Failed to connect to API: {e}")

with tab_query:
    st.subheader("Fine‑grained Query & Filters")
    st.write(
        "Use natural language to filter your feed. For example: "
        "**'Show me recent science fiction posts'** or **'Only posts with high engagement about AI'**. "
        "This demo displays parsed intent as a placeholder."
    )
    if user_query:
        st.info(f"Parsed Intent (demo): search for tech posts")
    else:
        st.warning("Enter a query in the sidebar to see the parsed intent.")

with tab_metrics:
    st.subheader("Live Policy Experiment Metrics (Demo)")
    st.write(
        "Monitor the performance of different ranking policies in real time. "
        "These demo metrics show click‑through rate (CTR) and dwell time for two policy arms."
    )
    # Demo data for visualisation
    metrics_df = pd.DataFrame({
        'Policy': ['Policy A', 'Policy B'],
        'CTR': [0.125, 0.133],
        'Dwell Time (s)': [30, 28]
    })
    ctr_chart = (
        alt.Chart(metrics_df)
        .mark_bar(color='#2563eb')
        .encode(
            x='Policy',
            y=alt.Y('CTR', axis=alt.Axis(format='%', title='Click‑Through Rate')),
            tooltip=['Policy', alt.Tooltip('CTR', format='.1%')]
        )
        .properties(title='CTR by Policy')
    )
    dwell_chart = (
        alt.Chart(metrics_df)
        .mark_bar(color='#10b981')
        .encode(
            x='Policy',
            y=alt.Y('Dwell Time (s)', axis=alt.Axis(title='Dwell Time (seconds)')),
            tooltip=['Policy', 'Dwell Time (s)']
        )
        .properties(title='Average Dwell Time by Policy')
    )
    st.altair_chart(ctr_chart, use_container_width=True)
    st.altair_chart(dwell_chart, use_container_width=True)
    st.write("### Raw Metrics")
    st.dataframe(metrics_df.set_index('Policy'))