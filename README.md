# 🚀 OmniRank: Extended Multi‑Modal Feed Ranking & Assistant Platform

This repository contains **OmniRank**, an extended and polished version of the multi‑modal feed ranking & AI assistant platform.  OmniRank merges the best ideas from the original `OmniRank`, `AuroraRank` and `MetaFeedX` concepts and takes them several steps further.  It is designed to mirror the scale and complexity found at leading tech companies – think FANG‑level systems – while still being approachable enough to run locally for experimentation.

Major highlights:

* **Unique UI/UX**: A bespoke Streamlit dashboard (see `ui/app_ui_extended.py`) with a modern colour palette, tabbed navigation, responsive two‑column feed layout and real‑time charts.  The interface infers your environment from uploaded images, parses natural language queries and lets you log events directly from the UI.
* **Scalable Data & Models**: Generate or ingest tens of millions of events, train deep recommendation models, graph neural networks and contextual bandits, and observe the impact of your policies live.
* **Modular Design**: Swap out models, plug in real message brokers (Kafka, Pulsar), and extend the UI with React/FastAPI if desired.

* **Data Scale**: Generate or ingest **tens of millions** of interaction events across millions of users and posts.  Support both synthetic data and integration with external datasets.
* **Rich Context**: Incorporate **multi‑modal signals** such as images, text, audio, and user metadata.  Use large language models to infer user intent and summarise content.  Use vision models to understand scenes and objects.
* **Advanced Models**:
  * **Two‑Tower Recommender**: A scalable deep learning architecture trained on implicit feedback.  Embeds users and items separately and serves recommendations via nearest‑neighbour search.
  * **Graph Neural Networks (GNNs)**: Capture social network structure and propagate influence through follower/friend relationships.
  * **Contextual Bandits & Reinforcement Learning**: Optimise multiple objectives (e.g., click‑through rate, dwell time, diversity) with online learning.
* **Stream Processing**: Simulate or connect to a real event stream (Apache Kafka, Pulsar, etc.) for real‑time ingestion, logging, and model updates.
* **Production‑ready UI**: A sleek web interface built with **Streamlit** (or optionally React + FastAPI) that supports multiple users, configurable feeds, analytics dashboards, and interactive experiments.

The goal of this project is not merely to run locally but to **demonstrate architecture patterns** and **scalable design** used at FANG‑level companies.  It’s a sandbox for experimenting with full‑stack ML systems, from data generation through to user‑facing product.

## 🧱 Repository Structure

```
omni_rank_extended_project/
│   README.md                # This file
│   requirements.txt        # Python dependencies
│
├── data_generation/
│   └── generate_large_scale_data.py  # Parameterised synthetic data generator
│
├── models/
│   ├── two_tower_recommender.py      # Scalable deep recommendation model
│   ├── graph_module.py               # Graph neural network and social graph logic
│   └── policy_controller.py          # Contextual bandit and reinforcement learning logic
│
├── services/
│   ├── streaming_service.py          # Real‑time event ingestion simulation
│   └── api_server.py                 # Optional FastAPI backend for the UI and model serving
│
├── ui/
│   └── app_ui_extended.py            # Streamlit app with a unique and polished UI
│
└── utils/
    └── data_utils.py                 # Helpers for reading/writing Parquet and interfacing with big‑data stores
```

## 📦 Installation

This project assumes **Python 3.10+** and optionally an installation of **PyTorch** or **TensorFlow** with GPU support for training deep models.

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# If using GPUs
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
```

### Optional Dependencies

* **PySpark** for distributed data processing.
* **FAISS** for efficient vector search in the two‑tower recommender.
* **DGL** or **PyG** for graph neural network implementation.
* **Kafka** client libraries for stream ingestion.

## 🛠 Generating Large‑Scale Data

Use `generate_large_scale_data.py` to synthesise a dataset of arbitrary size.  You can specify the number of users, posts, days of activity, and event distribution.  Data is stored in **Parquet** format by default, ready for fast loading into Spark or Pandas.

```bash
python data_generation/generate_large_scale_data.py --users 1000000 --posts 10000000 --days 30 --events 50000000 --output_dir data
```

The script can also ingest real datasets (e.g., MovieLens, public social graph dumps) and convert them into the expected schema.

## 🤖 Training the Two‑Tower Recommender

The `models/two_tower_recommender.py` module defines a deep recommendation model.  It follows the industry‑standard two‑tower architecture: a user encoder and an item encoder trained jointly using negative sampling and implicit feedback loss.  To train:

```bash
python -m models.two_tower_recommender --train_path data/events.parquet --model_dir models/checkpoints --epochs 5
```

The module is designed to scale: it supports distributed data loaders, mixed‑precision training, and exporting embeddings for ANN search with **FAISS**.

## 🔗 Modeling the Social Graph

`models/graph_module.py` demonstrates how to build a simple social graph from follower relations and train a graph neural network (GNN) to produce user embeddings that capture network effects.  These embeddings can then be fused with behavioural embeddings from the two‑tower recommender.

## 🎛 Contextual Bandits & RL

`models/policy_controller.py` contains classes for policy optimisation.  The system tracks metrics (CTR, dwell time, etc.) for each policy arm and updates its parameters using contextual bandit algorithms (e.g., LinUCB, Thompson Sampling).  A reinforcement learning loop can be configured to adapt to user feedback in real time.

## 🌐 Stream Processing

`services/streaming_service.py` simulates an event pipeline.  It can emit interaction events to a message bus (e.g., Kafka) and consume them for logging and online learning.  The design is modular, so you can swap the simulation with a real broker.

## 📊 UI & Experiment Dashboard

OmniRank ships with a **distinctive Streamlit dashboard** that was designed with UX in mind.  The UI is organised into tabs – **Feed**, **Query & Filters** and **Experiment Metrics** – and includes the following capabilities:

* 🎨 **Modern design**: custom CSS for colours, fonts and button styles.  A welcome banner summarises what the platform can do.
* 📷 **Environment inference**: upload an image (e.g., of your room) and the system will display a placeholder environment classification.  In a real deployment, this would call a vision model.
* 💬 **Natural language queries**: type queries such as “Show me trending science posts” and view the parsed intent (demo).  This simulates connecting to an LLM.
* 📑 **Personalised feed**: click a button to fetch recommendations and view them in a two‑column card layout with images.  You can log events (views) directly from the feed cards.
* 📈 **Metrics dashboard**: explore interactive bar charts for CTR and dwell time per policy arm and inspect raw tables of metrics.  The charts are built with **Altair**.

For a more production‑ready experience, you can replace the Streamlit UI with a **React** front‑end and a **FastAPI** backend (`services/api_server.py`).

## 🧩 Extending Further

This project is designed as a **platform** for experimentation.  Here are some ideas to take it even further:

* **Recommendation Diversity & Fairness**: Integrate objectives and constraints to ensure exposure to diverse content and avoid filter bubbles.
* **Multilingual Support**: Use translation models to support users across languages and cultures.
* **Content Integrity & Safety**: Integrate classifiers for toxicity, misinformation, and spam.  Apply safety filters in the ranking pipeline.
* **Generative Features**: Use generative models to produce personalised summaries of posts, generate alt‑text for images, or even create content suggestions.
* **Deployment to Cloud**: Containerise the services and deploy to Kubernetes or serverless platforms.  Use managed databases (e.g., BigQuery, Snowflake) and model serving frameworks (e.g., TorchServe).


## ▶️ Quickstart

If you just want to run the demo UI and play around with the ranking pipeline, follow these steps:

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies (may take a while)
pip install -r requirements.txt

# 3. (Optional) install deep learning frameworks if you plan to train models
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118

# 4. Launch the Streamlit dashboard
streamlit run ui/app_ui_extended.py

# 5. Open http://localhost:8501 in your browser.  Use the sidebar to set a user ID
#    and upload an image.  Click 'Get Recommendations' to see the personalised feed.

# 6. To train the recommender or generate data, see the sections above.
```

When you are ready to deploy the models or run at scale, consult the scripts in `models/` and `services/` for training and serving.
