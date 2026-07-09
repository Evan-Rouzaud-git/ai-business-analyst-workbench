# AI Business Analyst Workbench

> *Transform raw business data into actionable insights — fully local, no LLM dependency.*

---

## 🎥 Demo

![Demo GIF](docs/demo.gif)

📺 **[Watch the full video demonstration](https://github.com/Evan-Rouzaud-git/ai-business-analyst-workbench/releases/tag/demo/demo.mp4)**

---

## Why not using ChatGPT?

LLMs are powerful, but they're not always the right tool for business analysis:

- **Data Privacy** — Sensitive tickets can't be sent to third-party APIs
- **Cost** — Analyzing thousands of tickets with LLMs becomes expensive
- **Reproducibility** — Results vary between runs, making tracking impossible
- **Interpretability** — Stakeholders need to understand *why* a recommendation was made

This project uses **lightweight embeddings**, **clustering algorithms**, and **business heuristics** — delivering the same value as an LLM analyst, but running **completely offline** with **deterministic**, **explainable** results.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🧠 **Semantic Clustering** | Groups similar tickets using `sentence-transformers` embeddings |
| 📊 **Interactive Dashboard** | Real-time visualization with Plotly + Streamlit |
| 💰 **ROI Simulator** | Estimate cost savings per theme with customizable parameters |
| 🎯 **Priority Scoring** | Multi-criteria scoring (volume, growth, response rate, resolution ease) |
| 📈 **Forecasting** | Predict ticket volume for the next 4 weeks |
| ⚠️ **Risk Detection** | Identify tickets with high risk of non-response |
| 🔍 **Full Traceability** | Every recommendation is explainable and auditable |

---

## 🏗️ Architecture

```
CSV Input → Embeddings → Clustering → Priority Scoring → ROI → Dashboard
```

1. **Embedding** — Encode ticket descriptions with `all-MiniLM-L6-v2`
2. **Clustering** — Agglomerative clustering (30 fine clusters → 6 macro themes)
3. **Topic Labeling** — Extract key nouns per cluster using spaCy (French)
4. **Scoring** — 5 criteria: volume, gain, response rate, growth, resolution ease
5. **ROI** — Calculate cost, potential savings, and ROI per theme
6. **Forecast** — Linear regression on weekly volume (4 weeks ahead)

---

## 🛠️ Tech Stack

- **UI**: Streamlit
- **Embeddings**: sentence-transformers / all-MiniLM-L6-v2
- **Clustering**: scikit-learn (AgglomerativeClustering)
- **NLP**: spaCy (fr_core_news_sm)
- **Visualization**: Plotly
- **Data**: pandas, numpy

---

## 📊 Dataset

Derived from **[SignalConso](https://signal.conso.gouv.fr/)** — the French public service for consumer complaints.  
The data is reused under the **Etalab Open Licence 1.0**.

**Generation:** 1,000 rows extracted, then a local LLM generated realistic complaint descriptions while preserving original structure (category, status, tags, dates).

---

## 🚀 Installation

```bash
# Clone
git clone https://github.com/your-username/ai-business-analyst-workbench.git
cd ai-business-analyst-workbench

# Setup
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install
pip install -r requirements.txt
python -m spacy download fr_core_news_sm

# Run
streamlit run streamlit_app.py
```

---

## 🎮 Usage

1. **Load data** — Sample dataset or your own CSV
2. **Click "Analyze"** — Processing takes seconds
3. **Explore 5 tabs**:
   - **Summary** — 15-second executive view + top 3 priorities
   - **Analysis** — Category breakdown, weekly trends, tags
   - **Macro-themes** — Detailed theme view with examples
   - **ROI** — Simulate cost savings per theme
   - **Forecast** — Weekly volume prediction + risk alerts

---

## 🔧 Customization

| Parameter | Where | Description |
|-----------|-------|-------------|
| `n_macro` | `clustering.py` | Number of high-level themes (default: 6) |
| `n_clusters_fin` | `clustering.py` | Fine clusters before aggregation (default: 30) |
| `model_name` | `clustering.py` | Embedding model (e.g., `all-mpnet-base-v2`) |
| `hourly_rate` | ROI tab | Agent cost per hour |
| Priority thresholds | `streamlit_app.py` | Adjust volume/gain/response rate thresholds |

---

## 🔮 Improvement Ideas

- **Specialized connectors** for Zendesk, Salesforce, Intercom, email inboxes, databases
- **Export reports** to PDF/Excel
- **Customizable scoring weights** (volume vs. growth vs. cost)
- **Anomaly detection** for unexpected spikes
- **User feedback loop** to track recommendation outcomes

---

## 🙋 About the Author

This project demonstrates:

- **Full-stack data science** — from embeddings to UI
- **Business acumen** — turning data into ROI
- **Self-contained development** — no cloud dependencies
- **Clean architecture** — modular, maintainable, explainable

📧 [avanrouzaud@gmail.com](evanrouzaud@gmail.com)  
🔗 [www.linkedin.com/in/evanrouzaud](www.linkedin.com/in/evanrouzaud)  

---

*Built with ❤️ for data-driven decision making.*
