# 🎬 Movie Recommendation System with Cold-Start Handling

A movie recommendation engine built on **MovieLens 100K** using multiple collaborative filtering approaches, evaluated with both rating-accuracy and ranking metrics. The key engineering contribution is a **Bayesian popularity fallback** that handles cold-start users — a real production challenge that pure CF systems fail at silently.

---

## 🎯 What This Project Actually Does

Most beginner recommender projects train SVD, report RMSE, and stop. This project goes further in three ways:

1. **Benchmarks 4 models** (Baseline, User-CF, Item-CF, SVD) on both accuracy metrics (RMSE, MAE) and ranking metrics (Precision@K, Recall@K) — because minimising rating error and surfacing relevant items are different goals
2. **Quantifies the cold-start problem** — 22% of users have fewer than 5 ratings and would receive degraded CF predictions; these users are explicitly identified and routed to a separate strategy
3. **Bayesian cold-start ranker** — rather than falling back to raw average ratings (which favours low-count movies), uses a confidence-weighted score that shrinks estimates toward the global mean

---

## ⚠️ Honest Limitations

This is a portfolio project, not a production system. Known gaps worth acknowledging:

- **Not truly hybrid:** the system uses a *switching* strategy (CF for warm users, popularity for cold users), not a blended model. A true hybrid would combine SVD scores with content-based similarity (e.g. genre cosine similarity) into a single score
- **Dataset scale:** MovieLens 100K is intentionally small for fast iteration; a production system would use 1M+ interactions and require approximate nearest-neighbour search for inference
- **No temporal modelling:** user preferences drift over time; this model treats all historical ratings as equally relevant
- **Offline evaluation only:** Precision@K and Recall@K are proxies; real-world performance requires online A/B testing

---

## 🧠 Models & Methodology

### Collaborative Filtering

| Model | Method | Key Hyperparameters |
|---|---|---|
| Baseline | ALS global bias | 10 epochs |
| User-CF | KNN with Pearson similarity | k=40, shrinkage=100 |
| Item-CF | KNN with Pearson similarity | k=40, shrinkage=100 |
| SVD | Matrix factorization | 100 factors, 20 epochs, lr=0.005, reg=0.02 |

All models trained on an **80/20 stratified split**. Evaluation uses:
- **RMSE / MAE** — rating prediction accuracy
- **Precision@K, Recall@K** (K = 5, 10, 20) — ranking quality with relevance threshold of 4.0 stars

### Cold-Start Strategy

Users with fewer than 5 ratings are served via a Bayesian-smoothed popularity score:

![Formula](https://latex.codecogs.com/svg.image?\color{LightBlue}\hat{s}(i)=\frac{n_i\cdot\bar{r}_i+C\cdot\mu}{n_i+C})

Where $n_i$ is the rating count for movie $i$, $\bar{r}_i$ its mean rating, $\mu$ the global mean, and $C$ the mean rating count across all movies (the confidence weight). This prevents low-count movies from appearing artificially high-rated.

Cold-start users can also declare genre preferences, which filters the popularity ranking to genre-matching films.

---

## 📊 Key Findings

- **SVD outperforms KNN** on RMSE by ~2–3% — latent factor models generalise better on sparse data
- **Item-CF > User-CF** in ranking metrics — item similarity is more stable than user similarity at this data scale
- **Long-tail confirmed:** top 20% of movies account for 80% of all ratings — a direct motivation for the cold-start fallback
- **22% cold-start users, 35% cold-start movies** — significant enough to require an explicit strategy, not an afterthought

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/your-username/movie-recommender.git
cd movie-recommender
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Windows users:** if `scikit-surprise` fails, install via conda:
> ```bash
> conda install -c conda-forge scikit-surprise
> ```

### 4. Run the app

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

> The MovieLens 100K dataset (~5 MB) downloads automatically on first run. If the network blocks it, a statistically equivalent synthetic dataset is generated locally — no manual steps needed.

---

## 📁 Project Structure

```
movie-recommender/
├── app.py               # Streamlit application (6-page interactive dashboard)
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Interactive web dashboard |
| `scikit-surprise` | CF model implementations (SVD, KNN, Baseline) |
| `pandas` / `numpy` | Data manipulation and matrix operations |
| `scipy` | Sparse CSR matrix for memory-efficient storage |
| `scikit-learn` | Label encoding for user/item indices |
| `matplotlib` / `seaborn` | Visualisations |
| `requests` | Dataset download |

---

## 🛠️ Running in VS Code

1. Open the project folder (`File → Open Folder`)
2. Open the integrated terminal (`` Ctrl+` ``)
3. Run:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```
4. App opens at `http://localhost:8501`

---

## 🔭 What I Would Do With More Time

- Replace SVD with **Neural Collaborative Filtering (NCF)** or a **two-tower model** for better representation learning
- Add **content-based features** (genre similarity, release year) and blend with CF scores to make the hybrid claim legitimate
- Scale to **MovieLens 1M** and benchmark approximate nearest-neighbour inference latency
- Add **temporal decay** to weight recent ratings more heavily than old ones
- Run **offline simulation of A/B test** using a time-based holdout to measure business metrics (CTR proxy, coverage, novelty)

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 📚 References

- Harper & Konstan (2015). [The MovieLens Datasets](https://doi.org/10.1145/2827872)
- Koren, Bell & Volinsky (2009). [Matrix Factorization Techniques for Recommender Systems](https://ieeexplore.ieee.org/document/5197422)
- Ricci et al. (2011). *Recommender Systems Handbook*
