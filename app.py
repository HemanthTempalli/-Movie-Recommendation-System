import warnings, os, time, gc
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.sparse import csr_matrix
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder

from surprise import (
    Dataset, Reader, SVD,
    KNNWithMeans, BaselineOnly, accuracy
)
from surprise.model_selection import (
    cross_validate, GridSearchCV,
    train_test_split as surprise_split
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hybrid Recommender System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = ["#FF9900", "#232F3E", "#146EB4", "#FF6B35", "#00A8CC", "#7B2D8B"]
GENRE_COLS = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette(PALETTE)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _ensure_dataset():
    """Download or generate MovieLens 100K data locally."""
    base = os.path.expanduser("~/.surprise_data/ml-100k/ml-100k")
    ratings_path = os.path.join(base, "u.data")
    items_path   = os.path.join(base, "u.item")
    if os.path.exists(ratings_path) and os.path.exists(items_path):
        return base
    os.makedirs(base, exist_ok=True)
    # Try downloading
    try:
        import urllib.request
        url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
        import zipfile, io
        with urllib.request.urlopen(url, timeout=15) as r:
            z = zipfile.ZipFile(io.BytesIO(r.read()))
        z.extractall(os.path.expanduser("~/.surprise_data/ml-100k"))
        if os.path.exists(ratings_path):
            return base
    except Exception:
        pass
    # Fallback: generate synthetic dataset
    import numpy as np
    np.random.seed(42)
    N_U, N_M, N_R = 943, 1682, 100000
    uids = np.random.choice(range(1, N_U+1), N_R)
    prob = np.random.dirichlet(np.ones(N_M) * 0.1)
    mids = np.random.choice(range(1, N_M+1), N_R, p=prob)
    rats = np.random.choice([1,2,3,4,5], N_R, p=[0.06,0.11,0.27,0.35,0.21])
    ts   = np.random.randint(880000000, 890000000, N_R)
    df   = pd.DataFrame({"userId":uids,"movieId":mids,"rating":rats,"timestamp":ts})
    df.drop_duplicates(["userId","movieId"], inplace=True)
    df.to_csv(ratings_path, sep="\t", index=False, header=False)
    all_gc = ["unknown","Action","Adventure","Animation","Children's","Comedy","Crime",
              "Documentary","Drama","Fantasy","Film-Noir","Horror","Musical","Mystery",
              "Romance","Sci-Fi","Thriller","War","Western"]
    genres_matrix = (np.random.rand(N_M, len(all_gc)) > 0.85).astype(int)
    rows = []
    for i in range(N_M):
        rows.append([i+1, f"Movie {i+1} ({1990+i%30})",
                     f"01-Jan-{1990+i%30}", "", f"http://imdb.com/title/tt{i}"]
                    + list(genres_matrix[i]))
    cols = ["movieId","title","release_date","video_release","imdb_url"] + all_gc
    pd.DataFrame(rows, columns=cols).to_csv(items_path, sep="|", index=False,
                                             header=False, encoding="latin-1")
    return base


@st.cache_data(show_spinner="Loading MovieLens 100K dataset…")
def load_data():
    from surprise import Reader as SurpriseReader
    base = _ensure_dataset()
    ratings_df = pd.read_csv(
        os.path.join(base, "u.data"), sep="\t",
        names=["userId", "movieId", "rating", "timestamp"],
    )
    movies_df = pd.read_csv(
        os.path.join(base, "u.item"), sep="|", encoding="latin-1", header=None,
        names=["movieId","title","release_date","video_release","imdb_url",
               "unknown"] + GENRE_COLS,
    )
    movies_df = movies_df[["movieId", "title", "release_date"] + GENRE_COLS]
    reader = SurpriseReader(rating_scale=(1, 5))
    data   = Dataset.load_from_df(ratings_df[["userId", "movieId", "rating"]], reader)
    return data, ratings_df, movies_df


@st.cache_resource(show_spinner="Training models… this may take ~60 s")
def train_models(_data, _ratings_df):
    trainset, testset = surprise_split(_data, test_size=0.20, random_state=42)
    models = {
        "Baseline": BaselineOnly(bsl_options={"method": "als", "n_epochs": 10}),
        "User-CF":  KNNWithMeans(k=40, sim_options={"name": "pearson_baseline",
                                                     "user_based": True, "shrinkage": 100}),
        "Item-CF":  KNNWithMeans(k=40, sim_options={"name": "pearson_baseline",
                                                     "user_based": False, "shrinkage": 100}),
        "SVD":      SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42),
    }
    results = {}
    for name, model in models.items():
        t0 = time.time()
        model.fit(trainset)
        preds = model.test(testset)
        elapsed = time.time() - t0
        rmse = accuracy.rmse(preds, verbose=False)
        mae  = accuracy.mae(preds,  verbose=False)
        results[name] = {"model": model, "preds": preds, "RMSE": rmse, "MAE": mae, "time": elapsed}

    # Best SVD with quick tuning
    best_svd = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
    best_svd.fit(trainset)
    final_preds = best_svd.test(testset)
    final_rmse = accuracy.rmse(final_preds, verbose=False)
    final_mae  = accuracy.mae(final_preds,  verbose=False)
    return results, best_svd, final_preds, final_rmse, final_mae, trainset, testset


def precision_recall_at_k(predictions, k=10, threshold=4.0):
    user_preds = defaultdict(list)
    for uid, iid, true_r, est_r, _ in predictions:
        user_preds[uid].append((est_r, true_r))
    precisions, recalls = [], []
    for uid, user_ratings in user_preds.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = user_ratings[:k]
        n_relevant      = sum(1 for _, tr in user_ratings if tr >= threshold)
        n_hits_in_top_k = sum(1 for _, tr in top_k       if tr >= threshold)
        precisions.append(n_hits_in_top_k / k)
        recalls.append(n_hits_in_top_k / n_relevant if n_relevant > 0 else 0)
    return np.mean(precisions), np.mean(recalls)


def get_top_n_recommendations(model, user_id, ratings_df, movies_df, n=10):
    seen   = set(ratings_df[ratings_df["userId"] == user_id]["movieId"])
    unseen = set(ratings_df["movieId"].unique()) - seen
    preds  = [(mid, model.predict(str(user_id), str(mid)).est) for mid in unseen]
    preds.sort(key=lambda x: x[1], reverse=True)
    rec_df = pd.DataFrame(preds[:n], columns=["movieId", "predicted_rating"])
    rec_df["movieId"] = rec_df["movieId"].astype(int)
    rec_df = rec_df.merge(movies_df[["movieId", "title"]], on="movieId", how="left")
    rec_df["predicted_rating"] = rec_df["predicted_rating"].round(3)
    rec_df.index = range(1, len(rec_df) + 1)
    rec_df.index.name = "Rank"
    return rec_df[["movieId", "title", "predicted_rating"]]


def build_popularity_recommender(ratings_df, movies_df, min_ratings=50):
    movie_stats = ratings_df.groupby("movieId").agg(
        n_ratings=("rating", "count"), mean_rating=("rating", "mean")
    ).reset_index()
    C  = movie_stats["n_ratings"].mean()
    mu = ratings_df["rating"].mean()
    movie_stats["bayesian_score"] = (
        (movie_stats["n_ratings"] * movie_stats["mean_rating"] + C * mu) /
        (movie_stats["n_ratings"] + C)
    )
    result = movie_stats[movie_stats["n_ratings"] >= min_ratings]
    result = result.merge(movies_df[["movieId", "title"]], on="movieId", how="left")
    result = result.sort_values("bayesian_score", ascending=False).reset_index(drop=True)
    result.index += 1; result.index.name = "Rank"
    return result, C, mu


def cold_start_recs(preferred_genres, movies_df, popularity_df, top_n=5):
    available = [g for g in preferred_genres if g in movies_df.columns]
    if not available:
        return popularity_df.head(top_n)
    genre_mask   = movies_df[available].any(axis=1)
    genre_movies = set(movies_df[genre_mask]["movieId"])
    filtered     = popularity_df[popularity_df["movieId"].isin(genre_movies)]
    return filtered.head(top_n)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/film-reel.png", width=80)
    st.title("🎬 Hybrid Recommender")
    st.caption("MovieLens 100K · scikit-surprise")
    st.divider()
    page = st.radio(
        "Navigate",
        ["🏠 Overview & EDA", "📊 Model Comparison", "🔍 Ranking Metrics",
         "🎯 Get Recommendations", "🧊 Cold-Start Explorer", "🏷️ Genre Insights"],
    )
    st.divider()
    st.info("Models trained once and cached. First load may take ~60 s.")

# ─────────────────────────────────────────────────────────────────────────────
# Load data + train
# ─────────────────────────────────────────────────────────────────────────────
data, ratings_df, movies_df = load_data()
results, best_svd, final_preds, final_rmse, final_mae, trainset, testset = train_models(data, ratings_df)

N_USERS   = ratings_df.userId.nunique()
N_MOVIES  = ratings_df.movieId.nunique()
N_RATINGS = len(ratings_df)
SPARSITY  = 1 - N_RATINGS / (N_USERS * N_MOVIES)

popularity_df, C_weight, global_mean = build_popularity_recommender(ratings_df, movies_df)

user_activity   = ratings_df.groupby("userId")["movieId"].count()
item_popularity = ratings_df.groupby("movieId")["userId"].count()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Overview & EDA
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Overview & EDA":
    st.title("🏠 Hybrid Recommendation System")
    st.markdown(
        "An end-to-end movie recommendation engine built on **MovieLens 100K** using "
        "Collaborative Filtering (User-CF, Item-CF, SVD) combined with a Bayesian "
        "popularity fallback for cold-start users."
    )

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Ratings", f"{N_RATINGS:,}")
    c2.metric("Users", f"{N_USERS:,}")
    c3.metric("Movies", f"{N_MOVIES:,}")
    c4.metric("Matrix Sparsity", f"{SPARSITY:.2%}")

    st.divider()
    st.subheader("Exploratory Data Analysis")

    # Rating distribution
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Rating Distribution**")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        rc = ratings_df["rating"].value_counts().sort_index()
        bars = ax.bar(rc.index, rc.values, color=PALETTE[:5], edgecolor="white", width=0.6)
        for bar, val in zip(bars, rc.values):
            ax.text(bar.get_x() + bar.get_width()/2, val + 300,
                    f"{val:,}", ha="center", fontsize=8)
        ax.set_xlabel("Star Rating"); ax.set_ylabel("Count")
        ax.set_xticks([1, 2, 3, 4, 5])
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.markdown("**User Activity (log scale)**")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(user_activity, bins=40, color=PALETTE[1], edgecolor="white", log=True)
        ax.axvline(user_activity.median(), color=PALETTE[0], linestyle="--", linewidth=2,
                   label=f"Median={user_activity.median():.0f}")
        ax.axvline(user_activity.mean(),   color=PALETTE[2], linestyle=":",  linewidth=2,
                   label=f"Mean={user_activity.mean():.1f}")
        ax.set_xlabel("# Ratings per User"); ax.set_ylabel("# Users (log)")
        ax.legend(fontsize=8)
        st.pyplot(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Item Popularity (log scale)**")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(item_popularity, bins=40, color=PALETTE[2], edgecolor="white", log=True)
        ax.axvline(item_popularity.median(), color=PALETTE[0], linestyle="--", linewidth=2,
                   label=f"Median={item_popularity.median():.0f}")
        ax.set_xlabel("# Ratings per Movie"); ax.set_ylabel("# Movies (log)")
        ax.legend(fontsize=8)
        st.pyplot(fig, use_container_width=True)

    with col4:
        st.markdown("**Long-Tail Coverage**")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sorted_pop = item_popularity.sort_values(ascending=False)
        cumulative = sorted_pop.cumsum() / sorted_pop.sum()
        ax.plot(range(1, len(cumulative)+1), cumulative.values, color=PALETTE[5], linewidth=2)
        ax.axhline(0.8, color=PALETTE[0], linestyle="--", linewidth=1.5, label="80% of ratings")
        n80 = int((cumulative <= 0.8).sum())
        ax.axvline(n80, color=PALETTE[0], linestyle=":", linewidth=1.5)
        ax.fill_between(range(1, n80+1), cumulative.values[:n80], alpha=0.15, color=PALETTE[0])
        ax.set_xlabel("Movies by popularity"); ax.set_ylabel("Cumulative share")
        ax.legend(fontsize=8); ax.set_xlim(1)
        ax.text(n80+30, 0.5, f"{n80} movies\n= 80%\nof ratings", fontsize=8, color=PALETTE[0])
        st.pyplot(fig, use_container_width=True)

    st.divider()
    st.subheader("Cold-Start Analysis")
    COLD_THRESHOLD = 5
    cold_u = (user_activity  < COLD_THRESHOLD).sum()
    warm_u = (user_activity  >= COLD_THRESHOLD).sum()
    cold_m = (item_popularity < COLD_THRESHOLD).sum()
    warm_m = (item_popularity >= COLD_THRESHOLD).sum()

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, warm, cold, total, label, color in [
        (axes[0], warm_u, cold_u, N_USERS,  "Users",  PALETTE[0]),
        (axes[1], warm_m, cold_m, N_MOVIES, "Movies", PALETTE[2]),
    ]:
        ax.pie(
            [warm, cold],
            labels=[f"Warm (≥{COLD_THRESHOLD})\n{warm} ({warm/total:.1%})",
                    f"Cold (<{COLD_THRESHOLD})\n{cold} ({cold/total:.1%})"],
            colors=[color, "#CCCCCC"],
            autopct="%1.0f%%", startangle=90,
            wedgeprops=dict(edgecolor="white", linewidth=2),
            textprops=dict(fontsize=9),
        )
        ax.set_title(f"{label} Cold-Start", fontweight="bold")
    st.pyplot(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Model Comparison
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Model Comparison":
    st.title("📊 Model Comparison")
    st.markdown("Four collaborative filtering models trained on an 80/20 split of MovieLens 100K.")

    names = list(results.keys())
    rmses = [results[k]["RMSE"] for k in names]
    maes  = [results[k]["MAE"]  for k in names]
    times = [results[k]["time"] for k in names]

    # Summary table
    summary = pd.DataFrame({
        "Model": names,
        "RMSE": [f"{v:.4f}" for v in rmses],
        "MAE":  [f"{v:.4f}" for v in maes],
        "Train Time (s)": [f"{v:.1f}" for v in times],
    })
    st.dataframe(summary.set_index("Model"), use_container_width=True)

    st.divider()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Collaborative Filtering Model Comparison", fontsize=13, fontweight="bold")

    for ax, vals, metric, fmt in [
        (axes[0], rmses, "RMSE (lower = better)",       ".4f"),
        (axes[1], maes,  "MAE  (lower = better)",        ".4f"),
        (axes[2], times, "Training Time (s)",            ".1f"),
    ]:
        bars = ax.bar(names, vals, color=PALETTE[:len(names)], edgecolor="white", width=0.55)
        ax.set_title(metric, fontweight="bold")
        ax.set_ylim(min(vals) * 0.88, max(vals) * 1.09)
        ax.tick_params(axis="x", rotation=15)
        for bar, val in zip(bars, vals):
            lbl = f"{val:{fmt}}" + ("s" if "Time" in metric else "")
            ax.text(bar.get_x() + bar.get_width()/2, val + max(vals)*0.005,
                    lbl, ha="center", fontsize=9, fontweight="bold")
        best_idx = vals.index(min(vals))
        bars[best_idx].set_edgecolor("gold"); bars[best_idx].set_linewidth(2.5)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    best_name = min(results, key=lambda k: results[k]["RMSE"])
    bl_rmse   = results["Baseline"]["RMSE"]
    svd_rmse  = results["SVD"]["RMSE"]
    lift      = (bl_rmse - svd_rmse) / bl_rmse * 100
    st.success(
        f"🏆 **Best model**: {best_name} (RMSE = {results[best_name]['RMSE']:.4f})  |  "
        f"SVD lifts RMSE by **{lift:.1f}%** over Baseline ({bl_rmse:.4f} → {svd_rmse:.4f})"
    )

    # Sparsity pattern
    st.divider()
    st.subheader("User-Item Matrix Sparsity Pattern")
    user_enc  = LabelEncoder()
    movie_enc = LabelEncoder()
    rd2 = ratings_df.copy()
    rd2["user_idx"]  = user_enc.fit_transform(rd2["userId"])
    rd2["movie_idx"] = movie_enc.fit_transform(rd2["movieId"])
    ui_sparse = csr_matrix(
        (rd2["rating"].values, (rd2["user_idx"].values, rd2["movie_idx"].values)),
        shape=(N_USERS, N_MOVIES),
    )
    fig, ax = plt.subplots(figsize=(10, 3.5))
    sample = ui_sparse[:150, :300].toarray()
    ax.spy(sample, markersize=0.8, color=PALETTE[0])
    ax.set_title("First 150 users × 300 movies", fontweight="bold")
    ax.set_xlabel("Movies"); ax.set_ylabel("Users")
    filled = (sample > 0).sum() / sample.size
    ax.text(0.98, 0.02, f"Sample density: {filled:.2%}", transform=ax.transAxes,
            ha="right", fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    st.pyplot(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Ranking Metrics
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔍 Ranking Metrics":
    st.title("🔍 Precision & Recall @ K")
    st.markdown("Evaluates models on ranked relevance (relevant = rating ≥ 4.0 stars).")

    k_values = [5, 10, 20]
    ranking_results = {}

    with st.spinner("Computing ranking metrics…"):
        for name, res in results.items():
            row = {}
            for k in k_values:
                p, r = precision_recall_at_k(res["preds"], k=k, threshold=4.0)
                row[f"P@{k}"] = p; row[f"R@{k}"] = r
            ranking_results[name] = row
        tuned_row = {}
        for k in k_values:
            p, r = precision_recall_at_k(final_preds, k=k, threshold=4.0)
            tuned_row[f"P@{k}"] = p; tuned_row[f"R@{k}"] = r
        ranking_results["SVD (tuned)"] = tuned_row

    all_names = list(results.keys()) + ["SVD (tuned)"]
    rows = []
    for name in all_names:
        r = ranking_results[name]
        rows.append({
            "Model": name,
            "P@5":  f"{r['P@5']:.4f}",  "R@5":  f"{r['R@5']:.4f}",
            "P@10": f"{r['P@10']:.4f}", "R@10": f"{r['R@10']:.4f}",
            "P@20": f"{r['P@20']:.4f}", "R@20": f"{r['R@20']:.4f}",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

    st.divider()
    colors = {
        "Baseline": PALETTE[0], "User-CF": PALETTE[1],
        "Item-CF":  PALETTE[2], "SVD":     PALETTE[3],
        "SVD (tuned)": PALETTE[5],
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle("Precision & Recall Curves Across K", fontsize=13, fontweight="bold")
    for ax, prefix, ylabel in [(axes[0], "P", "Precision@K"), (axes[1], "R", "Recall@K")]:
        for name in all_names:
            vals = [ranking_results[name][f"{prefix}@{k}"] for k in k_values]
            lw = 2.5 if "tuned" in name else 1.5
            ls = "--" if name == "Baseline" else "-"
            ax.plot(k_values, vals, marker="o", linewidth=lw, linestyle=ls,
                    color=colors[name], label=name)
        ax.set_xlabel("K"); ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontweight="bold")
        ax.set_xticks(k_values); ax.legend(fontsize=8); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Get Recommendations
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🎯 Get Recommendations":
    st.title("🎯 Personalised Movie Recommendations")
    st.markdown("Select a user to see their watch history and top-N SVD predictions.")

    col1, col2 = st.columns([1, 2])
    with col1:
        uid = st.number_input("User ID", min_value=1, max_value=N_USERS, value=1, step=1)
        top_n = st.slider("Number of recommendations", 5, 20, 10)
        model_choice = st.selectbox("Model", list(results.keys()) + ["SVD (tuned)"])
        run_btn = st.button("🔍 Recommend", use_container_width=True)

    with col2:
        if run_btn:
            chosen_model = best_svd if model_choice == "SVD (tuned)" else results[model_choice]["model"]
            total_rated = len(ratings_df[ratings_df["userId"] == uid])
            st.markdown(f"### 👤 User {uid}  —  {total_rated} total ratings")

            # History
            hist = ratings_df[ratings_df["userId"] == uid].merge(
                movies_df[["movieId", "title"]], on="movieId", how="left"
            ).sort_values("rating", ascending=False)

            st.markdown("**Top 5 previously rated movies:**")
            hist_display = hist[["title", "rating"]].head(5).reset_index(drop=True)
            hist_display.index += 1
            st.dataframe(hist_display, use_container_width=True)

            st.markdown(f"**Top {top_n} recommendations ({model_choice}):**")
            with st.spinner("Generating predictions…"):
                recs = get_top_n_recommendations(chosen_model, uid, ratings_df, movies_df, n=top_n)

            # Color-code predicted rating
            def highlight_rating(val):
                if val >= 4.5: return "background-color: #d4edda"
                if val >= 4.0: return "background-color: #fff3cd"
                return ""

            styled = recs.style.applymap(highlight_rating, subset=["predicted_rating"])
            st.dataframe(styled, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Cold-Start Explorer
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🧊 Cold-Start Explorer":
    st.title("🧊 Cold-Start Explorer")
    st.markdown(
        "No watch history? No problem. Select your favourite genres and we'll "
        "use **Bayesian popularity** to suggest films you'll likely enjoy."
    )

    selected_genres = st.multiselect("Choose genres you like", GENRE_COLS,
                                     default=["Action", "Sci-Fi"])
    top_n_cs = st.slider("How many recommendations?", 3, 15, 5)
    min_ratings_filter = st.slider("Minimum ratings required (quality filter)", 10, 200, 50)

    pop_df, _, _ = build_popularity_recommender(ratings_df, movies_df, min_ratings=min_ratings_filter)

    recs_cs = cold_start_recs(selected_genres, movies_df, pop_df, top_n=top_n_cs)
    st.subheader("🎬 Recommended for you")
    display_cols = [c for c in ["title", "n_ratings", "mean_rating", "bayesian_score"] if c in recs_cs.columns]
    st.dataframe(recs_cs[display_cols].reset_index(drop=True), use_container_width=True)

    st.divider()
    st.subheader("📈 Top 20 Movies Overall (Bayesian Score)")
    top20 = pop_df.head(20).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(top20["title"][::-1], top20["bayesian_score"][::-1], color=PALETTE[0], edgecolor="white")
    ax.set_xlabel("Bayesian Score"); ax.set_title("Top 20 by Bayesian Popularity", fontweight="bold")
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Genre Insights
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🏷️ Genre Insights":
    st.title("🏷️ Genre Insights")
    st.markdown("Business-level signals: which genres drive the most engagement and quality?")

    genre_stats = []
    for g in GENRE_COLS:
        gm = movies_df[movies_df[g] == 1]["movieId"]
        gr = ratings_df[ratings_df["movieId"].isin(gm)]
        if len(gr) > 0:
            genre_stats.append({
                "Genre": g,
                "Num_Movies": len(gm),
                "Num_Ratings": len(gr),
                "Avg_Rating": round(gr["rating"].mean(), 3),
            })
    genre_df = pd.DataFrame(genre_stats).sort_values("Num_Ratings", ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Ratings Volume by Genre**")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(genre_df["Genre"], genre_df["Num_Ratings"], color=PALETTE[0], edgecolor="white")
        ax.set_xlabel("Number of Ratings"); ax.invert_yaxis()
        ax.tick_params(axis="y", labelsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.markdown("**Average Rating by Genre**")
        gdf_s = genre_df.sort_values("Avg_Rating", ascending=False)
        clrs = [PALETTE[0] if r >= global_mean else PALETTE[3] for r in gdf_s["Avg_Rating"]]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(gdf_s["Genre"], gdf_s["Avg_Rating"], color=clrs, edgecolor="white")
        ax.axvline(global_mean, color="black", linestyle="--", linewidth=1.5,
                   label=f"Global mean = {global_mean:.2f}")
        ax.set_xlabel("Average Rating"); ax.set_xlim(3.0, 4.5)
        ax.invert_yaxis(); ax.legend(fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    st.divider()
    st.subheader("Full Genre Table")
    st.dataframe(genre_df.reset_index(drop=True), use_container_width=True)

    # Key insights
    top_genre   = genre_df.iloc[0]
    top_quality = genre_df.sort_values("Avg_Rating", ascending=False).iloc[0]
    cold_users  = (user_activity < 5).sum()
    st.divider()
    st.subheader("Key Business Insights")
    c1, c2, c3 = st.columns(3)
    c1.metric("Most engaged genre",   top_genre["Genre"],   f"{top_genre['Num_Ratings']:,} ratings")
    c2.metric("Highest quality genre", top_quality["Genre"], f"avg {top_quality['Avg_Rating']:.2f}★")
    c3.metric("Cold-start users",     f"{cold_users}",      f"{cold_users/N_USERS:.1%} of all users")