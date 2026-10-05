"""
Customer Segmentation Dashboard
===============================
E-commerce RFM analysis + K-Means clustering, built with Streamlit and Plotly Express.

Run with:  streamlit run app.py

Design notes
------------
* RFM metrics and clusters are computed on the *full* cleaned history so that a
  customer's segment stays stable. The sidebar filters (country, date range, segment)
  only change which transactions / customers are *displayed*.
* Only the number of clusters (k) re-runs the model; everything else is a cheap filter.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# 0. PAGE CONFIG, PATHS & THEME
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🛍️",
    layout="wide",
)

DATA_PATH = Path(__file__).parent / "data" / "online_retail.csv"
EXPECTED_COLUMNS = [
    "InvoiceNo", "StockCode", "Description", "Quantity",
    "InvoiceDate", "UnitPrice", "CustomerID", "Country",
]

# One consistent colour palette used by every chart.
PRIMARY = "#4C78A8"  # single-series charts
SEQUENTIAL = "Blues"  # heatmaps
SEGMENT_BASE_COLORS = {
    "Champions": "#2E9E6B",
    "Loyal Customers": "#4C78A8",
    "Potential Loyalists": "#F2B134",
    "At Risk": "#E8743B",
    "Lost / Hibernating": "#8C8C99",
}
EXTRA_COLORS = ["#7A5CA8", "#17A2B8", "#C2548A", "#8C6D31"]  # for duplicated tiers when k > 5

px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = [PRIMARY, "#2E9E6B", "#F2B134", "#E8743B", "#8C8C99"]

# Segment names from best to worst. Used to label clusters.
SEGMENT_TIERS = ["Champions", "Loyal Customers", "Potential Loyalists", "At Risk", "Lost / Hibernating"]

# Non-product lines (postage, manual adjustments, fees) excluded from product charts only.
NON_PRODUCT_CODES = {"POST", "M", "DOT", "C2", "BANK CHARGES", "PADS", "AMAZONFEE", "S", "D", "CRUK", "B"}

MARKETING_ACTIONS = {
    "Champions": "Reward them: VIP perks, early access to new products, referral and review programmes.",
    "Loyal Customers": "Upsell and cross-sell: bundles, loyalty points and membership tiers.",
    "Potential Loyalists": "Nurture: personalised recommendations and a nudge toward a second or third purchase.",
    "At Risk": "Win back: time-limited discounts and 'we miss you' emails before they lapse.",
    "Lost / Hibernating": "Low-cost reactivation: a final win-back offer, then suppress to save marketing spend.",
}


def base_name(label: str) -> str:
    """Strip the ' (n)' suffix added when k > 5 forces duplicate tier names."""
    return label.split(" (")[0]


# --------------------------------------------------------------------------- #
# 1. DATA LOADING & PREPROCESSING (cached)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading and cleaning data…")
def load_and_clean(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the raw CSV, clean it and return (clean_df, cleaning_summary).

    Cleaning steps (applied sequentially, row counts recorded after each):
      1. drop duplicate rows
      2. drop rows with missing CustomerID
      3. drop cancelled invoices (InvoiceNo starts with 'C')
      4. drop rows with Quantity <= 0 or UnitPrice <= 0
    Then: parse dates, cast CustomerID to int, fill missing Description,
    and add Revenue and Month columns.
    """
    raw = pd.read_csv(path, dtype={"InvoiceNo": str, "StockCode": str})
    steps = [("Raw rows", len(raw))]

    df = raw.drop_duplicates()
    steps.append(("After dropping duplicates", len(df)))

    df = df.dropna(subset=["CustomerID"])
    steps.append(("After dropping missing CustomerID", len(df)))

    df = df[~df["InvoiceNo"].str.startswith("C")]
    steps.append(("After removing cancelled invoices", len(df)))

    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    steps.append(("After removing Quantity/UnitPrice <= 0", len(df)))

    df = df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["Description"] = df["Description"].fillna("UNKNOWN").str.strip()
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    df = df.reset_index(drop=True)

    summary = pd.DataFrame(steps, columns=["Step", "Rows"])
    summary["Rows removed"] = (summary["Rows"].shift(1) - summary["Rows"]).fillna(0).astype(int)
    return df, summary


# --------------------------------------------------------------------------- #
# 2. RFM + CLUSTERING (cached)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Computing RFM features…")
def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Build one row per customer with Recency, Frequency, Monetary (+ main country)."""
    snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda s: (snapshot - s.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum"),
    )
    # Main country = the country where the customer spent the most.
    country_rev = df.groupby(["CustomerID", "Country"])["Revenue"].sum().reset_index()
    main_country = country_rev.loc[country_rev.groupby("CustomerID")["Revenue"].idxmax()]
    rfm["Country"] = main_country.set_index("CustomerID")["Country"]
    return rfm.reset_index()


def _scale(rfm: pd.DataFrame) -> np.ndarray:
    """log1p-transform (tames skew) then standardise R, F, M."""
    logged = np.log1p(rfm[["Recency", "Frequency", "Monetary"]])
    return StandardScaler().fit_transform(logged)


def name_clusters(rfm: pd.DataFrame, labels: np.ndarray) -> dict[int, str]:
    """Automatically name clusters from their RFM averages.

    Each cluster gets a composite score = z(Frequency) + z(Monetary) - z(Recency)
    (low recency is good). Clusters are ranked best→worst and spread evenly across
    the five tiers. With k > 5, tiers repeat and get a numeric suffix to stay unique.
    """
    tmp = rfm[["Recency", "Frequency", "Monetary"]].copy()
    tmp["cluster"] = labels
    means = tmp.groupby("cluster").mean()
    z = (means - means.mean()) / means.std(ddof=0).replace(0, 1)
    score = z["Frequency"] + z["Monetary"] - z["Recency"]
    ranked = score.sort_values(ascending=False).index.tolist()

    k = len(ranked)
    tier_idx = [0] if k == 1 else [round(i * (len(SEGMENT_TIERS) - 1) / (k - 1)) for i in range(k)]
    names = {}
    for cluster, idx in zip(ranked, tier_idx):
        names[cluster] = SEGMENT_TIERS[idx]

    # Disambiguate duplicates (only possible when k > 5).
    for tier in set(names.values()):
        members = [c for c in ranked if names[c] == tier]
        if len(members) > 1:
            for n, c in enumerate(members, start=1):
                names[c] = f"{tier} ({n})"
    return names


@st.cache_data(show_spinner="Running K-Means…")
def run_kmeans(rfm: pd.DataFrame, k: int) -> dict:
    """Fit K-Means for a given k and return labels, names, inertia and silhouette."""
    X = _scale(rfm)
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    sil = silhouette_score(X, km.labels_, sample_size=3000, random_state=42)
    names = name_clusters(rfm, km.labels_)
    segment = pd.Series(km.labels_).map(names)
    # Ordered best→worst, used for consistent chart ordering.
    order = sorted(set(names.values()), key=lambda n: _rank_key(n))
    return {"segment": segment.values, "inertia": km.inertia_, "silhouette": sil, "order": order}


def _rank_key(label: str) -> tuple[int, int]:
    """Sort key: tier position, then numeric suffix."""
    tier = SEGMENT_TIERS.index(base_name(label))
    suffix = int(label.split("(")[1].rstrip(")")) if "(" in label else 0
    return tier, suffix


@st.cache_data(show_spinner="Evaluating k = 2…8…")
def k_sweep(rfm: pd.DataFrame) -> pd.DataFrame:
    """Inertia and silhouette for k = 2..8 (powers the elbow / silhouette charts)."""
    X = _scale(rfm)
    rows = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        sil = silhouette_score(X, km.labels_, sample_size=3000, random_state=42)
        rows.append({"k": k, "Inertia": km.inertia_, "Silhouette": sil})
    return pd.DataFrame(rows)


def segment_color_map(order: list[str]) -> dict[str, str]:
    """Stable colour per segment; duplicated tiers (k > 5) get extra colours."""
    cmap, extras = {}, iter(EXTRA_COLORS)
    seen = set()
    for label in order:
        b = base_name(label)
        if b not in seen:
            cmap[label] = SEGMENT_BASE_COLORS[b]
            seen.add(b)
        else:
            cmap[label] = next(extras)
    return cmap


# --------------------------------------------------------------------------- #
# 3. LOAD DATA (with friendly error handling)
# --------------------------------------------------------------------------- #
st.title("🛍️ Customer Segmentation Dashboard")
st.caption("RFM analysis + K-Means clustering on online retail transactions.")

if not DATA_PATH.exists():
    st.error(
        f"**Dataset not found.** Please place the CSV at `{DATA_PATH}` "
        "(expected columns: " + ", ".join(EXPECTED_COLUMNS) + ") and refresh the page."
    )
    st.stop()

try:
    df, cleaning_summary = load_and_clean(str(DATA_PATH))
except Exception as exc:  # noqa: BLE001 - show any parsing problem nicely
    st.error(f"Could not read the dataset: `{exc}`")
    st.stop()

missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
if missing:
    st.error(f"The CSV is missing required columns: {', '.join(missing)}")
    st.stop()

rfm = compute_rfm(df)

# --------------------------------------------------------------------------- #
# 4. SIDEBAR FILTERS
# --------------------------------------------------------------------------- #
st.sidebar.header("Filters")
k = st.sidebar.slider("Number of clusters (k)", 2, 8, 4)

model = run_kmeans(rfm, k)
rfm = rfm.assign(Segment=model["segment"])
seg_order = model["order"]
seg_colors = segment_color_map(seg_order)

selected_segments = st.sidebar.multiselect("Customer segment", seg_order, default=seg_order)

all_countries = sorted(df["Country"].unique())
selected_countries = st.sidebar.multiselect("Country", all_countries, default=all_countries)

min_date, max_date = df["InvoiceDate"].min().date(), df["InvoiceDate"].max().date()
date_range = st.sidebar.date_input(
    "Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date
)

top_n = st.sidebar.slider("Top N products", 5, 20, 10)

st.sidebar.caption("CustomerID is anonymised. No personal data is used.")

# date_input returns a 1-tuple while the user is still choosing the end date.
if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
    st.info("Select an end date to apply the date range.")
    st.stop()

# --------------------------------------------------------------------------- #
# 5. APPLY FILTERS
# --------------------------------------------------------------------------- #
seg_lookup = rfm.set_index("CustomerID")["Segment"]
data = df.assign(Segment=df["CustomerID"].map(seg_lookup))

start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
mask = (
    data["Segment"].isin(selected_segments)
    & data["Country"].isin(selected_countries)
    & (data["InvoiceDate"] >= start)
    & (data["InvoiceDate"] < end)
)
fdata = data[mask]

if fdata.empty:
    st.warning("No data matches the current filters. Try widening the date range or selecting more segments/countries.")
    st.stop()

# Customer-level view: customers that have transactions inside the filtered set.
fcust = rfm[rfm["CustomerID"].isin(fdata["CustomerID"].unique())]

# --------------------------------------------------------------------------- #
# 6. KPI CARDS
# --------------------------------------------------------------------------- #
total_revenue = fdata["Revenue"].sum()
n_customers = fdata["CustomerID"].nunique()
n_orders = fdata["InvoiceNo"].nunique()
aov = total_revenue / n_orders

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Revenue", f"£{total_revenue:,.0f}")
c2.metric("Customers", f"{n_customers:,}")
c3.metric("Orders", f"{n_orders:,}")
c4.metric("Average Order Value", f"£{aov:,.2f}")
c5.metric("Silhouette Score", f"{model['silhouette']:.3f}", help=f"K-Means with k={k} (sample of 3,000 customers).")

# --------------------------------------------------------------------------- #
# 7. TABS
# --------------------------------------------------------------------------- #
tab_overview, tab_segments, tab_products, tab_model = st.tabs(
    ["📊 Overview", "👥 Segments", "📦 Product Demand", "⚙️ Model & Data"]
)

# ---- Tab 1: Overview ------------------------------------------------------- #
with tab_overview:
    monthly = fdata.groupby("Month", as_index=False)["Revenue"].sum()
    fig = px.line(monthly, x="Month", y="Revenue", markers=True, title="Monthly revenue (£)")
    fig.update_layout(xaxis_title=None, yaxis_title="Revenue (£)")
    st.plotly_chart(fig, width="stretch")
    st.caption("The final month in the dataset (Dec 2011) is only partially covered.")

    left, right = st.columns(2)
    with left:
        by_country = (
            fdata.groupby("Country", as_index=False)["Revenue"].sum()
            .nlargest(10, "Revenue").sort_values("Revenue")
        )
        fig = px.bar(by_country, x="Revenue", y="Country", orientation="h", title="Top 10 countries by revenue (£)")
        fig.update_layout(yaxis_title=None, xaxis_title="Revenue (£)")
        st.plotly_chart(fig, width="stretch")
        st.caption("The UK dominates; use the Country filter to explore smaller markets.")
    with right:
        seg_counts = fcust["Segment"].value_counts().rename_axis("Segment").reset_index(name="Customers")
        fig = px.pie(seg_counts, names="Segment", values="Customers", hole=0.55,
                     color="Segment", color_discrete_map=seg_colors,
                     category_orders={"Segment": seg_order}, title="Customers per segment")
        st.plotly_chart(fig, width="stretch")

    seg_rev = fdata.groupby("Segment", as_index=False)["Revenue"].sum()
    fig = px.pie(seg_rev, names="Segment", values="Revenue", hole=0.55,
                 color="Segment", color_discrete_map=seg_colors,
                 category_orders={"Segment": seg_order}, title="Revenue per segment")
    st.plotly_chart(fig, width="stretch")

# ---- Tab 2: Segments ------------------------------------------------------- #
with tab_segments:
    profile = (
        fcust.groupby("Segment")
        .agg(Customers=("CustomerID", "count"),
             **{"Avg Recency (days)": ("Recency", "mean"),
                "Avg Frequency (orders)": ("Frequency", "mean"),
                "Avg Monetary (£)": ("Monetary", "mean")})
        .reindex([s for s in seg_order if s in fcust["Segment"].values])
        .round(1)
        .reset_index()
    )
    st.subheader("Segment profile")
    st.dataframe(profile, width="stretch", hide_index=True)

    sample = fcust.sample(min(3000, len(fcust)), random_state=42)
    fig = px.scatter_3d(
        sample, x="Recency", y="Frequency", z="Monetary", color="Segment",
        color_discrete_map=seg_colors, category_orders={"Segment": seg_order},
        log_x=True, log_y=True, log_z=True, opacity=0.7,
        hover_data=["CustomerID"], title="3D RFM view (log scale, sample of up to 3,000 customers)",
    )
    fig.update_traces(marker_size=3)
    fig.update_layout(height=600, margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    with left:
        fig = px.scatter(
            fcust, x="Frequency", y="Monetary", color="Segment", log_x=True, log_y=True,
            color_discrete_map=seg_colors, category_orders={"Segment": seg_order},
            opacity=0.6, hover_data=["CustomerID", "Recency"], title="Frequency vs Monetary",
        )
        fig.update_layout(xaxis_title="Frequency (orders, log)", yaxis_title="Monetary (£, log)")
        st.plotly_chart(fig, width="stretch")
    with right:
        melted = fcust.melt(id_vars=["CustomerID", "Segment"], value_vars=["Recency", "Frequency", "Monetary"],
                            var_name="Metric", value_name="Value")
        fig = px.box(melted, x="Segment", y="Value", color="Segment", facet_col="Metric",
                     color_discrete_map=seg_colors, category_orders={"Segment": seg_order, "Metric": ["Recency", "Frequency", "Monetary"]},
                     title="RFM distribution by segment (log scale)")
        fig.update_yaxes(matches=None, type="log", showticklabels=True)
        fig.update_xaxes(title=None, showticklabels=False)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Suggested marketing actions")
    present = [s for s in seg_order if s in fcust["Segment"].values]
    cols = st.columns(min(len(present), 3))
    for i, seg in enumerate(present):
        with cols[i % len(cols)].container(border=True):
            st.markdown(f"**{seg}**")
            st.write(MARKETING_ACTIONS[base_name(seg)])

# ---- Tab 3: Product Demand ------------------------------------------------- #
with tab_products:
    products = fdata[~fdata["StockCode"].str.upper().isin(NON_PRODUCT_CODES)]
    st.caption("Non-product lines (postage, manual adjustments, bank charges) are excluded from this tab.")

    if products.empty:
        st.warning("No product sales match the current filters.")
    else:
        agg = products.groupby("Description").agg(Units=("Quantity", "sum"), Revenue=("Revenue", "sum"))
        top_units = agg.nlargest(top_n, "Units").reset_index().sort_values("Units")
        top_rev = agg.nlargest(top_n, "Revenue").reset_index().sort_values("Revenue")

        left, right = st.columns(2)
        with left:
            fig = px.bar(top_units, x="Units", y="Description", orientation="h", title=f"Top {top_n} products by units sold")
            fig.update_layout(yaxis_title=None)
            st.plotly_chart(fig, width="stretch")
        with right:
            fig = px.bar(top_rev, x="Revenue", y="Description", orientation="h", title=f"Top {top_n} products by revenue (£)")
            fig.update_layout(yaxis_title=None, xaxis_title="Revenue (£)")
            st.plotly_chart(fig, width="stretch")

        # Segment x product heatmap: each cell = the product's share of that segment's revenue,
        # so segments of different sizes are comparable.
        seg_totals = products.groupby("Segment")["Revenue"].sum()
        top_names = top_rev["Description"].tolist()[::-1]
        cell = (
            products[products["Description"].isin(top_names)]
            .pivot_table(index="Segment", columns="Description", values="Revenue", aggfunc="sum", fill_value=0)
            .reindex(index=[s for s in seg_order if s in seg_totals.index], columns=top_names, fill_value=0)
        )
        share = cell.div(seg_totals.reindex(cell.index), axis=0) * 100
        fig = px.imshow(share.round(2), text_auto=".1f", aspect="auto", color_continuous_scale=SEQUENTIAL,
                        labels=dict(color="% of segment revenue"),
                        title=f"Which segment buys the top {top_n} products? (% of each segment's product revenue)")
        fig.update_layout(xaxis_title=None, yaxis_title=None)
        fig.update_xaxes(tickangle=-40)
        st.plotly_chart(fig, width="stretch")

        # Orders by weekday x hour (unique invoices).
        orders = fdata.drop_duplicates("InvoiceNo").assign(
            Day=lambda d: d["InvoiceDate"].dt.day_name(), Hour=lambda d: d["InvoiceDate"].dt.hour)
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        grid = (orders.groupby(["Day", "Hour"]).size().unstack(fill_value=0)
                .reindex(index=days, fill_value=0))
        grid = grid.reindex(columns=range(int(orders["Hour"].min()), int(orders["Hour"].max()) + 1), fill_value=0)
        fig = px.imshow(grid, aspect="auto", color_continuous_scale=SEQUENTIAL, text_auto=True,
                        labels=dict(x="Hour of day", y="Day", color="Orders"), title="Orders by day of week and hour")
        st.plotly_chart(fig, width="stretch")
        st.caption("The source data contains no Saturday trading.")

# ---- Tab 4: Model & Data --------------------------------------------------- #
with tab_model:
    sweep = k_sweep(rfm.drop(columns="Segment"))
    left, right = st.columns(2)
    with left:
        fig = px.line(sweep, x="k", y="Inertia", markers=True, title="Elbow chart (inertia vs k)")
        fig.add_vline(x=k, line_dash="dash", line_color="#E8743B", annotation_text=f"k = {k}")
        st.plotly_chart(fig, width="stretch")
    with right:
        fig = px.line(sweep, x="k", y="Silhouette", markers=True, title="Silhouette score vs k")
        fig.add_vline(x=k, line_dash="dash", line_color="#E8743B", annotation_text=f"k = {k}")
        st.plotly_chart(fig, width="stretch")
    st.caption(f"Current model: k = {k}, inertia = {model['inertia']:,.1f}, silhouette = {model['silhouette']:.3f}.")

    st.subheader("Data cleaning summary")
    raw_rows, clean_rows = cleaning_summary["Rows"].iloc[0], cleaning_summary["Rows"].iloc[-1]
    m1, m2, m3 = st.columns(3)
    m1.metric("Raw rows", f"{raw_rows:,}")
    m2.metric("Cleaned rows", f"{clean_rows:,}")
    m3.metric("Rows removed", f"{raw_rows - clean_rows:,} ({(raw_rows - clean_rows) / raw_rows:.1%})")
    st.dataframe(cleaning_summary, width="stretch", hide_index=True)

    st.subheader("Segmented customers")
    table = fcust[["CustomerID", "Country", "Recency", "Frequency", "Monetary", "Segment"]].copy()
    table["Monetary"] = table["Monetary"].round(2)
    st.dataframe(table, width="stretch", hide_index=True)
    st.download_button(
        "⬇️ Download segmented customers (CSV)",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name=f"customer_segments_k{k}.csv",
        mime="text/csv",
    )
