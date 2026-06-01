import streamlit as st
import json
import pandas as pd
import os

# ---------------------------
# PAGE CONFIG
# ---------------------------

st.set_page_config(
    page_title="Purple Retail Intelligence Platform",
    layout="wide"
)

st.title(" Purple Retail Intelligence Platform")
st.caption(
    "AI Powered Visitor Tracking | Heatmap Analytics | Smart Store Insights"
)

# ---------------------------
# LOAD EVENTS
# ---------------------------

with open("data/events.json", "r") as f:
    events = json.load(f)

df = pd.DataFrame(events)

unique_visitors = len(df)
events_count = len(df)

# ---------------------------
# TEMPORARY VALUES
# ---------------------------

peak_occupancy = 5
staff_present = 5

entry_count = unique_visitors
exit_count = max(unique_visitors - 2, 0)
inside_store = entry_count - exit_count

store_score = 85

# ---------------------------
# TOP METRICS
# ---------------------------

st.subheader("📊 Store Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("👥 Unique Visitors", unique_visitors)

with col2:
    st.metric("📋 Events", events_count)

with col3:
    st.metric("📈 Peak Occupancy", peak_occupancy)

with col4:
    st.metric("👩 Staff Present", staff_present)

# ---------------------------
# ENTRY EXIT METRICS
# ---------------------------

st.subheader("🚪 Entry Analytics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🚶 Entries", entry_count)

with col2:
    st.metric("🚪 Exits", exit_count)

with col3:
    st.metric("🏬 Current Occupancy", inside_store)

# ---------------------------
# ALERTS
# ---------------------------

st.subheader("🚨 Smart Alerts")

if peak_occupancy >= 5:
    st.error(
        "High Footfall Detected - Additional Staff Recommended"
    )
else:
    st.success(
        "Store Occupancy Normal"
    )

# ---------------------------
# VISITOR TREND
# ---------------------------

st.subheader("📈 Visitor Analytics")

visitor_df = pd.DataFrame({
    "Visitors": [
        max(unique_visitors // 4, 1),
        max(unique_visitors // 2, 1),
        max((unique_visitors * 3) // 4, 1),
        unique_visitors
    ]
})

st.line_chart(visitor_df)

# ---------------------------
# STORE HEALTH SCORE
# ---------------------------

st.subheader("🏆 Store Health Score")

st.progress(store_score / 100)

st.metric(
    "Overall Store Score",
    f"{store_score}/100"
)

# ---------------------------
# ANALYTICS SUMMARY
# ---------------------------

st.subheader("📊 Analytics Summary")

st.write(f"👥 Total Unique Visitors: {unique_visitors}")
st.write(f"📈 Peak Occupancy: {peak_occupancy}")
st.write(f"👩 Staff Present: {staff_present}")
st.write(f"📷 Camera Monitored: CAM2")
st.write(f"🏬 Current Occupancy: {inside_store}")

# ---------------------------
# HEATMAP + INSIGHTS
# ---------------------------

st.subheader("🔥 Store Movement Heatmap")

col1, col2 = st.columns([2, 1])

with col1:

    if os.path.exists("data/heatmap.png"):
        st.image(
            "data/heatmap.png",
            use_container_width=True
        )
    else:
        st.warning("Heatmap image not found")

with col2:

    st.markdown("""
    ### 🤖 AI Heatmap Insights

    ✅ Most movement near entrance

    ✅ Customers spend more time
    in hotspot zones

    ✅ Promotional displays can be
    placed near high traffic areas

    ✅ Improve visibility in cold zones

    ✅ Optimize staff allocation
    during busy hours
    """)

# ---------------------------
# AI RECOMMENDATIONS
# ---------------------------

st.subheader("🤖 AI Recommendations")

if unique_visitors > 10:

    st.info("""
    • Store traffic is healthy

    • Increase staffing during peak hours

    • Customers spend more time near entrance

    • Place promotional products in hotspot zones

    • Improve engagement in low-traffic areas
    """)

else:

    st.warning("""
    Visitor traffic is low.

    Consider:
    • Running promotions
    • Improving product visibility
    • Increasing customer engagement
    """)

# ---------------------------
# EVENT LOG
# ---------------------------

st.subheader("📑 Event Log")

st.dataframe(
    df,
    use_container_width=True
)

# ---------------------------
# FOOTER
# ---------------------------

st.markdown("---")
st.markdown(
    "Built with YOLOv8 + OpenCV + FastAPI + Streamlit "
)
sales_df = pd.read_csv("data/sales.csv")
st.write(sales_df.columns.tolist())
# ---------------------------
# SALES ANALYTICS
# ---------------------------

st.subheader("💰 Sales Intelligence")

try:
    sales_df = pd.read_csv("data/sales.csv")

    total_transactions = sales_df["invoice_number"].nunique()

    total_gmv = pd.to_numeric(
        sales_df["GMV"],
        errors="coerce"
    ).fillna(0).sum()

    conversion_rate = (
        total_transactions /
        max(unique_visitors, 1)
    ) * 100

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "🧾 Transactions",
            total_transactions
        )

    with col2:
        st.metric(
            "💰 Total GMV",
            f"₹{total_gmv:,.0f}"
        )

    with col3:
        st.metric(
            "🎯 Conversion Rate",
            f"{conversion_rate:.1f}%"
        )

    # -----------------------
    # TOP BRANDS
    # -----------------------

    st.subheader("🏷 Top Selling Brands")

    top_brands = (
        sales_df["brand_name"]
        .astype(str)
        .value_counts()
        .head(10)
    )

    st.bar_chart(top_brands)

    # -----------------------
    # TOP PRODUCTS
    # -----------------------

    st.subheader("🛍 Top Selling Products")

    top_products = (
        sales_df["product_name"]
        .astype(str)
        .value_counts()
        .head(10)
    )

    st.bar_chart(top_products)

    # -----------------------
    # TOP CATEGORIES
    # -----------------------

    st.subheader("📦 Top Categories")

    top_categories = (
        sales_df["sub_category"]
        .astype(str)
        .value_counts()
        .head(10)
    )

    st.bar_chart(top_categories)

    # -----------------------
    # PEAK SALES HOURS
    # -----------------------

    sales_df["hour"] = pd.to_datetime(
        sales_df["order_time"],
        format="%H:%M:%S",
        errors="coerce"
    ).dt.hour

    hourly_sales = (
        sales_df.groupby("hour")
        .size()
    )

    st.subheader("⏰ Peak Sales Hours")

    st.line_chart(hourly_sales)

    peak_hour = hourly_sales.idxmax()

    # -----------------------
    # AI INSIGHTS
    # -----------------------

    top_brand = (
        sales_df["brand_name"]
        .value_counts()
        .idxmax()
    )

    top_category = (
        sales_df["sub_category"]
        .value_counts()
        .idxmax()
    )

    st.subheader("🤖 AI Retail Insights")

    st.success(f"""
    Peak sales hour: {peak_hour}:00

    Best performing brand: {top_brand}

    Highest selling category: {top_category}

    Total transactions: {total_transactions}

    Conversion rate: {conversion_rate:.1f}%

    Recommendation:
    Deploy more staff during peak hours
    and place promotions near hotspot zones.
    """)

except Exception as e:

    st.error(
        f"Sales file error: {e}"
    )
