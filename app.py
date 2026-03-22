import streamlit as st
import pandas as pd
import plotly.express as px

# 🎨 COLOR SETTINGS
COLOR_THEME = px.colors.qualitative.Bold
TEMPLATE = "plotly_white"
# ----------------------------
# Page configuration
# ----------------------------
st.set_page_config(
    page_title="Real Estate Buyer Segmentation Dashboard",
    layout="wide"
)

# ----------------------------
# Load dataset
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("clustered_buyers_output.csv")
    return df

df = load_data()

# ----------------------------
# Title
# ----------------------------
st.title("Machine Learning Based Buyer Segmentation and Investment Profiling")
st.markdown("### Real Estate Market Intelligence Dashboard")

# ----------------------------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar filters
# ----------------------------
st.sidebar.header("Filter Buyers")

filtered_df = df.copy()

if 'country' in df.columns:
    country_options = sorted(df['country'].dropna().astype(str).unique())
    selected_country = st.sidebar.multiselect("Select Country", country_options)
    if selected_country:
        filtered_df = filtered_df[filtered_df['country'].astype(str).isin(selected_country)]

if 'region' in df.columns:
    region_options = sorted(df['region'].dropna().astype(str).unique())
    selected_region = st.sidebar.multiselect("Select Region", region_options)
    if selected_region:
        filtered_df = filtered_df[filtered_df['region'].astype(str).isin(selected_region)]

if 'acquisition_purpose' in df.columns:
    purpose_options = sorted(df['acquisition_purpose'].dropna().astype(str).unique())
    selected_purpose = st.sidebar.multiselect("Select Acquisition Purpose", purpose_options)
    if selected_purpose:
        filtered_df = filtered_df[filtered_df['acquisition_purpose'].astype(str).isin(selected_purpose)]

if 'client_type' in df.columns:
    client_type_options = sorted(df['client_type'].dropna().astype(str).unique())
    selected_client_type = st.sidebar.multiselect("Select Client Type", client_type_options)
    if selected_client_type:
        filtered_df = filtered_df[filtered_df['client_type'].astype(str).isin(selected_client_type)]

# Optional extra filter
if 'buyer_segment' in df.columns:
    segment_options = sorted(df['buyer_segment'].dropna().astype(str).unique())
    selected_segment = st.sidebar.multiselect("Select Buyer Segment", segment_options)
    if selected_segment:
        filtered_df = filtered_df[filtered_df['buyer_segment'].astype(str).isin(selected_segment)]

# ----------------------------
# Top metrics
# ----------------------------
st.subheader("Overview Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
        <div style="
            background-color:#1f77b4;
            padding:20px;
            border-radius:10px;
            text-align:center;
            color:white;">
            <h3>Total Buyers</h3>
            <h2>{len(filtered_df)}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div style="
            background-color:#2ca02c;
            padding:20px;
            border-radius:10px;
            text-align:center;
            color:white;">
            <h3>Total Segments</h3>
            <h2>{filtered_df['buyer_segment'].nunique()}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div style="
            background-color:#d62728;
            padding:20px;
            border-radius:10px;
            text-align:center;
            color:white;">
            <h3>Total Investment</h3>
            <h2>{filtered_df['total_investment_value'].sum():,.0f}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
# ----------------------------
# Buyer Segmentation Overview
# ----------------------------
st.subheader("Buyer Segmentation Overview")

if 'buyer_segment' in filtered_df.columns:
    segment_count = filtered_df['buyer_segment'].value_counts().reset_index()
    segment_count.columns = ['buyer_segment', 'count']

    fig_segment = px.bar(
        segment_count,
        x='buyer_segment',
        y='count',
        color='buyer_segment',
        color_discrete_sequence=COLOR_THEME,
        title="Cluster Distribution by Buyer Segment",
        text='count'
    )
    st.plotly_chart(fig_segment, use_container_width=True)

# ----------------------------
fig_segment.update_layout(
    template="plotly_white",
    title_font_size=20
)
# Investor Behavior Dashboard
# ----------------------------
st.subheader("Investor Behavior Dashboard")

col4, col5 = st.columns(2)

with col4:
    if 'buyer_segment' in filtered_df.columns and 'total_investment_value' in filtered_df.columns:
        spend_df = filtered_df.groupby('buyer_segment', as_index=False)['total_investment_value'].mean()
        fig_spend = px.bar(
            spend_df,
            x='buyer_segment',
            y='total_investment_value',
            color='buyer_segment',
            color_discrete_sequence=COLOR_THEME,
            title="Average Investment Value by Segment"
        )
        st.plotly_chart(fig_spend, use_container_width=True)

with col5:
    if 'buyer_segment' in filtered_df.columns and 'total_properties_bought' in filtered_df.columns:
        prop_df = filtered_df.groupby('buyer_segment', as_index=False)['total_properties_bought'].mean()
        fig_prop = px.bar(
            prop_df,
            x='buyer_segment',
            y='total_properties_bought',
            color='buyer_segment',
            color_discrete_sequence=COLOR_THEME,
            title="Average Properties Bought by Segment"
        )
        st.plotly_chart(fig_prop, use_container_width=True)

# ----------------------------
fig_segment.update_layout(
    template="plotly_white",
    title_font_size=20
)
# Loan behavior and purpose
# ----------------------------
col6, col7 = st.columns(2)

with col6:
    if 'buyer_segment' in filtered_df.columns and 'loan_applied' in filtered_df.columns:
        loan_chart = px.histogram(
            filtered_df,
            x='buyer_segment',
            color='loan_applied',
            barmode='group',
            color_discrete_sequence=px.colors.qualitative.Set2,
            title="Loan Behavior by Segment"
        )
        st.plotly_chart(loan_chart, use_container_width=True)

with col7:
    if 'buyer_segment' in filtered_df.columns and 'acquisition_purpose' in filtered_df.columns:
        purpose_chart = px.histogram(
            filtered_df,
            x='buyer_segment',
            color='acquisition_purpose',
            barmode='group',
            color_discrete_sequence=px.colors.qualitative.Pastel,
            title="Acquisition Purpose by Segment"
        )
        st.plotly_chart(purpose_chart, use_container_width=True)

# ----------------------------
fig_segment.update_layout(
    template="plotly_white",
    title_font_size=20
)
# Geographic Buyer Analysis
# ----------------------------
st.subheader("Geographic Buyer Analysis")

col8, col9 = st.columns(2)

with col8:
    if 'region' in filtered_df.columns and 'buyer_segment' in filtered_df.columns:
        region_chart = px.histogram(
            filtered_df,
            x='region',
            color='buyer_segment',
            barmode='group',
            color_discrete_sequence=COLOR_THEME,
            title="Buyer Segments by Region"
        )
        st.plotly_chart(region_chart, use_container_width=True)

with col9:
    if 'country' in filtered_df.columns and 'buyer_segment' in filtered_df.columns:
        country_counts = filtered_df['country'].value_counts().reset_index().head(10)
        country_counts.columns = ['country', 'count']
        fig_country = px.bar(
            country_counts,
            x='country',
            y='count',
            color='country',
            color_discrete_sequence=px.colors.qualitative.Safe,
            title="Top 10 Countries by Buyer Count"
        )
        st.plotly_chart(fig_country, use_container_width=True)

# ----------------------------
fig_segment.update_layout(
    template="plotly_white",
    title_font_size=20
)
# Segment Insights Panel
# ----------------------------
st.subheader("Segment Insights Panel")

if 'buyer_segment' in filtered_df.columns:
    insight_cols = []
    for col in ['age', 'satisfaction_score', 'total_properties_bought', 'total_investment_value', 'avg_property_value', 'avg_floor_area']:
        if col in filtered_df.columns:
            insight_cols.append(col)

    if insight_cols:
        insights_df = filtered_df.groupby('buyer_segment')[insight_cols].mean().round(2)
        st.dataframe(insights_df, use_container_width=True)

# ----------------------------
# Raw data preview
# ----------------------------
st.subheader("Filtered Data Preview")
st.dataframe(filtered_df.head(100), use_container_width=True)

# ----------------------------
# Author section
# ----------------------------
st.markdown("---")
st.markdown("### Author")
st.write("Kruthika Penugonda")
st.write("Data Analyst Intern")
st.write("Unified Mentor Pvt. Ltd")
st.write("March 2026")