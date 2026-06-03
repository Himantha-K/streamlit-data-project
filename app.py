import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Set up page configuration for an eye-catching dashboard
st.set_page_config(
    page_title="AI Pattern Finder",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-title { font-size: 42px; font-weight: 800; color: #1E3A8A; margin-bottom: 10px; }
    .subtitle { font-size: 18px; color: #4B5563; margin-bottom: 30px; }
    .metric-card { background-color: #F3F4F6; padding: 20px; border-radius: 10px; border-left: 5px solid #3B82F6; }
    </style>
""", unsafe_allow_html=True)

# App Headers
st.markdown('<div class="main-title">📊 Smart Pattern Finder</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Mine data trends, correlations, and insights instantly from any CSV URL.</div>', unsafe_allow_html=True)
st.sidebar.header("⚙️ Data Configuration")

# 1. User Input: CSV Link
csv_url = st.sidebar.text_input(
    "Paste CSV File URL:",
    placeholder="https://raw.githubusercontent.com/datasets/population/master/data/population.csv"
)

@st.cache_data
def load_data(url):
    """Loads and caches data to make the website lightning fast."""
    return pd.read_csv(url)

if csv_url:
    try:
        # Load the data
        with st.spinner("Mining the dataset for patterns..."):
            df = load_data(csv_url)
        
        st.sidebar.success("Data successfully loaded!")
        
        # --- SECTION 1: DATA OVERVIEW ---
        st.subheader("📋 Dataset Preview & Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Total Rows (Records)", value=df.shape[0])
        with col2:
            st.metric(label="Total Columns (Features)", value=df.shape[1])
        with col3:
            st.metric(label="Missing Values detected", value=df.isnull().sum().sum())
            
        with st.expander("👀 View Raw Dataset"):
            st.dataframe(df.head(100), use_container_width=True)

        # Separate Numeric and Categorical columns for analysis
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # --- SECTION 2: DATA MINING & PATTERN ANALYSIS ---
        st.markdown("---")
        st.subheader("🔍 Trend & Pattern Analysis")
        
        tab1, tab2, tab3 = st.tabs(["📈 Distribution & Trends", "🔗 Correlations & Relationships", "📊 Categorical Breakdowns"])

        # TAB 1: Trends & Distributions
        with tab1:
            if num_cols:
                st.markdown("### Numerical Data Distributions")
                selected_num = st.selectbox("Select a column to view its trend/distribution:", num_cols)
                
                # Highlight basic stats
                mean_val = df[selected_num].mean()
                max_val = df[selected_num].max()
                
                c1, c2 = st.columns([1, 3])
                with c1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>Stats for {selected_num}</h4>
                        <p><b>Average:</b> {mean_val:,.2f}</p>
                        <p><b>Peak Value:</b> {max_val:,.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with c2:
                    # Eye-catching histogram/line trend
                    fig_hist = px.histogram(df, x=selected_num, marginal="box", 
                                            title=f"Distribution Pattern of {selected_num}",
                                            color_discrete_sequence=['#3B82F6'])
                    fig_hist.update_layout(template="plotly_white")
                    st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.warning("No numerical columns found to chart trends.")

        # TAB 2: Correlations (Finding links between data points)
        with tab2:
            if len(num_cols) >= 2:
                st.markdown("### Pattern Hunting: Finding Relationships")
                col_x = st.selectbox("Select X-Axis Variable:", num_cols, index=0)
                col_y = st.selectbox("Select Y-Axis Variable:", num_cols, index=min(1, len(num_cols)-1))
                
                color_by = st.selectbox("Color code points by (Categorical Column):", [None] + cat_cols)
                
                fig_scatter = px.scatter(df, x=col_x, y=col_y, color=color_by, 
                                         title=f"Relationship Pattern: {col_x} vs {col_y}",
                                         trendline="ols" if color_by is None else None) # Adds a trend line if no color grouping
                fig_scatter.update_layout(template="plotly_white")
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Corelation Matrix Heatmap
                if len(num_cols) > 2:
                    st.markdown("### Global Correlation Matrix")
                    corr = df[num_cols].corr()
                    fig_heat = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                                         title="Strength of Patterns Between All Numeric Variables")
                    st.plotly_chart(fig_heat, use_container_width=True)
            else:
                st.warning("You need at least 2 numerical columns to look for cross-variable patterns.")

        # TAB 3: Categorical Breakdowns
        with tab3:
            if cat_cols:
                st.markdown("### Composition & Ranking Analysis")
                selected_cat = st.selectbox("Select Category to Analyze:", cat_cols)
                
                # Count the frequency of patterns
                cat_counts = df[selected_cat].value_counts().reset_index()
                cat_counts.columns = [selected_cat, 'Count']
                
                # Top 10 breakdown
                fig_bar = px.bar(cat_counts.head(15), x=selected_cat, y='Count', 
                                 title=f"Top Volume Breakdowns for {selected_cat}",
                                 color='Count', color_continuous_scale='Viridis')
                fig_bar.update_layout(template="plotly_white")
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.warning("No categorical columns found to break down.")

    except Exception as e:
        st.error(f"Error parsing the CSV file. Please check the URL. Details: {e}")

else:
    # Landing View before user enters a link
    st.info("💡 **How to start:** Paste a direct link to a CSV file in the sidebar text box to extract patterns automatically.")
    
    # Showcase standard example link user can test with
    st.markdown("""
    ### Try it with this sample link:
    Copy and paste this URL into the sidebar input to see it act like a website instantly:
    ```text
    [https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv)
    ```
    """)