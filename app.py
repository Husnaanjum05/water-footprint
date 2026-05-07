import streamlit as st
import pandas as pd
import plotly.express as px
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="AgriWater AI Dashboard", layout="wide", page_icon="💧")

# --- 2. API KEY SETUP ---
# It's best to set this in Streamlit Cloud Secrets: Settings -> Secrets
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("Please add your GOOGLE_API_KEY to Streamlit Secrets.")
    st.stop()

# --- 3. DATA LOADING ---
@st.cache_data
def load_agri_data():
    # Mock data representing Liters of water per Kilogram of product
    data = {
        'Crop': ['Almonds', 'Wheat', 'Rice', 'Maize', 'Soybeans', 'Coffee', 'Chocolate'],
        'Category': ['Nuts', 'Cereals', 'Cereals', 'Cereals', 'Pulses', 'Luxury', 'Luxury'],
        'Blue_Water': [10200, 342, 1025, 189, 100, 150, 1000],
        'Green_Water': [1500, 1277, 2248, 700, 1800, 17000, 15000],
        'Grey_Water': [1100, 208, 137, 150, 200, 1500, 1100],
    }
    df = pd.DataFrame(data)
    df['Total_Footprint'] = df['Blue_Water'] + df['Green_Water'] + df['Grey_Water']
    return df

# Initialize the dataframe
df = load_agri_data()

# --- 4. SIDEBAR NAVIGATION ---
st.sidebar.title("💧 Water Footprint Analytics")
st.sidebar.markdown("Explore the environmental impact of agricultural products.")

st.sidebar.subheader("Global Filters")
selected_category = st.sidebar.multiselect(
    "Filter by Category", 
    df['Category'].unique(), 
    default=df['Category'].unique()
)
filtered_df = df[df['Category'].isin(selected_category)]

# --- 5. MAIN DASHBOARD UI ---
st.title("Interactive Water Footprint Dashboard")

# Metric Summary Row
m1, m2, m3 = st.columns(3)
m1.metric("Selected Crops", len(filtered_df))
m2.metric("Highest Total Footprint", f"{filtered_df['Total_Footprint'].max():,} L/kg")
m3.metric("Avg Blue Water", f"{int(filtered_df['Blue_Water'].mean()):,} L/kg")

# Tabs for Organization
tab1, tab2, tab3 = st.tabs(["📊 Visual Analytics", "🤖 AI Data Expert", "📄 Raw Data"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Water Composition (Stacked)")
        fig = px.bar(
            filtered_df, 
            x='Crop', 
            y=['Blue_Water', 'Green_Water', 'Grey_Water'], 
            barmode='stack',
            labels={'value': 'Liters per Kilogram', 'variable': 'Water Type'},
            color_discrete_map={
                'Blue_Water': '#3498db', 
                'Green_Water': '#2ecc71', 
                'Grey_Water': '#95a5a6'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Total Footprint Distribution")
        fig_pie = px.pie(filtered_df, values='Total_Footprint', names='Crop', hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

with tab2:
    st.subheader("Ask Gemini about your Data")
    st.info("Ask questions like: 'Which cereal has the highest green water?' or 'Compare Coffee and Chocolate water usage.'")
    
    user_query = st.text_input("Enter your query for the AI Agent:")
    
    if user_query:
        # Initialize Gemini Model
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        
        # Create Agent with correct 2026 tool-calling pattern
        agent = create_pandas_dataframe_agent(
            llm, 
            df, 
            verbose=True, 
            agent_type="tool-calling", 
            allow_dangerous_code=True
        )
        
        with st.spinner("Gemini is analyzing the data..."):
            try:
                response = agent.run(user_query)
                st.write("### AI Response:")
                st.success(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")

with tab3:
    st.subheader("Dataset Preview")
    st.dataframe(filtered_df, use_container_width=True)
