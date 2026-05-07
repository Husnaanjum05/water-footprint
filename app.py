import streamlit as st
import pandas as pd
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# 1. LOAD DATA FIRST
@st.cache_data
def load_data():
    # Example data - replace with your actual CSV loading
    data = {
        'Crop': ['Almonds', 'Rice', 'Wheat'],
        'Blue_Water': [10200, 1025, 342],
        'Total_Footprint': [16000, 2500, 1300]
    }
    return pd.DataFrame(data)

# This creates the 'df' variable
df = load_data() 

# 2. INITIALIZE LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# 3. CREATE AGENT (Using the 'df' defined above)
agent = create_pandas_dataframe_agent(
    llm, 
    df, # This is line 19 - it will now find the 'df' variable
    verbose=True, 
    agent_type="tool-calling", 
    allow_dangerous_code=True
)
# --- 1. CONFIGURATION & API SETUP ---
st.set_page_config(page_title="AgriWater AI Dashboard", layout="wide", page_icon="💧")

# Accessing the Google API Key (ensure it's in your .env or streamlit secrets)
# os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY" 
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("Please set your GOOGLE_API_KEY in the Streamlit secrets or environment variables.")
    st.stop()

# --- 2. DATA ENGINE ---
@st.cache_data
def load_agri_data():
    # Placeholder: In a real scenario, you'd load your 'water_data.csv'
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

df = load_agri_data()

# --- 3. SIDEBAR NAVIGATION & FILTERS ---
st.sidebar.title("💧 Water Analytics")
st.sidebar.info("Analyze the hidden water in global agriculture using AI.")

st.sidebar.subheader("Global Filters")
selected_category = st.sidebar.multiselect("Filter by Category", df['Category'].unique(), default=df['Category'].unique())
filtered_df = df[df['Category'].isin(selected_category)]

# --- 4. MAIN DASHBOARD UI ---
st.title("Interactive Agricultural Water Footprint")

# Metric Row
m1, m2, m3 = st.columns(3)
m1.metric("Selected Crops", len(filtered_df))
m2.metric("Max Blue Water (L/kg)", f"{filtered_df['Blue_Water'].max():,}")
m3.metric("Avg Sustainability", "Moderate")

# Visualization Tabs
tab1, tab2, tab3 = st.tabs(["📊 Visual Analytics", "🔬 AI Data Expert", "📄 Raw Data"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Water Composition (Stacked)")
        fig = px.bar(filtered_df, x='Crop', y=['Blue_Water', 'Green_Water', 'Grey_Water'], 
                     barmode='stack', color_discrete_sequence=['#3498db', '#2ecc71', '#95a5a6'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Total Footprint Comparison")
        fig_pie = px.pie(filtered_df, values='Total_Footprint', names='Crop', hole=0.3)
        st.plotly_chart(fig_pie, use_container_width=True)

with tab2:
    st.subheader("🤖 Ask Gemini about this Data")
    st.write("Use natural language to query the dataset (e.g., 'Which crop has the lowest grey water?')")
    
    user_query = st.text_input("Enter your question:")
    
    if user_query:
        # Initialize Gemini via LangChain
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        
        # Create an agent that can interact with the dataframe
        agent = create_pandas_dataframe_agent(
            llm, 
            df, 
            verbose=True, 
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            allow_dangerous_code=True # Required for data manipulation
        )
        
        with st.spinner("Gemini is thinking..."):
            try:
                response = agent.run(user_query)
                st.success(response)
            except Exception as e:
                st.error(f"Error: {str(e)}")

with tab3:
    st.dataframe(filtered_df, use_container_width=True)
