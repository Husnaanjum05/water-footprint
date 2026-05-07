import streamlit as st
import pandas as pd
import os
import plotly.express as px  # <--- MUST BE HERE
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# --- 1. DATA SETUP (Must happen before the Agent) ---
@st.cache_data
def load_data():
    data = {
        'Crop': ['Almonds', 'Wheat', 'Rice'],
        'Blue_Water': [10200, 342, 1025],
        'Green_Water': [1500, 1277, 2248],
        'Grey_Water': [1100, 208, 137]
    }
    df = pd.DataFrame(data)
    df['Total_Footprint'] = df.iloc[:, 1:].sum(axis=1)
    return df

df = load_data()
filtered_df = df # Or your sidebar filter logic

# --- 2. AI AGENT SETUP ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", azure=False) # Ensure API key is in secrets
agent = create_pandas_dataframe_agent(
    llm, 
    df, 
    verbose=True, 
    agent_type="tool-calling", 
    allow_dangerous_code=True
)

# --- 3. VISUALIZATION (Line 82 area) ---
st.subheader("Water Composition by Crop")
# px is now defined from the import above
fig = px.bar(
    filtered_df, 
    x='Crop', 
    y=['Blue_Water', 'Green_Water', 'Grey_Water'],
    title="Liters per Kilogram",
    barmode='stack'
)
st.plotly_chart(fig, use_container_width=True)
