import streamlit as st
import pandas as pd
import altair as alt
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate

# --- Dashboard Title ---
st.title("🌍 Analytical Dashboard: Water Footprint in Agriculture & Animal Products")
st.write("Explore and compare water usage across crops and animal products with interactive charts and AI explanations.")

# --- LangChain Setup ---
llm = OpenAI(temperature=0.3)

prompt_template = PromptTemplate(
    input_variables=["topic"],
    template="Explain the water footprint of {topic} in simple analytical terms."
)

# --- Sample Data (liters per kg) ---
data = {
    "Crop": ["Rice", "Wheat", "Maize", "Sugarcane", "Pulses"],
    "Water_Footprint": [2500, 1800, 1200, 1500, 4000]
}

animal_data = {
    "Animal Product": ["Milk", "Chicken", "Eggs", "Mutton", "Beef"],
    "Water_Footprint": [1000, 4300, 3300, 6000, 15000]
}

df_crops = pd.DataFrame(data)
df_animals = pd.DataFrame(animal_data)

# --- Sidebar Controls ---
st.sidebar.header("Dashboard Controls")
view = st.sidebar.radio("Choose Analysis:", ["Crop Comparison", "Animal Product Comparison", "Combined Analysis"])

# --- Crop Analysis ---
if view == "Crop Comparison":
    st.subheader("Crop Water Footprint Analysis")
    chart = alt.Chart(df_crops).mark_bar().encode(
        x="Crop",
        y="Water_Footprint",
        color="Crop"
    ).properties(width=600)
    st.altair_chart(chart)
    
    crop = st.selectbox("Select a Crop for AI Explanation:", df_crops["Crop"])
    response = llm(prompt_template.format(topic=crop))
    st.write(response)

# --- Animal Product Analysis ---
elif view == "Animal Product Comparison":
    st.subheader("Animal Product Water Footprint Analysis")
    chart = alt.Chart(df_animals).mark_bar().encode(
        x="Animal Product",
        y="Water_Footprint",
        color="Animal Product"
    ).properties(width=600)
    st.altair_chart(chart)
    
    animal = st.selectbox("Select an Animal Product for AI Explanation:", df_animals["Animal Product"])
    response = llm(prompt_template.format(topic=animal))
    st.write(response)

# --- Combined Analysis ---
elif view == "Combined Analysis":
    st.subheader("Combined Crop vs Animal Product Analysis")
    combined_df = pd.DataFrame({
        "Category": ["Rice", "Wheat", "Maize", "Sugarcane", "Pulses", "Milk", "Chicken", "Eggs", "Mutton", "Beef"],
        "Water_Footprint": [2500, 1800, 1200, 1500, 4000, 1000, 4300, 3300, 6000, 15000]
    })
    
    chart = alt.Chart(combined_df).mark_bar().encode(
        x="Category",
        y="Water_Footprint",
        color="Category"
    ).properties(width=700)
    st.altair_chart(chart)
    
    st.write("This combined view highlights how animal products generally require significantly more water compared to crops.")
