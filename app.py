import streamlit as st
import pandas as pd
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# --- Dashboard Title ---
st.title("🌍 Water Footprint of Agricultural Products")
st.write("Interactive dashboard to explore water usage in crops and animal products.")

# --- Sidebar for Navigation ---
st.sidebar.header("Dashboard Controls")
section = st.sidebar.radio("Choose Section:", 
                           ["Introduction", "Crop Water Footprint", "Animal Products", "Factors", "Solutions", "Charts"])

# --- LangChain Setup ---
llm = OpenAI(temperature=0.3)
prompt_template = PromptTemplate(
    input_variables=["topic"],
    template="Explain the water footprint of {topic} in simple terms."
)

# --- Data for Charts ---
crop_data = {
    "Rice": 2500,
    "Wheat": 1800,
    "Maize": 1200,
    "Sugarcane": 3500,
    "Pulses": 900
}

animal_data = {
    "Milk": 1000,
    "Chicken": 4300,
    "Eggs": 3300,
    "Mutton": 6100,
    "Beef": 15400
}

# --- Dashboard Sections ---
if section == "Introduction":
    st.subheader("Introduction")
    st.write("Agriculture is one of the largest users of freshwater. "
             "Studying water footprint helps us understand and manage water usage efficiently.")

elif section == "Crop Water Footprint":
    st.subheader("Crop Water Footprint")
    crop = st.selectbox("Select a Crop:", list(crop_data.keys()))
    response = llm(prompt_template.format(topic=crop))
    st.write(response)

elif section == "Animal Products":
    st.subheader("Animal Products")
    animal = st.selectbox("Select an Animal Product:", list(animal_data.keys()))
    response = llm(prompt_template.format(topic=animal))
    st.write(response)

elif section == "Factors":
    st.subheader("Factors Affecting Water Footprint")
    st.write("""
    - Climate conditions (rainfall, temperature)  
    - Type of crop or animal product  
    - Irrigation methods  
    - Soil type and quality  
    - Farming practices  
    - Use of fertilizers and pesticides  
    """)

elif section == "Solutions":
    st.subheader("Solutions to Reduce Water Footprint")
    st.write("""
    ✅ Efficient irrigation (drip irrigation)  
    ✅ Rainwater harvesting  
    ✅ Choosing crops with lower water needs  
    ✅ Reducing fertilizers and pesticides  
    ✅ Recycling water in farming  
    ✅ Minimizing food waste  
    """)

elif section == "Charts":
    st.subheader("Water Footprint Comparisons")

    # Crop Chart
    st.write("### Crop Water Footprint (liters/kg)")
    crop_df = pd.DataFrame(list(crop_data.items()), columns=["Crop", "Liters/kg"])
    st.bar_chart(crop_df.set_index("Crop"))

    # Animal Chart
    st.write("### Animal Product Water Footprint (liters/kg)")
    animal_df = pd.DataFrame(list(animal_data.items()), columns=["Animal Product", "Liters/kg"])
    st.bar_chart(animal_df.set_index("Animal Product"))
