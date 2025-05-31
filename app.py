import streamlit as st
import numpy as np
import pickle
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier

# Load model
with open("wine_quality_model.pkl", "rb") as f:
    model = pickle.load(f)

print(type(model))
print(model)

# --- Custom CSS for background and card effects ---
st.markdown(
    """
    <style>
    body {
        background-image: url('https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=1500&q=80');
        background-size: cover;
        background-attachment: fixed;
    }
    .title-card {
        background: rgba(255,255,255,0.85);
        border-radius: 24px;
        box-shadow: 0 6px 32px 0 rgba(0,0,0,0.18);
        padding: 32px 16px 24px 16px;
        margin-bottom: 32px;
        text-align: center;
    }
    .input-card {
        background: rgba(255,255,255,0.92);
        border-radius: 18px;
        box-shadow: 0 4px 24px 0 rgba(0,0,0,0.10);
        padding: 24px 18px 18px 18px;
        margin-bottom: 24px;
    }
    .input-label {
        font-weight: 600;
        color: #7B3F00;
        font-size: 1.1em;
        display: flex;
        align-items: center;
    }
    .input-icon {
        margin-right: 8px;
        font-size: 1.2em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Title Section with Logo and Background ---
st.markdown(
    """
    <div class='title-card'>
        <img src='https://cdn-icons-png.flaticon.com/512/3595/3595455.png' width='80' style='margin-bottom:10px;' />
        <h1 style='font-size:2.8em; margin-bottom:0;'>🍇 Wine Quality Classifier 🍷</h1>
        <p style='font-size:1.2em; color:#7B3F00;'>Predict wine quality and explore wine data in style!</p>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Animated, Colorful Input Section ---
input_icons = [
    "🧪", "🧪", "🍋", "🍬", "🧂", "🧂", "🧂", "💧", "🧫", "🧂", "🍷"
]
input_tooltips = [
    "Amount of fixed acids in g/dm³.",
    "Amount of volatile acids in g/dm³.",
    "Amount of citric acid in g/dm³.",
    "Amount of residual sugar in g/dm³.",
    "Amount of chlorides in g/dm³.",
    "Free SO₂ in mg/dm³.",
    "Total SO₂ in mg/dm³.",
    "Density of the wine (g/cm³).",
    "pH value.",
    "Sulphates in g/dm³.",
    "Alcohol content (% vol)."
]
input_fields = [
    ("Fixed Acidity", 4.0, 16.0, 7.4),
    ("Volatile Acidity", 0.1, 1.5, 0.7),
    ("Citric Acid", 0.0, 1.0, 0.0),
    ("Residual Sugar", 0.5, 15.0, 1.9),
    ("Chlorides", 0.01, 0.2, 0.076),
    ("Free Sulfur Dioxide", 1, 75, 11),
    ("Total Sulfur Dioxide", 6, 300, 34),
    ("Density", 0.990, 1.005, 0.9978),
    ("pH", 2.5, 4.5, 3.51),
    ("Sulphates", 0.3, 2.0, 0.56),
    ("Alcohol", 8.0, 15.0, 9.4)
]

with st.container():
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center; color:#7B3F00;'>🍷 Enter Wine Chemical Properties</h3>", unsafe_allow_html=True)
    input_values = []
    progress = st.progress(0)
    for idx, (label, minv, maxv, default) in enumerate(input_fields):
        slot = st.empty()
        time.sleep(0.06)
        icon = input_icons[idx]
        tooltip = input_tooltips[idx]
        st.markdown(f"<span class='input-label'><span class='input-icon'>{icon}</span>{label} <span style='color:#888;' title='{tooltip}'>ⓘ</span></span>", unsafe_allow_html=True)
        if label in ["Free Sulfur Dioxide", "Total Sulfur Dioxide"]:
            val = slot.number_input("", int(minv), int(maxv), int(default), key=label)
        else:
            val = slot.number_input("", float(minv), float(maxv), float(default), key=label)
        input_values.append(val)
        progress.progress(int((idx+1)/len(input_fields)*100))
    st.markdown("</div>", unsafe_allow_html=True)

(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, 
 free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol) = input_values

# Add a progress bar for a more interactive feel
progress_bar = st.progress(0)
for i in range(100):
    progress_bar.progress(i + 1)

# Predict button
if st.button("Predict Wine Quality"):
    input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid,
                            residual_sugar, chlorides, free_sulfur_dioxide,
                            total_sulfur_dioxide, density, pH, sulphates, alcohol]])
    prediction = model.predict(input_data)[0]

    quality_map = {0: "Low", 1: "Medium", 2: "High"}
    st.success(f"Predicted Quality: **{quality_map[prediction]}**")

# Enhanced creative card-style layout for wine display with st.image for local images and expanders
st.subheader("🍷 Sample Wines")
wine_data = {
    "Red Wine": {
        "image": "RedWine_01.jpeg",
        "description": "A rich and full-bodied red wine with notes of dark fruit and spices.",
        "color": "#8B0000",
        "details": "Pairs well with steak, lamb, and hard cheeses. Region: Bordeaux, France. Fun fact: Red wine gets its color from grape skins!"
    },
    "White Wine": {
        "image": "WhiteWIne-01.webp",
        "description": "A crisp and refreshing white wine with citrus and floral notes.",
        "color": "#F5DEB3",
        "details": "Pairs well with fish, chicken, and soft cheeses. Region: Marlborough, New Zealand. Fun fact: White wine is usually served chilled!"
    },
    "Rosé Wine": {
        "image": "RoseWine_01.jpeg",
        "description": "A light and fruity rosé wine with hints of berries and melon.",
        "color": "#FFC0CB",
        "details": "Pairs well with salads, seafood, and light pasta. Region: Provence, France. Fun fact: Rosé is made by allowing red grape skins to touch wine for only a short time!"
    }
}

cols = st.columns(3)
for idx, (wine_type, details) in enumerate(wine_data.items()):
    with cols[idx]:
        st.markdown(f"<div style='border:3px solid {details['color']}; border-radius:18px; padding:16px; box-shadow:0 4px 16px 0 rgba(0,0,0,0.10); text-align:center;'>", unsafe_allow_html=True)
        st.markdown(f"### {'🍷' if wine_type=='Red Wine' else '🥂' if wine_type=='White Wine' else '🌹'} {wine_type}")
        st.image(details["image"], caption=wine_type, use_container_width=True)
        st.markdown(f"<p style='font-size:1.05em;margin-bottom:0;'>{details['description']}</p>", unsafe_allow_html=True)
        with st.expander("Learn More"):
            st.write(details["details"])
        st.markdown("</div>", unsafe_allow_html=True)

# Add an advanced tabular section with animations
st.subheader("Wine Quality Data")
quality_data = {
    "Low": {"range": "0-3", "avg_rating": 2.5},
    "Medium": {"range": "4-6", "avg_rating": 3.5},
    "High": {"range": "7-10", "avg_rating": 4.5}
}

# Create a DataFrame for the tabular data
df = pd.DataFrame(quality_data).T
df.index.name = "Quality"
df.columns = ["Quality Range", "Average Rating"]

# Display the DataFrame with animations
st.dataframe(df, use_container_width=True)

# Add a progress bar for animation
progress_bar = st.progress(0)
for i in range(100):
    progress_bar.progress(i + 1)

# Save the model
with open("wine_quality_model.pkl", "wb") as f:
    pickle.dump(model, f)
