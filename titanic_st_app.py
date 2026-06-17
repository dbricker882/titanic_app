import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
from PIL import Image

# Page configuration
st.set_page_config(layout="wide")

# Load your pre-trained model
with open('survival_model.pkl', 'rb') as f:
    lm2 = pickle.load(f)


# Load feature importance from an Excel file
def load_feature_importance(file_path):
    return pd.read_excel(file_path)


# Load the feature importance DataFrame
final_fi = load_feature_importance("feature_importance.xlsx")  # Replace with your file path

# Sidebar setup
# image_sidebar = Image.open('Pic 1.png')  # Replace with your image file
# st.sidebar.image(image_sidebar, use_column_width=True)
st.sidebar.header('Passenger Information')


# Feature selection on sidebar
def get_user_input():
    gender = st.sidebar.selectbox('Gender', ["female", "male"])
    age = st.sidebar.number_input('Age', min_value=1, max_value=99, step=1, value=47)
    pclass = st.sidebar.selectbox('Passenger Class', [1, 2, 3])
    fare = st.sidebar.number_input('Fare', min_value=1.00, max_value=99.00, step=0.01, value=7.57)
    sibsp = st.sidebar.number_input('Siblings On Board', min_value=1, max_value=10, step=1, value=1)
    parch = st.sidebar.number_input('Parch', min_value=1, max_value=10, step=1, value=2)
    embark = st.sidebar.selectbox('Embarked From', ['C', 'Q', 'S'])

    user_data = {
        'PClass': pclass,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Age': age,
        f'Sex_{gender}': 1,
        f'Embarked_{embark}': 1
    }
    return user_data


# Top banner
# image_banner = Image.open('Pic 2.png')  # Replace with your image file
# st.image(image_banner, use_column_width=True)

# Centered title
st.markdown("<h1 style='text-align: center;'>Will You Survive The Titanic?</h1>", unsafe_allow_html=True)

# Split layout into two columns
left_col, right_col = st.columns(2)

# Left column: Feature Importance Interactive Bar Chart
with left_col:
    st.header("Feature Importance")

    # Sort feature importance DataFrame by 'Feature Importance Score'
    final_fi_sorted = final_fi.sort_values(by='Feature Importance Score', ascending=True)

    # Create interactive bar chart with Plotly
    fig = px.bar(
        final_fi_sorted,
        x='Feature Importance Score',
        y='Variable',
        orientation='h',
        title="Feature Importance",
        labels={'Feature Importance Score': 'Importance', 'Variable': 'Feature'},
        text='Feature Importance Score',
        color_discrete_sequence=['#48a3b4']  # Custom bar color
    )
    fig.update_layout(
        xaxis_title="Feature Importance Score",
        yaxis_title="Variable",
        template="plotly_white",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

# Right column: Prediction Interface
with right_col:
    st.header("Should You Get On The Ship?")

    # User inputs from sidebar
    user_data = get_user_input()


    # Transform the input into the required format
    def prepare_input(data, feature_list):
        input_data = {feature: data.get(feature, 0) for feature in feature_list}
        return np.array([list(input_data.values())])


    # Feature list (same order as used during model training)
    features = ['PClass', 'SibSp', 'Parch', 'Fare', 'Age', 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q',
                'Embarked_S']

    # Predict button
    if st.button("Predict"):
        input_array = prepare_input(user_data, features)
        prediction = lm2.predict(input_array)
        # st.subheader("Predicted Price")
        if prediction == 0:
            st.write(f"You will not survive, consider not being poor.")
        else:
            st.write("You will survive!")

# streamlit run Regr_model_cars.py