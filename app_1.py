import plotly.graph_objects as go
from flask import Flask, request, render_template, jsonify, redirect, url_for
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import logging

app = Flask(__name__)


# Logging setup
logging.basicConfig(level=logging.DEBUG)

# Load the model and preprocessing pipeline
logging.debug("Loading model and preprocessing pipeline.")
model = load_model('models/life_expectancy_model.h5')
pipeline = joblib.load('models/preprocessing_pipeline.pkl')
logging.debug("Model and preprocessing pipeline loaded successfully.")

#  feature names (excluding the target variable 'life_expectancy')
columns = [
    'Country', 'Year', 'Gender', 'Unemployment', 'Infant Mortality', 'GDP', 'GNI',
    'Clean fuels and cooking technologies', 'Per Capita', 'Mortality caused by road traffic injury',
    'Tuberculosis Incidence', 'DPT Immunization', 'HepB3 Immunization', 'Measles Immunization',
    'Hospital beds', 'Basic sanitation services', 'Tuberculosis treatment',
    'Urban population', 'Rural population', 'Non-communicable Mortality', 'Sucide Rate'
]

# default values for the form
default_values = {
    "Country": "Afghanistan",
    "Year": 2008,
    "Gender": "Female",
    "Unemployment": 14.04699993,
    "Infant Mortality": 75.3,
    "GDP": 5226778809,
    "GNI": 18240442245,
    "Clean fuels and cooking technologies": 10.4,
    "Per Capita": 211.3820742,
    "Mortality caused by road traffic injury": 15.2,
    "Tuberculosis Incidence": 189,
    "DPT Immunization": 50,
    "HepB3 Immunization": 65.07692308,
    "Measles Immunization": 48,
    "Hospital beds": 0.39,
    "Basic sanitation services": 26.06769471,
    "Tuberculosis treatment": 89,
    "Urban population": 22.5,
    "Rural population": 77.5,
    "Non-communicable Mortality": 40.2,
    "Sucide Rate": 4.6
}

# Tooltip descriptions for each feature
tool_tips = {
    "Country": "The country of the individual.",
    "Year": "The year of the data.",
    "Gender": "The gender of the individual (Male/Female).",
    "Unemployment": "The unemployment rate percentage.",
    "Infant Mortality": "Number of infant deaths per 1,000 live births.",
    "GDP": "Gross Domestic Product in USD.",
    "GNI": "Gross National Income in USD.",
    "Clean fuels and cooking technologies": "Percentage of population with access to clean cooking fuels.",
    "Per Capita": "Income per capita in USD.",
    "Mortality caused by road traffic injury": "Rate of deaths from road traffic injuries per 100,000 people.",
    "Tuberculosis Incidence": "Number of tuberculosis cases per 100,000 people.",
    "DPT Immunization": "Percentage of children immunized for DPT.",
    "HepB3 Immunization": "Percentage of children immunized for Hepatitis B.",
    "Measles Immunization": "Percentage of children immunized for measles.",
    "Hospital beds": "Number of hospital beds per 1,000 people.",
    "Basic sanitation services": "Percentage of population with access to basic sanitation.",
    "Tuberculosis treatment": "Success rate of tuberculosis treatment as a percentage.",
    "Urban population": "Percentage of population living in urban areas.",
    "Rural population": "Percentage of population living in rural areas.",
    "Non-communicable Mortality": "Mortality rate from non-communicable diseases per 100,000 people.",
    "Sucide Rate": "Suicide rate per 100,000 people."
}

# demographics data for research
@app.route('/demographics')
def demographics():
    try:
        # Load the dataset
        df = pd.read_csv('Data/final.csv')

        # Group by 'Country' and calculate mean for GDP, Infant Mortality, Unemployment Rate, and Urban Population
        df_mean = df.groupby('Country')[['GDP', 'Infant Mortality', 'Unemployment', 'Urban population']].mean().reset_index()

        # Round the values for better display
        df_mean = df_mean.round({'GDP': 2, 'Infant Mortality': 2, 'Unemployment': 2, 'Urban population': 2})

        # Convert to a list of dictionaries for rendering
        demographics_data = df_mean.to_dict(orient='records')

        logging.debug(f"Calculated mean demographics data: {demographics_data}")

        return render_template('demographics.html', demographics_data=demographics_data)

    except Exception as e:
        logging.error(f"Error loading demographics data: {e}")
        return jsonify({"error": f"Error loading demographics data: {e}"}), 500


# Colors for feature categories
colors = {
    'Health Indicators': 'rgba(255, 99, 132, 0.7)',
    'Economic Factors': 'rgba(54, 162, 235, 0.7)',
    'Infrastructure': 'rgba(75, 192, 192, 0.7)'
}

@app.route('/')
def home():
    return render_template('index.html', columns=columns, default_values=default_values,tool_tips=tool_tips)

def create_plot():
    # Example feature importance and input values (replace with actual model results)
    feature_importance = {
        'Infant Mortality': 20,
        'GDP': 15,
        'Unemployment': 30,
        'GNI': 12,
        'Tuberculosis Incidence': 10,
        'Basic sanitation services': 8,
        'Non-communicable Mortality': 9,
        'Clean fuels and cooking technologies': 7,
        'Per Capita': 14,
        'Mortality caused by road traffic injury': 6,
        'DPT Immunization': 13,
        'HepB3 Immunization': 11,
        'Measles Immunization': 9,
        'Hospital beds': 16,
        'Urban population': 10,
        'Rural population': 8,
        'Sucide Rate': 5
    }

    feature_categories = {
        'Health Indicators': ['Infant Mortality', 'Tuberculosis Incidence', 'DPT Immunization', 'HepB3 Immunization', 'Measles Immunization', 'Non-communicable Mortality', 'Sucide Rate'],
        'Economic Factors': ['GDP', 'GNI', 'Unemployment', 'Per Capita'],
        'Infrastructure': ['Clean fuels and cooking technologies', 'Hospital beds', 'Basic sanitation services', 'Mortality caused by road traffic injury']
    }

    feature_values = default_values

    fig = go.Figure()

    for category, features in feature_categories.items():
        importance_values = [feature_importance.get(feature, 0) for feature in features]
        input_values = [feature_values.get(feature, 0) for feature in features]

        # Bar for feature importance
        fig.add_trace(go.Bar(
            x=features,
            y=importance_values,
            name=f'{category} - Importance',
            marker=dict(color=colors[category]),
        ))

        # Scatter for input values
        fig.add_trace(go.Scatter(
            x=features,
            y=input_values,
            mode='markers',
            name=f'{category} - Input Values',
            marker=dict(size=10, color='black', symbol='circle')
        ))

    fig.update_layout(
        title="Life Expectancy: Feature Importance and Input Values",
        xaxis_title="Features",
        yaxis_title="Values / Importance",
        template="plotly_white",
        barmode='group',
        showlegend=True,
        xaxis=dict(tickangle=-45)
    )

    return fig.to_html(full_html=False)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logging.debug("Predict endpoint called.")

        # Parse form data
        form_data = request.form
        logging.debug(f"Form data received: {form_data}")

        # Extract values for each column
        data = [form_data.get(column, default_values[column]) for column in columns]

        # Convert to DataFrame
        df = pd.DataFrame([data], columns=columns)
        logging.debug(f"Input data converted to DataFrame: {df}")

        # Preprocess the data
        preprocessed_data = pipeline.transform(df)
        logging.debug(f"Preprocessed data: {preprocessed_data}")

        # Make prediction
        prediction = model.predict(preprocessed_data)
        logging.debug(f"Prediction: {prediction}")

        # Generate plot
        plot_html = create_plot()

        return render_template('result.html', prediction=round(prediction[0][0], 2), plot_html=plot_html)
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': f"Error during prediction: {e}"}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5001)
