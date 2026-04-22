# Solar Flare Predictor ☀️

A simple machine learning project that predicts **solar flare class** from recent space weather data.

## Objective

The goal of this project is to build an end-to-end pipeline that:
- collects solar activity data,
- prepares features for modeling,
- trains an LSTM model,
- runs inference to classify flare activity.

## Features

- 📥 **Data collection** from NOAA public APIs  
- 🧹 **Feature preprocessing** for model-ready datasets  
- 🧠 **Model training** with an LSTM-based predictor  
- 🔮 **Inference pipeline** to produce solar class predictions  
- 🖥️ **Two ways to run it**:
  - Terminal menu (`src/main.py`)
  - Streamlit web app (`src/app.py`)

## Project Structure

```text
src/
  main.py                  # Terminal menu to run pipeline steps
  app.py                   # Streamlit interface
  features_pipeline_main.py
  training_main.py
  inference_main.py
  features_pipeline/       # Data download + preprocessing
  training_pipeline/       # Dataset split + model training
  inference_pipeline/      # Prediction + classification

datas/                     # Input and generated datasets
ai_model/                  # Trained model and scaler files
docs/                      # Reports and notebook
```

## Quick Start

1. Clone the repository.
2. Install required Python packages (TensorFlow, pandas, scikit-learn, streamlit, etc.).
3. Run one of the options below.

### Option 1: Terminal Pipeline Menu

From the project root:

```bash
python src/main.py
```

### Option 2: Streamlit UI

From the `src` directory:

```bash
cd src
streamlit run app.py
```

## Pipeline Flow

1. **Features** → Downloads and preprocesses solar datasets  
2. **Training** → Trains and saves model artifacts  
3. **Inference** → Loads model and predicts solar flare class  

You can run each step individually or run all steps in sequence.
