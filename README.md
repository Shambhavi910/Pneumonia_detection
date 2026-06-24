# Pneumonia Detection

A production-oriented Streamlit dashboard for pneumonia screening from chest X-rays. The app uses a pre-trained DenseNet Keras model to provide quick inference, attention-based Grad-CAM explainability, and a clinical-style interface for demonstration purposes.

## Overview
This repository showcases an end-to-end pneumonia detection prototype. It allows a user to upload an X-ray image, receive an automated prediction, and inspect the model’s attention map. For cases predicted as pneumonia, the app displays targeted supportive care guidance.

## Features
- Upload chest X-ray scans and get instant model predictions
- Display confidence scores and clear result labels
- Generate Grad-CAM heatmaps for visual interpretability
- Professional dashboard layout with sample images and care guidance
- Designed for demonstration, clinical concept validation, and rapid prototyping

## Built With
- Python
- Streamlit
- TensorFlow / Keras
- Pillow
- NumPy

## Quick Start
1. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
2. Run the app:
   ```powershell
   streamlit run App/main.py
   ```
3. Open the local URL shown in the terminal.

## Repository Structure
- `App/main.py` — Streamlit application, inference logic, Grad-CAM generation, and UI layout
- `App/best_densenet.keras` — trained pneumonia detection model artifact
- `App/sample_image/` — sample chest X-rays displayed in the dashboard
- `Model/pneumonia-detection.ipynb` — research notebook for model development and analysis

## Usage Guidance
- Upload a valid frontal chest X-ray image
- Review the predicted label and probability
- Inspect the Grad-CAM overlay to understand model focus areas
- For pneumonia predictions, use the highlighted supportive care guidance as a concept-level suggestion

## Disclaimer
This project is provided for demonstration and educational use only. It is not intended for clinical diagnosis or treatment. Always consult a qualified medical professional for healthcare decisions.

## Future Improvements
- Deploy to a hosted Streamlit service or cloud platform
- Add support for additional chest pathology classes
- Include user authentication and secure data handling
- Build a prediction report export feature

## Contact
For questions or enhancements, update this README with your preferred contact details.
