Baby Cry Detection using OpenL3 Embeddings

This project performs audio classification to detect baby cries using OpenL3 audio embeddings and traditional machine learning models like Random Forest and SVM.
It extracts 512-dimensional OpenL3 embeddings from .wav audio files and classifies them based on the type of sound (e.g., crying, laughing, neutral, etc.).

📘 Table of Contents

Overview

Features

Project Structure

Installation

How It Works

Model Training

Results

Saving & Loading Models

Future Improvements

License

🚀 Overview

This project aims to automatically classify baby audio clips (e.g., cries, babbles, laughs) using deep audio embeddings extracted from OpenL3, which is a deep learning model trained on audio-video correspondence tasks.

It then applies classical ML algorithms such as:

Random Forest

Support Vector Machine (SVM)

to classify the embeddings into respective sound classes.

✨ Features

✅ Uses OpenL3 to extract 512-dimensional embeddings
✅ Includes dimensionality reduction using PCA
✅ Supports Random Forest and SVM classifiers
✅ Saves trained models and preprocessing objects
✅ Modular, easy-to-extend pipeline
