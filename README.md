# Wall Surface Smoothness Inspector 🧱

A computer vision pipeline for detecting and quantifying wall surface roughness using YOLO-based detection and feature-driven regression.

## Project Overview

This project provides a two-stage solution for automated wall quality inspection:
1. **Detection**: Localize rough surface regions using YOLOv8
2. **Quantification**: Calculate smoothness scores (0-100) using handcrafted computer vision features


## Key Features

- 🎯 **YOLOv8 Integration**: Accurate rough region localization
- 📊 **Feature-Based Scoring**: Edge density, texture analysis, fractal dimension
- 📈 **Interpretable Metrics**: Physically meaningful features (not black-box)
- 📷 **Visual Reporting**: Annotated images with scores and bounding boxes

## Architecture

```mermaid
graph TD
    A[Input Image] --> B{YOLO Detection}
    B -->|Rough Regions| C[Feature Extraction]
    C --> D[Regression Model]
    D --> E[Smoothness Score]
    B -->|No Detections| F[Perfect Score]
    E --> G[Visualization]
    F --> G
