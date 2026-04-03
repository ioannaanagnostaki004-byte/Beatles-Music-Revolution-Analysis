# Beatles-Music-Revolution-Analysis

This project explores the musical evolution of The Beatles between 1962 and 1970, correlating their audio features with the socio-political climate of the 60s.

## Data Sources
* **The Beatles Songs Dataset:** [Kaggle - Beatles Songs Dataset](https://www.kaggle.com/datasets/devedzic/the-beatles-songs-dataset)
* **Spotify Dataset 1921-2020, 160k+ Tracks:** [Kaggle - Spotify Dataset](https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-1921-2020-160k-tracks)
* **Social Events Dataset:** Custom curated dataset mapping significant 1960s socio-political events.

## Project Overview
Using data mining and machine learning, this analysis investigates how historical events (like the Cold War or social unrest) mirrored the band's sonic shifts (Energy, Loudness, Acousticness).

### Key Features:
* **Data Preparation:** Merging musical metadata with historical event datasets.
* **PCA (Principal Component Analysis):** Reducing dimensionality to visualize musical clusters.
* **Naive Bayes Classification:** Predicting historical periods based on audio features (Accuracy: 23.46% - reflective of complex lyrical/musical overlaps).
* **Clustering (K-Means & EM):** Identifying distinct musical phases in the band's career.

## Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn.
* **Tools:** Jupyter/Quarto for reporting.

## Highlights from the Report
* **Musical Mirror:** Analysis shows that songs like "Across the Universe" (1968) statistically correlate with high-intensity social events.
* **Legacy vs. Baseline:** The study compares the Beatles' enduring popularity (Score: 50) against the 1960s artist baseline (Score: 28).

---
*Developed as part of the Big Data Mining course at the Department of Computer Science and Biomedical Informatics.*
