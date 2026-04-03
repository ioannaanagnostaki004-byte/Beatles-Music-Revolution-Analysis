import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, silhouette_score, confusion_matrix

# --- 1. ΠΡΟΕΤΟΙΜΑΣΙΑ ΔΕΔΟΜΕΝΩΝ ---
df_beatles = pd.read_csv('beatles_songs.csv')
df_social = pd.read_csv('social_events.csv')

# ΚΑΘΑΡΙΣΜΟΣ: Μετατροπή όλων των στηλών σε μικρά γράμματα και αφαίρεση κενών
df_beatles.columns = df_beatles.columns.str.strip().str.lower()
df_social.columns = df_social.columns.str.strip().str.lower()

# ΜΕΤΑΤΡΟΠΗ ΤΟΥ YEAR ΣΕ ΑΡΙΘΜΟ (Η λύση στο σφάλμα)
df_beatles['year'] = pd.to_numeric(df_beatles['year'], errors='coerce')
df_social['year'] = pd.to_numeric(df_social['year'], errors='coerce')

# Αφαίρεση τυχόν κενών γραμμών που μπορεί να δημιουργήθηκαν
df_beatles = df_beatles.dropna(subset=['year'])
df_social = df_social.dropna(subset=['year'])

# Μετατροπή σε ακέραιο (int) για σιγουριά
df_beatles['year'] = df_beatles['year'].astype(int)
df_beatles['year'] = df_beatles['year'].astype(int)
df_social['year'] = df_social['year'].astype(int)

# Φόρτωση Spotify Data (Baseline)
use_cols = ['valence', 'year', 'energy', 'loudness', 'artists', 'popularity', 'danceability', 'acousticness']
df_spotify = pd.read_csv('data.csv', usecols=use_cols, encoding='ISO-8859-1')
df_baseline = df_spotify[(df_spotify['year'] >= 1962) & (df_spotify['year'] <= 1970)].copy()
df_baseline = df_baseline[~df_baseline['artists'].str.contains('The Beatles', case=False)]

# Ενοποίηση Beatles με Social Events
df_merged = pd.merge(df_beatles, df_social, on='year')

# Ορισμός χαρακτηριστικών για όλη την ανάλυση
features = ['energy', 'loudness', 'valence', 'danceability', 'acousticness']
X_main = df_merged[features]
y_main = df_merged['label'] 

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_main)

# --- 2. PCA: ΒΕΛΤΙΣΤΗ r-ΔΙΑΣΤΑΤΗ & 2-ΔΙΑΣΤΑΤΗ ΠΡΟΣΕΓΓΙΣΗ ---
pca_full = PCA().fit(X_scaled)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)
r_optimal = np.where(cum_var > 0.85)[0][0] + 1  # Σημείωση: r_optimal είναι η βέλτιστη r-διάστατη προσέγγιση

# 2D PCA Προβολή
pca_2d = PCA(n_components=2)
X_pca = pca_2d.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
# Χρησιμοποιούμε loop για να φτιάξουμε το υπόμνημα ανά έτος χειροκίνητα
for label in df_merged['label'].unique():
    mask = df_merged['label'] == label
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=label, alpha=0.7)
#for year in sorted(df_beatles['Year'].unique()):
#    mask = df_beatles['Year'] == year
#    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=year, alpha=0.7)

plt.title('2D PCA: Η Μουσική Διαδρομή των Beatles (1962-1970)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(title='Έτος', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# 

# --- 3. ΚΑΤΗΓΟΡΙΟΠΟΙΗΣΗ NAIVE BAYES (Κοινωνική Αναταραχή) ---
if 'title' in df_merged.columns:
    title_col = 'title'
elif 'name' in df_merged.columns:
    title_col = 'name'
else:
    # Αν δεν βρει τίποτα, παίρνει την πρώτη στήλη που είναι κείμενο
    title_col = df_merged.select_dtypes(include=['object']).columns[0]

X_train, X_test, y_train, y_test, titles_train, titles_test = train_test_split(
    X_scaled, y_main, df_merged[title_col], test_size=0.20, random_state=42
)
# Εκπαίδευση
nb = GaussianNB()
nb.fit(X_train, y_train)

# Πρόβλεψη
y_pred = nb.predict(X_test)

# Δημιουργία αρχείου TXT 
results_df = pd.DataFrame({
    'Song': titles_test,
    'Actual': y_test,
    'Predicted': y_pred
})
results_df['Match'] = results_df['Actual'] == results_df['Predicted']

with open('naive_bayes_results.txt', 'w', encoding='utf-8') as f:
    f.write(f"Naive Bayes Accuracy: {nb.score(X_test, y_test):.2%}\n")
    f.write("-" * 50 + "\n")
    f.write(results_df.to_string(index=False))


# --- 4. CLUSTERING STABILITY (K-means vs EM) ---
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=50)
plt.title('K-means Clustering των Τραγουδιών (Stability Analysis)')
plt.colorbar(label='Cluster ID')
plt.show()

print(f"Silhouette Score: {silhouette_score(X_scaled, clusters):.3f}")

# --- 5. ΔΙΑΧΡΟΝΙΚΗ ΕΠΙΠΤΩΣΗ (Σύγκριση Popularity) ---
# Σύγκριση μέσων όρων δημοτικότητας
labels = ['Beatles', 'Others (60s)']
means = [df_beatles['popularity'].mean(), df_baseline['popularity'].mean()]

plt.figure(figsize=(8, 5))
plt.bar(labels, means, color=['navy', 'gray'])
plt.ylabel('Μέση Δημοτικότητα στο Spotify')
plt.title('Διαχρονικό Influence: Beatles vs Συγχρόνων τους')
plt.show()