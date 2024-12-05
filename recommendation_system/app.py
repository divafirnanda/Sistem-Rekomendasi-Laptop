from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Memuat dataset
laptop_data = pd.read_csv("laptop_prices.csv")

# Mengonversi kolom tertentu ke tipe data string
columns_to_string = ["CPU_model", "Ram", "PrimaryStorageType", "GPU_company", "Company", "Inches"]
for col in columns_to_string:
    if col in laptop_data.columns:
        laptop_data[col] = laptop_data[col].astype(str)

# Fungsi untuk merekomendasikan laptop menggunakan metode hybrid
def recommend_laptops(requirements):
    filtered = laptop_data.copy()

    # Filter berdasarkan input pengguna
    for key, value in requirements.items():
        if value and value != "Any":
            if key == "Processor" and "CPU_company" in filtered.columns:
                filtered = filtered[filtered["CPU_company"].str.contains(value, case=False, na=False)]
            elif key == "RAM" and "Ram" in filtered.columns:
                filtered = filtered[filtered["Ram"].str.contains(value, case=False, na=False)]
            elif key == "Storage" and "PrimaryStorageType" in filtered.columns:
                filtered = filtered[filtered["PrimaryStorageType"].str.contains(value, case=False, na=False)]
            elif key == "GPU" and "GPU_company" in filtered.columns:
                filtered = filtered[filtered["GPU_company"].str.contains(value, case=False, na=False)]
            elif key == "Screen Size" and "Inches" in filtered.columns:
                filtered = filtered[filtered["Inches"].str.contains(value, case=False, na=False)]
            elif key == "Brand" and "Company" in filtered.columns:
                filtered = filtered[filtered["Company"].str.contains(value, case=False, na=False)]
    
    # Jika tidak ada filter yang diterapkan atau semua filter tidak ada yang cocok, mengembalikan 5 laptop teratas default
    if filtered.empty:
        return laptop_data.head(5).to_dict(orient="records")

    # Filtering berbasis konten (Content-based filtering)
    # Menggunakan TF-IDF untuk mengukur kesamaan antar laptop berdasarkan deskripsi fitur-fiturnya
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(filtered.apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1))
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Membuat vektor preferensi pengguna
    # Vektor preferensi pengguna ini digunakan untuk collaborative filtering
    user_preferences = {key: value for key, value in requirements.items() if value and value != "Any"}
    user_vector = np.zeros(tfidf_matrix.shape[1])
    for key, value in user_preferences.items():
        if key in filtered.columns:
            indices = filtered.columns.get_loc(key)
            user_vector[indices] = 1

    # Menghitung skor hybrid
    # Skor hybrid dihitung sebagai kombinasi dari skor kesamaan berbasis konten (cosine_sim.mean(axis=0)) dan preferensi pengguna (user_vector)
    # Kombinasi ini memberikan bobot yang sama antara dua metode, menghasilkan rekomendasi yang lebih baik karena mempertimbangkan kesamaan fitur dan preferensi spesifik pengguna
    hybrid_score = 0.5 * cosine_sim.mean(axis=0) + 0.5 * user_vector[:cosine_sim.shape[0]]

    # Mendapatkan 5 rekomendasi laptop teratas berdasarkan skor hybrid
    top_indices = hybrid_score.argsort()[-5:][::-1]
    recommendations = filtered.iloc[top_indices]

    return recommendations.to_dict(orient="records")

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/rekomendasi", methods=["GET", "POST"])
def rekomendasi():
    recommendations = []
    if request.method == "POST":
        # Mengambil data dari form
        requirements = {
            "Processor": request.form.get("processor"),
            "RAM": request.form.get("ram"),
            "Storage": request.form.get("storage"),
            "GPU": request.form.get("gpu"),
            "Screen Size": request.form.get("screen_size"),
            "Brand": request.form.get("brand"),
        }
        # Mendapatkan rekomendasi
        recommendations = recommend_laptops(requirements)

    return render_template("index2.html", recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
