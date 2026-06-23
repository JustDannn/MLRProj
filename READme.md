# LLM Performance Analysis & Multiple Linear Regression (MLR)

Project ini bertujuan untuk menganalisis performa beberapa model Large Language Model (LLM) lokal yang dijalankan melalui Ollama. Analisis dilakukan dengan membandingkan metrik performa (seperti latensi dan jumlah token) saat LLM memproses teks dalam tiga bahasa berbeda: **Inggris, Jawa, dan Sunda**.

Hasil dari eksperimen ini kemudian dianalisis menggunakan _Multiple Linear Regression_ (MLR) untuk melihat variabel apa saja yang paling memengaruhi waktu respons (latensi) dari sebuah model.

## Struktur Direktori

```text
.
├── Data
│   ├── data.csv            # Dataset mentah (teks dalam 3 bahasa)
│   ├── data_withAI.csv     # Dataset hasil eksperimen dengan metrik performa AI
│   └── embedingNews.py     # Script untuk scraping API berita dan translasi
├── Model
│   └── API_hit.py          # Script untuk menjalankan inferensi LLM via Ollama
├── Notebook
│   ├── 00_EDA.ipynb        # Exploratory Data Analysis
│   └── 01_MLR_Analysis.ipynb # Analisis Multiple Linear Regression
└── READme.md
```
