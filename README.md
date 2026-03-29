# Steam reviews sentiment analysis
Dataset: https://www.kaggle.com/datasets/najzeko/steam-reviews-2021

Proyek ini bertujuan untuk membangun model klasifikasi sentimen yang dapat memprediksi label ulasan berdasarkan isi teks-nya, sekaligus mengungkap kata-kata dan pola bahasa yang paling berpengaruh dalam menentukan sentimen komunitas gamer.

### Tujuan spesifik proyek:
•	Membangun pipeline NLP (Natural Languange Processing) end-to-end dari data mentah hingga model siap pakai.
•	Menangani ketidakseimbangan kelas ekstrem (87.5% positif vs 12.5% negatif) dalam data.
•	Melatih dan membandingkan dua model: Logistic Regression dan Naive Bayes.
•	Menganalisis sentimen komunitas pada 5 game ikonik dan membandingkannya dengan rating Steam asli.
•	Menyediakan demo interaktif untuk inferensi real-time.

### Preprocessing & Quality Filtering
Data mentah mengandung berbagai noise yang harus dibersihkan sebelum dapat digunakan oleh model. Pipeline pembersihan diterapkan secara berurutan untuk menghindari kebocoran data:
•	Filter bahasa: hanya ulasan berbahasa Inggris yang dipertahankan.
•	Filter playtime: minimal 300 menit (5 jam) untuk menyingkirkan ulasan di dalam waktu refund.
•	Filter panjang: minimal 5 kata agar model mendapat konteks yang cukup.
•	Filter votes_funny ≤ votes_helpful: menyingkirkan ulasan meme/troll yang tidak mencerminkan opini nyata.
•	Teks: lowercase → hapus HTML/URL/non-ASCII → hapus tanda baca dan angka → hapus stopwords (kecuali negasi) → lemmatisasi.

### Penanganan Class Imbalance
Distribusi kelas asli dataset adalah 87.5% positif dan 12.5% negatif. Melatih model langsung pada distribusi ini akan menghasilkan model yang selalu memprediksi 'Recommended' dan mencapai akurasi tinggi secara artifisial (≈87.5%) tanpa benar-benar belajar. Solusinya adalah undersampling yang seimbang dengan mengambil masing-masing 25.000 sampel dari kedua kelas menggunakan random_state=42 untuk reprodusibilitas.

### Vektorisasi menggunakan TF-IDF
TF-IDF (Term Frequency–Inverse Document Frequency) dipilih sebagai representasi teks karena bekerja efisien dengan teks pendek-menengah, memberikan bobot lebih tinggi pada kata-kata yang spesifik dan diskriminatif, serta kompatibel secara langsung dengan Logistic Regression. Konfigurasi yang digunakan:
•	max_features=15.000: menjaga dimensi vektor tetap manageable.
•	ngram_range=(1,2): mencakup unigram dan bigram — penting untuk menangkap frasa seperti 'not worth' atau 'highly recommend'.
•	sublinear_tf=True: kompresi logaritmik frekuensi term sehingga kemunculan 100x tidak 100x lebih berbobot dari kemunculan 1x.
•	min_df=5, max_df=0.95: membuang kata yang sangat langka (noise) dan sangat umum

### Model
Dua model dilatih dan dibandingkan. Logistic Regression dipilih sebagai model utama karena koefisiennya dapat langsung diinterpretasikan sebagai feature importance yang memungkinkan analisis kata-kata paling berpengaruh per kelas. Naive Bayes digunakan sebagai baseline perbandingan.
