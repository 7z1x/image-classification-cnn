# 🐾 Klasifikasi Gambar Hewan — CNN dengan TensorFlow

Membangun CNN dari nol untuk memisahkan wajah **cat**, **dog**, dan **wild** pada dataset AFHQ, lalu mengekspornya ke tiga format siap-deploy: SavedModel, TensorFlow.js, dan TFLite.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Sequential%20CNN-D00000?logo=keras&logoColor=white)](https://keras.io/)
[![Dataset](https://img.shields.io/badge/Dataset-AFHQ%20Animal%20Faces-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/andrewmvd/animal-faces)

**📦 Dataset:** [Kaggle — Animal Faces (AFHQ)](https://www.kaggle.com/datasets/andrewmvd/animal-faces)

---

## 📌 Ringkasan Eksekutif

| Pertanyaan | Jawaban |
|---|---|
| Apa masalahnya? | Klasifikasi 3 kelas gambar hewan (**cat / dog / wild**) dari 16.130 gambar 224×224 |
| Model apa yang dipakai? | **CNN Sequential** 4 blok Conv+BN+MaxPool → Flatten → Dense(512) → Softmax |
| Hasilnya? | **Akurasi test 99,01%** (2.396 dari 2.420 benar, hanya **24 salah**) |
| Seberapa besar modelnya? | **26,1 juta parameter** (~99 MB float32) — 98,5% di antaranya di layer `Flatten→Dense` |
| Siap deploy? | **Secara format ya** — tiga format berhasil diekspor. Secara praktik, lihat catatan di bawah |

> **Insight utama:** akurasi 99% di sini **bukan** prestasi rekayasa model, melainkan sifat dari datasetnya. AFHQ adalah data kualitas riset dengan wajah hewan yang terpusat, pencahayaan seragam, dan latar bersih — kelasnya sangat mudah dipisahkan. Yang benar-benar menentukan hasil adalah dua hal lain: **augmentasi agresif** (rotasi 40°, shift, shear, zoom, flip, brightness) yang mencegah overfitting pada 11 ribu gambar latih, dan **callback `ReduceLROnPlateau`** yang menurunkan learning rate dari 1e-3 hingga 3,2e-7. Tanpa keduanya, CNN sebesar ini akan menghafal data latih dalam beberapa epoch.

---

## 🎯 Business Understanding

### Latar Belakang

Klasifikasi gambar hewan adalah fondasi bagi banyak sistem nyata: moderasi konten (menyaring gambar hewan dari feed yang salah kategori), pencarian visual, penandaan otomatis pada galeri foto, dan pengelolaan katalog untuk platform e-commerce produk hewan.

AFHQ (Animal Faces HQ) adalah dataset benchmark yang dirancang untuk tugas generasi wajah hewan, tetapi strukturnya — tiga kelas, resolusi tinggi, latar bersih — membuatnya juga cocok sebagai **batu uji pipeline klasifikasi gambar end-to-end**. Proyek ini membangun pipeline lengkap tersebut: dari data mentah hingga model yang siap dipanggil dari browser.

### Problem Statement

1. Bagaimana membangun model CNN yang mampu mengklasifikasikan gambar hewan ke dalam tiga kelas (cat, dog, wild) dengan akurasi tinggi?
2. Teknik apa yang diperlukan agar CNN tidak overfitting pada dataset berukuran menengah?
3. Bagaimana mengekspor model hasil latihan ke format yang dapat dipakai di web, mobile, dan server?

### Goals

- Membangun CNN Sequential dengan Conv2D + BatchNormalization + MaxPooling.
- Menerapkan augmentasi data untuk meningkatkan generalisasi.
- Mengekspor model ke **SavedModel**, **TensorFlow.js**, dan **TFLite** beserta `label.txt`.

### Metrik Keberhasilan

- **Metrik bisnis:** akurasi klasifikasi yang cukup tinggi untuk dipakai sebagai filter otomatis tahap pertama.
- **Metrik model:** **akurasi test** pada data yang tidak pernah dilihat model, dilengkapi **confusion matrix** dan **precision/recall per kelas** — karena akurasi agregat saja bisa menyembunyikan kelas yang lemah.

### Cakupan

**Termasuk:** penggabungan dataset, EDA visual, split 70/15/15, augmentasi, CNN kustom, callback training, evaluasi, ekspor 3 format.
**Tidak termasuk:** transfer learning (VGG/ResNet/EfficientNet), hyperparameter tuning sistematis, cross-validation, validasi silang antar-dataset, deployment ke layanan produksi, kuantisasi model.

---

## 🗂️ Dataset

| Atribut | Keterangan |
|---|---|
| **Sumber** | [Kaggle — Animal Faces](https://www.kaggle.com/datasets/andrewmvd/animal-faces) (AFHQ, Clova AI Research) |
| **Struktur asal** | `afhq/train/` dan `afhq/val/`, masing-masing berisi `cat`, `dog`, `wild` |
| **Total setelah digabung** | **16.130** gambar |
| **Kelas** | 3 — `cat`, `dog`, `wild` |
| **Resolusi input** | 224 × 224 × 3 |
| **Split akhir** | 70% train / 15% val / 15% test |

**Distribusi split (dari output notebook):**

| Split | Jumlah gambar | Proporsi |
|---|---:|---:|
| Train | **11.291** | 70,0% |
| Validation | **2.419** | 15,0% |
| Test | **2.420** | 15,0% |

**Distribusi kelas pada test set:**

| Kelas | Jumlah | Proporsi |
|---|---:|---:|
| cat | 848 | **35,0%** |
| dog | 786 | 32,5% |
| wild | 786 | 32,5% |

Distribusinya hampir seimbang, sehingga akurasi bisa dipakai sebagai metrik utama tanpa jebakan kelas mayoritas — berbeda dengan kasus churn atau dropout. Baseline naif (selalu jawab "cat") hanya mencapai 35,04%, jadi akurasi 99% memang bermakna.

**Kelompok gambar:**

- **cat** — wajah kucing domestik, berbagai ras dan warna
- **dog** — wajah anjing, berbagai ras
- **wild** — wajah hewan liar (singa, harimau, serigala, rubah, dan sejenisnya)

---

## 🔬 Metodologi

```
Data Loading → Gabung Train+Val → EDA → Split 70:15:15 → Augmentasi
→ CNN (4 blok Conv+BN+Pool) → Training + Callback → Evaluasi → Ekspor 3 Format
```

1. **Data Loading** — unduh `andrewmvd/animal-faces` via Kaggle API, ekstrak ke `afhq/`.
2. **Penggabungan** — salin `afhq/train/*` dan `afhq/val/*` ke satu folder `animal/dataset/` agar bisa di-*split* ulang secara stratifikasi.
3. **EDA** — tampilkan 5 sampel acak per kelas, hitung distribusi kelas dengan `os.walk`.
4. **Split** — dua tahap `train_test_split` dengan `stratify` dan `random_state=123`: 70% train, lalu 30% sisanya dibagi rata menjadi 15% val dan 15% test.
5. **Augmentasi** — hanya pada train set:
   - `rotation_range=40`, `width_shift_range=0.2`, `height_shift_range=0.2`
   - `shear_range=0.2`, `zoom_range=0.2`, `horizontal_flip=True`
   - `brightness_range=[0.8, 1.2]`, `fill_mode='nearest'`
   - `rescale=1./255` pada train, val, dan test
6. **Arsitektur CNN:**

   | Blok | Layer | Output shape |
   |---|---|---|
   | 1 | Conv2D(32, 3×3) + BN + MaxPool | 112×112×32 |
   | 2 | Conv2D(64, 3×3) + BN + MaxPool | 56×56×64 |
   | 3 | Conv2D(128, 3×3) + BN + MaxPool | 28×28×128 |
   | 4 | Conv2D(256, 3×3) + BN + MaxPool | 14×14×256 |
   | — | Flatten | 50.176 |
   | 5 | Dense(512) + Dropout(0.5) | 512 |
   | 6 | Dense(3, softmax) | 3 |

   **Total: 26.081.539 parameter** (~99 MB dalam float32).

7. **Kompilasi** — Adam (`learning_rate=0.001`), `categorical_crossentropy`, metrik `accuracy`.
8. **Training** — maksimal 50 epoch dengan tiga callback:
   - `EarlyStopping(patience=5, restore_best_weights=True)`
   - `ReduceLROnPlateau(factor=0.2, patience=3)`
   - `ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_accuracy')`
9. **Evaluasi** — akurasi test, confusion matrix, classification report.
10. **Ekspor** — `model.export()` untuk SavedModel, `tfjs.converters.save_keras_model()` untuk TFJS, `TFLiteConverter` untuk TFLite, plus `label.txt`.

**Kenapa `padding='same'` di semua Conv2D?** Tanpa padding, setiap konvolusi 3×3 memangkas 2 piksel di tiap dimensi. Dengan empat blok, resolusi akan menyusut lebih cepat dan informasi di tepi gambar hilang. `padding='same'` menjaga dimensi sampai MaxPool yang memang bertugas menurunkannya.

**Kenapa BatchNormalization setelah setiap Conv?** BN menstabilkan distribusi aktivasi antar layer, sehingga learning rate 1e-3 bisa dipakai tanpa divergensi. Pada run pertama, loss epoch 1 sempat mencapai 10,46 — tanda bobot awal belum stabil — dan BN berperan menahannya agar tidak meledak.

**Kenapa Dropout(0.5) hanya sebelum layer output?** Ini satu-satunya titik di mana model punya kapasitas menghafal berlebih: `Flatten→Dense(512)` menyumbang **98,5% dari seluruh parameter**. Dropout di situ adalah pertahanan paling efisien.

---

## 📊 Temuan Utama (EDA)

### 1. Kelas hampir seimbang, baseline rendah

Dengan 35,0% / 32,5% / 32,5%, tidak ada kelas mayoritas yang mendominasi. Baseline "selalu jawab cat" hanya 35,04% — jauh di bawah 99%. Ini membuat akurasi tinggi di sini **benar-benar berarti**, berbeda dengan dataset timpang seperti ulasan aplikasi atau churn pelanggan.

### 2. Karakteristik visual AFHQ sangat bersahabat untuk CNN

Gambar AFHQ memiliki wajah hewan yang terpusat, pencahayaan relatif seragam, dan latar yang bersih. Tidak ada occlusions berat, tidak ada skala ekstrem, tidak ada variasi sudut yang liar. Inilah alasan utama akurasi bisa mencapai 99% — **bukan karena arsitekturnya canggih**, melainkan karena tugasnya memang mudah pada data ini. Lihat [Keterbatasan](#-keterbatasan--asumsi) poin 2.

### 3. Kurva training memperlihatkan peran callback secara langsung

Learning rate turun tiga kali sepanjang training:

| Epoch | Learning rate | Pemicu |
|---|---|---|
| 1–9 | 1,0e-3 | awal |
| 10–14 | 2,0e-4 | `ReduceLROnPlateau` ×1 |
| 15–23 | 4,0e-5 | ×2 |
| 24–31 | 8,0e-6 | ×3 |
| 32–44 | 1,6e-6 → 3,2e-7 | ×4, ×5 |

Setiap penurunan learning rate diikuti lonjakan akurasi validasi. Contoh paling jelas: epoch 9 val_accuracy 0,7871 → epoch 10 (LR turun ke 2e-4) melonjak ke **0,9756**. Pola ini menunjukkan model sudah berada di lembah yang benar tetapi masih "melompat-lompat" karena langkah terlalu besar.

### 4. Validasi awal sangat fluktuatif

| Epoch | val_accuracy |
|---|---:|
| 1 | 0,5511 |
| 2 | 0,8818 |
| 3 | **0,6858** |
| 4 | 0,9091 |
| 7 | 0,7921 |
| 8 | **0,7230** |
| 9 | 0,7871 |
| 10 | **0,9756** |

Lonjakan naik-turun sebesar 20 poin antar epoch bukan tanda model buruk, melainkan konsekuensi dari learning rate yang terlalu besar untuk tahap akhir. Inilah tepatnya masalah yang diselesaikan `ReduceLROnPlateau` — dan mengapa callback ini bukan sekadar pelengkap.

---

## 🤖 Hasil Model

### Performa akhir

| Metrik | Nilai |
|---|---:|
| **Akurasi test** | **99,01%** (2.396 / 2.420) |
| **Test loss** | 0,0379 |
| Total salah klasifikasi | **24 gambar** |
| Baseline naif (selalu "cat") | 35,04% |
| Selisih di atas baseline | **+63,97 poin** |

**Interval kepercayaan 95% (Wilson)** untuk akurasi 2.396/2.420: **0,9853 – 0,9933**. Dengan 2.420 sampel uji, estimasi ini cukup presisi — berbeda dengan kasus test set kecil yang rentangnya lebar.

### Classification report per kelas

| Kelas | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| cat | 0,99 | **1,00** | 1,00 | 848 |
| dog | 0,98 | 0,99 | 0,99 | 786 |
| wild | 0,99 | **0,98** | 0,99 | 786 |
| **Macro avg** | **0,99** | **0,99** | **0,99** | 2.420 |

**Cara membaca:** kelas **cat** sempurna (recall 1,00 — tidak ada satu pun kucing yang terlewat), sementara **wild** paling sering salah (recall 0,98, sekitar 16 dari 786). Ini masuk akal: kelas `wild` adalah kumpulan banyak spesies berbeda (singa, harimau, serigala, rubah), sehingga variasi intra-kelasnya paling besar. Precision 0,99 pada semua kelas berarti kesalahan tidak menumpuk ke satu arah — model tidak "melempar" tebakan ke kelas tertentu.

### Reproducibility: dua run menghasilkan angka berbeda

Repo ini memuat tiga notebook dengan isi kode identik tetapi status eksekusi berbeda:

| Notebook | Platform | Test accuracy | Epoch berhenti |
|---|---|---:|---:|
| `project-klasifikasi-gambar-a548ybm523.ipynb` | Kaggle | **99,01%** | 44 |
| `Project_klasifikasi_gambar(A548YBM523).ipynb` | Colab | **98,60%** | 25 |
| `Project_klasifikasi_gambar(A548YBM523) (1).ipynb` | — | *tanpa output* | — |

Selisih **0,41 poin** (10 gambar) antar run. Penyebabnya: `train_test_split` sudah deterministik (`random_state=123`), tetapi **inisialisasi bobot Keras tidak di-seed**, sehingga training berjalan berbeda setiap kali. Selain itu, `EarlyStopping` berhenti di epoch yang berbeda (44 vs 25) karena memantau `val_loss` yang bergerak tidak monoton.

Ini bukan masalah serius untuk dataset sebesar ini — kedua run sama-sama di atas 98,5% — tetapi penting dicatat: **angka 99,01% bukan angka yang akan selalu Anda dapatkan jika menjalankan ulang notebook.**

### Model yang diekspor

| Format | Fungsi | Perkiraan ukuran |
|---|---|---:|
| **SavedModel** | `model.export()` — untuk serving TF/Python | ~99 MB |
| **TensorFlow.js** | `tfjs.converters.save_keras_model()` — untuk browser | ~99 MB |
| **TFLite** | `TFLiteConverter` — untuk mobile/edge | ~99 MB |
| `label.txt` | daftar kelas (`cat`, `dog`, `wild`) | < 1 KB |

> **Catatan penting:** TFLite di sini **belum dikuantisasi**. Ukurannya masih ~99 MB, sama dengan model float32 penuh — belum ada manfaat ukuran untuk perangkat mobile. Kuantisasi int8 akan menyusutkan model ke ~25 MB (sekitar 4× lebih kecil) dengan penurunan akurasi yang biasanya kecil. Lihat [Keterbatasan](#-keterbatasan--asumsi) poin 5.

---

## ⚠️ Keterbatasan & Asumsi

### 1. Dataset asli digabung, sehingga tidak sebanding dengan benchmark AFHQ

Cell 6 menyalin **keduanya** — `afhq/train` dan `afhq/val` — ke satu folder, lalu melakukan split ulang 70/15/15. Secara teknis tidak ada kebocoran (gambar train dan val AFHQ memang berbeda), tetapi ada dua konsekuensi:

- **Hasil tidak bisa dibandingkan** dengan paper atau model lain yang memakai split resmi AFHQ. Angka 99% ini hanya berlaku untuk split buatan sendiri.
- **Validation set resmi AFHQ (1.500 gambar) kehilangan fungsinya** sebagai tolok ukur independen. Model tidak pernah diuji pada split yang disepakati komunitas.

Total gambar yang ditemukan notebook adalah 16.130, bukan 16.500 seperti AFHQ asli — selisih 370 gambar, kemungkinan karena berkas yang gagal diekstrak atau varian dataset Kaggle yang berbeda.

### 2. Akurasi 99% mencerminkan kemudahan data, bukan kualitas arsitektur

Ini poin paling penting untuk tidak salah membaca hasil. AFHQ punya wajah terpusat, latar bersih, dan pencahayaan seragam. Model sederhana pun akan mencapai akurasi sangat tinggi. Yang **tidak** dibuktikan oleh angka ini:

- Model belum diuji pada foto hewan dari dunia nyata (latar ramai, sudut miring, sebagian tertutup, resolusi rendah).
- Tidak ada bukti bahwa arsitektur ini lebih baik dari model yang jauh lebih kecil — tidak ada perbandingan sama sekali.

Sebagai gambaran: sebuah CNN 26 juta parameter dengan 98,5% parameternya di satu layer Dense adalah desain yang **boros**. Arsitektur modern akan memakai GlobalAveragePooling (mengurangi parameter Dense dari 25,7 juta menjadi ~131 ribu) dengan akurasi yang kemungkinan setara.

### 3. Tidak ada hyperparameter tuning

Semua hyperparameter dipilih sekali: learning rate 1e-3, 4 blok konvolusi, filter 32→64→128→256, Dense 512, Dropout 0,5, batch 32. Tidak ada pencarian sistematis. Arsitektur ini adalah titik awal yang berhasil, bukan hasil optimasi.

### 4. Tidak ada cross-validation

Hanya satu split (`random_state=123`). Tidak ada estimasi variabilitas performa antar split. Ditambah inisialisasi bobot yang tidak di-seed, angka 99,01% sebaiknya dibaca sebagai **"sekitar 98,5–99%"**, bukan angka eksak.

### 5. TFLite belum dikuantisasi

`TFLiteConverter.from_keras_model()` tanpa `optimizations` menghasilkan model float32 penuh (~99 MB). Untuk tujuan deployment mobile, ini belum memenuhi janji "TFLite" — yang seharusnya lebih kecil dan lebih cepat. Diperlukan `converter.optimizations = [tf.lite.Optimize.DEFAULT]` plus representatif dataset untuk kuantisasi penuh.

### 6. Augmentasi bisa merusak gambar wajah hewan

`rotation_range=40` dan `horizontal_flip=True` diterapkan tanpa pengecualian. Untuk klasifikasi wajah, rotasi 40° cukup ekstrem dan bisa menghasilkan gambar yang tidak lagi realistis. `horizontal_flip` umumnya aman untuk hewan, tetapi perlu diingat bahwa pencerminan mengubah sisi wajah — pada tugas yang lebih halus (misalnya identifikasi individu), augmentasi ini bisa merugikan. Di sini hasilnya tetap baik, tetapi kombinasinya dipilih tanpa eksperimen ablasi.

### 7. `ModelCheckpoint` menyimpan model terbaik, tetapi yang dievaluasi bukan model terbaik

`ModelCheckpoint` menyimpan berdasarkan `monitor='val_accuracy'`, sementara `EarlyStopping` memantau `val_loss` (default). Kedua metrik ini tidak selalu bergerak searah. Selain itu, evaluasi di cell 16 dijalankan pada `model` di memori — yang setelah `restore_best_weights=True` memang berisi bobot terbaik menurut `val_loss`. Model yang tersimpan di `best_model.keras` bisa jadi **berbeda** dari model yang dievaluasi, karena dipilih dengan kriteria lain.

### 8. Kelas `wild` adalah label payung yang kabur

"wild" mencakup banyak spesies berbeda yang secara visual lebih beragam daripada "cat" atau "dog". Model tidak memisahkan singa dari serigala — dan memang tidak diminta. Untuk penggunaan nyata, kategori sekasar ini jarang cukup; klasifikasi spesies akan memerlukan dataset dan pendekatan berbeda.

### 9. Tidak ada analisis kesalahan kualitatif

Confusion matrix menunjukkan **24 kesalahan**, tetapi tidak ada satu pun gambar yang diperiksa secara visual untuk memahami *mengapa* model salah. Tanpa itu, tidak diketahui apakah kesalahan terjadi pada gambar yang ambigu, gambar berkualitas rendah, atau kelas yang memang sulit — informasi yang penting untuk perbaikan berikutnya.

---

## 💡 Rekomendasi

### Untuk perbaikan model (prioritas utama)

1. **Ganti `Flatten` dengan `GlobalAveragePooling2D`.** Ini memangkas parameter dari 26,1 juta menjadi sekitar 400 ribu (98% lebih kecil) tanpa kehilangan akurasi yang berarti, dan secara drastis mengurangi risiko overfitting pada dataset yang lebih kecil.
2. **Tambahkan kuantisasi TFLite.** Satu baris `converter.optimizations = [tf.lite.Optimize.DEFAULT]` akan menyusutkan model dari ~99 MB ke ~25 MB — syarat minimum agar klaim "siap mobile" bisa dipertahankan.
3. **Seed semua sumber nondeterminisme** (`tf.keras.utils.set_random_seed(42)`, `tf.config.experimental.enable_op_determinism()`) agar hasil bisa direproduksi persis.
4. **Selaraskan kriteria `EarlyStopping` dan `ModelCheckpoint`** — gunakan `val_accuracy` untuk keduanya, atau `val_loss` untuk keduanya. Evaluasi model yang benar-benar tersimpan, bukan model di memori.
5. **Uji transfer learning** (MobileNetV2 atau EfficientNetB0 dengan `include_top=False`). Perbandingan akurasi dan ukuran model akan menunjukkan apakah CNN dari nol ini memang pilihan terbaik, atau hanya pilihan yang paling mudah ditulis.
6. **Jalankan ablasi augmentasi.** Ukur kontribusi masing-masing transformasi. Rotasi 40° mungkin berlebihan; `brightness_range` mungkin yang paling berjasa.

### Untuk validasi (sebelum klaim apa pun)

1. **Uji pada data di luar AFHQ** — foto hewan dari internet, kamera ponsel, atau dataset lain. Inilah satu-satunya cara mengetahui apakah akurasi 99% bertahan di dunia nyata.
2. **Periksa 24 gambar yang salah** secara visual, kelompokkan penyebabnya.
3. **Gunakan split resmi AFHQ** jika ingin hasilnya bisa dibandingkan dengan publikasi lain.

### KPI yang Disarankan

| KPI | Baseline | Target |
|---|---|---|
| Akurasi test (split sendiri) | 99,01% | ≥ 99% stabil antar 5 seed |
| Akurasi pada data eksternal | **belum diukur** | ≥ 90% |
| Jumlah parameter model | 26,1 juta | **< 1 juta** (dengan GAP) |
| Ukuran model TFLite | ~99 MB | **< 10 MB** (kuantisasi + GAP) |
| Waktu inference per gambar | belum diukur | **< 50 ms** di CPU |
| Variasi akurasi antar run | 0,41 poin | **< 0,1 poin** (setelah seeding) |

---

## 🚀 Cara Menjalankan

### 1. Prasyarat

- Python **3.10+**
- Akun Kaggle + API token (`kaggle.json`) — notebook mengunduh dataset via Kaggle API

### 2. Install dependensi

```bash
pip install -r requirements.txt
```

### 3. Siapkan dataset

```bash
kaggle datasets download -d andrewmvd/animal-faces
unzip animal-faces.zip
```

> Struktur yang diharapkan: `afhq/train/{cat,dog,wild}` dan `afhq/val/{cat,dog,wild}`.

### 4. Jalankan

**Opsi A — Notebook Kaggle (disarankan, ini yang menghasilkan angka 99,01%):**

```bash
jupyter notebook project-klasifikasi-gambar-a548ybm523.ipynb
```

**Opsi B — Notebook Colab:**

```bash
jupyter notebook "Project_klasifikasi_gambar(A548YBM523).ipynb"
```

> **Catatan:** versi Colab memakai `files.upload()` dan path `/content/`; versi Kaggle memakai `UserSecretsClient()` dan path `/kaggle/working/`. Sesuaikan dengan lingkungan Anda. Training penuh memakan **~2–3 jam** pada GPU (44 epoch × ~145 detik).

### 5. Output

Setelah selesai, folder `submissions/` berisi:

```
submissions/
├── saved_model/      # TensorFlow SavedModel
├── tfjs_model/       # model.json + weight shards
└── tflite/
    ├── model.tflite
    └── label.txt
```

---

## 🗂️ Struktur Proyek

```
image-classification-cnn/
│
├── project-klasifikasi-gambar-a548ybm523.ipynb      # Run Kaggle — 99,01% (lengkap dengan output)
├── Project_klasifikasi_gambar(A548YBM523).ipynb     # Run Colab — 98,60%
├── Project_klasifikasi_gambar(A548YBM523) (1).ipynb # Versi tanpa output
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Kategori | Tools |
|---|---|
| Bahasa | Python 3.10+ |
| Deep learning | TensorFlow 2.x / Keras (Conv2D, BatchNormalization, MaxPooling2D, Dense, Dropout) |
| Augmentasi | Keras `ImageDataGenerator` |
| Analisis data | pandas, NumPy |
| Evaluasi | scikit-learn (confusion_matrix, classification_report) |
| Visualisasi | Matplotlib, Seaborn |
| Ekspor model | SavedModel, TensorFlow.js, TFLite |
| Environment | Kaggle Notebook / Google Colab |

---

## 👤 Author

**Zulfahmi M. Ardianto**
Laskar ID: A548YBM523

*Project ini merupakan bagian dari portfolio Data Analyst.*
