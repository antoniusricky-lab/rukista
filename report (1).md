# Analisis Kompetisi Schematics NPC Data Science

## Ringkasan eksekutif

Kompetisi ini adalah **klasifikasi audio open-set**. Model harus mengenali tiga spesies yang tersedia pada data latih (`Bondol Jawa`, `Cendrawasih`, `Kutilang`) dan menolak suara dari spesies lain sebagai `Unknown`. Metrik resmi adalah **Macro F1 empat kelas**, sehingga kelas langka dan keputusan `Unknown` mempunyai bobot yang sama dengan kelas mayoritas.

Dataset berisi 1.826 rekaman latih dan 814 rekaman uji. Seluruh audio adalah FLAC stereo 48 kHz PCM 24-bit dengan durasi sekitar 5,3–14,9 detik. Data latih sangat timpang: Bondol Jawa 1.124 (61,6%), Kutilang 619 (33,9%), dan Cendrawasih hanya 83 (4,5%). Ada 15 `source_group`; setiap grup hanya berisi satu spesies. Karena rekaman dari sumber yang sama berpotensi sangat mirip, validasi acak dapat membocorkan karakteristik sumber.

Baseline menggunakan 319 fitur waveform/spektral/log-mel/MFCC dan `ExtraTreesClassifier` dengan class weighting. Validasi utama membagi `source_group` ke empat fold dan memastikan setiap fold memiliki ketiga kelas. Hasil OOF untuk **tiga kelas dikenal** adalah Macro F1 **0,8028**. Random stratified CV menghasilkan 0,8666, selisih +0,0639 yang menunjukkan optimisme jika grup diabaikan. F1 per kelas pada group CV adalah 0,9578 (Bondol Jawa), 0,5128 (Cendrawasih), dan 0,9378 (Kutilang).

Tidak ada skor CV empat kelas yang jujur karena data latih tidak menyediakan satu pun contoh `Unknown`. Threshold rejection 0,5739 pada baseline adalah heuristik yang membatasi penolakan kelas dikenal pada sekitar 10% OOF. Baseline memprediksi 288 dari 814 rekaman uji (35,4%) sebagai `Unknown`; angka ini **bukan estimasi prevalensi yang tervalidasi**. Submission sudah dibuat, tetapi belum dinilai di leaderboard.

Risiko terbesar adalah: validasi bocor bila `source_group` diabaikan, hanya empat grup untuk dua kelas, distribusi train–test berbeda (domain-classifier AUC 0,8397), kelas Cendrawasih sangat kecil, serta tidak adanya data `Unknown` untuk kalibrasi threshold. Prioritas tertinggi adalah memperkuat validasi berbasis grup, menggunakan model segment-level/log-mel yang sensitif terhadap panggilan burung singkat, dan membangun open-set detector terkalibrasi.

## Asumsi dan informasi yang belum diketahui

- Tujuan kerja diasumsikan: membangun baseline kuat dan reproducible serta rencana menuju skor leaderboard setinggi mungkin. Target skor spesifik belum diberikan.
- Aturan penggunaan data eksternal, pretrained weights, internet, dan pseudo-label test **belum diketahui** karena tidak ada README/rules lengkap di folder. Eksperimen yang membutuhkan hal tersebut harus menunggu verifikasi aturan.
- Batas waktu inferensi, batas hardware, jumlah submission per hari, lisensi data, dan deadline kompetisi **belum diketahui**.
- Screenshot menyatakan public leaderboard memakai sekitar 30% data uji dan private leaderboard sekitar 70%.
- Nilai CV di laporan ini hanya bukti lokal. Tidak ada klaim skor leaderboard sampai submission benar-benar dinilai.

## Struktur folder dan inventaris

```text
DS_Schematics/
├── kaggle.zip
├── Data Schematics/
│   ├── sample_submission.csv
│   ├── train/
│   │   ├── metadata.csv
│   │   └── audio/                  # 1.826 FLAC
│   └── test/
│       └── audio/                  # 814 FLAC
├── screenshot overview dan data/  # 5 PNG dokumentasi kompetisi
└── analysis_output/                # seluruh hasil analisis; data asli tidak diubah
```

| Lokasi | Jumlah | Ukuran | Format/keterangan |
|---|---:|---:|---|
| `kaggle.zip` | 1 | 5.220.807.189 byte (4,86 GiB) | Arsip sumber |
| `Data Schematics/train/audio/` | 1.826 | 3,34 GiB termasuk metadata folder | FLAC |
| `Data Schematics/test/audio/` | 814 | 1,52 GiB | FLAC |
| `Data Schematics/train/metadata.csv` | 1 | 407.205 byte | CSV, 1.826 baris × 4 kolom |
| `Data Schematics/sample_submission.csv` | 1 | 55.366 byte | CSV, 814 baris × 2 kolom |
| `screenshot overview dan data/` | 5 | 930.544 byte | PNG dokumentasi |

Isi non-audio di dalam `kaggle.zip` hanya `train/metadata.csv` dan `sample_submission.csv`; tidak ada README atau data dictionary tambahan. Lima screenshot telah dibaca sebagai sumber deskripsi, evaluasi, struktur data, dan aturan submission.

## Pemahaman masalah

### Tujuan sederhana

Untuk setiap rekaman uji, model harus menentukan apakah suara burung merupakan Bondol Jawa, Cendrawasih, Kutilang, atau spesies lain (`Unknown`) di tengah hujan, angin, serangga, dan suara lingkungan lain.

### Jenis task

- Input: audio multikanal berdurasi variabel.
- Output: satu label kategorikal per file.
- Task utama: multiclass audio classification.
- Komponen khusus: open-set recognition / novelty detection karena kelas `Unknown` tidak memiliki contoh berlabel pada data latih.

### Metrik dan implikasi

Macro F1 menghitung F1 tiap kelas lalu memberi bobot sama kepada empat kelas. Implikasinya:

1. Akurasi total tidak cukup. Kegagalan pada Cendrawasih atau Unknown dapat menurunkan skor besar walaupun Bondol Jawa dominan.
2. Threshold `Unknown` adalah hyperparameter utama. Terlalu rendah menyebabkan spesies asing masuk ke kelas dikenal; terlalu tinggi menolak terlalu banyak kelas dikenal.
3. Class weighting, balanced sampling, dan evaluasi per kelas wajib.
4. Threshold tidak boleh dituning pada random CV atau test label yang tidak tersedia.

### Format submission

Submission harus memiliki tepat dua kolom: `file_id` dan `label`. Setiap ID uji harus muncul tepat sekali, tidak boleh ada ID tambahan/hilang, dan label harus persis salah satu dari:

```text
Bondol Jawa
Cendrawasih
Kutilang
Unknown
```

Urutan baris tidak memengaruhi penilaian. File baseline telah melalui pemeriksaan integritas ini.

## Pemahaman dan kualitas data

### Data dictionary

| Kolom | Peran | Temuan |
|---|---|---|
| `file_id` | ID rekaman; string 64 karakter | Unik pada train dan test; tidak ada overlap train–test |
| `audio_path` | Path relatif file train | Semua 1.826 referensi ada; tidak ada orphan file |
| `label` | Target tiga kelas dikenal | Tidak ada missing pada train; seluruh label sample submission kosong sebagai template |
| `source_group` | Kelompok rekaman dari sumber asli yang sama | 15 grup; setiap grup hanya memiliki satu label |

### Kelengkapan, duplikasi, dan konsistensi

- Tidak ada missing value pada empat kolom metadata train.
- Tidak ada duplikat baris, `file_id`, atau `audio_path` pada train.
- Tidak ada duplikat `file_id` pada test.
- Seluruh 2.640 file audio memiliki SHA-256 isi yang unik; tidak ada exact duplicate di dalam split maupun lintas train–test.
- Nama `file_id` tidak sama dengan SHA-256 byte FLAC hasil ekstraksi. Karena itu audit duplikat dilakukan terhadap isi file, bukan hanya nama.
- Seluruh file konsisten: 48.000 Hz, 2 channel, FLAC PCM_24.
- Durasi train rata-rata 10,038 detik (min 5,269; median 10,029; max 14,900). Test rata-rata 10,091 detik (min 5,549; median 10,090; max 14,900).

### Imbalance dan grup

![Distribusi target](figures/target_distribution.png)

| Label | Rekaman | Proporsi | Source group |
|---|---:|---:|---:|
| Bondol Jawa | 1.124 | 61,6% | 7 |
| Cendrawasih | 83 | 4,5% | 4 |
| Kutilang | 619 | 33,9% | 4 |

Cendrawasih hanya memiliki 83 rekaman dan empat sumber. Satu grup bahkan hanya berisi tiga rekaman. Estimasi performa kelas ini mempunyai variansi tinggi dan sangat sensitif terhadap grup yang masuk fold validasi.

![Ukuran source group](figures/source_group_sizes.png)

### Outlier dan kualitas audio

Aturan deskriptif 1,5×IQR pada enam indikator kualitas menandai 146 file, terutama karena `silence_fraction` (102) dan RMS (30). Durasi tidak memiliki outlier IQR. Rekaman bertanda bukan otomatis data buruk: hujan, angin, bagian sunyi, dan panggilan burung yang sporadis memang bagian dari masalah. Karena itu file tidak dihapus. Daftar rinci ada di `results/audio_quality_outliers.csv`.

## EDA mendalam

### Univariat dan bivariat

- Durasi hampir sama di ketiga kelas dan bukan pembeda utama.
- Fitur terpenting baseline didominasi variasi energi log-mel pada pita menengah-tinggi, spectral flux, rolloff, dan rasio energi 2–6 kHz. Ini konsisten dengan panggilan burung yang bersifat lokal dan berubah cepat.
- Banyak fitur log-mel berdekatan sangat berkorelasi. Hal ini tidak bermasalah besar untuk Extra Trees, tetapi perlu regularisasi atau reduksi dimensi pada model linear.
- Performa berdasarkan kuartil durasi relatif stabil (akurasi mikro 0,930–0,954), sehingga kegagalan utama bukan panjang rekaman.

![Korelasi fitur](figures/feature_correlation_heatmap.png)

### Distribution shift train–test

Domain classifier yang membedakan train versus test dari 319 fitur audio mencapai out-of-fold ROC AUC **0,8397**. Nilai 0,5 akan berarti distribusi sulit dibedakan; 0,84 menunjukkan shift yang material. Fitur paling bergeser adalah spectral flux dan standar deviasi sejumlah band log-mel.

![Shift train test](figures/train_test_shift_top_features.png)

PCA masih menunjukkan overlap yang luas, jadi shift bukan pemisahan total. Namun test lebih terkonsentrasi pada area tertentu. Penjelasan yang mungkin mencakup perbedaan sumber rekaman, noise, prevalensi spesies, atau keberadaan kelas Unknown; penyebab pastinya belum dapat ditentukan tanpa metadata test.

![PCA train test](figures/pca_train_test.png)

Nearest-neighbor lintas split pada fitur agregat menemukan pasangan sangat dekat, tetapi tidak ada duplikat byte. Pasangan tersebut adalah kandidat inspeksi audio, bukan bukti leakage.

## Risiko dan jebakan

1. **Source leakage.** `source_group` murni terhadap label dan tidak tersedia pada test. Memasukkannya sebagai fitur akan menjadi kebocoran langsung. Memecah satu grup ke train dan validation juga memberi skor optimistis.
2. **Random-CV mismatch.** Random CV menghasilkan Macro F1 0,8666, sedangkan group-safe OOF 0,8028. Random CV tidak digunakan untuk memilih model.
3. **Unknown tidak terobservasi.** Tidak ada cara menghitung Macro F1 empat kelas secara jujur. Leave-one-species-out hanya proxy dan dapat terlalu mudah atau terlalu sulit dibanding Unknown sebenarnya.
4. **Sedikit grup.** Hanya empat grup untuk Cendrawasih dan Kutilang. Estimasi fold sangat beragam: Macro F1 fold 0,6098–0,9402.
5. **Class imbalance.** Cendrawasih hanya 4,5% data. Accuracy tinggi dapat menyembunyikan recall yang buruk.
6. **Train–test shift.** Domain AUC 0,8397 mengindikasikan CV lokal mungkin tidak sepenuhnya mencerminkan leaderboard.
7. **Public leaderboard overfitting.** Public hanya sekitar 30% test. Tuning threshold berdasarkan sedikit submission dapat merusak private score 70%.
8. **Near-duplicate/source style.** Fitur sangat dekat lintas split perlu audit audio fingerprint lanjutan, tetapi exact duplicate sudah dikesampingkan.

## Strategi validasi yang direkomendasikan

Validasi utama adalah **4-fold Balanced Group CV**:

- semua rekaman dengan `source_group` sama tetap di fold yang sama;
- setiap fold memiliki minimal satu grup dari setiap kelas;
- grup besar didistribusikan secara greedy untuk mengurangi ketimpangan ukuran fold;
- empat fold dipilih karena jumlah minimum grup per label adalah empat.

Metrik yang dilaporkan per eksperimen:

- Macro F1 tiga kelas dikenal pada OOF gabungan;
- F1 dan recall per kelas, terutama Cendrawasih;
- mean dan standar deviasi Macro F1 antar-fold;
- error per `source_group`;
- known-rejection rate untuk threshold Unknown;
- proxy open-set leave-one-species-out, selalu diberi label sebagai proxy.

Untuk model final, ulangi alokasi grup dengan beberapa tie-breaking/seed yang sah atau gunakan bootstrap pada level grup untuk interval ketidakpastian. Jangan gunakan random split sebagai bukti utama.

## Baseline

### Pipeline

1. Stereo dirata-ratakan menjadi mono dan di-resample deterministik ke 16 kHz.
2. Setiap file diringkas menjadi 319 fitur: RMS, zero-crossing, centroid, bandwidth, rolloff, flatness, entropy, flux, rasio energi pita, 48 band log-mel, 20 MFCC, dan ringkasan temporal tiga segmen.
3. Missing/non-finite diimputasi median train (pada data saat ini tidak ada missing fitur).
4. `ExtraTreesClassifier`: 500 tree, `class_weight="balanced"`, `min_samples_leaf=2`, `max_features="sqrt"`, seed 42.
5. Threshold Unknown dipilih dari proxy leave-one-species-out, lalu dibatasi agar tidak menolak lebih dari sekitar 10% known OOF. Threshold akhir 0,5739.

### Hasil CV

| Evaluasi | Nilai |
|---|---:|
| Group-safe OOF Macro F1, tiga kelas dikenal | **0,8028** |
| Group-safe balanced accuracy | 0,7646 |
| Mean Macro F1 per group fold | 0,7383 ± 0,1473 |
| Random stratified OOF Macro F1 | 0,8666 |
| Gap random minus group | +0,0639 |

| Kelas | Precision | Recall | F1 |
|---|---:|---:|---:|
| Bondol Jawa | 0,9573 | 0,9582 | 0,9578 |
| Cendrawasih | 0,8824 | 0,3614 | **0,5128** |
| Kutilang | 0,9040 | 0,9742 | 0,9378 |

![Confusion matrix](figures/cv_confusion_matrix.png)

Model hampir selalu membedakan Bondol Jawa dan Kutilang, tetapi hanya menemukan 30 dari 83 Cendrawasih. Tiga dari empat source group Cendrawasih sangat sulit: akurasi grup masing-masing 0%, 6,25%, dan 20%; satu grup mencapai 75,86%. Ini menunjukkan model mempelajari karakteristik sumber dan belum cukup invariant.

![Feature importance](figures/feature_importance.png)

### Heuristik Unknown dan submission

Eksperimen pseudo-Unknown menghasilkan threshold optimal yang tidak stabil (0,58–0,93), bukti bahwa threshold belum teridentifikasi dengan baik. Baseline memakai threshold 0,5739 untuk membatasi known OOF rejection sekitar 10,0%. Prediksi test:

| Label prediksi | Jumlah | Proporsi |
|---|---:|---:|
| Bondol Jawa | 283 | 34,8% |
| Cendrawasih | 9 | 1,1% |
| Kutilang | 234 | 28,7% |
| Unknown | 288 | 35,4% |

Distribusi tersebut tidak boleh dianggap sebagai distribusi ground truth. File `results/baseline_submission.csv` siap diunggah, tetapi belum memiliki skor leaderboard.

## Rekomendasi

1. Jadikan group-safe CV satu-satunya gate promosi model. Laporkan random CV hanya sebagai diagnosis leakage.
2. Tingkatkan recall Cendrawasih melalui segment-level training, balanced sampler/focal loss, dan augmentasi yang tidak menghapus sinyal kelas minor.
3. Gunakan log-mel CNN/CRNN dengan pooling top-k atau attention agar panggilan singkat tidak hilang dalam agregasi seluruh file.
4. Bangun open-set score dari gabungan calibrated confidence, embedding distance/Mahalanobis, dan energy score. Tuning dengan nested leave-one-species-out proxy dan known-rejection constraints.
5. Tangani shift dengan augmentasi noise/reverb/gain, source-style normalization, dan evaluasi feature ablation terhadap fitur yang paling membedakan train–test.
6. Jika aturan mengizinkan pretrained weights/external data, prioritaskan embedding AudioSet/BirdCLEF/BirdNET/PANNs lalu klasifier ringan. Status aturan harus dikonfirmasi dulu.
7. Ensemble model yang error-nya berbeda: CNN segment-level + pretrained embedding + handcrafted Extra Trees. Optimalkan bobot hanya dengan group OOF.
8. Gunakan public leaderboard secara hemat untuk menguji 2–3 keluarga threshold Unknown, bukan tuning granular.

Urutan eksperimen rinci ada di `experiment_plan.md`.

## Reproduksi

Semua seed diset ke 42. Versi utama: Python 3.12.14, NumPy 2.3.5, pandas 3.0.1, SciPy 1.16.2, scikit-learn 1.7.2, matplotlib 3.10.7, seaborn 0.13.2, SoundFile 0.13.1.

```powershell
cd C:\Users\yanli\OneDrive\Documents\DS_Schematics
python -m pip install -r .\analysis_output\requirements.txt
python .\analysis_output\eda.py --n-jobs 4
python .\analysis_output\baseline.py --n-jobs 4 --n-estimators 500
```

Script memakai cache di `analysis_output/cache/`; gunakan `--force-features` bila ingin mengekstrak ulang semua fitur. Data asli tidak pernah ditulis atau diubah.

