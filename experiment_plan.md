# Rencana Eksperimen Schematics NPC

## Prinsip keputusan

- Metrik promosi utama: Macro F1 OOF tiga kelas pada 4-fold Balanced Group CV.
- Guardrail: F1/recall Cendrawasih, standar deviasi antar-fold, dan error per `source_group`.
- Open-set dilaporkan terpisah sebagai proxy sampai ada feedback leaderboard atau data Unknown berlabel.
- Satu eksperimen mengubah satu kelompok keputusan. Simpan konfigurasi, seed, OOF probability, threshold, waktu, dan versi library.
- Promosikan perubahan jika memberi peningkatan konsisten (panduan awal ≥0,01 Macro F1 pada lebih dari satu alokasi/seed) tanpa merusak Cendrawasih atau meningkatkan known rejection secara tidak terkendali.
- Jangan memilih model dari random CV atau satu public leaderboard score.

## Prioritas eksperimen

| ID | Prioritas | Eksperimen | Hipotesis | Evaluasi / gate | Dampak | Usaha |
|---|---:|---|---|---|---|---|
| E01 | P0 | Stabilitas group CV | Skor sangat sensitif terhadap 15 grup | Ulangi pembagian grup yang valid dan group bootstrap; catat interval Macro F1/per-class F1 | Tinggi | Rendah |
| E02 | P0 | Ablasi fitur domain-shift | Spectral flux/log-mel std dapat membawa source style | Hapus/normalisasi top domain features; pertahankan jika group CV naik dan domain AUC turun | Sedang–tinggi | Rendah |
| E03 | P0 | Segment-level handcrafted baseline | Panggilan burung singkat hilang dalam agregasi full-file | Ekstrak fitur per segmen 2–5 detik; agregasi max/top-k; group CV | Tinggi | Sedang |
| E04 | P0 | Rare-class treatment | Cendrawasih gagal karena hanya 83 contoh/4 grup | Coba balanced bootstrap, class weights, focal-style sample weights; gate utama recall/F1 Cendrawasih | Tinggi | Rendah–sedang |
| E05 | P1 | CNN/CRNN log-mel dari nol | Representasi lokal waktu–frekuensi lebih tepat daripada fitur agregat | EfficientNet kecil/CRNN, SpecAugment, mixup, group CV, seed berulang | Tinggi | Tinggi |
| E06 | P1 | Pretrained audio/bird embeddings | Transfer learning dapat mengatasi data kecil | **Hanya jika aturan mengizinkan**: PANNs, BirdNET/BirdCLEF, AudioSet/CLAP; classifier linear/GBM | Sangat tinggi | Sedang |
| E07 | P1 | Open-set score ensemble | Max probability saja buruk untuk OOD | Gabungkan temperature scaling, energy, Mahalanobis/prototype distance, kNN density | Sangat tinggi | Sedang |
| E08 | P1 | Audio preprocessing/augmentation | Hujan, angin, gain, dan channel memicu shift | Bandpass, pre-emphasis, channel-best vs mono, gain/noise/reverb, time/frequency masking | Tinggi | Sedang |
| E09 | P2 | Domain adaptation tanpa label | Train–test AUC 0,8397 menandakan covariate shift | CORAL/whitening, domain-adversarial training, importance weighting; jangan gunakan label test | Sedang | Tinggi |
| E10 | P2 | Ensemble OOF | Model berbeda menangkap error berbeda | Blend CNN + pretrained embedding + handcrafted dengan bobot dari group OOF | Tinggi | Sedang |
| E11 | P3 | Pseudo-label test | Test prediction confidence dapat menambah data | **Hanya jika aturan mengizinkan**; conservative high-confidence, satu iterasi, audit collapse | Tidak pasti | Tinggi |

## Tahap 1 — mengunci evaluasi

1. Bekukan daftar fold pada level `source_group` dan simpan manifest.
2. Buat 20–100 bootstrap resample pada level grup untuk interval Macro F1 dan F1 per kelas.
3. Bandingkan agregat OOF dan rata-rata fold; laporkan keduanya karena ukuran fold berbeda.
4. Tambahkan metrik calibration: log loss, expected calibration error, reliability plot.
5. Buat dashboard eksperimen CSV dengan kolom: `experiment_id`, `code_hash`, `seed`, `fold_manifest`, `known_macro_f1`, `cendrawasih_f1`, `fold_std`, `proxy_open_f1`, `known_reject_rate`, `runtime`.

Kriteria selesai: ranking eksperimen stabil terhadap alokasi grup yang wajar dan tidak bergantung pada satu grup Cendrawasih.

## Tahap 2 — memperkuat kelas dikenal

### Segmentasi dan pooling

- Potong audio menjadi jendela 3 detik dengan overlap 50%.
- Latih/prediksi per jendela lalu coba mean, max, top-2 mean, dan attention pooling ke level file.
- Tambahkan background-aware sampling: pilih sebagian jendela energi rendah agar model belajar noise, tetapi jangan memberi label Unknown secara otomatis.
- Bandingkan resampling 16 kHz vs 32 kHz; panggilan frekuensi tinggi mungkin hilang pada 16 kHz.

### Model

1. Logistic regression / linear SVM pada embedding sebagai sanity check.
2. LightGBM/CatBoost/Extra Trees pada fitur segment-level.
3. EfficientNet-B0 atau ConvNeXt-Tiny pada log-mel.
4. CRNN/attention pooling bila pola waktu penting.

### Kelas minor

- Balanced batch sampler di level kelas dan grup.
- Class-balanced loss atau focal loss.
- Mixup hanya antar contoh; uji apakah mixup lintas kelas merusak ciri spesies.
- Augmentasi Cendrawasih harus mempertahankan pitch/temporal call; hindari pitch shift besar tanpa bukti biologis.

## Tahap 3 — open-set recognition

Karena Unknown tidak ada pada train, jalankan beberapa proxy dan jangan menggabungkannya menjadi satu angka seolah-olah ground truth:

1. Leave-one-species-out: tahan satu spesies sebagai pseudo-Unknown.
2. Leave-one-source-group-out hard negatives: ukur confidence pada grup yang gaya rekamannya baru.
3. Background-only crops dari rekaman train sebagai noise rejection proxy. Periksa manual agar tidak berisi panggilan target.
4. Jika aturan mengizinkan data eksternal, gunakan panggilan spesies non-target dan environmental audio sebagai OOD validation terpisah.

Kandidat score:

- maximum calibrated probability;
- margin probabilitas kelas pertama–kedua;
- energy score dari logits;
- jarak Mahalanobis ke centroid kelas di embedding;
- kNN density atau Local Outlier Factor;
- OpenMax / Weibull tail fitting.

Threshold dipilih dengan constraint, misalnya known rejection 2%, 5%, 10%, dan 15%. Buat empat submission threshold yang berbeda hanya jika kuota kompetisi cukup; gunakan hasil leaderboard sebagai bukti kasar, bukan tuning kontinu.

## Tahap 4 — shift dan ensemble

- Jalankan domain classifier per kelompok fitur untuk mengetahui apakah shift berasal dari level/gain, noise temporal, atau pita frekuensi.
- Uji per-recording standardization, PCEN, dan band-wise normalization.
- Buang fitur dengan domain importance tinggi satu per satu; pertahankan hanya bila group CV tidak turun.
- Ensemble menggunakan OOF probabilities yang sejajar. Optimalkan bobot pada group OOF dengan regularisasi dan batas non-negatif.
- Untuk Unknown, ensemble skor OOD setelah calibration, bukan rata-rata label final.

## Urutan eksekusi yang disarankan

1. E01: validasi stabil dan bootstrap grup.
2. E02 + E04: ablation shift dan rare-class treatment pada baseline murah.
3. E03: segment-level features + top-k pooling.
4. E08: preprocessing/augmentation kecil dan terkontrol.
5. Verifikasi aturan eksternal/pretrained.
6. E06 bila legal; jika tidak, lanjut E05.
7. E07: open-set score dan threshold sensitivity.
8. E10: ensemble OOF.
9. E09/E11 hanya setelah baseline kuat dan aturan jelas.

## Artefak minimum setiap eksperimen

- config JSON/YAML lengkap;
- fold manifest dan seed;
- OOF probability per file;
- Macro F1, per-class precision/recall/F1, confusion matrix;
- error per `source_group` dan durasi;
- feature/model importance bila tersedia;
- proxy Unknown curve: threshold vs known rejection vs pseudo-Unknown recall;
- test probability dan submission dengan hash file;
- catatan status: `promote`, `retain for ensemble`, atau `reject`, beserta alasan.

