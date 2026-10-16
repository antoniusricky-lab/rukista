# Log Eksperimen Schematics NPC

## Kontrak evaluasi

- Metrik leaderboard: Macro F1 empat kelas, termasuk Unknown.
- Metrik lokal utama: Macro F1 tiga kelas dikenal pada 4-fold `BalancedGroupKFold` yang sama dengan baseline.
- Seed utama: 42.
- Gate promosi model known-class: delta F1 tidak negatif pada setiap fold, mean F1 meningkat, standar deviasi tidak memburuk secara material, dan F1 Cendrawasih tidak turun.
- Open-set dilaporkan terpisah karena train tidak memiliki Unknown berlabel. Tidak akan dibuat klaim CV empat kelas tanpa ground truth.
- Satu eksperimen hanya boleh mengubah satu keputusan dari baseline.

## Hasil yang sudah ada

| ID | Perubahan | Hipotesis | CV Macro F1 mean ± SD antar-fold | OOF agregat | Delta | Waktu | Keputusan |
|---|---|---|---:|---:|---:|---:|---|
| BASELINE | Kontrol: ExtraTrees 500, fitur handcrafted full-file 16 kHz, threshold 0,57387 | Titik acuan | 0,73828 ± 0,14731 | 0,80279 | — | Tidak tercatat | Dipertahankan sebagai kontrol; LB publik 0,62271 |
| DIAG-00 | Tidak mengubah model; error slices, calibration, TreeSHAP OOF, dan audit shift | Menentukan akar masalah sebelum eksperimen | 0,73828 ± 0,14731 | 0,80279 | 0,00000 | 18,10 detik | Selesai; menemukan over-rejection, rare-class instability, dan domain shortcut |
| EXP-01 | Threshold Unknown 0,57387 → 0,45; model dan probability tetap | Mengurangi false Unknown, terutama kandidat Cendrawasih | 0,72878 ± 0,13621 setelah rejection | Tidak sebanding langsung | -0,00951 | 0,066 detik | **Ditolak**; LB 0,55009, turun 0,07262 dari baseline |
| EXP-02 | Hapus tepat 20 fitur dengan KS train–test shift terbesar | Mengurangi shortcut sumber/noise | 0,67361 ± 0,07785 | 0,69876 | -0,06467 | 2,92 detik | **Ditolak**; seluruh empat fold turun dan tidak dibuat submission |
| EXP-03 | Ganti global class weighting dengan balanced bootstrap per pohon | Memperbaiki boundary kelas minor tanpa mengubah fitur | 0,79072 ± 0,12984 | 0,83242 | +0,05243 | 6,63 detik | **Diterima lokal**; 4/4 fold dan 3/3 seed konfirmasi naik, menunggu LB |

Catatan: public leaderboard menggunakan sekitar 30% test. Snapshot top-5 adalah 0,65629, sehingga baseline tertinggal 0,03358. Hanya satu pasangan CV–LB tersedia; korelasi belum dapat dihitung.

## Antrean eksperimen

Status semua baris di bawah adalah **menunggu persetujuan/eksekusi**. Skor belum diisi dan tidak boleh ditafsirkan sebagai hasil.

| ID | Satu perubahan yang diuji | Hipotesis | Dampak | Usaha | Risiko utama | Status |
|---|---|---|---|---|---|---|
| EXP-01 | Threshold Unknown 0,57387 → 0,45 | Mengurangi false Unknown, terutama kandidat Cendrawasih | Tinggi | Rendah | Tidak ada label Unknown lokal; public-LB overfit | Ditolak; LB 0,55009 |
| EXP-02 | Hapus 20 fitur dengan shift KS terbesar | Mengurangi shortcut sumber/noise | Sedang–tinggi | Rendah | Sinyal spesies ikut terhapus | Ditolak; 4/4 fold turun |
| EXP-03 | Balanced bootstrap per pohon menggantikan class weight baseline | Memperbaiki recall Cendrawasih | Tinggi | Rendah–sedang | Skala confidence/Unknown ikut berubah | Diterima lokal; v03 dibuat |
| EXP-04 | Resampling fitur 16 kHz → 32 kHz | Memulihkan informasi 8–16 kHz | Sedang–tinggi | Sedang | Komputasi naik | Menunggu |
| EXP-05 | Top-2 pooling segmen 3 detik menggantikan agregasi full-file | Menonjolkan panggilan singkat | Tinggi | Sedang | Segmentasi menambah variansi | Menunggu |
| EXP-06 | Temperature scaling cross-fitted | Menstabilkan confidence dan threshold | Sedang–tinggi | Sedang | Tidak menyelesaikan OOD sendiri | Menunggu |
| EXP-07 | Satu model boosting menggantikan ExtraTrees | Boundary fitur lebih baik | Sedang | Sedang | Overfit pada 15 grup | Ditunda sampai fitur matang |
| EXP-08 | OOD score jarak centroid menggantikan max probability | Memisahkan novelty dari confidence | Tinggi | Sedang | Proxy Unknown tidak representatif | Menunggu protokol OOD |
| EXP-09 | Pretrained embedding menggantikan fitur handcrafted | Transfer lintas sumber | Tinggi | Tinggi | Legalitas pretrained belum diketahui | Diblokir aturan |
| EXP-10 | Satu iterasi pseudo-label | Adaptasi domain test | Tidak pasti | Tinggi | Confirmation bias dan aturan | Prioritas terakhir |

## EXP-01 — threshold Unknown 0,45

- Kode: `analysis_output/experiments/exp_01_unknown_threshold_045.py`
- Satu perubahan: threshold Unknown diturunkan dari 0,5738699 menjadi 0,45.
- Tidak berubah: fitur, model, seed, fold, probability kelas, dan urutan test.
- Runtime inti: 0,066 detik.
- Submission: `analysis_output/submissions/submission_v01_threshold_045.csv`.
- SHA-256: `d2fe4101073f3c33e8c13024ae9269bfdae6e2df5ea6462dae522b6a428234db`.

| Fold | Baseline F1 | F1 setelah rejection | Delta | Known rejection |
|---:|---:|---:|---:|---:|
| 1 | 0,752326 | 0,732359 | -0,019967 | 2,73% |
| 2 | 0,940151 | 0,919161 | -0,020990 | 1,24% |
| 3 | 0,609783 | 0,614667 | +0,004884 | 2,40% |
| 4 | 0,650868 | 0,648920 | -0,001948 | 2,93% |
| **Mean ± SD** | **0,738282 ± 0,147311** | **0,728777 ± 0,136207** | **-0,009505** | — |

Distribusi label test berubah sebagai berikut:

| Label | Baseline v00 | EXP-01 v01 | Delta |
|---|---:|---:|---:|
| Bondol Jawa | 283 | 362 | +79 |
| Cendrawasih | 9 | 38 | +29 |
| Kutilang | 234 | 326 | +92 |
| Unknown | 288 | 88 | -200 |

Hasil leaderboard publik: **0,55009**, atau **-0,07262** dari baseline 0,62271.

Keputusan akhir: **ditolak**. Hasil lokal turun pada tiga dari empat fold dan leaderboard juga turun tajam. Baseline v00 tetap menjadi champion. Jangan melanjutkan tuning threshold ke arah yang lebih rendah berdasarkan public leaderboard.

## EXP-02 — ablation 20 fitur dengan shift terbesar

- Kode: `analysis_output/experiments/exp_02_drop_shift_features.py`.
- Satu perubahan: menghapus tepat 20 fitur dengan statistik KS train–test terbesar.
- Tidak berubah: cached feature values, median imputation, model ExtraTrees 500 trees, seed, fold, dan seluruh hyperparameter.
- Fitur: 319 → 299.
- Runtime inti: 2,92 detik.
- Artefak: `analysis_output/experiments/results/exp_02_drop_shift_features/`.

| Fold | Baseline F1 | EXP-02 F1 | Delta |
|---:|---:|---:|---:|
| 1 | 0,752326 | 0,724194 | -0,028132 |
| 2 | 0,940151 | 0,747209 | -0,192942 |
| 3 | 0,609783 | 0,576200 | -0,033583 |
| 4 | 0,650868 | 0,646830 | -0,004038 |
| **Mean ± SD** | **0,738282 ± 0,147311** | **0,673608 ± 0,077849** | **-0,064674** |

OOF aggregate Macro F1 turun menjadi 0,698765 dan F1 Cendrawasih turun dari 0,5128 menjadi 0,210526. Hipotesis shortcut hanya benar sebagian: fitur yang bergeser juga membawa sinyal kelas yang penting, sehingga ablation massal terlalu destruktif.

Keputusan: **ditolak** karena keempat fold turun. Strict gate gagal dan kode sengaja tidak membuat `submission_v02_drop_shift20.csv`; folder submission tetap hanya berisi v00 dan v01.

## EXP-03 — balanced bootstrap per pohon

- Kode utama: `analysis_output/experiments/exp_03_balanced_bootstrap.py`.
- Validasi seed: `analysis_output/experiments/validate_exp_03_seed_stability.py`.
- Satu perubahan: mekanisme `class_weight="balanced"` diganti balanced bootstrap per pohon. Setiap pohon tetap ExtraTree dengan parameter baseline dan menerima jumlah baris total kira-kira sama, tetapi jumlah sampel tiap kelas dibuat sama dengan replacement.
- Tidak berubah: 319 fitur, median imputation, manifest 4 fold, 500 pohon, parameter base tree, dan threshold numerik Unknown 0,5738699.
- Runtime seed 42: 6,63 detik.
- Submission: `analysis_output/submissions/submission_v03_balanced_bootstrap.csv`.
- SHA-256: `7cbb6845e9e16d1172094e3c9b98264cce12e6f59e6820c86af77b21cc67b24f`.

| Fold | Baseline F1 | EXP-03 F1 | Delta | Cendrawasih F1 baseline → EXP-03 |
|---:|---:|---:|---:|---:|
| 1 | 0,752326 | 0,804500 | +0,052174 | 0,333333 → 0,478261 |
| 2 | 0,940151 | 0,961274 | +0,021122 | 0,846154 → 0,909091 |
| 3 | 0,609783 | 0,652151 | +0,042368 | 0,100000 → 0,240000 |
| 4 | 0,650868 | 0,744942 | +0,094074 | 0,000000 → 0,285714 |
| **Mean ± SD** | **0,738282 ± 0,147311** | **0,790717 ± 0,129840** | **+0,052435** | — |

Metrik OOF tambahan:

| Metrik | Baseline | EXP-03 | Delta |
|---|---:|---:|---:|
| Macro F1 agregat | 0,802790 | 0,832421 | +0,029631 |
| Accuracy | 0,936473 | 0,938664 | +0,002191 |
| Balanced accuracy | 0,764594 | 0,803701 | +0,039107 |
| F1 Cendrawasih | 0,512821 | 0,601504 | +0,088683 |
| Recall Cendrawasih | 0,361446 | 0,481928 | +0,120482 |
| Mean recall antar-sumber Cendrawasih | 0,255280 | 0,424297 | +0,169017 |
| Median recall antar-sumber Cendrawasih | 0,131250 | 0,323810 | +0,192560 |

Validasi paired seed tambahan:

| Seed | Delta mean fold | Semua fold naik | F1 Cendrawasih baseline → EXP-03 | Gate seed |
|---:|---:|---|---:|---|
| 13 | +0,039842 | Ya | 0,533333 → 0,590909 | Lolos |
| 137 | +0,066387 | Ya | 0,512821 → 0,616541 | Lolos |
| 2026 | +0,068607 | Ya | 0,516667 → 0,600000 | Lolos |

Rata-rata paired improvement pada tiga seed konfirmasi adalah +0,058279; delta fold terkecil dari seluruh seed tetap positif, +0,014227.

Distribusi submission v03: 249 Bondol Jawa, 17 Cendrawasih, 243 Kutilang, dan 305 Unknown. Dibanding v00, 65 dari 814 label berubah: 41 known menjadi Unknown dan 24 Unknown menjadi known. Perubahan ini terjadi karena balanced bootstrap mengubah confidence walaupun threshold numeriknya tetap. Oleh karena itu peningkatan lokal dapat dipastikan, tetapi skor leaderboard belum dapat dijamin tanpa ground truth Unknown.

Keputusan: **diterima sebagai challenger lokal dan layak satu kali submission leaderboard**. Baseline v00 tetap champion resmi sampai skor v03 melampaui 0,62271.

## Template hasil per eksperimen

Setiap eksperimen yang dijalankan akan menambah satu baris dengan:

- hash/config kode dan versi library;
- skor keempat fold, mean ± sample SD, OOF agregat, serta delta terhadap baseline;
- precision/recall/F1 per kelas, terutama Cendrawasih;
- waktu eksekusi;
- untuk open-set: known rejection per fold, distribusi label test, dan feedback leaderboard bila disubmit;
- keputusan `diterima`, `ditolak`, atau `perlu diselidiki`, beserta alasan.
