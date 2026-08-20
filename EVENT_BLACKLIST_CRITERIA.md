# Event Blacklist Criteria

Dokumen ini menjadi rujukan tetap saat menentukan event berita mana yang layak dimasukkan ke blacklist EA atau ke blacklist training-side.

## Tujuan

Blacklist digunakan untuk membuang event yang secara konsisten merusak performa model.

Blacklist **bukan** dipakai untuk:
- membuang semua event yang total profit-nya negatif
- bereaksi berlebihan pada sampel kecil
- mengejar equity curve bagus dengan mengorbankan terlalu banyak peluang trade

## Data Minimum yang Harus Dicek per Event

Untuk setiap event, hitung minimal:

- `trade_count`
- `win_count`
- `loss_count`
- `win_rate`
- `total_profit`
- `avg_profit_per_trade`
- `largest_loss`
- `largest_win`

Kalau memungkinkan, tambahkan:

- `profit_factor_per_event`
- `median_profit_per_trade`
- `recent_period_consistency`

## Prinsip Utama

### 1. Sampel kecil tidak boleh langsung diblacklist

Aturan:
- `trade_count <= 2`: jangan blacklist
- `trade_count = 3 atau 4`: hanya blacklist jika hasilnya sangat buruk dan konsisten
- `trade_count >= 5`: baru layak dinilai serius

Alasan:
- 1-2 trade tidak cukup untuk menyimpulkan karakter event
- 3-4 trade masih rawan noise

### 2. Total profit negatif saja tidak cukup

Event tidak otomatis diblacklist hanya karena:
- `total_profit < 0`

Event baru menjadi kandidat blacklist jika negatifnya disertai:
- jumlah trade cukup
- loss rate tinggi
- average profit per trade negatif
- pola loss konsisten

### 3. Event frekuensi tinggi dinilai lebih ketat

Jika event muncul sangat sering, jangan mudah diblok hanya karena total profit negatif kecil.

Contoh:
- `trade_count = 60`
- `total_profit = -250`

Ini belum tentu layak blacklist karena:
- dampak per trade kecil
- event mungkin masih berkontribusi pada banyak trade yang valid

Untuk event frekuensi tinggi, lebih penting melihat:
- `avg_profit_per_trade`
- `win_rate`
- apakah loss benar-benar struktural

### 4. Konsistensi buruk lebih penting daripada 1 loss besar

Hindari blacklist event yang total negatif hanya karena satu trade besar.

Contoh buruk untuk dasar blacklist:
- `trade_count = 6`
- `5` win kecil
- `1` loss sangat besar

Itu belum tentu event buruk; bisa jadi masalah money management atau stop loss.

Yang lebih layak diblacklist:
- banyak loss
- sedikit win
- average trade negatif
- pola buruk berulang

## Kriteria Operasional

Gunakan prioritas berikut.

### A. Jangan blacklist

Jangan blacklist jika salah satu kondisi ini terjadi:

- `trade_count <= 2`
- `trade_count <= 4` dan hasil campuran
- `trade_count` tinggi tetapi `avg_profit_per_trade` mendekati nol
- total negatif kecil dibanding frekuensinya
- ada indikasi event masih memberi peluang profit yang berarti

### B. Kandidat blacklist lemah

Masuk observasi dulu, belum langsung diblok:

- `trade_count = 3 sampai 5`
- `loss_count > win_count`
- `total_profit < 0`
- tetapi magnitudo loss belum besar

Tindakan:
- simpan sebagai watchlist
- cek lagi setelah hasil backtest/model lain bertambah

### C. Kandidat blacklist kuat

Layak dipertimbangkan blacklist jika seluruh atau hampir seluruh kondisi ini terpenuhi:

- `trade_count >= 5`
- `loss_rate >= 70%`
- `total_profit` jelas negatif
- `avg_profit_per_trade < 0`
- tidak ada indikasi bahwa profit sesekali mampu menutup pola loss

### D. Kandidat blacklist sangat kuat

Sangat layak diblacklist jika:

- `trade_count >= 6`
- `loss_rate >= 75%`
- `win_count` sangat sedikit atau nol
- `avg_profit_per_trade` jelas negatif
- `total_profit` termasuk salah satu yang terburuk di model itu

## Skema Keputusan yang Disarankan

### Rule 1: Sampel minimum

- Jika `trade_count <= 2` -> `JANGAN BLACKLIST`
- Jika `trade_count 3-4` -> hanya lanjut jika loss sangat dominan
- Jika `trade_count >= 5` -> evaluasi penuh

### Rule 2: Loss dominance

Event masuk shortlist jika:
- `loss_rate >= 70%`

### Rule 3: Profit quality

Event tetap masuk shortlist hanya jika:
- `avg_profit_per_trade < 0`
- dan `total_profit < 0`

### Rule 4: Frekuensi tinggi

Jika `trade_count >= 20`, gunakan syarat lebih ketat:
- `loss_rate >= 75%`
- `avg_profit_per_trade` cukup negatif
- total loss material

Tujuan aturan ini:
- event yang sangat sering muncul tidak diblok hanya karena sedikit negatif

## Skema Scoring Sederhana

Untuk audit cepat, gunakan skor berikut:

- `trade_count >= 5` -> `+1`
- `trade_count >= 10` -> `+1`
- `loss_rate >= 70%` -> `+1`
- `loss_rate >= 80%` -> `+1`
- `total_profit < 0` -> `+1`
- `avg_profit_per_trade < 0` -> `+1`
- event termasuk 5 terburuk berdasarkan `total_profit` pada model itu -> `+1`

Interpretasi:
- `0-2`: jangan blacklist
- `3-4`: watchlist / perlu audit manual
- `5+`: layak blacklist

## Pembedaan EA Blacklist vs Training Blacklist

### EA Blacklist

Gunakan lebih konservatif.

Cocok untuk:
- event yang jelas buruk
- hasil buruk konsisten
- perlu diblok langsung saat inference

Karena EA blacklist memotong trade secara langsung, jangan terlalu banyak.

### Training-side Blacklist / Weight Suppression

Bisa sedikit lebih longgar daripada EA blacklist.

Cocok untuk:
- event yang tidak selalu buruk, tetapi sering mengganggu pembelajaran model
- event yang ingin diturunkan pengaruhnya, bukan dihapus total dari eksekusi live

## Prosedur Audit yang Direkomendasikan

Urutan kerja:

1. Kelompokkan hasil per `symbol` dan per `model`
2. Hitung statistik per `event_name`
3. Urutkan berdasarkan:
   - `total_profit` terburuk
   - lalu cek `trade_count`
   - lalu cek `loss_rate`
4. Buang event dengan sampel terlalu kecil
5. Pisahkan:
   - `watchlist`
   - `EA blacklist`
   - `training-side suppression`
6. Setelah blacklist diterapkan, backtest ulang
7. Bandingkan:
   - total net profit
   - drawdown
   - jumlah trade
   - profit factor
8. Jika trade turun terlalu tajam, blacklist terlalu agresif

## Red Flags

Hindari blacklist jika:

- event hanya rugi karena 1 trade ekstrem
- event punya banyak trade dan hasilnya hanya sedikit negatif
- event sering muncul dan masih memberi banyak win
- alasan blacklist hanya karena "kelihatan jelek"

## Prinsip Akhir

Blacklist yang baik harus:

- selektif
- bisa dijelaskan
- berbasis sampel
- tidak menghapus terlalu banyak trade
- meningkatkan kualitas trade, bukan sekadar mengurangi jumlah trade

Jika ragu:
- masukkan ke watchlist dulu
- jangan langsung blacklist
