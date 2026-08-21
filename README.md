# Deploy ke Streamlit Community Cloud 

## Langkah 1 — Siapkan artifacts.zip (cukup PatchCore, bukan semua)

PatchCore itu satu-satunya metode yang dipakai deployment (metode final, hasil eksperimen).
Autoencoder dan PaDiM **tidak perlu ikut** — biar ukuran file lebih kecil.

Di Google Drive kamu, masuk ke folder `Kelompok 1_Nalara/artifacts/`. Untuk **tiap kategori**,
di dalamnya ada `autoencoder.pt`, `patchcore.pkl`, `padim.pkl`, `config.json` — kamu cuma perlu
`patchcore.pkl` dan `config.json`.

Cara paling praktis: download folder `artifacts/` apa adanya ke laptop, lalu di laptop:
1. Hapus semua file `autoencoder.pt` dan `padim.pkl` dari tiap folder kategori (sisakan
   `patchcore.pkl` + `config.json` saja per kategori)
2. Zip ulang folder `artifacts/` itu jadi `artifacts.zip` (pastikan strukturnya
   `artifacts/bottle/patchcore.pkl`, `artifacts/bottle/config.json`, dst — bukan makin ada
   folder pembungkus tambahan)

## Langkah 2 — Upload artifacts.zip ke Drive & ambil File ID

1. Upload `artifacts.zip` ke Google Drive kamu (folder manapun)
2. Klik kanan file itu → **Share** → ubah jadi **"Anyone with the link"** (Viewer)
3. Klik **Copy link** — bentuknya seperti:
   `https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQrStUvWxYz/view?usp=sharing`
4. **File ID** adalah bagian di antara `/d/` dan `/view` — di contoh itu:
   `1AbCdEfGhIjKlMnOpQrStUvWxYz`
5. Simpan File ID ini, dipakai di Langkah 5.

## Langkah 3 — Buat akun GitHub (kalau belum ada) & buat repo baru

1. Daftar gratis di [github.com](https://github.com) kalau belum punya akun
2. Klik **New repository** → kasih nama (misal `smart-inspection-app`) → **Public** atau
   **Private** (dua-duanya bisa dipakai Streamlit Cloud) → Create repository

## Langkah 4 — Upload 2 file ini ke repo GitHub

Upload `app.py` dan `requirements.txt` (yang ada di folder ini) ke repo yang baru dibuat —
klik **Add file → Upload files** di halaman repo GitHub, drag-drop kedua file, lalu Commit.

## Langkah 5 — Deploy di Streamlit Community Cloud

1. Buka [share.streamlit.io](https://share.streamlit.io) → sign in pakai akun GitHub
2. Klik **Create app** → pilih repo yang tadi dibuat → **Main file path**: `app.py` → Deploy
3. Sebelum/sesudah deploy, buka **Settings → Secrets** aplikasi ini, isi:
   ```
   ARTIFACTS_DRIVE_FILE_ID = "1AbCdEfGhIjKlMnOpQrStUvWxYz"
   ```
   (ganti dengan File ID dari Langkah 2)
4. Tunggu proses build (beberapa menit, install PyTorch itu berat). Setelah selesai,
   aplikasi otomatis download `artifacts.zip` dari Drive saat pertama kali dibuka.

Selesai — kamu dapat URL permanen bentuknya `https://nama-app-kamu.streamlit.app`,
tidak perlu Colab sama sekali untuk mengaksesnya.

## Cara pakai setelah deploy

Di sidebar: pilih kategori, lalu **atur threshold manual** — upload dulu 1 gambar,
lihat "Anomaly score" yang keluar, baru sesuaikan `tau_low`/`tau_high` di sekitar
angka itu (tau_low sedikit di bawah, tau_high sedikit di atas skor gambar normal).

## Troubleshooting

- **App crash / "Oh no" error saat load model** → kemungkinan kehabisan RAM (limit 1GB
  tier gratis). Coba refresh, atau pertimbangkan upgrade ke tier berbayar Streamlit Cloud
  kalau ini sering terjadi.
- **"Gagal download artifacts.zip"** → cek lagi sharing permission Drive-nya benar-benar
  "Anyone with the link", dan File ID di Secrets sudah benar (tanpa spasi/karakter tambahan).
- **App "sleeping"** → Streamlit Cloud gratis menidurkan app yang lama tidak diakses;
  buka linknya, tunggu ~30 detik, otomatis bangun lagi (data pengaturan/riwayat sesi
  sebelumnya akan reset).
