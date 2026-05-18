# Praktikum Kecerdasan Buatan - Pertemuan 8

**Identitas Mahasiswa:**
- **Nama:** MUHAMMAD FATHAN RAMDANI
- **NIM:** H1D024026
- **Shift Awal:** A
- **Shift Akhir:** B
- **Mata Kuliah:** PRAKTIKUM KECERDASAN BUATAN

---

Repositori ini berisi implementasi Convolutional Neural Network (CNN) untuk mengklasifikasi gambar Rock-Paper-Scissors (batu-gunting-kertas) menggunakan TensorFlow dan Keras. Tugas ini merupakan bagian dari praktikum Kecerdasan Buatan (Pertemuan 8).

## Deskripsi Tugas
Tugas ini mengharuskan pembuatan model CNN yang dapat mengenali gambar tangan yang membentuk batu, kertas, atau gunting. Arsitektur dan alur program diimplementasikan secara spesifik sesuai dengan instruksi yang terdapat di dalam file PDF panduan praktikum.

Arsitektur model CNN yang dibangun:
- Convolutional Layer (32 filter, kernel 3x3, aktivasi ReLU) + Max Pooling (2x2)
- Convolutional Layer (64 filter, kernel 3x3, aktivasi ReLU) + Max Pooling (2x2)
- Convolutional Layer (128 filter, kernel 3x3, aktivasi ReLU) + Max Pooling (2x2)
- Flatten Layer
- Dense Layer (512 neuron, aktivasi ReLU)
- Output Dense Layer (3 neuron, aktivasi Softmax)

## Persyaratan
- Python 3.x
- TensorFlow
- NumPy
- Pandas
- Dataset "rockpaperscissors" (diunduh terpisah)

## Cara Menjalankan
1. Pastikan library yang dibutuhkan sudah terinstall (`pip install tensorflow numpy pandas`).
2. Unduh dataset *rockpaperscissors* dari [Kaggle](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors).
3. Ekstrak dataset dan pastikan foldernya berada di satu lokasi dengan script ini dengan struktur direktori `rockpaperscissors/rock`, `rockpaperscissors/paper`, dan `rockpaperscissors/scissors`.
4. Jalankan script `Praktikum8.py`:
   ```bash
   python Praktikum8.py
   ```

## Hasil
Program akan melatih model selama 10 epoch menggunakan `ImageDataGenerator` dan akan menampilkan akurasi serta loss pada data validasi di akhir pelatihan. Hasil riwayat pelatihan lengkap dapat dilihat di `hasil.txt`.
