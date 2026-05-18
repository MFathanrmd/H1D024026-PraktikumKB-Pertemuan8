# Praktikum Kecerdasan Buatan - Pertemuan 8

Repositori ini berisi implementasi Convolutional Neural Network (CNN) untuk mengklasifikasi gambar Rock-Paper-Scissors (batu-gunting-kertas) menggunakan TensorFlow dan Keras. Tugas ini merupakan bagian dari praktikum Kecerdasan Buatan (Pertemuan 8).

## Deskripsi Tugas
Tugas ini mengharuskan pembuatan model CNN yang dapat mengenali gambar tangan yang membentuk batu, kertas, atau gunting. Dataset yang digunakan adalah dataset rockpaperscissors yang diunduh dari Kaggle.

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
- Dataset "rockpaperscissors" diletakkan di dalam folder yang sama dengan script.

## Cara Menjalankan
1. Pastikan library yang dibutuhkan sudah terinstall (`pip install tensorflow numpy pandas`).
2. Pastikan dataset rockpaperscissors sudah berada di folder proyek dengan struktur `rockpaperscissors/rock`, `rockpaperscissors/paper`, dan `rockpaperscissors/scissors`.
3. Jalankan script `Praktikum8.py`:
   ```bash
   python Praktikum8.py
   ```

## Hasil
Program akan melatih model selama 10 epoch menggunakan `ImageDataGenerator` dan akan menampilkan akurasi serta loss pada data validasi di akhir pelatihan.
