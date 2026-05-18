import os
# Menyembunyikan log TensorFlow agar output lebih bersih
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import logging
import warnings
logging.getLogger('absl').setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.FATAL)
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore

def main():
    print("*" * 60)
    print(" PROGRAM PRAKTIKUM 8: JST 3 (CONVOLUTIONAL NEURAL NETWORK) ")
    print("*" * 60)

    # 1. Menyiapkan Data Latih dan Validasi dengan ImageDataGenerator
    dataset_path = "./rockpaperscissors"
    
    print("\n--- [TAHAP 1] Inisialisasi Data Generator ---")
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.20
    )

    train_data = datagen.flow_from_directory(
        dataset_path,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        classes=['rock', 'paper', 'scissors'],
        subset='training'
    )

    valid_data = datagen.flow_from_directory(
        dataset_path,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        classes=['rock', 'paper', 'scissors'],
        subset='validation'
    )

    # 2. Menyusun Arsitektur Model CNN
    print("\n--- [TAHAP 2] Membangun Arsitektur CNN ---")
    model_cnn = Sequential([
        Input(shape=(150, 150, 3)),
        Conv2D(32, kernel_size=(3,3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3,3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=(3,3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(3, activation='softmax')
    ])

    print("\n> Ringkasan Model:")
    model_cnn.summary()

    # 3. Proses Kompilasi
    print("\n--- [TAHAP 3] Mengkompilasi Model ---")
    model_cnn.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    # 4. Pelatihan Model (Fitting)
    print("\n--- [TAHAP 4] Pelatihan Model (10 Epoch) ---")
    hist = model_cnn.fit(
        train_data,
        validation_data=valid_data,
        epochs=10
    )

    # 5. Evaluasi Kinerja
    print("\n--- [TAHAP 5] Hasil Evaluasi Data Validasi ---")
    v_loss, v_acc = model_cnn.evaluate(valid_data)
    print(f'Loss Validasi  : {v_loss}')
    print(f'Akurasi Validasi: {v_acc}')

    # 6. Prediksi
    print("\n--- [TAHAP 6] Uji Prediksi Model ---")
    pred = model_cnn.predict(valid_data)
    print("Array Hasil Prediksi (Probabilitas):")
    print(pred)

    print("\n[INFO] Eksekusi Praktikum 8 Selesai.")

if __name__ == "__main__":
    main()