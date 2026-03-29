# ============================================================
# COLAB KURULUM: Veri setini otomatik indirme
# ============================================================
import os
import shutil
import glob
import zipfile

def setup_dataset():
    """HAM10000 veri setini indir ve hazırla."""
    if os.path.exists('dataset/HAM10000_metadata.csv'):
        print("✅ Dataset zaten mevcut, indirme atlanıyor.")
        return

    os.makedirs('dataset', exist_ok=True)

    # Kaggle üzerinden doğrudan indirme (opendatasets kütüphanesi)
    print("📥 Veri seti indiriliyor...")
    os.system('pip install -q opendatasets')
    import opendatasets as od
    od.download('https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000', data_dir='.')

    # İndirilen klasörün adını bul ve dosyaları taşı
    kaggle_dir = 'skin-cancer-mnist-ham10000'
    if os.path.exists(kaggle_dir):
        # CSV dosyasını taşı
        for csv_file in glob.glob(os.path.join(kaggle_dir, '*.csv')):
            shutil.copy(csv_file, 'dataset/')

        # Resimleri tek klasörde topla
        os.makedirs('dataset/images', exist_ok=True)

        # part_1 ve part_2 klasörlerindeki resimleri taşı
        for part_dir in glob.glob(os.path.join(kaggle_dir, 'HAM10000_images_part_*')):
            for img_file in glob.glob(os.path.join(part_dir, '*.jpg')):
                dest = os.path.join('dataset/images', os.path.basename(img_file))
                if not os.path.exists(dest):
                    shutil.move(img_file, dest)

        # Eğer resimler doğrudan klasör altındaysa
        for img_file in glob.glob(os.path.join(kaggle_dir, '*.jpg')):
            dest = os.path.join('dataset/images', os.path.basename(img_file))
            if not os.path.exists(dest):
                shutil.move(img_file, dest)

    img_count = len(glob.glob('dataset/images/*.jpg'))
    print(f"✅ Dataset hazır! {img_count} görüntü bulundu.")

# Veri setini indir ve hazırla
setup_dataset()

# ============================================================
# ANA KOD
# ============================================================
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Veriseti dizini
DATA_DIR = 'dataset'
CSV_PATH = os.path.join(DATA_DIR, 'HAM10000_metadata.csv')
IMG_DIR = os.path.join(DATA_DIR, 'images')

# Model parametreleri
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32
INITIAL_EPOCHS = 20   # Phase 1: Sadece head eğitimi
FINE_TUNE_EPOCHS = 20 # Phase 2: Fine-tuning
NUM_CLASSES = 7

# Sınıflar (Dataset etiketleri)
CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']


def load_data():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"{CSV_PATH} bulunamadı. Lütfen dataset'i indirip 'dataset' klasörüne yerleştirin.")

    df = pd.read_csv(CSV_PATH)
    df['image_path'] = df['image_id'].apply(lambda x: os.path.join(IMG_DIR, x + '.jpg'))

    # Sınıf dağılımını göster
    print("\nSınıf Dağılımı:")
    print(df['dx'].value_counts())
    print()

    # Veriyi eğitim ve doğrulama olarak ikiye ayırma
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['dx'])

    return train_df, val_df


def get_smoothed_class_weights(train_df, smoothing=0.5):
    """
    Yumuşatılmış sınıf ağırlıkları.
    smoothing=0 → tüm ağırlıklar eşit (1.0)
    smoothing=1 → tam 'balanced' (çok agresif)
    smoothing=0.5 → dengeli ve kararlı (önerilen)
    """
    class_counts = train_df['dx'].value_counts()
    total = len(train_df)
    n_classes = len(CLASSES)

    class_weight_dict = {}
    for i, cls in enumerate(CLASSES):
        count = class_counts.get(cls, 1)
        # Balanced weight hesapla, sonra smoothing uygula
        balanced_weight = total / (n_classes * count)
        # 1.0 (eşit) ile balanced_weight arasında interpolasyon
        smoothed_weight = 1.0 + smoothing * (balanced_weight - 1.0)
        class_weight_dict[i] = smoothed_weight

    print("\nSınıf Ağırlıkları (Yumuşatılmış):")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls}: {class_weight_dict[i]:.3f}")

    return class_weight_dict


def build_model(num_classes):
    # MobileNetV2 base model
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
    )

    # Phase 1: Özellik çıkarıcı katmanları dondur
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # Sınıflandırma başı (head)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def main():
    print("Veri Seti Yükleniyor...")
    train_df, val_df = load_data()

    print(f"Eğitim Seti Boyutu: {len(train_df)}")
    print(f"Doğrulama Seti Boyutu: {len(val_df)}")

    # === Yumuşatılmış Sınıf Ağırlıkları ===
    # smoothing=0.5 → dengeli ama eğitimi kararsız hale getirmeyecek kadar hafif
    class_weight_dict = get_smoothed_class_weights(train_df, smoothing=0.5)

    # === Veri Artırma (Data Augmentation) ===
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.85, 1.15],
        fill_mode='nearest'
    )

    # Validation seti için veri artırma YAPILMAZ
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="image_path",
        y_col="dx",
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASSES
    )

    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col="image_path",
        y_col="dx",
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASSES,
        shuffle=False
    )

    # ============================================================
    # PHASE 1: Head Eğitimi (Base model dondurulmuş)
    # ============================================================
    print("\n" + "="*50)
    print("PHASE 1: Head Eğitimi (Base model dondurulmuş)")
    print("="*50)

    model = build_model(NUM_CLASSES)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1
    )

    history1 = model.fit(
        train_generator,
        epochs=INITIAL_EPOCHS,
        validation_data=val_generator,
        callbacks=[early_stop, reduce_lr],
        class_weight=class_weight_dict
    )

    # ============================================================
    # PHASE 2: Fine-Tuning (Base model'in son katmanları açılıyor)
    # ============================================================
    print("\n" + "="*50)
    print("PHASE 2: Fine-Tuning")
    print("="*50)

    # MobileNetV2'nin son ~30 katmanını eğitilebilir yap (daha muhafazakâr)
    fine_tune_at = 120  # 155 katmandan sadece son 35'i açıyoruz
    for layer in model.layers[:fine_tune_at]:
        layer.trainable = False
    for layer in model.layers[fine_tune_at:]:
        layer.trainable = True

    # Fine-tuning için çok düşük learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stop_ft = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=7, restore_best_weights=True
    )
    reduce_lr_ft = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1
    )

    history2 = model.fit(
        train_generator,
        epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
        initial_epoch=len(history1.history['accuracy']),
        validation_data=val_generator,
        callbacks=[early_stop_ft, reduce_lr_ft],
        class_weight=class_weight_dict
    )

    # ============================================================
    # DEĞERLENDİRME
    # ============================================================
    print("\n" + "="*50)
    print("DEĞERLENDİRME SONUÇLARI")
    print("="*50)

    # Sınıflandırma raporu
    val_generator.reset()
    y_pred_probs = model.predict(val_generator, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = val_generator.classes

    print("\nSınıflandırma Raporu:")
    print(classification_report(y_true, y_pred, target_names=CLASSES))

    # Karmaşıklık matrisi
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Karmaşıklık Matrisi (Confusion Matrix)')
    plt.xlabel('Yapay Zekanın Tahminleri')
    plt.ylabel('Gerçekleşen Hastalıklar')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()

    # Eğitim grafiği
    acc = history1.history['accuracy'] + history2.history['accuracy']
    val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    loss = history1.history['loss'] + history2.history['loss']
    val_loss = history1.history['val_loss'] + history2.history['val_loss']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = range(len(acc))
    ft_start = len(history1.history['accuracy'])

    ax1.plot(epochs_range, acc, 'b-', label='Eğitim Başarısı (Train)')
    ax1.plot(epochs_range, val_acc, 'orange', label='Doğrulama Başarısı (Validation)')
    ax1.axvline(x=ft_start, color='green', linestyle='--', alpha=0.7, label='Fine-tuning Başlangıcı')
    ax1.set_title('Model Başarısı (Accuracy) Gelişimi')
    ax1.set_xlabel('Epok (Epoch)')
    ax1.set_ylabel('Başarı Oranı')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_range, loss, 'b-', label='Eğitim Kaybı (Train)')
    ax2.plot(epochs_range, val_loss, 'orange', label='Doğrulama Kaybı (Validation)')
    ax2.axvline(x=ft_start, color='green', linestyle='--', alpha=0.7, label='Fine-tuning Başlangıcı')
    ax2.set_title('Model Kayıpları (Loss) Gelişimi')
    ax2.set_xlabel('Epok (Epoch)')
    ax2.set_ylabel('Kayıp Oranı')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.show()

    # ============================================================
    # MODEL KAYDETME
    # ============================================================
    h5_model_path = 'skin_cancer_model.keras'
    model.save(h5_model_path)
    print(f"\nAna Model Kaydedildi: {h5_model_path}")

    print("Model TensorFlow Lite (.tflite) Formatına Dönüştürülüyor...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    tflite_model_path = 'skin_cancer_model.tflite'
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

    print(f"Mobil Uygulama Modeli Kaydedildi: {tflite_model_path}")
    print("İşlem Başarılı! 'skin_cancer_model.tflite' dosyasını Android/iOS uygulamanızda kullanabilirsiniz.")


if __name__ == '__main__':
    main()
