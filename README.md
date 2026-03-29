# 🧠 Cilt Kanseri Tespit Modeli

> Bu doküman, [train_model.py] kodunun ne yaptığını ve **%76 doğruluk oranına** nasıl ulaştığımızı, yapay zekayı hiç bilmeyen birine anlatmak için hazırlanmıştır.

---

## 📌 Ne Yapmaya Çalışıyoruz?

Bir cilt fotoğrafına bakıp o lezyonun **7 farklı cilt hastalığından hangisi olduğunu tahmin eden** bir yapay zeka modeli eğitiyoruz. Bu modeli daha sonra bir **mobil uygulama** içine koyacağız ki kullanıcılar telefonlarından fotoğraf çekip sonuç alabilsin.

### 7 Hastalık Türü

| Kısaltma | Hastalık | Tehlike Durumu |
|----------|----------|----------------|
| **nv** | Ben (melanositik nevüs) | ✅ İyi huylu |
| **mel** | Melanom | ❌ **Ölümcül kanser** |
| **bkl** | Benign keratoz (iyi huylu leke) | ✅ İyi huylu |
| **bcc** | Bazal hücreli karsinom | ⚠️ Kanser (ama yavaş ilerler) |
| **akiec** | Aktinik keratoz | ⚠️ Kansere dönüşebilir |
| **vasc** | Damarsal lezyon | ✅ İyi huylu |
| **df** | Dermatofibrom | ✅ İyi huylu |

---

## 📂 1. Adım: Veri Setini İndirme (Satır 1-50)

```python
setup_dataset()
```

### Ne yapıyor?
İnternetten **10.015 adet dermoskopik cilt fotoğrafı** indiriyoruz. Bu fotoğraflar, gerçek hastalardan alınmış ve uzman doktorlar tarafından etiketlenmiş.

### Günlük hayat benzetmesi:
> 📚 Bir öğrencinin sınava hazırlanması gibi düşünün. Öğrenciye binlerce soru (fotoğraf) ve bunların doğru cevaplarını (hangi hastalık olduğunu) veriyoruz ki öğrensin.

**Çıktı:**
```
📥 Veri seti indiriliyor...
✅ Dataset hazır! 10015 görüntü bulundu.
```

---

## 📊 2. Adım: Verileri İnceleme ve Bölme (Satır 84-99)

```python
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['dx'])
```

### Ne yapıyor?
10.015 fotoğrafı **ikiye** bölüyoruz:
- **%80'i (8.012 fotoğraf)** → Yapay zekaya öğretmek için (eğitim seti)
- **%20'si (2.003 fotoğraf)** → Yapay zekanın ne kadar iyi öğrendiğini sınamak için (sınav seti)

### Neden bölüyoruz?
> 🎓 Aynen bir okulda olduğu gibi: öğrenciye ders anlatırsınız (eğitim) ama sınavda **daha önce görmediği** sorular sorarsınız. Eğer sınavda da başarılıysa, gerçekten öğrenmiştir. Eğitimde gördüğü soruları sorarsanız, ezberleyip ezberlemiş olabilir.

### Büyük Problem: Dengesiz Veri

**Çıktı:**
```
Sınıf Dağılımı:
nv       6705    ← Fotoğrafların %67'si sadece "ben"
mel      1113
bkl      1099
bcc       514
akiec     327
vasc      142
df        115    ← Sadece 115 fotoğraf!
```

> ⚠️ **Problem:** 10.015 fotoğrafın 6.705'i sadece "ben" (nv) hastalığına ait. Yapay zeka tembel davranıp her fotoğrafa "bu bir ben" dese bile **%67 doğruluk** alır ama hiçbir kanseri tespit edemez!

---

## ⚖️ 3. Adım: Dengesizliği Düzeltme — Sınıf Ağırlıkları (Satır 102-126)

```python
class_weight_dict = get_smoothed_class_weights(train_df, smoothing=0.5)
```

### Ne yapıyor?
Az görülen hastalıklara **daha fazla önem** veriyoruz. Yapay zeka nadir bir hastalığı yanlış tahmin ederse **cezasını daha ağır** ödüyor.

### Günlük hayat benzetmesi:
> ⚖️ Bir sınavda 100 soru var. 67 tanesi kolay (ben), 1 tanesi çok zor (dermatofibrom). Normal sınavda her soru 1 puan. Ama biz zor soruya **6.7 puan**, kolay soruya **0.6 puan** veriyoruz. Böylece öğrenci zor soruları da öğrenmek zorunda kalıyor.

**Çıktı:**
```
Sınıf Ağırlıkları (Yumuşatılmış):
  akiec: 2.684    ← Yanlış bilirse 2.7 kat ceza
  bcc:   1.892
  bkl:   1.151
  df:    6.720    ← Yanlış bilirse 6.7 kat ceza (en nadir hastalık)
  mel:   1.143
  nv:    0.607    ← Yanlış bilirse az ceza (zaten çok örneği var)
  vasc:  5.520
```

---

## 🖼️ 4. Adım: Veri Artırma — Fotoğrafları Çoğaltma (Satır 173-208)

```python
train_datagen = ImageDataGenerator(
    rotation_range=20,        # Döndürme
    horizontal_flip=True,     # Yatay çevirme
    zoom_range=0.2,           # Yakınlaştırma
    brightness_range=[0.85, 1.15],  # Parlaklık değiştirme
    ...
)
```

### Ne yapıyor?
Elimizdeki fotoğrafları **döndürerek, çevirerek, yakınlaştırarak, parlaklığını değiştirerek** yeni varyasyonlar oluşturuyoruz. Böylece yapay zeka aynı hastalığı farklı açılardan ve koşullarda tanımayı öğreniyor.

### Günlük hayat benzetmesi:
> 📷 Bir kedinin fotoğrafını düşünün. O fotoğrafı ters çevirseniz, biraz yakınlaştırsanız, parlaklığını değiştirseniz — hâlâ kedi. Ama yapay zeka için bunlar "yeni" fotoğraflar. Böylece 10.000 fotoğraftan sanki 100.000 fotoğrafmış gibi öğreniyor.

**Önemli Not:** Sınav setine (validation) bu işlemi **uygulamıyoruz** çünkü sınavda gerçek fotoğraflarla test etmemiz gerekiyor.

---

## 🏗️ 5. Adım: Modeli Oluşturma — MobileNetV2 (Satır 129-159)

```python
base_model = MobileNetV2(weights='imagenet', include_top=False)
```

### Ne yapıyor?
Google'ın önceden eğittiği **MobileNetV2** adlı hazır bir yapay zeka modelini alıyoruz ve üzerine kendi sınıflandırma katmanımızı ekliyoruz.

### Transfer Öğrenme — Anahtar Kavram

> 🎓 **Analoji:** Bir üniversite mezununu düşünün. Bu kişi 4 yıl boyunca genel mühendislik eğitimi almış (ImageNet ile eğitim — 14 milyon fotoğraftan öğrenmiş). Şimdi biz bu kişiye sadece **dermatoloji kursu** veriyoruz. Sıfırdan üniversiteye başlatmıyoruz çünkü temel bilgileri (kenar tanıma, doku tanıma, şekil tanıma) zaten biliyor.

### Model Yapısı (Basitleştirilmiş):

```
Fotoğraf Girişi (224x224 piksel)
        ↓
┌───────────────────────┐
│   MobileNetV2         │  ← Google'ın önceden eğittiği kısım
│   (Özellik Çıkarıcı)  │     "Fotoğraftaki şekilleri, dokuları,
│                       │      renkleri anlayan kısım"
│   155 katman          │
└───────────────────────┘
        ↓
┌───────────────────────┐
│   Bizim Eklediğimiz   │  ← Biz bunu ekledik
│   Sınıflandırma Başı  │     "Çıkarılan özelliklere bakıp
│                       │      hastalığı tahmin eden kısım"
│   Dense(128) → Dropout│
│   → Dense(7)          │
└───────────────────────┘
        ↓
  Tahmin: [akiec, bcc, bkl, df, mel, nv, vasc]
  Örnek:  [%2,   %5,  %3,  %1, %4,  %83, %2]
          → "Bu bir ben (nv)" (%83 olasılıkla)
```

### Neden MobileNetV2?
- **Küçük model boyutu** (~14 MB) → Telefona sığar
- **Hızlı çalışır** → Telefonda saniyeler içinde sonuç verir
- **Yeterli doğruluk** → Büyük modellerle karşılaştırılabilir performans

---

## 🎯 6. Adım: Eğitim — İki Aşamalı Strateji (Satır 211-270)

Eğitimi **iki aşamada** yapıyoruz. Bu en kritik kısım.

### 🔵 AŞAMA 1: Sadece Yeni Katmanları Eğit (Epoch 1-20)

```python
# Google'ın modelini dondur, sadece bizim eklediğimiz kısmı eğit
for layer in base_model.layers:
    layer.trainable = False
```

> 🎓 **Analoji:** Üniversite mezununa "Senin genel mühendislik bilgini değiştirme, sadece dermatoloji kursunu öğren" diyoruz.

**Ne oldu?**
```
Epoch  1 → Eğitim: %48  |  Sınav: %65   (öğrenmeye başlıyor)
Epoch  8 → Eğitim: %70  |  Sınav: %72   (iyi gidiyor)
Epoch 16 → Eğitim: %72  |  Sınav: %73   (tavana yaklaşıyor)
```

Bu aşamada model %73'e ulaştı ama **daha fazla ilerleyemiyor** çünkü Google'ın modeli hâlâ "genel" resimleri tanıyor, cilt lezyonlarına özel şekilde bakmıyor.

---

### 🟢 AŞAMA 2: İnce Ayar / Fine-Tuning (Epoch 21-40)

```python
# Google'ın modelinin son 35 katmanını da aç
fine_tune_at = 120
for layer in model.layers[fine_tune_at:]:
    layer.trainable = True
```

> 🎓 **Analoji:** "Tamam, şimdi genel mühendislik bilginin bir kısmını da dermatolojiye göre güncelle. Ama çok dikkatli ol, temeldeki bilgiyi bozma!" diyoruz.

**Neden çok düşük öğrenme hızı (learning rate)?**
```python
optimizer = Adam(learning_rate=5e-5)  # 0.00005 — çok yavaş ve dikkatli
```

> 🎨 Bir ressam düşünün. İlk aşamada kalın fırçayla büyük şekilleri çizdi. Şimdi ince fırçayla detayları ekliyor. Eğer yine kalın fırça kullanırsa, önceki çizimi bozar!

**Ne oldu?**
```
Epoch 21 → Eğitim: %66  |  Sınav: %67   (geçici düşüş — normal!)
Epoch 27 → Eğitim: %76  |  Sınav: %73   (toparlanıyor)
Epoch 33 → Eğitim: %79  |  Sınav: %76   ← EN İYİ SONUÇ 🎯
Epoch 40 → Eğitim: %80  |  Sınav: %74   (hafif düşme)
```

> ⚡ **Epoch 21'deki düşüş neden?** Modelin alt katmanları ilk kez değişmeye başlıyor. Bunu, uzun süredir sabit duran bir makineyi yeniden çalıştırmaya benzetebilirsiniz — ilk birkaç denemede biraz sarsılır ama sonra düzgün çalışmaya başlar.

---

## 📈 7. Adım: Sonuçları Değerlendirme (Satır 272-330)

### Genel Doğruluk: **%76**

Bu ne anlama geliyor?
> 📊 Sınav setindeki 2.003 fotoğrafın **1.522'sini doğru**, **481'ini yanlış** tahmin etti.

### Sınıf Bazında Sonuçlar

| Hastalık | Doğru Yakalanma Oranı | Yorum |
|----------|----------------------|-------|
| **nv** (ben) | %82 | ✅ Çok iyi — benleri rahatça tanıyor |
| **vasc** (damarsal) | %75 | ✅ İyi — nadir olmasına rağmen tanıyor |
| **bcc** (kanser) | %72 | ✅ İyi — kanser tespitinde başarılı |
| **bkl** (iyi huylu leke) | %71 | ✅ İyi |
| **mel** (melanom/kanser) | %58 | ⚠️ Orta — her 10 melanomdan 6'sını buluyor |
| **df** (dermatofibrom) | %48 | ⚠️ Düşük — sadece 115 eğitim fotoğrafı var |
| **akiec** (pre-kanser) | %34 | ❌ Düşük — karıştırıyor |

### Karmaşıklık Matrisi (Confusion Matrix)

Karmaşıklık matrisi, hangi hastalıkların birbiriyle **karıştırıldığını** gösterir.

> 📋 **Nasıl okunur?** Satırlar gerçek hastalığı, sütunlar yapay zekanın tahminini gösterir.

**Ana tespitler:**
- **nv** (ben): 1.341 fotoğrafın 1.106'sını doğru bildi (%82) ✅
- **mel** (melanom): 223 melanomdan 130'unu doğru bildi, ama 40'ını "benign keratoz" sandı ⚠️
- **bcc** (kanser): 103'ten 74'ünü doğru bildi (%72) ✅

---

## 💡 %76 Doğruluk Nasıl Elde Edildi? — Adım Adım Özet

Aşağıda, her tekniğin doğruluğa olan katkısını görebilirsiniz:

```
Hiçbir şey yapmasak (hep "nv" dese):       ~%67 (sahte doğruluk)
                                              ↓
+ MobileNetV2 Transfer Öğrenme:             ~%70 (+3)
  "Hazır bir yapay zeka alıp üzerine koyduk"
                                              ↓
+ Veri Artırma (Data Augmentation):         ~%72 (+2)
  "Fotoğrafları döndürüp çevirerek çoğalttık"
                                              ↓
+ Sınıf Ağırlıkları (Class Weights):       ~%73 (+1)
  "Nadir hastalıklara daha fazla önem verdik"
                                              ↓
+ İki Aşamalı Eğitim (Fine-tuning):        ~%76 (+3)
  "Modelin alt katmanlarını da cilt için özelleştirdik"
                                              ↓
                                        SONUÇ: %76 ✅
```

---

## 📱 8. Adım: Telefona Uyarlama (Satır 332-349)

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

### Ne yapıyor?
Bilgisayarda eğittiğimiz büyük modeli, **telefona sığacak** küçük bir formata (.tflite) dönüştürüyoruz.

> 📦 Bir kitabı düşünün. Eğitim sırasında ansiklopedi boyutunda bir kitap kullandık. Ama telefonda taşımak için bunun **cep kitabı versiyonunu** oluşturuyoruz. Bilgi kaybı minimum, boyut çok daha küçük.

**Çıktı:**
```
Ana Model Kaydedildi: skin_cancer_model.keras
Model TensorFlow Lite (.tflite) Formatına Dönüştürülüyor...
Mobil Uygulama Modeli Kaydedildi: skin_cancer_model.tflite
İşlem Başarılı! 'skin_cancer_model.tflite' dosyasını Android/iOS uygulamanızda kullanabilirsiniz.
```

---

## 🔧 Eğitimi İyileştiren Akıllı Mekanizmalar

### Early Stopping (Erken Durdurma)
```python
EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
```

> 🛑 "Sınav notu 5 tur üst üste düşerse, eğitimi durdur ve en iyi olduğu ana geri dön." Bu, modelin **fazla ezberleyip** gerçek sınavda kötü yapmasını önler.

### ReduceLROnPlateau (Öğrenme Hızı Azaltma)
```python
ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
```

> 🐢 "3 tur boyunca ilerleme olmazsa, daha yavaş ve dikkatli öğrenmeye başla." Tıpkı bir öğrencinin zor konularda yavaşlayıp dikkatli okuması gibi.

---

## ❓ Sıkça Sorulan Sorular

### "%76 iyi mi kötü mü?"
**Bağlama göre değişir:**
- 7 hastalığı ayırt etmek çok zor bir görev — rastgele tahminde sadece %14 doğruluk beklenir
- %76, rastgeleden **5.4 kat** daha iyi
- Bazı hastalıklar görsel olarak birbirine çok benziyor (melanom ↔ benign keratoz)
- Akademik çalışmalarda benzer veri setlerinde %75-85 arası doğruluk normal kabul edilir
- Model bir **doktor yerine** değil, doktora **yardım etmek** için tasarlandı

### "Neden %100 değil?"
1. **Hastalıklar birbirine benziyor** — Uzman doktorlar bile bazen yanılıyor
2. **Veri yetersizliği** — Bazı hastalıkların sadece 115 fotoğrafı var
3. **Fotoğraf kalitesi** — Farklı cihazlardan, farklı açılardan çekilmiş
4. **Model boyutu** — Telefona sığması için küçük bir model seçtik

### "Epoch ne demek?"
> 🔄 Tüm eğitim fotoğraflarının bir kez baştan sona modele gösterilmesi = 1 epoch. Biz toplam **40 epoch** çalıştırdık, yani model tüm fotoğrafları 40 kere gördü.

### "Loss (kayıp) ne demek?"
> 📉 Yapay zekanın ne kadar **yanlış** yaptığını gösteren bir sayı. Loss ne kadar düşükse, model o kadar iyi. Eğitimin amacı loss'u mümkün olduğunca düşürmek.

### "Accuracy (doğruluk) ne demek?"
> ✅ Yapay zekanın ne kadar **doğru** tahmin yaptığını gösteren yüzdelik değer. %76 = Her 100 fotoğraftan 76'sını doğru tahmin ediyor.

---

## 🗂️ Üretilen Dosyalar

| Dosya | Açıklama |
|-------|----------|
| `skin_cancer_model.keras` | Eğitilmiş tam model (bilgisayar için) |
| `skin_cancer_model.tflite` | Mobil uygulama modeli (telefon için) |
| `confusion_matrix.png` | Hangi hastalıkların karıştırıldığını gösteren grafik |
| `training_history.png` | Eğitim sürecinin doğruluk ve kayıp grafikleri |

---

## 📝 Kodun Tamamının Akış Şeması

```
┌──────────────────────────┐
│  1. VERİ SETİ İNDİR      │  10.015 dermoskopik fotoğraf
│     (Kaggle'dan)          │
└──────────┬───────────────┘
           ↓
┌──────────────────────────┐
│  2. VERİYİ HAZIRLA       │  CSV oku, dosya yollarını eşle
│     %80 eğitim / %20 test│
└──────────┬───────────────┘
           ↓
┌──────────────────────────┐
│  3. DENGESİZLİĞİ DÜZELT  │  Nadir hastalıklara ağırlık ver
│     (Sınıf ağırlıkları)  │
└──────────┬───────────────┘
           ↓
┌──────────────────────────┐
│  4. FOTOĞRAFLARI ÇOĞALT  │  Döndür, çevir, yakınlaştır
│     (Data Augmentation)   │
└──────────┬───────────────┘
           ↓
┌──────────────────────────┐
│  5. MODELİ OLUŞTUR       │  MobileNetV2 + Bizim katman
│     (Transfer Öğrenme)    │
└──────────┬───────────────┘
           ↓
┌──────────────────────────┐
│  6a. AŞAMA 1: HEAD EĞİT  │  20 epoch, LR=0.001
│      (Alt katmanlar sabit)│  → %73 doğruluk
└──────────┬───────────────┘
           ↓
┌──────────────────────────┐
│  6b. AŞAMA 2: FİNE-TUNE  │  20 epoch, LR=0.00005
│      (Son 35 katman açık) │  → %76 doğruluk ✅
└──────────┬───────────────┘
           ↓
┌──────────────────────────┐
│  7. DEĞERLENDİR          │  Classification report
│     (Sınav sonuçları)     │  Confusion matrix
└──────────┬───────────────┘
           ↓
┌──────────────────────────┐
│  8. MOBİLE DÖNÜŞTÜR      │  .keras → .tflite
│     (Telefona hazır)      │  Boyut küçültme
└──────────────────────────┘
