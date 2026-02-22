---
description: FPGA Tabanlı ECG/Kardiyak İzleme Projesi — Tam Geliştirme Workflow'u
---

# 🫀 FPGA Tabanlı ECG/Kardiyak İzleme Projesi — Workflow

Bu workflow, projenin 4 seviyesini (Akademik Prototip → Ürünleşme) kapsayan adım adım geliştirme rehberidir.

---

## Ön Hazırlık (Hafta 0)

### Geliştirme Ortamı Kurulumu

1. **Python ortamı kur**
   ```bash
   python -m venv ecg_env
   ecg_env\Scripts\activate
   pip install tensorflow numpy scipy wfdb matplotlib scikit-learn pandas
   ```
   > `wfdb` — PhysioNet veritabanlarını okumak için resmi Python kütüphanesi

2. **FPGA araçlarını kur**
   - Xilinx Vivado + Vitis HLS (ücretsiz WebPACK sürümü yeterli)
   - Hedef cihaz: Zynq-7020 (Pynq-Z2) veya Artix-7 (Basys3/Nexys A7)
   - Pynq-Z2 kullanılacaksa: Pynq SD kart imajını hazırla (pynq.io)

3. **Donanım temin et**
   | Parça | Tahmini Fiyat | Kaynak |
   |-------|--------------|--------|
   | AD8232 ECG Modülü + Elektrotlar | ~5-10$ | AliExpress, Amazon |
   | Pynq-Z2 Geliştirme Kartı | ~100-120$ | TUL, Digilent |
   | Artix-7 (Nexys A7 / Basys3) alternatif | ~150-250$ | Digilent |
   | Tek-kullanımlık ECG elektrotları (50 adet) | ~5$ | Tıbbi malzeme |
   | BLE modülü (Seviye 2+) | ~5-10$ | HM-10 veya nRF52 |

4. **Proje dizin yapısını oluştur**
   ```
   ECG/
   ├── data/                    # Ham ve işlenmiş veri
   │   ├── raw/                 # PhysioNet'ten indirilen ham veriler
   │   └── processed/           # Ön-işlenmiş veriler
   ├── model/                   # ML model eğitimi
   │   ├── train.py
   │   ├── evaluate.py
   │   ├── quantize.py
   │   └── export/              # Dışa aktarılan modeller (.tflite, .h5, weights)
   ├── fpga/                    # FPGA tasarımları
   │   ├── rtl/                 # Verilog/VHDL kaynak dosyaları
   │   │   ├── ecg_top.v
   │   │   ├── qrs_detector.v
   │   │   ├── cnn_accelerator.v
   │   │   └── uart_tx.v
   │   ├── hls/                 # Vitis HLS C/C++ kaynakları
   │   │   ├── cnn_inference.cpp
   │   │   └── cnn_inference.h
   │   ├── tb/                  # Testbench dosyaları
   │   ├── constraints/         # .xdc pin tanımları
   │   └── vivado_project/      # Vivado proje dosyaları
   ├── app/                     # Companion mobil/web uygulaması
   │   ├── ble_receiver/
   │   └── dashboard/
   ├── docs/                    # Dokümantasyon
   │   ├── architecture.md
   │   ├── papers/              # Referans makaleler
   │   └── datasheets/          # AD8232, XADC vb. datasheet'leri
   ├── scripts/                 # Yardımcı scriptler
   └── README.md
   ```

---

## SEVİYE 1: Akademik Prototip (Hafta 1-12)

### Faz 1A: Veri Hazırlığı (Hafta 1-2)

1. **PhysioNet MIT-BIH veritabanını indir**
   ```python
   import wfdb
   # MIT-BIH Arrhythmia Database — 48 kayıt, 360 Hz örnekleme
   wfdb.dl_database('mitdb', dl_dir='data/raw/mitdb')
   ```

2. **Verileri ön-işle**
   - Bandpass filtre (0.5-40 Hz) ile gürültü temizliği
   - R-peak etrafında beat segmentasyonu (±128 veya ±180 örnek)
   - AAMI standardına göre 5 sınıfa haritalama:
     - **N** — Normal beat
     - **S** — Supraventriküler ektopik
     - **V** — Ventriküler ektopik (PVC)
     - **F** — Füzyon beat
     - **Q** — Bilinmeyen / paced beat
   - Sınıf dengesizliği için SMOTE veya oversampling

3. **Eğitim/test bölümlemesi**
   - Hasta bazlı bölümleme (data leakage önleme)
   - DS1 (eğitim): Kayıt 101, 106, 108, 109, 112, ...
   - DS2 (test): Kayıt 100, 103, 105, 111, 113, ...

### Faz 1B: Model Eğitimi (Hafta 3-5)

1. **1D-CNN modeli tasarla ve eğit**
   ```python
   # Hafif mimari — FPGA'e taşınabilir olması için
   model = tf.keras.Sequential([
       tf.keras.layers.Conv1D(16, 7, activation='relu', input_shape=(256, 1)),
       tf.keras.layers.MaxPooling1D(2),
       tf.keras.layers.Conv1D(32, 5, activation='relu'),
       tf.keras.layers.MaxPooling1D(2),
       tf.keras.layers.Conv1D(64, 3, activation='relu'),
       tf.keras.layers.GlobalAveragePooling1D(),
       tf.keras.layers.Dense(32, activation='relu'),
       tf.keras.layers.Dropout(0.3),
       tf.keras.layers.Dense(5, activation='softmax')
   ])
   ```

2. **Alternatif: Lightweight LSTM**
   - Bidirectional LSTM (32 units) + Dense katman
   - Daha az parametre, ancak FPGA'de CNN'e göre daha zor implemente edilir

3. **Baseline sonuçları kaydet**
   - Accuracy, Precision, Recall, F1-Score (sınıf bazlı)
   - Confusion matrix
   - Hedef: >95% overall accuracy, >85% V-sınıfı recall

### Faz 1C: Model Quantization (Hafta 5-6)

1. **Post-Training Quantization (INT8)**
   ```python
   converter = tf.lite.TFLiteConverter.from_keras_model(model)
   converter.optimizations = [tf.lite.Optimize.DEFAULT]
   
   # Representative dataset ile tam INT8 quantization
   def representative_dataset():
       for data in calibration_data:
           yield [data.astype(np.float32)]
   
   converter.representative_dataset = representative_dataset
   converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
   converter.inference_input_type = tf.int8
   converter.inference_output_type = tf.int8
   
   tflite_model = converter.convert()
   ```

2. **Quantization sonrası doğrulama**
   - FP32 vs INT8 accuracy karşılaştırması (<%2 düşüş kabul edilebilir)
   - Model boyutu karşılaştırması

3. **Ağırlıkları dışa aktar**
   - HLS/RTL'de kullanılmak üzere weights'leri .h veya .mem dosyasına yaz

### Faz 1D: FPGA Implementasyon (Hafta 6-10)

1. **Yol A: Vitis HLS ile (önerilen, daha hızlı)**
   - C/C++ ile CNN inference fonksiyonu yaz
   - `#pragma HLS PIPELINE` ve `#pragma HLS ARRAY_PARTITION` ile optimize et
   - Fixed-point (ap_fixed<16,8>) kullan
   - C/RTL Co-simulation ile doğrula
   - IP olarak export et → Vivado'ya entegre et

2. **Yol B: Elle yazılmış RTL ile (daha fazla kontrol)**
   - Her katman için ayrı modül: `conv1d_layer.v`, `maxpool_layer.v`, `dense_layer.v`
   - FSM ile katmanlar arası veri akışı kontrolü
   - Testbench ile Python çıktılarıyla karşılaştır
   - Ağırlıklar BRAM'e yüklenecek

3. **Zynq entegrasyonu (Pynq-Z2)**
   - XADC üzerinden AD8232 analog sinyalini oku
   - AXI-Lite ile PS↔PL iletişimi
   - Pynq overlay olarak yükle
   - Jupyter Notebook ile kontrol ve görselleştirme

### Faz 1E: Test ve Benchmark (Hafta 10-12)

1. **Performans ölçümleri**
   - **Latency**: Tek bir beat'in sınıflandırılma süresi (hedef: <10ms)
   - **Throughput**: Saniyede kaç beat sınıflandırılabilir
   - **Güç tüketimi**: Vivado Power Report
   - **Kaynak kullanımı**: LUT, FF, DSP, BRAM yüzdeleri

2. **Karşılaştırma tablosu oluştur**
   | Metrik | FPGA (Zynq) | GPU (Jetson Nano) | MCU (STM32H7) |
   |--------|-------------|-------------------|----------------|
   | Latency | ? ms | ? ms | ? ms |
   | Güç | ? mW | ? mW | ? mW |
   | Accuracy | ? % | ? % | ? % |
   | Boyut | ? mm² | ? mm² | ? mm² |

3. **Makale/rapor taslağını yaz**

---

## SEVİYE 2: Giyilebilir Prototip (Hafta 13-26)

### Faz 2A: Donanım Tasarımı (Hafta 13-16)

1. **Özel PCB tasarımı (KiCad)**
   - AD8232 ECG analog frontend
   - Zynq-7010 veya iCE40 UltraPlus (düşük güç FPGA)
   - nRF52832 BLE SoC
   - LiPo pil yönetim devresi (BQ25185)
   - MicroSD kart arayüzü
   - Toplam PCB boyutu hedefi: 40x60mm

2. **Güç bütçesi hesapla**
   | Bileşen | Aktif | Uyku |
   |---------|-------|------|
   | FPGA | ~50mW | ~5mW |
   | AD8232 | ~0.3mW | ~0.3mW |
   | BLE | ~15mW (TX) | ~0.01mW |
   | SD Kart | ~30mW (yazma) | ~0.1mW |
   | **Toplam** | ~95mW | ~5.4mW |
   - 200mAh LiPo ile: ~7.5 saat aktif, ~5 gün uyku

### Faz 2B: Gerçek Zamanlı Sinyal İşleme (Hafta 16-20)

1. **R-peak algılama modülü (RTL)**
   - Pan-Tompkins algoritması FPGA implementasyonu
   - Bandpass filtre → Türev → Kare alma → Hareketli ortalama → Eşikleme
   - Adaptif eşik mekanizması

2. **HRV hesaplama modülü**
   - RR-interval hesaplama (ardışık R-peak'ler arası süre)
   - Zaman-domain metrikler: SDNN, RMSSD, pNN50
   - Frekans-domain: LF/HF oranı (basit FFT veya Welch metodu)

3. **Anomali tespit**
   - AF taraması: RR-interval irregülaritesi (Shannon entropy)
   - Bradikarsi/taşikardi alarmları
   - PVC sayısı ve yüzdesi

### Faz 2C: BLE İletişim ve Uygulama (Hafta 20-24)

1. **FPGA → BLE veri protokolü**
   - UART üzerinden FPGA → nRF52 iletişimi
   - BLE GATT servisleri tanımla:
     - Heart Rate Service (0x180D) — standart BLE profili
     - Custom ECG Data Service — ham sinyal streaming
     - Alert Service — anomali bildirimleri

2. **Mobil uygulama (Flutter veya React Native)**
   - Gerçek zamanlı ECG sinyal görselleştirme
   - HR ve HRV trend grafikleri
   - Push notification ile anomali alarmları
   - Günlük/haftalık sağlık raporu

### Faz 2D: Güç Optimizasyonu (Hafta 24-26)

1. **FPGA düşük güç stratejileri**
   - Clock gating: Sinyal yokken saat sinyalini durdur
   - Duty cycling: Her 1 saniyede 0.1 saniyelik analiz penceresi
   - Dinamik frekans ölçekleme

2. **Sistem seviyesi güç yönetimi**
   - Uyku modları: deep sleep ↔ active (motion-triggered wake-up)
   - BLE advertisement interval optimizasyonu
   - SD karta batch yazma (sürekli yazma yerine tampon kullan)

---

## SEVİYE 3: Klinik Değer Taşıyan Sistem (Hafta 27-52)

### Faz 3A: Çok Kanallı ECG (Hafta 27-34)

1. **12-derivasyonlu ölçüm sistemi**
   - 3 adet ADS1298 (Texas Instruments) — 8 kanallı 24-bit ADC
   - Eşzamanlı örnekleme, 500 Hz/kanal
   - SPI üzerinden FPGA'e veri transferi

2. **Paralel sinyal işleme pipeline**
   ```
   Kanal 1 ──→ [Filtre] → [QRS Tespit] → [ST Analiz] ─→ ╗
   Kanal 2 ──→ [Filtre] → [QRS Tespit] → [ST Analiz] ─→ ║
   ...                                                      ║→ [Birleştirme] → [Sınıflandırma]
   Kanal 12 ─→ [Filtre] → [QRS Tespit] → [ST Analiz] ─→ ╝
   ```

3. **ST-segment analizi**
   - ST elevasyonu/depresyonu tespiti (miyokard enfarktüsü göstergesi)
   - J-noktası tespiti
   - ST-trend izleme

### Faz 3B: Gelişmiş ML Modelleri (Hafta 34-40)

1. **Multi-lead CNN**
   - 12 kanal girişli 2D-CNN (zaman × kanal)
   - Daha kapsamlı arritmi sınıflandırma (15+ sınıf)

2. **Temporal analiz**
   - Uzun süreli ritim bozukluğu tespiti
   - Paroksismal AF tanımlama
   - QT uzaması tespiti

### Faz 3C: Doktor Rapor Sistemi (Hafta 40-46)

1. **Otomatik rapor oluşturma**
   - 24 saatlik özet: toplam beat sayısı, ortalama HR, min/max HR
   - Arritmi olayları listesi (zaman damgalı)
   - ST-segment değişiklikleri
   - HRV trendi
   - PDF çıktısı

### Faz 3D: Klinik Validasyon (Hafta 46-52)

1. **Hastane iş birliği**
   - Etik kurul onayı
   - Kontrollü ortamda test (en az 50 hasta)
   - Referans cihaz ile karşılaştırma (sensitivity, specificity)
   - Bland-Altman analizi

---

## SEVİYE 4: Ürünleşme Yolu (Hafta 52+)

### Düzenleyici ve Ticari

1. **CE işaretlemesi süreci** (Wellness device olarak)
   - Risk değerlendirmesi (ISO 14971)
   - Teknik dosya hazırlığı
   - "Bu bir tıbbi cihaz değildir" disclaimer'ı

2. **Türkiye destek başvuruları**
   - TÜBİTAK 1512 Girişimcilik Desteği (150K TL hibe)
   - KOSGEB Ar-Ge İnovasyon Desteği
   - Teknopark şirket kurulumu

3. **Üretim hazırlığı**
   - Industrial PCB tasarımı (DFM)
   - Kasa tasarımı (3D baskı → enjeksiyon kalıp)
   - EMC testleri
   - Küçük seri üretim (100-500 adet pilot)

---

## Genel Kurallar

- Her faz sonunda `docs/` altına ilerleme raporu yaz
- Git ile versiyon kontrolü kullan (ancak büyük veri dosyalarını `.gitignore`'a ekle)
- Testbench'leri her RTL modülü için yaz — CI/CD pipeline'a entegre et
- Haftalık hedefler belirle ve `README.md`'de güncelle
