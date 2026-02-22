# 🫀 FPGA Tabanlı ECG/Kardiyak İzleme Sistemi

FPGA üzerinde gerçek zamanlı arritmia sınıflandırma ve kardiyak izleme projesi.

## Proje Yapısı

```
ECG/
├── data/                    # Ham ve işlenmiş ECG verileri
│   ├── raw/                 # PhysioNet'ten indirilen ham kayıtlar
│   └── processed/           # Ön-işlenmiş, segmentlenmiş veriler
├── model/                   # ML model eğitimi ve dışa aktarma
│   └── export/              # Quantize edilmiş modeller (.tflite, weights)
├── fpga/                    # FPGA tasarım dosyaları
│   ├── rtl/                 # Verilog/VHDL kaynak kodları
│   ├── hls/                 # Vitis HLS C/C++ kaynakları
│   ├── tb/                  # Testbench dosyaları
│   ├── constraints/         # .xdc pin kısıtlama dosyaları
│   └── vivado_project/      # Vivado proje dosyaları
├── app/                     # Companion uygulama
│   ├── ble_receiver/        # BLE üzerinden veri alma
│   └── dashboard/           # Web/mobil gösterge paneli
├── docs/                    # Dokümantasyon
│   ├── papers/              # Referans akademik makaleler
│   └── datasheets/          # Donanım datasheet'leri
├── scripts/                 # Yardımcı scriptler
└── .agent/workflows/        # Proje workflow rehberi
```

## Hızlı Başlangıç

### 1. Python Ortamı
```bash
python -m venv ecg_env
ecg_env\Scripts\activate
pip install tensorflow numpy scipy wfdb matplotlib scikit-learn neurokit2
```

### 2. Veri İndirme
```python
import wfdb
wfdb.dl_database('mitdb', dl_dir='data/raw/mitdb')
```

### 3. Proje Seviyeleri
| Seviye | Süre | Açıklama |
|--------|------|----------|
| 1 | 1-3 ay | Akademik Prototip — Arritmia sınıflandırıcı on FPGA |
| 2 | 3-6 ay | Giyilebilir Prototip — Sürekli izleme + BLE |
| 3 | 6-12 ay | Klinik Sistem — Çok kanallı akıllı Holter |
| 4 | 12+ ay | Ürünleşme — Wellness device olarak pazar girişi |

## Hedef Donanım
- **FPGA**: Pynq-Z2 (Zynq-7020) veya Nexys A7 (Artix-7)
- **ECG Frontend**: AD8232 analog modül
- **İletişim**: BLE (HM-10 / nRF52832)

## 🚀 Eklenebilecek Özellikler

### Seviye 1: Akademik Prototip
- **Transfer Learning**: PTB-XL üzerinde pretrain edilmiş model → MIT-BIH'e fine-tune.
- **Explainability (XAI)**: Grad-CAM ile modelin ECG'nin neresine baktığını görselleştirme.
- **Multi-model karşılaştırma**: Aynı FPGA üzerinde 1D-CNN vs LSTM vs Random Forest benchmark'ı.
- **Edge AI benchmark süiti**: Latency-accuracy Pareto eğrisi.
- **Otomatik rapor oluşturucu**: Model performansını LaTeX/PDF olarak dışa aktaran script.

### Seviye 2: Giyilebilir Prototip
- **Stres seviyesi tahmini**: HRV'den LF/HF oranı → stres sınıflandırması.
- **Uyku kalitesi analizi**: HRV trendi ile uyku fazlarını (derin/hafif/REM) tahmin etme.
- **Hareket artefaktı reddi**: Akselerometre ile fiziksel hareket gürültüsünü filtreleme.
- **Kablosuz OTA güncelleme**: BLE üzerinden FPGA bitstream veya model ağırlıkları güncelleme.
- **Çoklu kullanıcı profili**: Farklı kişiler için kalibrasyon ve baseline kaydetme.
- **Gerçek zamanlı ECG streaming**: WebSocket üzerinden canlı ECG görselleştirme.

### Seviye 3: Klinik Sistem
- **Miyokard iskemisi tespiti**: ST-segment depresyonu → erken uyarı sistemi.
- **QT uzaması izleme**: İlaç yan etkisi takibi.
- **Atriyal flutter tespiti**: Sawtooth patern tanıma.
- **Pacemaker spike tespiti**: Pacemaker hastalarında doğru beat sınıflandırma.
- **Federated Learning**: Cihazlar arası veri paylaşımı olmadan model güncelleme.
- **Edge-Cloud hibrit mimari**: Basit tespitler FPGA'da, karmaşık analiz bulutta.

### Seviye 4: Ürünleşme
- **Aile sağlık paneli**: Aile fertlerinin kalp sağlığını merkezden izleme paneli.
- **Acil durum SOS**: Kritik arritmi tespiti durumunda otomatik bildirim/yardım çağrısı.
- **EHR Entegrasyonu**: HL7 FHIR standardı ile hastane sistemlerine veri gönderimi.
- **Telemonitor modu**: Doktorun hastayı uzaktan gerçek zamanlı izlemesi.
- **AI Chatbot entegrasyonu**: ECG verilerini doğal dil ile sorgulama.
- **Yaşlı düşme tespiti**: Akselerometre + ECG birleşimi ile düşme tespiti.

## Lisans
MIT License
