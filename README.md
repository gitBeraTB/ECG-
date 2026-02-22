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

## Lisans
MIT License
