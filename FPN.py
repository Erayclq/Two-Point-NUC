import numpy as np
import matplotlib.pyplot as plt
import warnings

# --- 1. NUC SINIFIMIZ  ---
class TwoPointNUC:
    def __init__(self, width=640, height=480):
        self.gain_map = np.ones((height, width), dtype=np.float32)
        self.offset_map = np.zeros((height, width), dtype=np.float32)
        self.is_calibrated = False

    def calibrate(self, cold_frame, hot_frame):
        target_cold = np.mean(cold_frame)
        target_hot = np.mean(hot_frame)
        target_diff = target_hot - target_cold

        pixel_diff = hot_frame.astype(np.float32) - cold_frame.astype(np.float32)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw_gain = target_diff / pixel_diff

        self.gain_map = np.where(pixel_diff == 0, 1.0, raw_gain)
        self.offset_map = target_cold - (self.gain_map * cold_frame)
        self.is_calibrated = True

    def apply_nuc(self, raw_frame):
        if not self.is_calibrated:
            return raw_frame
        corrected = (raw_frame.astype(np.float32) * self.gain_map) + self.offset_map
        np.clip(corrected, 0, 65535, out=corrected)
        return corrected.astype(np.uint16)

# --- 2. SENTETİK TEST ORTAMI ---
WIDTH, HEIGHT = 200, 200

# Adım A: Sensörün fiziksel kusurlarını simüle et (Gizli hatalar)
# Gain hataları 0.8 ile 1.2 arasında, Offset hataları -30 ile +30 arasında olsun
true_gain_error = np.random.uniform(0.8, 1.2, (HEIGHT, WIDTH))
true_offset_error = np.random.uniform(-30, 30, (HEIGHT, WIDTH))

def boz_goruntuyu(ideal_goruntu):
    """Kusursuz bir görüntüyü alıp, sensörün fiziksel hatalarıyla kirletir"""
    bozuk = (ideal_goruntu * true_gain_error) + true_offset_error
    return bozuk

# Adım B: Kalibrasyon için düz "Kara Cisim" (Blackbody) referansları üret
ideal_soguk = np.full((HEIGHT, WIDTH), 50.0)  # Gerçekte 50 birimlik soğuk yüzey
ideal_sicak = np.full((HEIGHT, WIDTH), 200.0) # Gerçekte 200 birimlik sıcak yüzey

# Sensör bu düz yüzeylere bakınca hataları yüzünden bozuk okur
bozuk_soguk = boz_goruntuyu(ideal_soguk)
bozuk_sicak = boz_goruntuyu(ideal_sicak)

# Adım C: Kameranın bakacağı asıl manzarayı (İdeal Görüntü) üret
# Ortası sıcak, kenarları soğuk dairesel bir ısı dağılımı (Gaussian Blob) çizelim
X, Y = np.meshgrid(np.linspace(-1, 1, WIDTH), np.linspace(-1, 1, HEIGHT))
ideal_manzara = np.exp(-(X**2 + Y**2) * 5) * 200 + 30 
bozuk_manzara = boz_goruntuyu(ideal_manzara)

# --- 3. ALGORİTMAYI ÇALIŞTIRMA VE TEST ETME ---
nuc_sistemi = TwoPointNUC(width=WIDTH, height=HEIGHT)

# 1. Kamerayı fabrikada kalibre et
nuc_sistemi.calibrate(bozuk_soguk, bozuk_sicak)

# 2. Kalibre edilmiş kamerayla bozuk manzarayı temizle
temizlenmis_manzara = nuc_sistemi.apply_nuc(bozuk_manzara)

# --- 4. GÖRSELLEŞTİRME VE KARŞILAŞTIRMA ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. Beklenen Çıktı (Ground Truth)
axes[0].imshow(ideal_manzara, cmap='inferno', vmin=0, vmax=255)
axes[0].set_title("1. İdeal Manzara (Beklenen)")
axes[0].axis('off')

# 2. Ham Girdi (Fixed Pattern Noise içeren)
axes[1].imshow(bozuk_manzara, cmap='inferno', vmin=0, vmax=255)
axes[1].set_title("2. Sensörün Gördüğü (Bozuk Girdi)")
axes[1].axis('off')

# 3. Bizim Çıktımız (NUC Uygulanmış)
axes[2].imshow(temizlenmis_manzara, cmap='inferno', vmin=0, vmax=255)
axes[2].set_title("3. NUC Çiktisi (Temizlenmiş)")
axes[2].axis('off')

plt.tight_layout()
plt.show()

# Matematiksel Karşılaştırma (Mean Squared Error)
fark = np.mean((ideal_manzara - temizlenmis_manzara) ** 2)
print(f"İdeal Görüntü ile NUC Çiktisi Arasındaki Ortalama Hata (MSE): {fark:.4f}")
