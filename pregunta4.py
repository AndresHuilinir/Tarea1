import numpy as np
import librosa     
import soundfile as sf  
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import matplotlib.pyplot as plt

# Una mausquerramienta que no ayudara mas tarde *risa del miki maus*
def normalizar(x):
    max_abs = np.max(np.abs(x))
    if max_abs == 0:
        return x
    return (x / max_abs) * 0.99

# Cargar audio 
archivo_entrada = "Audio_Andres_Huilinir_Sanchez.wav"
senal, fs = librosa.load(archivo_entrada, sr=None, mono=True)

# Guardar info básica
n_muestras = len(senal)
tiempo = np.arange(n_muestras) / fs   # Vector de tiempo para graficar

# Transformada de Fourier
espectro_original = np.fft.rfft(senal)
frecuencias = np.fft.rfftfreq(n_muestras, d=1/fs)

# Filtros 
tipo = ["baja", "banda", "alta"]
rangos = [(frecuencias <= 1000),(frecuencias >= 300) & (frecuencias <= 3400),(frecuencias >= 1000)]
senales_filtradas = {}
espectros_filtrados = {}
for i in range(len(tipo)):
    senal_filtrada = np.fft.irfft(espectro_original*rangos[i], n=n_muestras)
    senales_filtradas[tipo[i]] = senal_filtrada
    espectros_filtrados[tipo[i]] = espectro_original*rangos[i]
    archivo_salida = f"Audio_Andres_Huilinir_Sanchez_{tipo[i]}.wav"
    if not os.path.exists(archivo_salida):
        sf.write(archivo_salida, normalizar(senal_filtrada), fs)

# Magnitudes 
eps = 1e-12
mag_original_db = 20*np.log10(np.abs(espectro_original) + eps)
comparaciones = [
    ("Pasa bajos", senales_filtradas["baja"],  20*np.log10(np.abs(espectros_filtrados["baja"]) + eps)),
    ("Pasa banda", senales_filtradas["banda"], 20*np.log10(np.abs(espectros_filtrados["banda"]) + eps)),
    ("Pasa altos", senales_filtradas["alta"], 20*np.log10(np.abs(espectros_filtrados["alta"]) + eps))
]
idx_max = min(n_muestras, int(5*fs))

for nombre, senal_f, mag_f in comparaciones:
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    axs[0].plot(tiempo[:idx_max], senal[:idx_max], label="Original", color=plt.cm.inferno(0.2))
    axs[0].plot(tiempo[:idx_max], senal_f[:idx_max], label=nombre, color=plt.cm.inferno(0.7))
    axs[0].set_title(f"Onda temporal: Original vs {nombre}")
    axs[0].set_xlabel("Tiempo [s]")
    axs[0].set_ylabel("Amplitud")
    axs[0].legend()
    axs[1].plot(frecuencias, mag_original_db, label="Original (dB)", color=plt.cm.inferno(0.2))
    axs[1].plot(frecuencias, mag_f, label=f"{nombre} (dB)", color=plt.cm.inferno(0.7))
    axs[1].set_title(f"Espectro: Original vs {nombre}")
    axs[1].set_xlabel("Frecuencia [Hz]")
    axs[1].set_ylabel("Magnitud [dB]")
    axs[1].legend()
    plt.suptitle(f"Comparación: Original vs {nombre}")
    plt.tight_layout()
    plt.show()