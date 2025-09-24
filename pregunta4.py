import numpy as np
import librosa       # para cargar audio
import soundfile as sf  # para guardar audio filtrado
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---- cargar audio ----
# para leer la señal directamente en mono y con su frecuencia original
archivo_entrada = "Audio_Andres_Huilinir_Sanchez.wav"
senal, fs = librosa.load(archivo_entrada, sr=None, mono=True)

# guardar info básica
n_muestras = len(senal)
tiempo = np.arange(n_muestras) / fs   # vector de tiempo para graficar

# ---- transformada de fourier ----
# para pasar al dominio de frecuencia
espectro_original = np.fft.rfft(senal)
frecuencias = np.fft.rfftfreq(n_muestras, d=1/fs)

# ---- crear filtros ----
# pasa bajos: hasta 1000 Hz
mask_pasabajos = (frecuencias <= 1000)
# pasa banda: 300-3400 Hz
mask_pasabanda = (frecuencias >= 300) & (frecuencias <= 3400)
# pasa altos: desde 1000 Hz hacia arriba
mask_pasaaltos = (frecuencias >= 1000)

# ---- aplicar filtros ----
# multiplicar espectro por la máscara
espectro_baja = espectro_original * mask_pasabajos
espectro_banda = espectro_original * mask_pasabanda
espectro_alta = espectro_original * mask_pasaaltos

# ---- volver a tiempo con ifft ----
# para escuchar los resultados
senal_baja = np.fft.irfft(espectro_baja, n=n_muestras)
senal_banda = np.fft.irfft(espectro_banda, n=n_muestras)
senal_alta = np.fft.irfft(espectro_alta, n=n_muestras)

# ---- normalizar antes de guardar ----
def normalizar(x):
    max_abs = np.max(np.abs(x))
    if max_abs == 0:
        return x
    return (x / max_abs) * 0.99

# ---- guardar archivos filtrados ----
sf.write("Audio_Andres_Huilinir_Sanchez_baja.wav", normalizar(senal_baja), fs)   # guarda pasa bajos
sf.write("Audio_Andres_Huilinir_Sanchez_banda.wav", normalizar(senal_banda), fs) # guarda pasa banda
sf.write("Audio_Andres_Huilinir_Sanchez_alta.wav", normalizar(senal_alta), fs)   # guarda pasa altos

# ---- preparar espectros en dB para graficar ----
eps = 1e-12
mag_original_db = 20 * np.log10(np.abs(espectro_original) + eps)
mag_baja_db = 20 * np.log10(np.abs(espectro_baja) + eps)
mag_banda_db = 20 * np.log10(np.abs(espectro_banda) + eps)
mag_alta_db = 20 * np.log10(np.abs(espectro_alta) + eps)

# ---- graficar comparaciones ----
primeros_segundos = 5
idx_max = min(n_muestras, int(primeros_segundos * fs))

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "onda temporal: original vs pasa bajos",
        "espectro: original vs pasa bajos",
        "onda temporal: original vs pasa banda",
        "espectro: original vs pasa banda"
    )
)

# onda temporal: original vs pasa bajos
fig.add_trace(go.Scatter(x=tiempo[:idx_max], y=senal[:idx_max], name="original"), row=1, col=1) # original
fig.add_trace(go.Scatter(x=tiempo[:idx_max], y=senal_baja[:idx_max], name="pasa bajos"), row=1, col=1)

# espectro: original vs pasa bajos
fig.add_trace(go.Scatter(x=frecuencias, y=mag_original_db, name="original (dB)"), row=1, col=2)
fig.add_trace(go.Scatter(x=frecuencias, y=mag_baja_db, name="pasa bajos (dB)"), row=1, col=2)

# onda temporal: original vs pasa banda
fig.add_trace(go.Scatter(x=tiempo[:idx_max], y=senal[:idx_max], name="original", showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=tiempo[:idx_max], y=senal_banda[:idx_max], name="pasa banda"), row=2, col=1)

# espectro: original vs pasa banda
fig.add_trace(go.Scatter(x=frecuencias, y=mag_original_db, name="original (dB)", showlegend=False), row=2, col=2)
fig.add_trace(go.Scatter(x=frecuencias, y=mag_banda_db, name="pasa banda (dB)"), row=2, col=2)

fig.update_layout(height=800, width=1100, title="comparación: original vs filtros (pasa bajos y pasa banda)")
fig.show()

# ---- pasa altos separado ----
fig2 = make_subplots(rows=1, cols=2, subplot_titles=("onda temporal: original vs pasa altos", "espectro: original vs pasa altos"))
fig2.add_trace(go.Scatter(x=tiempo[:idx_max], y=senal[:idx_max], name="original"), row=1, col=1)
fig2.add_trace(go.Scatter(x=tiempo[:idx_max], y=senal_alta[:idx_max], name="pasa altos"), row=1, col=1)
fig2.add_trace(go.Scatter(x=frecuencias, y=mag_original_db, name="original (dB)"), row=1, col=2)
fig2.add_trace(go.Scatter(x=frecuencias, y=mag_alta_db, name="pasa altos (dB)"), row=1, col=2)
fig2.update_layout(height=400, width=1000, title="comparación: original vs pasa altos")
fig2.show()
