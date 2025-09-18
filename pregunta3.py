import matplotlib.pyplot as plt
import numpy as np

# Definir el intervalo [0,2] con 500 muestras
t = np.linspace(0, 2, 500)  
dt = t[1] - t[0]  # Paso temporal
fs = 1/dt # Frecuencia de muestreo

# La señal x(t)
x_t = np.sin(2*np.pi*3*t) + (1/5)*np.sin(2*np.pi*7*t) + (3/10)*np.sin(2*np.pi*11*t) + (1/5)*np.sin(2*np.pi*17*t)

X_k = np.fft.fft(x_t) # Transformada discreta de Fourier
N = len(X_k) # Numero de muestras
freqs = np.fft.fftfreq(N, d=dt) # Vector de frecuencias en Hz

# Normalizacion de la fft
X_mag = np.abs(X_k)/N

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

# Señal en el dominio del tiempo
ax1.plot(t, x_t, color='darkmagenta')
ax1.set_title('Gráfica de la señal x(t)') 
ax1.set_xlabel('Tiempo (t)') 
ax1.set_ylabel('Señal x(t)') 
ax1.grid(True) 
ax1.axhline(y=0, color='black', linewidth=1) 
ax1.axvline(x=0, color='black', linewidth=1) 

# Espectro en el dominio de la frecuencia (Hz)
ax2.plot(freqs, X_mag)
ax2.set_title('Espectro de la señal') 
ax2.set_xlabel('Frecuencia (Hz)')
ax2.set_ylabel('Amplitud') 
ax2.grid(True) 
ax2.axhline(y=0, color='black', linewidth=1) 
ax2.axvline(x=0, color='black', linewidth=1) 
ax2.set_xlim(-25, 25)

plt.tight_layout()
plt.show()
