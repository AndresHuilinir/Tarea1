import matplotlib.pyplot as plt
import numpy as np

# Definir el intervalo [0,2] con 500 muestras
t = np.linspace(0, 2, 500)  

# La función x(t)
x_t = np.sin(2*np.pi*3*t) + (1/5)*np.sin(2*np.pi*7*t) + (3/10)*np.sin(2*np.pi*11*t) + (1/5)*np.sin(2*np.pi*17*t)

X_k = np.fft.fft(x_t)   # Transformada Discreta de Fourier
N = len(X_k)            # Número de muestras

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

# Señal en el dominio del tiempo
ax1.plot(t, x_t, color='darkmagenta')
ax1.set_title('Gráfica de la función x(t)') 
ax1.set_xlabel('Tiempo (t)') 
ax1.set_ylabel('Señal x(t)') 
ax1.grid(True) 
ax1.axhline(y=0, color='black', linewidth=1) 
ax1.axvline(x=0, color='black', linewidth=1) 
ax1.legend(['x(t)']) 

# Espectro en el dominio de la frecuencia (índices de la DFT)
ax2.stem(range(N), np.abs(X_k), basefmt=" ")
ax2.set_title('Transformada Discreta de Fourier (DFT)') 
ax2.set_ylabel('Frecuencia (f)') 
ax2.grid(True) 

plt.tight_layout()
plt.show()
