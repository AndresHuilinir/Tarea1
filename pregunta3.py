import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

# Definir el intervalo [0,2] con 500 muestras
t = np.linspace(0, 2, 500)  
dt = t[1] - t[0]  # Paso temporal
fs = 1/dt # Frecuencia de muestreo

# La señal x(t)
x_t_1 = np.sin(2*np.pi*3*t)
x_t_2 = (1/5)*np.sin(2*np.pi*7*t)
x_t_3 = (3/10)*np.sin(2*np.pi*11*t)
x_t_4 = (1/5)*np.sin(2*np.pi*17*t)

x_t = x_t_1 + x_t_2 + x_t_3 + x_t_4

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

fig3d = go.Figure()

# Señal en el tiempo
fig3d.add_trace(go.Scatter3d(
    x=t, y=np.zeros_like(t), z=x_t,
    mode='lines', line=dict(color='darkmagenta', width=10),
    name='Señal en tiempo'
))

# Señales separadas
lista_senos = [x_t_1,x_t_2,x_t_3,x_t_4]
lista_n_freq = [3,7,11,17]
lista_colores =['#008000','#009900','#00ba00','#00f300']
lista_nombres =["Sen(2π⋅3t)","(1/5)Sen(2π⋅7t)","(3/10)Sen(2π⋅11t)","(1/5)Sen(2π⋅17t)"]


for i in range(4):
    fig3d.add_trace(go.Scatter3d(
        x=t, y=np.full_like(t, lista_n_freq[i]), z=lista_senos[i],
        mode='lines', line=dict(color=lista_colores[i],width=5),
        name=lista_nombres[i]
    ))
    fig3d.add_trace(go.Scatter3d(
        x=t, y=np.full_like(t, -1*lista_n_freq[i]), z=lista_senos[i],
        mode='lines', line=dict(color=lista_colores[i], width=5),
        showlegend=False
    ))

# Espectro de la señal
fig3d.add_trace(go.Scatter3d(
    x=np.zeros_like(freqs), y=freqs, z=X_mag,
    mode='lines', line=dict(color='steelblue', width=10),
    name='Espectro FFT'
))

# Linea x
fig3d.add_trace(go.Scatter3d(
    x=[0, 2], y=[0, 0], z=[0, 0],
    mode='lines', line=dict(color='black', width=10),
    showlegend=False
))

# Linea y
fig3d.add_trace(go.Scatter3d(
    x=[0, 0], y=[-25, 25], z=[0, 0],
    mode='lines', line=dict(color='black', width=4),
    showlegend=False
))

# Linea z
fig3d.add_trace(go.Scatter3d(
    x=[0, 0], y=[0, 0], z=[min(np.min(x_t), np.min(X_mag)), max(np.max(x_t), np.max(X_mag))],
    mode='lines', line=dict(color='black', width=4),
    showlegend=False
))

# Ajustes de layout
fig3d.update_layout(
    scene=dict(
        xaxis_title='Tiempo (t)',
        yaxis_title='Frecuencia (Hz)',
        zaxis_title='Amplitud',
        xaxis=dict(range=[0,2]),
        yaxis=dict(range=[-25,25]),
    ),
    width=900,
    height=600
)

fig3d.show(renderer="browser")