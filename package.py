import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

''' Sample Frequency: Vs = 1MHz (1000kHz), Amplitude = 2.5 MHz'''
'''loading data and adding time'''
#Data with Signal frequency across multiple Nyquist Zones

#Zone 0
with np.load('Z0_150kHz.npz') as data: #Vo = 150kHz 
    arr_z01 = data['arr_0']
with np.load('Z0_250kHz.npz') as data: #Vo = 250 kHz 
    arr_z02 = data['arr_0']
with np.load('Z0_350kHz.npz') as data: #Vo = 350 kHz
    arr_z03 = data['arr_0']
#Zone 1
with np.load('Z1_650kHz.npz') as data: #Vo = 650kHz 
    arr_z11 = data['arr_0']
with np.load('Z1_750kHz.npz') as data: #Vo = 750 kHz 
    arr_z12 = data['arr_0']
with np.load('Z1_850kHz.npz') as data: #Vo = 850 kHz
    arr_z13 = data['arr_0']
#Zone 2
with np.load('Z2_1150kHz.npz') as data: #Vo = 650kHz 
    arr_z21 = data['arr_0']
with np.load('Z2_1250kHz.npz') as data: #Vo = 750 kHz 
    arr_z22 = data['arr_0']
with np.load('Z2_1350kHz.npz') as data: #Vo = 850 kHz
    arr_z23 = data['arr_0']
#Zone 3
with np.load('Z3_1650kHz.npz') as data: #Vo = 650kHz 
    arr_z31 = data['arr_0']
with np.load('Z3_1750kHz.npz') as data: #Vo = 750 kHz 
    arr_z32 = data['arr_0']
with np.load('Z3_1850kHz.npz') as data: #Vo = 850 kHz
    arr_z33 = data['arr_0']

def time_axis(fs, N):
    return np.arange(N) / fs   # seconds

fs = 1e6

#time arrays
t_z01 = time_axis(fs, len(arr_z01[1])) * 1e3  # multiply by 1e3 for units of ms
t_z02 = time_axis(fs, len(arr_z02[1])) * 1e3
t_z03 = time_axis(fs, len(arr_z03[1])) * 1e3

t_z11 = time_axis(fs, len(arr_z11[1])) * 1e3
t_z12 = time_axis(fs, len(arr_z12[1])) * 1e3
t_z13 = time_axis(fs, len(arr_z13[1])) * 1e3

t_z21 = time_axis(fs, len(arr_z21[1])) * 1e3
t_z22 = time_axis(fs, len(arr_z22[1])) * 1e3
t_z23 = time_axis(fs, len(arr_z23[1])) * 1e3

t_z31 = time_axis(fs, len(arr_z31[1])) * 1e3
t_z32 = time_axis(fs, len(arr_z32[1])) * 1e3
t_z33 = time_axis(fs, len(arr_z33[1])) * 1e3


'''--------------------------------------------'''
'''Power and Voltage Spectra'''
dt = 1/fs

N = 2048

f_z01=150e6
f_z02=250e6
f_z03=350e6

f_z11=650e6 
f_z12=750e6
f_z13=850e6

f_z21=1150e6
f_z22=1250e6
f_z23=1350e6

f_z31=1650e6
f_z32=1750e6
f_z33=1850e6

# Fourier Transform to each data set
X_z01 = np.fft.fft(arr_z01[1])
X_z02 = np.fft.fft(arr_z02[1])
X_z03 = np.fft.fft(arr_z03[1])

X_z11 = np.fft.fft(arr_z11[1])
X_z12 = np.fft.fft(arr_z12[1])
X_z13 = np.fft.fft(arr_z13[1])

X_z21 = np.fft.fft(arr_z21[1])
X_z22 = np.fft.fft(arr_z22[1])
X_z23 = np.fft.fft(arr_z23[1])

X_z31 = np.fft.fft(arr_z31[1])
X_z32 = np.fft.fft(arr_z32[1])
X_z33 = np.fft.fft(arr_z33[1])

#shifting ft values
X_z01shift = np.fft.fftshift(X_z01)
X_z02shift = np.fft.fftshift(X_z02)
X_z03shift = np.fft.fftshift(X_z03)

X_z11shift = np.fft.fftshift(X_z11)
X_z12shift = np.fft.fftshift(X_z12)
X_z13shift = np.fft.fftshift(X_z13)

X_z21shift = np.fft.fftshift(X_z21)
X_z22shift = np.fft.fftshift(X_z22)
X_z23shift = np.fft.fftshift(X_z23)

X_z31shift = np.fft.fftshift(X_z31)
X_z32shift = np.fft.fftshift(X_z32)
X_z33shift = np.fft.fftshift(X_z33)

# Power spectra 
P_z01 = np.abs(X_z01shift)**2
P_z02 = np.abs(X_z02shift)**2
P_z03 = np.abs(X_z03shift)**2

P_z11 = np.abs(X_z11shift)**2
P_z12 = np.abs(X_z12shift)**2
P_z13 = np.abs(X_z13shift)**2

P_z21 = np.abs(X_z21shift)**2
P_z22 = np.abs(X_z22shift)**2
P_z23 = np.abs(X_z23shift)**2

P_z31 = np.abs(X_z31shift)**2
P_z32 = np.abs(X_z32shift)**2
P_z33 = np.abs(X_z33shift)**2

# Frequency
freq = np.fft.fftfreq(N, d=dt)
freq_shift = np.fft.fftshift(freq)

'''---------------------------------------------------------------------'''
'''theory alias frequency values vs. collected data comparison plot'''
'''---------------------------------------------------------------------'''
#note: AI helped me create the code for this plot (that is why it is much more efficient and less brute-force than the code for many of my other plots)'''

import numpy as np
import matplotlib.pyplot as plt

fs = 1e6  # Hz

datasets = [
    ("150 kHz", 150e3, arr_z01[1]),
    ("250 kHz", 250e3, arr_z02[1]),
    ("350 kHz", 350e3, arr_z03[1]),
    ("650 kHz", 650e3, arr_z11[1]),
    ("750 kHz", 750e3, arr_z12[1]),
    ("850 kHz", 850e3, arr_z13[1]),
    ("1150 kHz", 1150e3, arr_z21[1]),
    ("1250 kHz", 1250e3, arr_z22[1]),
    ("1350 kHz", 1350e3, arr_z23[1]),
    ("1650 kHz", 1650e3, arr_z31[1]),
    ("1750 kHz", 1750e3, arr_z32[1]),
    ("1850 kHz", 1850e3, arr_z33[1]),
]

def alias_theory(f0, fs):
    n = int(np.round(f0 / fs))
    fa = abs(f0 - n * fs)
    if fa > fs/2:
        fa = fs - fa
    return fa

def measured_alias_from_power(x, fs, remove_mean=True, fmin_hz=1e3):
    """
    Measure alias frequency from the POWER spectrum.
    - Uses rFFT (0..fs/2)
    - Ignores DC and optionally ignores very low frequencies (< fmin_hz)
      to avoid the huge peak at 0 Hz dominating.
    Returns: (f_peak_hz, df_hz)
    """
    x = np.asarray(x, dtype=float)
    if remove_mean:
        x = x - np.mean(x)

    N = len(x)
    df = fs / N

    X = np.fft.rfft(x)
    P = np.abs(X)**2
    f = np.fft.rfftfreq(N, d=1/fs)

    # mask out DC and very-low-f bins
    mask = f >= fmin_hz
    f2 = f[mask]
    P2 = P[mask]

    k = np.argmax(P2)
    f_peak = f2[k]

    return f_peak, df

# compute predicted and measured
f0s, fpreds, fmeas, dfs = [], [], [], []
for label, f0, sig in datasets:
    f_peak, df = measured_alias_from_power(sig, fs, remove_mean=True, fmin_hz=1e3)
    f0s.append(f0)
    fpreds.append(alias_theory(f0, fs))
    fmeas.append(f_peak)
    dfs.append(df)

f0s   = np.array(f0s)
fpreds = np.array(fpreds)
fmeas  = np.array(fmeas)
dfs    = np.array(dfs)

'''-------------------------------------'''
'''Autocorrelation Function'''
'''---------------------------------------------------------------------'''
x = arr_z02[1].astype(float)
x = x - np.mean(x) # to avoid DC dominating

N = len(x)


'''From time-domain'''
acf_td = np.correlate(x, x, mode='full')
lags_td = np.arange(-(N-1), N) * dt

# acf_td_norm = acf_td / np.max(acf_td) #normalizing for peak at 0

'''From power spectrum'''
# adding padding at beginning and end

Npad = 2*N

Xpad = np.fft.fft(x, n=Npad)
Ppad = np.abs(Xpad)**2

acf_fd = np.fft.ifft(Ppad).real 
acf_fd = np.fft.fftshift(acf_fd) #shifting
lags_fd = np.arange(-Npad//2, Npad//2) * dt

# mid = Npad // 2
# acf_fd_center = acf_fd[mid-(N-1): mid+N]
# lags_fd_center = lags_fd[mid-(N-1): mid+N]

# acf_fd_center_norm = acf_fd_center / np.max(acf_fd_center)

'''----------------------------------------------'''
'''LONG Autocorrelation function'''
'''-----------------------------------------------'''
# Zone 0
x1 = arr_z01[1].astype(float)
x1 = x1 - np.mean(x1)
N1 = len(x1)

acf_td1 = np.correlate(x1, x1, mode='full')
lags_td1 = np.arange(-(N1-1), N1) * dt

Npad1 = 2*N1
Xpad1 = np.fft.fft(x1, n=Npad1)
Ppad1 = np.abs(Xpad1)**2
acf_fd1 = np.fft.ifft(Ppad1).real
acf_fd1 = np.fft.fftshift(acf_fd1)
lags_fd1 = np.arange(-Npad1//2, Npad1//2) * dt

mid1 = Npad1//2
acf_fd1 = acf_fd1[mid1-(N1-1): mid1+N1]
lags_fd1 = lags_fd1[mid1-(N1-1): mid1+N1]

acf_td1 = acf_td1/np.max(acf_td1)
acf_fd1 = acf_fd1/np.max(acf_fd1)

# 250 kHz
x2 = arr_z02[1].astype(float); x2 = x2 - np.mean(x2)
N2 = len(x2)

acf_td2 = np.correlate(x2, x2, mode='full')
lags_td2 = np.arange(-(N2-1), N2) * dt

Npad2 = 2*N2
Xpad2 = np.fft.fft(x2, n=Npad2)
Ppad2 = np.abs(Xpad2)**2
acf_fd2 = np.fft.ifft(Ppad2).real
acf_fd2 = np.fft.fftshift(acf_fd2)
lags_fd2 = np.arange(-Npad2//2, Npad2//2) * dt

mid2 = Npad2//2
acf_fd2 = acf_fd2[mid2-(N2-1): mid2+N2]
lags_fd2 = lags_fd2[mid2-(N2-1): mid2+N2]

acf_td2 = acf_td2/np.max(acf_td2)
acf_fd2 = acf_fd2/np.max(acf_fd2)

# 350 kHz
x3 = arr_z03[1].astype(float); x3 = x3 - np.mean(x3)
N3 = len(x3)

acf_td3 = np.correlate(x3, x3, mode='full')
lags_td3 = np.arange(-(N3-1), N3) * dt

Npad3 = 2*N3
Xpad3 = np.fft.fft(x3, n=Npad3)
Ppad3 = np.abs(Xpad3)**2
acf_fd3 = np.fft.ifft(Ppad3).real
acf_fd3 = np.fft.fftshift(acf_fd3)
lags_fd3 = np.arange(-Npad3//2, Npad3//2) * dt

mid3 = Npad3//2
acf_fd3 = acf_fd3[mid3-(N3-1): mid3+N3]
lags_fd3 = lags_fd3[mid3-(N3-1): mid3+N3]

acf_td3 = acf_td3/np.max(acf_td3)
acf_fd3 = acf_fd3/np.max(acf_fd3)

# Zone 1
x4 = arr_z11[1].astype(float); x4 = x4 - np.mean(x4)
N4 = len(x4)

acf_td4 = np.correlate(x4, x4, mode='full')
lags_td4 = np.arange(-(N4-1), N4) * dt

Npad4 = 2*N4
Xpad4 = np.fft.fft(x4, n=Npad4)
Ppad4 = np.abs(Xpad4)**2
acf_fd4 = np.fft.ifft(Ppad4).real
acf_fd4 = np.fft.fftshift(acf_fd4)
lags_fd4 = np.arange(-Npad4//2, Npad4//2) * dt

mid4 = Npad4//2
acf_fd4 = acf_fd4[mid4-(N4-1): mid4+N4]
lags_fd4 = lags_fd4[mid4-(N4-1): mid4+N4]

acf_td4 = acf_td4/np.max(acf_td4)
acf_fd4 = acf_fd4/np.max(acf_fd4)

x5 = arr_z12[1].astype(float); x5 = x5 - np.mean(x5)
N5 = len(x5)

acf_td5 = np.correlate(x5, x5, mode='full')
lags_td5 = np.arange(-(N5-1), N5) * dt

Npad5 = 2*N5
Xpad5 = np.fft.fft(x5, n=Npad5)
Ppad5 = np.abs(Xpad5)**2
acf_fd5 = np.fft.ifft(Ppad5).real
acf_fd5 = np.fft.fftshift(acf_fd5)
lags_fd5 = np.arange(-Npad5//2, Npad5//2) * dt

mid5 = Npad5//2
acf_fd5 = acf_fd5[mid5-(N5-1): mid5+N5]
lags_fd5 = lags_fd5[mid5-(N5-1): mid5+N5]

acf_td5 = acf_td5/np.max(acf_td5)
acf_fd5 = acf_fd5/np.max(acf_fd5)

x6 = arr_z13[1].astype(float); x6 = x6 - np.mean(x6)
N6 = len(x6)

acf_td6 = np.correlate(x6, x6, mode='full')
lags_td6 = np.arange(-(N6-1), N6) * dt

Npad6 = 2*N6
Xpad6 = np.fft.fft(x6, n=Npad6)
Ppad6 = np.abs(Xpad6)**2
acf_fd6 = np.fft.ifft(Ppad6).real
acf_fd6 = np.fft.fftshift(acf_fd6)
lags_fd6 = np.arange(-Npad6//2, Npad6//2) * dt

mid6 = Npad6//2
acf_fd6 = acf_fd6[mid6-(N6-1): mid6+N6]
lags_fd6 = lags_fd6[mid6-(N6-1): mid6+N6]

acf_td6 = acf_td6/np.max(acf_td6)
acf_fd6 = acf_fd6/np.max(acf_fd6)

# Zone 2
x7 = arr_z21[1].astype(float); x7 = x7 - np.mean(x7)
N7 = len(x7)

acf_td7 = np.correlate(x7, x7, mode='full')
lags_td7 = np.arange(-(N7-1), N7) * dt

Npad7 = 2*N7
Xpad7 = np.fft.fft(x7, n=Npad7)
Ppad7 = np.abs(Xpad7)**2
acf_fd7 = np.fft.ifft(Ppad7).real
acf_fd7 = np.fft.fftshift(acf_fd7)
lags_fd7 = np.arange(-Npad7//2, Npad7//2) * dt

mid7 = Npad7//2
acf_fd7 = acf_fd7[mid7-(N7-1): mid7+N7]
lags_fd7 = lags_fd7[mid7-(N7-1): mid7+N7]

acf_td7 = acf_td7/np.max(acf_td7)
acf_fd7 = acf_fd7/np.max(acf_fd7)

x8 = arr_z22[1].astype(float); x8 = x8 - np.mean(x8)
N8 = len(x8)

acf_td8 = np.correlate(x8, x8, mode='full')
lags_td8 = np.arange(-(N8-1), N8) * dt

Npad8 = 2*N8
Xpad8 = np.fft.fft(x8, n=Npad8)
Ppad8 = np.abs(Xpad8)**2
acf_fd8 = np.fft.ifft(Ppad8).real
acf_fd8 = np.fft.fftshift(acf_fd8)
lags_fd8 = np.arange(-Npad8//2, Npad8//2) * dt

mid8 = Npad8//2
acf_fd8 = acf_fd8[mid8-(N8-1): mid8+N8]
lags_fd8 = lags_fd8[mid8-(N8-1): mid8+N8]

acf_td8 = acf_td8/np.max(acf_td8)
acf_fd8 = acf_fd8/np.max(acf_fd8)

x9 = arr_z23[1].astype(float); x9 = x9 - np.mean(x9)
N9 = len(x9)

acf_td9 = np.correlate(x9, x9, mode='full')
lags_td9 = np.arange(-(N9-1), N9) * dt

Npad9 = 2*N9
Xpad9 = np.fft.fft(x9, n=Npad9)
Ppad9 = np.abs(Xpad9)**2
acf_fd9 = np.fft.ifft(Ppad9).real
acf_fd9 = np.fft.fftshift(acf_fd9)
lags_fd9 = np.arange(-Npad9//2, Npad9//2) * dt

mid9 = Npad9//2
acf_fd9 = acf_fd9[mid9-(N9-1): mid9+N9]
lags_fd9 = lags_fd9[mid9-(N9-1): mid9+N9]

acf_td9 = acf_td9/np.max(acf_td9)
acf_fd9 = acf_fd9/np.max(acf_fd9)

# Zone 3
x10 = arr_z31[1].astype(float); x10 = x10 - np.mean(x10)
N10 = len(x10)

acf_td10 = np.correlate(x10, x10, mode='full')
lags_td10 = np.arange(-(N10-1), N10) * dt

Npad10 = 2*N10
Xpad10 = np.fft.fft(x10, n=Npad10)
Ppad10 = np.abs(Xpad10)**2
acf_fd10 = np.fft.ifft(Ppad10).real
acf_fd10 = np.fft.fftshift(acf_fd10)
lags_fd10 = np.arange(-Npad10//2, Npad10//2) * dt

mid10 = Npad10//2
acf_fd10 = acf_fd10[mid10-(N10-1): mid10+N10]
lags_fd10 = lags_fd10[mid10-(N10-1): mid10+N10]

acf_td10 = acf_td10/np.max(acf_td10)
acf_fd10 = acf_fd10/np.max(acf_fd10)

x11 = arr_z32[1].astype(float); x11 = x11 - np.mean(x11)
N11 = len(x11)

acf_td11 = np.correlate(x11, x11, mode='full')
lags_td11 = np.arange(-(N11-1), N11) * dt

Npad11 = 2*N11
Xpad11 = np.fft.fft(x11, n=Npad11)
Ppad11 = np.abs(Xpad11)**2
acf_fd11 = np.fft.ifft(Ppad11).real
acf_fd11 = np.fft.fftshift(acf_fd11)
lags_fd11 = np.arange(-Npad11//2, Npad11//2) * dt

mid11 = Npad11//2
acf_fd11 = acf_fd11[mid11-(N11-1): mid11+N11]
lags_fd11 = lags_fd11[mid11-(N11-1): mid11+N11]

acf_td11 = acf_td11/np.max(acf_td11)
acf_fd11 = acf_fd11/np.max(acf_fd11)

x12 = arr_z33[1].astype(float); x12 = x12 - np.mean(x12)
N12 = len(x12)

acf_td12 = np.correlate(x12, x12, mode='full')
lags_td12 = np.arange(-(N12-1), N12) * dt

Npad12 = 2*N12
Xpad12 = np.fft.fft(x12, n=Npad12)
Ppad12 = np.abs(Xpad12)**2
acf_fd12 = np.fft.ifft(Ppad12).real
acf_fd12 = np.fft.fftshift(acf_fd12)
lags_fd12 = np.arange(-Npad12//2, Npad12//2) * dt

mid12 = Npad12//2
acf_fd12 = acf_fd12[mid12-(N12-1): mid12+N12]
lags_fd12 = lags_fd12[mid12-(N12-1): mid12+N12]

acf_td12 = acf_td12/np.max(acf_td12)
acf_fd12 = acf_fd12/np.max(acf_fd12)


'''---------------------------------------'''
'''Leakage Power'''
'''---------------------------------------'''
import dft

# these values stay the same across all data sets
N = len(arr_z01[1]) # = 2048
t = np.arange(N) * dt # these stay the same across nyquist zones
f_fft = np.fft.fftfreq(N, d=dt)
f_fftshift = np.fft.fftshift(f_fft)

# plotting DFT for df >> fs/N
Nfreq = 20480 # trying 10*N (Nfreq >> N)
f_fine = np.linspace(-fs/2, fs/2, Nfreq, endpoint=False)


#DFT for 150kHz
f_out1, X_dft1 = dft.dft(x1, t=t, f=f_fine, vsamp=fs)
P_dft1 = np.abs(X_dft1)**2 
#FFT for 150kHz
x1 = arr_z01[1].astype(float) #rewriting for clarity
x1 = x1 - np.mean(x1)
X_fft1 = np.fft.fft(x1)
X_fftshift1 = np.fft.fftshift(X_fft1) # shifting
P_fftshift1 = np.abs(X_fftshift1)**2


#DFT for 650kHz
f_out6, X_dft6 = dft.dft(x6, t=t, f=f_fine, vsamp=fs)
P_dft6 = np.abs(X_dft6)**2 

#FFT for 650kHz
x6 = arr_z13[1].astype(float) 
x6 = x6 - np.mean(x6)
X_fft6 = np.fft.fft(x6)
X_fftshift6 = np.fft.fftshift(X_fft6)
P_fftshift6 = np.abs(X_fftshift6)**2


'''Frequency Resolution (5.6)'''





'''Noise at 3000000Hz'''



'''---------------------------------------'''
''' DSB Mixer'''
'''---------------------------------------'''
# data_high = np.load('run-sig1575kHz_1500kHz-sample1.0MHz-22mVpp.npz')
# print(data.files)
# data1_samprate = data[data.files[2]] #understanding what the data looks like
# print(data1_samprate)
# data1_sigrate = data[data.files[3]]
# print(data1_sigrate)
# data1_v_rms = data[data.files[4]]
# print(data1_v_rms)
# data1_v_max = data[data.files[5]]
# print(data1_v_max)
# data1_v_min = data[data.files[6]]
# print(data1_v_min)


#first configuration:
data1 = np.load('run-sig1575kHz_1500kHz-sample1.0MHz-22mVpp.npz')
x1 = data1['data_direct'][0].astype(float)
x1 = x1 - np.mean(x1)

fs1 = data1['sample_rate_mhz'] * 1e6   # 1 MHz
dt1 = 1/fs1

N1 = len(x1)
t1 = np.arange(N1) * dt

f1 = np.fft.fftfreq(N1, d=dt)
fshift1 = np.fft.fftshift(f1)

X1 = np.fft.fft(x1)
Xshift1 = np.fft.fftshift(X1)
P1 = np.abs(Xshift1)**2
P1 = P1 / np.max(P1)

# second set of data
data2 = np.load('run-sig1500kHz_1000kHz-sample2.0MHz-22mVpp.npz')
x2 = data2['data_direct'][0].astype(float)
x2 = x2 - np.mean(x2)

fs2 = data2['sample_rate_mhz'] * 1e6   # 1 MHz
dt2 = 1/fs2

N2 = len(x2)
t2 = np.arange(N2) * dt

f2 = np.fft.fftfreq(N2, d=dt)
fshift2 = np.fft.fftshift(f2)

X2 = np.fft.fftshift(np.fft.fft(x2))
Xshift2 = np.fft.fftshift(X2)
P2 = np.abs(X2)**2
P2 = P2 / np.max(P2)
