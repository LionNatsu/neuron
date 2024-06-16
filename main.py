#import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

V, n, m, h = 0, 0.3181736168219989, 0.0531348913086363, 0.5875799149114899

C = 1.0

gK, gNa, gL = 36, 120, 0.3
EK, ENa, EL = -12, 120, 10.6

emu_rate = 0.001
min_step = 0.01

t = -10.0
step = 0
t_all = []
V_all = []

while True:
    print(V, n, m, h)

    if 0 <= t and step % 10 == 0:
        t_all.append(t)
        V_all.append(V)

    if 10 <= t < 10.5:
        I = 20.0
    else:
        I = 0

    if step % 1000 == 0:
        plt.plot(t_all, V_all, color='blue')
        plt.pause(0.001)

    V_dot = I - gK * n ** 4 * (V - EK) - gNa * m ** 3 * h * (V - ENa) - gL * (V - EL)

    alpha_n = lambda V: 0.01 * (10 - V) / (np.exp((10 - V) / 10) - 1)
    beta_n = lambda V: 0.125 * np.exp(-V / 80)
    alpha_m = lambda V: 0.1 * (25 - V) / (np.exp((25 - V) / 10) - 1)
    beta_m = lambda V: 4 * np.exp(-V / 18)
    alpha_h = lambda V: 0.07 * np.exp(- V)
    beta_h = lambda V: 1 / (np.exp((30 - V) / 10) + 1)

    n_dot = alpha_n(V) * (1 - n) - beta_n(V) * n
    m_dot = alpha_m(V) * (1 - m) - beta_m(V) * m
    h_dot = alpha_h(V) * (1 - h) - beta_h(V) * h

    if t > 50: # np.abs(np.array([V_dot, n_dot, m_dot, h_dot])).max() < min_step:
        break

    if np.isnan(V_dot):
        break

    V += V_dot * emu_rate
    n += n_dot * emu_rate
    m += m_dot * emu_rate
    h += h_dot * emu_rate
    t += emu_rate
    step += 1

    n = np.clip(n, 0, 1)
    m = np.clip(m, 0, 1)
    h = np.clip(h, 0, 1)

plt.plot(t_all, V_all, color='blue')
plt.pause(0.001)
plt.show()