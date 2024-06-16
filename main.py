import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

V, n, m, h = -65, 0.3181736168219989, 0.0531348913086363, 0.5875799149114899

C = 1.0
I = 0

gK, gNa, gL = 36, 120, 0.3
EK, ENa, EL = -77, 55, -54.4

emu_rate = 0.001
min_step = 0.01

t = 0
step = 0
t_all = []
V_all = []

def gating_var_inf(voltage: torch.Tensor, half_voltage, slope_factor):
    return voltage.sub(half_voltage).mul(-1.0 / slope_factor).exp().add(1).reciprocal()
def gating_var_tau(voltage: torch.Tensor, max_voltage, deviation, tau_base, tau_amp):
    return voltage.sub(max_voltage).square().mul(-1.0 / deviation ** 2).exp().mul(tau_amp).add(tau_base)

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
    Vt = torch.tensor(V)
    n_dot = gating_var_inf(Vt, -53, 15).sub(n).div(gating_var_tau(Vt, -79, 50, 4.7, 1.1)).item()
    m_dot = gating_var_inf(Vt, -40, 15).sub(m).div(gating_var_tau(Vt, -38, 30, 0.46, 0.04)).item()
    h_dot = gating_var_inf(Vt, -62, -7).sub(h).div(gating_var_tau(Vt, -67, 20, 7.4, 1.2)).item()

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