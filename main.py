import torch
import torch.special
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.use("TkAgg")
palette = sns.color_palette("rocket_r")
torch.set_printoptions(precision=2, sci_mode=False)
torch.set_default_device('cuda')

V = torch.tensor([
    [-68.00],
    [-68.00],
    [-68.00],
    [-68.00]])

M = torch.tensor([
    [0.3, 0.0, 0.8],
    [0.4, 0.0, 0.8],
    [0.5, 0.0, 0.8],
    [0.5, 0.0, 0.6],
])

I_external = torch.zeros_like(V)

T_current = torch.tensor([
    [-77.00, 36.00],
    [55.00, 120.00],
    [-54.40, 0.30]]).T

T_gating_inf = torch.tensor([
    [-53.00, 1 / 15.00],
    [-40.00, 1 / 15.00],
    [-62.00, 1 / -7.00]]).T

T_gating_tau = torch.tensor([
    [-79.00, 1 / 50.00, 4.70, 1.10],
    [-38.00, 1 / 30.00, 0.46, 0.04],
    [-67.00, 1 / 20.00, 7.40, 1.20]]).T

C = 1.0
time_step = 0.01

t = 0
step = 0
data = {
    'time': [],
    'voltage': [],
    'exp': [],
}


while True:

    if 0 <= t and step % 10 == 0:
        for i in range(V.shape[0]):
            data['time'].append(t)
            data['voltage'].append(V[i].item())
            data['exp'].append(i)

    if step % 100 == 0:
        print(round(t), step, V.T)

    I = (V - T_current[0]) * T_current[1]
    G = torch.vstack([M[:, 0].pow(4), M[:, 1].pow(3) * M[:, 2], torch.ones(M.shape[0])]).T
    V_dot = (I_external - (I * G).sum(dim=1).unsqueeze(1)).div(C)

    gating_inf = torch.special.expit((V - T_gating_inf[0]) * T_gating_inf[1])
    gating_tau = (((V - T_gating_tau[0]) * T_gating_tau[1]).square().mul(-1).exp() * T_gating_tau[2] + T_gating_tau[3])
    M_dot = (gating_inf - M) / gating_tau


    if t > 50:
        break

    V = V + V_dot * time_step
    M = (M + M_dot * time_step).clip(0, 1)

    t += time_step
    step += 1

sns.relplot(data=data, x='time', y='voltage', hue='exp', palette=palette, kind='line')
print(V)
print(M)
plt.show()
