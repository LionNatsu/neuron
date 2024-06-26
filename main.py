import cv2
import numpy as np
import torch
import torch.special
import torchvision.transforms
# import matplotlib
# import matplotlib.pyplot as plt
# import seaborn as sns
import time

torch.set_printoptions(precision=2, sci_mode=False)
# torch.set_default_dtype(torch.bfloat16)
torch.set_default_device('cuda')
torch.manual_seed(0)
# matplotlib.use("TkAgg")
# palette = sns.color_palette("rocket_r")

N = 1000

# V = torch.zeros([N * N]).unsqueeze(1) - 68.0
# M = torch.tensor([[0.29, 0.08, 0.66]]).repeat(V.shape[0], 1)
# I_external = torch.zeros_like(V)

V = torch.randn([N * N]).unsqueeze(1) * 5 - 68.0
M = torch.tensor([[0.29, 0.08, 0.66]]).repeat(V.shape[0], 1)
I_external = torch.randn_like(V) * 0.1

T_current = torch.tensor([
    [-77.00, 36.00],
    [55.00, 120.00],
    [-68.00, 0.30]]).T

T_gating_inf = torch.tensor([
    [-53.00, 1 / 15.00],
    [-40.00, 1 / 10.00],
    [-62.00, 1 / -7.00]]).T

T_gating_tau = torch.tensor([
    [-79.00, 1 / 50.00, 4.70, 1.10],
    [-38.00, 1 / 30.00, 0.46, 0.04],
    [-67.00, 1 / 20.00, 7.40, 1.20]]).T

Blur = torchvision.transforms.GaussianBlur(kernel_size=[3, 3])

C = torch.tensor(1.0)
time_step = torch.tensor(0.0167)
t = 0.0

step = 0

# data = {
#     'time': [],
#     'voltage': [],
#     'exp': [],
# }

frame = []

start_time = time.perf_counter_ns()
while True:

    # if 0 <= t and step % 10 == 0:
    #     for i in range(V.shape[0]):
    #         data['time'].append(t)
    #         data['voltage'].append(V[i].item())
    #         data['exp'].append(i)

    # frame.append(V)

    if step % 10 == 0:
        img = V.reshape([N, N])
        img = (img / 120 * 127) + 127
        img = img.clip(0, 255).float().cpu().numpy().astype(np.uint8)
        frame.append(img)

    if 0 <= t < 2:
        I_external[N * N // 2 + N // 2] = 50

    # print(V, M)
    if step % 1000 == 0:
        img = V.reshape([N, N])
        img = (img / 120 * 127) + 127
        img = img.clip(0, 255).float().cpu().numpy().astype(np.uint8)
        cv2.imshow('update', cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS))
        cv2.waitKey(1)
        print(round((time.perf_counter_ns() - start_time) / 1000 / 1000), step, t)

    I_internal = (V - T_current[0]) * T_current[1]
    G = torch.vstack([M[:, 0].pow(4), M[:, 1].pow(3) * M[:, 2], torch.ones(M.shape[0])]).T
    I_total = I_external - (I_internal * G).sum(dim=1).unsqueeze(1)
    I_total = Blur(I_total.reshape([1, N, N])).reshape([N * N]).unsqueeze(1)
    V_dot = I_total.div(C)

    gating_inf = torch.special.expit((V - T_gating_inf[0]) * T_gating_inf[1])
    gating_tau = (((V - T_gating_tau[0]) * T_gating_tau[1]).square().mul(-1).exp() * T_gating_tau[2] + T_gating_tau[3])
    M_dot = (gating_inf - M) / gating_tau

    if t > 1200:
        break

    V = V + V_dot * time_step
    M = (M + M_dot * time_step).clip(0, 1)
    t += time_step.item()

    step += 1

writer = cv2.VideoWriter('pacemaker.mp4', cv2.VideoWriter.fourcc(*'avc1'), 60.0, (N, N))
writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)
for img in frame:
    writer.write(cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS))
writer.release()

print(round((time.perf_counter_ns() - start_time) / 1000 / 1000))
print(V.sum())

print(V, M)

# sns.relplot(data=data, x='time', y='voltage', hue='exp', palette=palette, kind='line')
# plt.show()
# for i in range(len(frame)):
#     frame[i] = frame[i].float().reshape([N, N]).cpu().numpy()
#
# while True:
#     for V in frame:
#         cv2.imshow('activation', V)
#         cv2.waitKey(1)
