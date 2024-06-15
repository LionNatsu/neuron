import torch

dataset_label = [
    'permeability', 'charge', 'outside(mmol/L)', 'inside(mmol/L)',
]

# Equilibrium: Na+, K+, Cl-, A-
dataset = [
    [0, 1, 117, 30],
    [1, 1, 3, 90],
    [1, -1, 120, 4],
    [0, -1, 0, 116]
]

# Lower the concentration of Cl-
dataset = [
    [0, 1, 117, 30],
    [1, 1, 3, 90],
    [1, -1, 60, 4],
    [0, -1, 60, 116]
]

# Increase the concentration of K+
dataset = [
    [0, 1, 114, 30],
    [1, 1, 6, 90],
    [1, -1, 120, 4],
    [0, -1, 0, 116],
]

ions = torch.tensor(dataset, dtype=torch.float)
while True:
    # Potentials per ion (Nernst equation)
    E = 58 * (ions[:, 2:4] + 1e-10).log10().diff().T[0] * ions[:, 1]

    # Overall voltage
    V = (ions[:, 0] * E).sum() / ions[:, 0].sum()

    # Currents per ion
    i = ions[:, 0] * (V - E)

    P = ions[:, 2].sum() - ions[:, 3].sum()

    print('S:', V, ions[:, 3])
    print('i:', i)
    print('P:', P)

    inside = ions[:, 3]

    # Ions flow in or out, proportional to currents
    step_ions = 0.2
    # A reasonable approximation: do not change outside at all.
    # Only change the inside concentration.
    new_inside = inside + step_ions * i / ions[:, 1]

    step_water = 0.005
    # Water balances osmotic pressure
    new_inside *= 1 + step_water * P.atan()

    if (new_inside - inside).norm(p=1) < 0.05:
        break

    ions[:, 3] = new_inside
