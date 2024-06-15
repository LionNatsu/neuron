import torch

dataset_label = [
    'permeability', 'charge', 'outside(mmol/L)', 'inside(mmol/L)',
]

# Equilibrium: Na+, K+, Cl-, A-
dataset_eq = torch.tensor([
    [0, 1, 117, 30],
    [1, 1, 3, 90],
    [1, -1, 120, 4],
    [0, -1, 0, 116]
], dtype=torch.float)

# Lower the concentration of Cl-
dataset_lo_cl = torch.tensor([
    [0, 1, 117, 30],
    [1, 1, 3, 90],
    [1, -1, 60, 4],
    [0, -1, 60, 116]
], dtype=torch.float)

# Increase the concentration of K+
dataset_hi_k = torch.tensor([
    [0, 1, 114, 30],
    [1, 1, 6, 90],
    [1, -1, 120, 4],
    [0, -1, 0, 116],
], dtype=torch.float)


def calc_potential(ions):
    # Potentials per ion (Nernst equation)
    inside_outside = ions[:, 2:4]
    charge = ions[:, 1]
    permeability = ions[:, 0]
    electrode_potential = -58 * (inside_outside + 1e-10).log10().diff().T[0] * charge
    # Overall voltage
    voltage = (permeability * electrode_potential).sum() / permeability.sum()
    return electrode_potential, voltage


def iterate(ions):
    while True:
        E, V = calc_potential(ions)

        # Currents per ion
        i = ions[:, 0] * (V - E)

        # Osmotic pressure difference is proportional to mmol/L.
        P = ions[:, 2].sum() - ions[:, 3].sum()

        print('---')
        print('concn\t', ions[:, 3])
        print('poten\t', E)
        print('voltg\t', V)
        print('osmtp\t', P)
        print('crrnt\t', i)

        inside = ions[:, 3]

        # Ions flow in or out, proportional to currents
        step_ions = 0.2
        # A reasonable approximation: do not change outside at all.
        # Only change the inside concentration.
        new_inside = inside - step_ions * i / ions[:, 1]

        step_water = 0.005
        # Water balances osmotic pressure.
        # Use atan function to compress the range even harder.
        new_inside *= 1 + step_water * P.atan()

        if (new_inside - inside).norm(p=1) < 0.05:
            break

        ions[:, 3] = new_inside


iterate(dataset_lo_cl)
