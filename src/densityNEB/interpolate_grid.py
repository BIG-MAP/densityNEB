import ase
import ase.cell
import numpy as np
import torch
import torch.nn.functional

from ase.calculators.vasp import VaspChargeDensity

class Interpolator():
    def __init__(self, grid: torch.Tensor, cell: ase.cell.Cell):
        grid = torch.unsqueeze(torch.unsqueeze(grid, 0), 0)

        # Images processing follows the convention that axes are on the interval
        # [-1, +1] and -1 and +1 are distinct points.
        # By adding a circular padding of 1 we get the correct scaling of the grid on the open interval [0, 1)
        # to the grid on the closed interval [0,1]
        self.padded_grid = torch.nn.functional.pad(grid, (0,1)*3, mode="circular")

        self.inv_cell_T = torch.tensor(np.linalg.inv(np.transpose(cell.complete())), device=grid.device, dtype=grid.dtype)

    def __call__(self, input_points: torch.Tensor):
        assert len(input_points.shape) == 2, "We assume Nx3 tensor"
        scaled_pos = self.inv_cell_T.matmul(input_points.T).T # [N, 3]

        scaled_pos = torch.remainder(scaled_pos, 1.0) # periodic boundary conditions
        scaled_pos = (scaled_pos - 0.5)*2

        # Unsqueeze from [N, 3] to [N, 1, 1, 1, 3]
        scaled_pos = torch.unsqueeze(scaled_pos, 1)
        scaled_pos = torch.unsqueeze(scaled_pos, 1)
        scaled_pos = torch.unsqueeze(scaled_pos, 1) 

        # The axes in grid sample are flipped, so it is z, y, x instead of x, y, z
        grid = torch.swapaxes(self.padded_grid, 2, 4) # [1, 1, Z, Y, X]

        vals = []
        for b in range(scaled_pos.shape[0]):
            interp_vals = torch.nn.functional.grid_sample(grid, torch.unsqueeze(scaled_pos[b],0), align_corners=True, padding_mode="border")
            interp_vals = torch.squeeze(interp_vals)
            vals.append(interp_vals)

        return torch.stack(vals, 0)

def calculate_grid_pos(density, origin, cell):
    # Calculate grid positions
    ngridpts = np.array(density.shape)  # grid matrix
    grid_pos = np.meshgrid(
        np.arange(ngridpts[0]) / (density.shape[0]),
        np.arange(ngridpts[1]) / (density.shape[1]),
        np.arange(ngridpts[2]) / (density.shape[2]),
        indexing="ij",
    )
    grid_pos = np.stack(grid_pos, 3)
    grid_pos = np.dot(grid_pos, cell)
    grid_pos = grid_pos + origin
    return grid_pos


def main():
    import matplotlib.pyplot as plt

    filename = "calculations/Li3AgB_Pm-3m/charge_density/LOCPOT"
    vasp_charge = VaspChargeDensity(filename)
    density = vasp_charge.chg[-1]
    atoms = vasp_charge.atoms[-1]
    grid_pos = calculate_grid_pos(
        density, np.zeros(3), atoms.get_cell()
    )

    interp = Interpolator(torch.tensor(density), atoms.get_cell())

    #points = [
    #    [0.,0.,0.],
    #    [1.,1.,1.],
    #]
    #points = torch.tensor(points, dtype=interp.grid.dtype)

    line_in_space = grid_pos[:,0,0]
    density_on_line = density[:,0,0]

    points = torch.tensor(line_in_space, dtype=interp.grid.dtype)
    print(points.shape)
    interped = interp(points)

    plt.figure()
    plt.plot(line_in_space[:,0], density_on_line, '.-')
    plt.plot(points[:,0], interped, 'o-')
    plt.show()



    print(density[10,20,40], density[10,5,5])
    print(interped)
    breakpoint()

if __name__ == "__main__":
    main()
