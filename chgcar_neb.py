from pathlib import Path

import ase.data
import matplotlib.pyplot as plt
import numpy as np
import torch
from ase.calculators.vasp import VaspChargeDensity
from densityNEB import fill, torchneb
from densityNEB.interpolate_grid import Interpolator, calculate_grid_pos
from mayavi import mlab

mlab.options.offscreen = True


def run_neb(vasp_charge_file: Path, output_dir: Path):
    vasp_charge = VaspChargeDensity(vasp_charge_file)
    density = vasp_charge.chg[-1]
    atoms = vasp_charge.atoms[-1]
    grid_pos = calculate_grid_pos(
        density, np.zeros(3), atoms.get_cell()
    )

    interp = Interpolator(torch.tensor(density), atoms.get_cell())

    atomic_numbers = atoms.get_atomic_numbers().tolist()
    occurences = sorted(
        [(x, atomic_numbers.count(x)) for x in set(atomic_numbers)],
        key=lambda x: x[1], reverse=True
    )
    target_atoms = occurences[0][0]

    li_indices = np.nonzero(atoms.get_atomic_numbers() == target_atoms)[0]
    li_indices = li_indices[0:2]
    li_positions = atoms.get_positions()[
        atoms.get_atomic_numbers() == target_atoms
    ]
    li_positions = li_positions[0:2]

    li_average = np.mean(li_positions, axis=0, keepdims=True)

    init_neb_positions = li_positions*0.90 + 0.10*li_average

    # Plot initial neb trajectory
    figure = mlab.figure()
    x, y, z = init_neb_positions.T
    gx, gy, gz = grid_pos[..., 0], grid_pos[..., 1], grid_pos[..., 2]
    lower_bound = np.log10(max(np.min(density), 0.0) + 0.001)
    upper_bound = min(0, np.log10(np.max(density) - 0.1))
    contours = np.logspace(lower_bound, upper_bound, 5).tolist()
    mlab.contour3d(
        gx, gy, gz, density, contours=contours, vmax=1.0, vmin=0.01,
        opacity=0.5, figure=figure
    )
    mlab.plot3d(x, y, z, figure=figure)

    momentum = 1e-2 if target_atoms == ase.data.atomic_numbers["Ca"] else 1e-1
    neb_optim_config = torchneb.OptimConfig(
        2000, torch.optim.SGD, {"lr": 0.05, "momentum": momentum}
    )
    # neb_optim_config = torchneb.OptimConfig(5000, torch.optim.Adam, {"lr":1e-3})
    neb_config = torchneb.NEBConfig(optim_config=neb_optim_config)
    neb_config.spring_constant = float('inf')
    neb_config.insert_method = fill.equal
    neb_config.insert_args = {"count": 20}
    model = interp

    path_coords = torch.tensor(init_neb_positions)

    # Calculate neb path
    result = torchneb.neb({
        "path_coords": path_coords,
        "target_distances": torch.ones(2)
    }, model, neb_config)
    dense_vals = np.array(result["dense_vals"])
    dense_coords = result["dense_coords"].cpu().numpy()

    # Find maximum between the two minima
    N = len(dense_vals)
    min1 = np.argmin(dense_vals[(N//4):(N//2)]) + N//4
    min2 = np.argmin(dense_vals[(N//2):-(N//4)]) + N//2
    max_index = np.argmax(dense_vals[min1:min2]) + min1
    if abs(max_index - N // 2) > 10:
        print(
            "WARNING: off-center max, max_index=%d, center=%d" % (
                max_index, N // 2)
        )
    max_coord = dense_coords[max_index]

    np.savetxt(output_dir / "neb_cost.txt", dense_vals)
    np.savetxt(output_dir / "neb_xyz.txt", dense_coords)
    np.savetxt(output_dir / "neb_cost_max_coord.txt",
               dense_vals[max_index, None])
    final_path = result["path_coords"].numpy()
    np.savetxt(output_dir / "final_path.txt", final_path)
    np.savetxt(output_dir / "li_positions.txt", li_positions)
    np.savetxt(output_dir / "li_indices.txt", li_indices, fmt="%d")
    np.savetxt(output_dir / "neb_max_coord.txt", max_coord)

    # Plot final path
    x, y, z = final_path.T
    mlab.plot3d(x, y, z, figure=figure)
    mlab.points3d(x, y, z, scale_factor=0.1,
                  color=(1.0, 0., 0.), figure=figure)
    mlab.savefig(str(output_dir / "neb.png"), figure=figure)
    mlab.close(figure)

    # Plot energy of trajectory
    fig = plt.figure()
    r = np.linspace(0, 1, len(result["dense_vals"]))
    plt.plot(r, dense_vals, '.')
    plt.plot(r[max_index], dense_vals[max_index], 'x', color="red")
    fig.savefig(output_dir / "trajectory.png")
    plt.close(fig)


def main():
    chgcar_paths = Path("chgcar_paths")

    # filename = "calculations/Li3BrO_Pm-3m/charge_density/CHGCAR"
    with open("chgcar_list.txt", "r") as flist:
        for line in flist:
            path = Path(line.strip())
            print(path)
            system_name = path.parts[1]
            output_path = chgcar_paths / system_name
            output_path.mkdir(exist_ok=True, parents=True)
            run_neb(path, output_path)


if __name__ == "__main__":
    main()
