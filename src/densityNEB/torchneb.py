# This code is based on https://github.com/fdraxler/PyTorch-AutoNEB/blob/master/torch_autoneb/neb_model.py
import logging
from typing import Union

import numpy as np
import torch
from torch import Tensor, linspace

from densityNEB import fill


# try:
#     from tqdm import tqdm as pbar
# except ModuleNotFoundError:
class pbar:
    def __init__(self, iterable=None, desc=None, total=None, *args, **kwargs):
        self.iterable = iterable

    def __iter__(self):
        yield from self.iterable

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def update(self, N=None):
        pass

class Eggcarton(torch.nn.Module):
    def forward(self, x):
        return (x * (2 * torch.pi)).cos().sum(dim=-1)

class OptimConfig:
    def __init__(self, nsteps: int, algorithm_type, algorithm_args: dict):
        self.nsteps = nsteps
        self.algorithm_type = algorithm_type
        self.algorithm_args = algorithm_args

class NEBConfig:
    def __init__(self, optim_config: OptimConfig):
        self.insert_method = fill.equal
        self.insert_args = {"count": 10}
        self.subsample_pivot_count = 100
        self.optim_config = optim_config
        self.spring_constant = float("inf")

class NEB():
    def __init__(self, model, path_coords: Tensor, target_distances: Tensor = None, spring_constant: Union[str, float] = "inf"):
        """
        Creates a NEB instance that is prepared for evaluating the band.
        """
        self.model = model
        self.path_coords = path_coords.clone()
        self.target_distances = target_distances

        self.spring_constant = float(spring_constant)

    def _assert_grad(self):
        if self.path_coords.grad is None:
            self.path_coords.grad = self.path_coords.new(self.path_coords.shape).zero_()

    def parameters(self):
        return [self.path_coords]

    def apply(self, gradient=False):
        npivots = self.path_coords.shape[0]
        losses = self.path_coords.new(npivots)

        # Redistribute if spring_constant == inf
        assert self.target_distances is not None or not gradient, "Cannot compute gradient if target distances are unavailable"
        if gradient and self.spring_constant == float("inf"):
            self.path_coords[:] = distribute_by_weights(self.path_coords, self.path_coords.shape[0], weights=self.target_distances).data
            # print(self.path_coords)

        # Assert gradient storage is available
        if gradient:
            self._assert_grad()
            self.path_coords.requires_grad_(True)

        # Compute losses and gradients
        if gradient:
            losses = self.model(self.path_coords)
        else:
            with torch.no_grad():
                losses = self.model(self.path_coords)
        if gradient:
            losses.sum().backward()
            # Make sure no gradient is there on the endpoints
            self.path_coords.grad[0].zero_()
            self.path_coords.grad[-1].zero_()

        self.path_coords.requires_grad_(False)

        # Compute NEB gradients as in (Henkelmann & Jonsson, 2000)
        if gradient:
            distances = (self.path_coords[:-1] - self.path_coords[1:]).norm(2, 1)
            for i in range(1, npivots - 1):
                d_prev, d_next = distances[i - 1].item(), distances[i].item()
                td_prev, td_next = self.target_distances[i - 1].item(), self.target_distances[i].item()
                l_prev, loss, l_next = losses[i - 1].item(), losses[i].item(), losses[i + 1].item()

                # Compute tangent
                tangent = self.compute_tangent(d_next, d_prev, i, l_next, l_prev, loss)

                # Project gradients perpendicular to tangent
                self.path_coords.grad[i] -= self.path_coords.grad[i].dot(tangent) * tangent

                assert self.spring_constant > 0
                if self.spring_constant < float("inf"):
                    # Spring force parallel to tangent
                    self.path_coords.grad[i] += ((d_prev - td_prev) - (d_next - td_next)) * self.spring_constant * tangent

        return losses.max().item()

    def compute_tangent(self, d_next, d_prev, i, l_next, l_prev, loss):
        if l_prev < loss > l_next or l_prev > loss < l_next:
            # Interpolate tangent at maxima/minima to make convergence smooth
            t_prev = (self.path_coords[i] - self.path_coords[i - 1]) / d_prev
            t_next = (self.path_coords[i + 1] - self.path_coords[i]) / d_next
            l_max = max(abs(loss - l_prev), abs(loss - l_next))
            l_min = min(abs(loss - l_prev), abs(loss - l_next))
            if l_prev > l_next:
                tangent = l_min * t_prev + l_max * t_next
            else:
                tangent = l_max * t_prev + l_min * t_next
            return tangent / (tangent.norm() + 1e-30)
        elif l_prev > l_next:
            # Tangent to the previous
            return (self.path_coords[i] - self.path_coords[i - 1]) / d_prev
        else:
            # Tangent to the next
            return (self.path_coords[i + 1] - self.path_coords[i]) / d_next

    def iterate_densely(self, sub_pivot_count=9):
        dense_pivot_count = (self.path_coords.shape[0] - 1) * (sub_pivot_count + 1) + 1
        alphas = linspace(0, 1, sub_pivot_count + 2)[:-1].to(self.path_coords.device)
        for i in pbar(range(dense_pivot_count), "Saddle analysis"):
            base_pivot = i // (sub_pivot_count + 1)
            sub_pivot = i % (sub_pivot_count + 1)

            if sub_pivot == 0:
                # Coords of pivot
                coords = self.path_coords[base_pivot]
            else:
                # Or interpolation between pivots
                alpha = alphas[sub_pivot]
                coords = self.path_coords[base_pivot] * (1 - alpha) + self.path_coords[base_pivot + 1] * alpha

            # Retrieve values from model analysis
            with torch.no_grad():
                val = self.model(coords.unsqueeze(0))
            yield coords, val

    def analyse(self, sub_pivot_count=19):
        # Collect stats here
        analysis = {}

        dense_vals = []
        dense_coords = []
        for coord, val in self.iterate_densely(sub_pivot_count):
            dense_vals.append(val.item())
            dense_coords.append(coord)

        # Compute lengths
        end_to_end_distance = (self.path_coords[-1] - self.path_coords[0]).norm(2)
        analysis["lengths"] = (self.path_coords[1:] - self.path_coords[:-1]).norm(2, 1) / end_to_end_distance
        analysis["length"] = end_to_end_distance
        analysis["dense_vals"] = dense_vals
        analysis["dense_coords"] = torch.stack(dense_coords, dim=0)

        return analysis


def distribute_by_weights(path: Tensor, nimages: int, path_target: Tensor = None, weights: Tensor = None, climbing_pivots: list = None):
    """
    Redistribute the pivots on the path so that they are spaced as given by the weights.
    """
    # Ensure storage for coordinates
    if path_target is None:
        path_target = path.new(nimages, path.shape[1])
    else:
        assert path_target is not path, "Source must be unequal to target for redistribution"
        assert path_target.shape[0] == nimages
    # Ensure weights
    if weights is None:
        weights = path.new(nimages - 1).fill_(1)
    else:
        assert len(weights.shape) == 1
        assert weights.shape[0] == nimages - 1

    # In climbing mode, reinterpolate only between the climbing images
    if climbing_pivots is not None:
        assert path.shape[0] == nimages, "Cannot change number of items when reinterpolating with respect to climbing images."
        assert len(climbing_pivots) == nimages
        assert all(isinstance(b, bool) for b in climbing_pivots), "Image must be climbing or not."
        start = 0
        for i, is_climbing in enumerate(climbing_pivots):
            if is_climbing or i == nimages - 1:
                distribute_by_weights(path[start:i + 1], i + 1 - start, path_target[start:i + 1], weights[start:i])
                start = i
        return path_target

    if path is path_target:
        # For the computation the original path is necessary
        path_source = path.clone()
    else:
        path_source = path

    # The current distances between elements on chain
    current_distances = (path_source[:-1] - path_source[1:]).norm(2, 1)
    target_positions = (weights / weights.sum()).cumsum(0) * current_distances.sum()  # Target positions of elements (spaced by weights)

    # Put each new item spaced by weights (measured along line) on the line
    last_idx = 0  # Index of previous pivot
    pos_prev = 0.  # Position of previous pivot on chain
    pos_next = current_distances[last_idx].item()  # Position of next pivot on chain
    path_target[0] = path_source[0]
    for i in range(1, nimages - 1):
        position = target_positions[i - 1].item()
        while position > pos_next:
            last_idx += 1
            pos_prev = pos_next
            pos_next += current_distances[last_idx].item()

        t = (position - pos_prev) / (pos_next - pos_prev)
        path_target[i] = (t * path_source[last_idx + 1] + (1 - t) * path_source[last_idx])
    path_target[nimages - 1] = path_source[-1]

    return path_target

def neb(previous_cycle_data, model, neb_config: NEBConfig, save_path_history=False) -> dict:
    # Initialise chain by inserting pivots
    start_path, target_distances = neb_config.insert_method(previous_cycle_data, **neb_config.insert_args)

    # Model
    neb_mod = NEB(model, start_path, target_distances, spring_constant=neb_config.spring_constant)

    # Load optimiser
    optim_config = neb_config.optim_config
    optimiser = optim_config.algorithm_type(neb_mod.parameters(), **optim_config.algorithm_args)

    # Optimise
    path_history = []
    for _ in pbar(range(optim_config.nsteps), "NEB"):
        optimiser.zero_grad()
        neb_mod.apply(gradient=True)
        optimiser.step()
        if save_path_history:
            path_history.append(neb_mod.path_coords.clone().to("cpu"))
    result = {
        "path_coords": neb_mod.path_coords.clone().to("cpu"),
        "target_distances": target_distances.to("cpu")
    }
    if save_path_history:
        result["path_history"] = path_history

    # Analyse
    analysis = neb_mod.analyse(neb_config.subsample_pivot_count)
    result.update(analysis)

    return result


def test_neb():
    import matplotlib.pyplot as plt

    minima = torch.tensor(np.array([[0.5, 0.5], [1.5, 1.2]]))

    neb_optim_config = OptimConfig(500, torch.optim.Adam, {})
    neb_config = NEBConfig(optim_config=neb_optim_config)
    neb_config.spring_constant = float('inf')
    neb_config.insert_method = fill.equal
    neb_config.insert_args = {"count": 20}
    model = Eggcarton()

    path_coords = minima  # [nimages, dim]

    result = neb({
        "path_coords": path_coords,
        "target_distances": torch.ones(1)
    }, model, neb_config)

    logging.disable(logging.CRITICAL)

    x1 = torch.linspace(0, 2, 100)
    grid_x, grid_y = torch.meshgrid(x1, x1, indexing="xy")
    grid_stacked = torch.stack((grid_x, grid_y), dim=-1)
    grid_eval = model(grid_stacked)

    plt.figure()
    plt.imshow(grid_eval, interpolation="none", origin="upper", extent=[0, 2, 2, 0])
    final_path = result["path_coords"]
    plt.plot(path_coords[:, 0], path_coords[:, 1], '-o')
    plt.plot(final_path[:, 0], final_path[:, 1], '-x')
    plt.figure()
    plt.plot(np.linspace(0, 1, len(result["dense_vals"])), result["dense_vals"], '.')
    plt.show()

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler("neblog.txt", mode="w"),
            logging.StreamHandler(),
        ],
    )
    test_neb()
