import torch
from jaxtyping import Float
from scipy.spatial.transform import Rotation as R
from torch import Tensor


@torch.no_grad()
def interpolate_pose(
    initial: Float[Tensor, "4 4"], final: Float[Tensor, "4 4"], t: float,
) -> Float[Tensor, "4 4"]:
    # Get the relative rotation.
    r_initial = initial[:3, :3]
    r_final = final[:3, :3]
    r_relative = r_final @ r_initial.T

    r_relative = r_relative.float()

    # Convert it to axis-angle to interpolate it.
    r_relative = R.from_matrix(r_relative.cpu().numpy()).as_rotvec()
    r_relative = R.from_rotvec(r_relative * t).as_matrix()
    r_relative = torch.tensor(r_relative, dtype=final.dtype, device=final.device)
    r_interpolated = r_relative @ r_initial

    # Interpolate the position.
    t_initial = initial[:3, 3]
    t_final = final[:3, 3]
    t_interpolated = t_initial + (t_final - t_initial) * t

    # Assemble the result.
    result = torch.zeros_like(initial)
    result[3, 3] = 1
    result[:3, :3] = r_interpolated
    result[:3, 3] = t_interpolated
    return result


@torch.no_grad()
def interpolate_intrinsics(
    initial: Float[Tensor, "3 3"], final: Float[Tensor, "3 3"], t: float,
) -> Float[Tensor, "3 3"]:
    return initial + (final - initial) * t


""" My code which takes in input and target poses and interpolates between them using the parameter t where 0 is the input pose and 1 is the target pose.

import torch
from scipy.spatial.transform import Rotation as R

@torch.no_grad()
def interpolate_pose(input_pose, target_pose, t):
    # Get the relative rotation.
    r_input = input_pose[:3, :3]
    r_target = target_pose[:3, :3]
    r_relative = r_target @ r_input.T

    # Convert it to axis-angle to interpolate it.
    r_relative_np = r_relative.cpu().numpy()
    r_relative_rotvec = R.from_matrix(r_relative_np).as_rotvec() * t
    r_interpolated_np = R.from_rotvec(r_relative_rotvec).as_matrix()

    # Convert back to tensor on the same device and dtype as target_pose
    r_interpolated = torch.from_numpy(r_interpolated_np).to(target_pose.device).type(target_pose.dtype)
    r_interpolated = r_interpolated @ r_input

    # Interpolate the position.
    t_input = input_pose[:3, 3]
    t_target = target_pose[:3, 3]
    t_interpolated = t_input + (t_target - t_input) * t

    # Assemble the result.
    result = torch.eye(4, dtype=input_pose.dtype, device=input_pose.device)
    result[:3, :3] = r_interpolated
    result[:3, 3] = t_interpolated
    return result
"""