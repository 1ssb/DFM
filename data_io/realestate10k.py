from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple, Union
import os, glob, json
import numpy as np
import torch, json
import torch.nn.functional as F
import torchvision.transforms as tf
from einops import rearrange, repeat
from torch.utils.data import Dataset
from numpy.random import default_rng
from utils import *
from geometry import get_opencv_pixel_coordinates
from numpy import random
import scipy
import cv2
from PIL import Image
from numpy.random import default_rng
from datetime import datetime

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

# TODO: For new images
# IMAGE_PATH = ''

Stage = Literal["train", "test", "val"]        


def perturb_pose(pose_matrix, magnitude):
    """
    Add perturbation to the translation part of a 4x4 homogeneous pose matrix.
    Verifies that the input is a 4x4 matrix, prints the original pose, generates random perturbations,
    applies them to the translation part of the matrix, and prints the perturbed pose.

    Parameters:
        pose_matrix (numpy.ndarray): 4x4 homogeneous pose matrix.
        magnitude (float): The magnitude of the perturbation.

    Returns:
        numpy.ndarray: The perturbed 4x4 pose matrix.
    """
    if pose_matrix.shape != (4, 4):
        print("The input matrix is not 4x4. Exiting.")
        return
    
    # print("Original Pose:")
    # print(pose_matrix)
    
    perturbations = np.random.uniform(-magnitude, magnitude, size=(3,))
    perturbed_pose = np.copy(pose_matrix)
    perturbed_pose[:3, 3] += perturbations

    # print("Perturbed Pose:")
    # print(perturbed_pose)
    return perturbed_pose

def rotation_pose(angle_degrees, clockwise=True, d='z'):
    # Convert angle from degrees to radians
    angle_radians = np.deg2rad(angle_degrees)
    
    # Negate the angle if the rotation should be anticlockwise
    if not clockwise:
        angle_radians = -angle_radians
    
    # Create the rotation matrix for the specified axis
    if d == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle_radians), -np.sin(angle_radians)],
            [0, np.sin(angle_radians), np.cos(angle_radians)]
        ])
    elif d == 'y':
        rotation_matrix = np.array([
            [np.cos(angle_radians), 0, np.sin(angle_radians)],
            [0, 1, 0],
            [-np.sin(angle_radians), 0, np.cos(angle_radians)]
        ])
    elif d == 'z':
        rotation_matrix = np.array([
            [np.cos(angle_radians), -np.sin(angle_radians), 0],
            [np.sin(angle_radians), np.cos(angle_radians), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError(f"Invalid axis: {d}. Axis must be 'x', 'y', or 'z'.")
    
    # Create the 4x4 pose matrix
    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = rotation_matrix
    
    return pose_matrix

#def return_intrinsics(path = IMAGE_PATH):
    # TODO: Complete this before EK train experiements for exact intrinsics
#  return


def strafe(pose_matrix, direction ='R', distance = 2):
    
    if direction not in ['L', 'R', 'F', 'B', 'U', 'D']:
        print("Invalid direction! Please use 'L', 'R', 'F', 'B', 'U', or 'D'.")
        return pose_matrix

    # Define movement vectors for each direction
    movement_vectors = {
        'L': np.array([1,  0,  0]),  # Left
        'R': np.array([-1,  0,  0]),  # Right
        'F': np.array([ 0,  0, -1]),  # Forward (-z axis)
        'B': np.array([ 0,  0,  1]),  # Backward
        'U': np.array([ 0,  1,  0]),  # Up
        'D': np.array([ 0, -1,  0])   # Down
    }

    movement = movement_vectors[direction] * distance

    # Check if pose_matrix is a TensorFlow tensor, PyTorch tensor, or NumPy array
    if isinstance(pose_matrix, torch.Tensor):
        # Convert movement to PyTorch tensor and perform update
        movement_tensor = torch.tensor(movement, dtype=pose_matrix.dtype)
        updated_pose = pose_matrix.clone()
        updated_pose[:3, 3] += movement_tensor
    else:
        # Handle as NumPy array
        updated_pose = np.copy(pose_matrix)
        updated_pose[:3, 3] += movement

    return updated_pose

class Camera(object):
    # ca = 0 
    # dis = 0
    # num = 0
    # c2w_mat = np.eye(4)
    # c2w_mat[:3, 3] = [0, 0, 0]  # Translation vector
    # pose_list = []
    
    def __init__(self, entry):
        fx, fy, cx, cy = entry[:4]
        # fx = 123.48 /128
        # fy = 125.02 /128
        
        assert np.allclose(cx, 0.5)
        assert np.allclose(cy, 0.5)
        
        # fx = fx * (640.0 / 360.0)
        # fy = fx = 0.48

        self.intrinsics = np.array(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32,
        )
        
        # Camera.ca+=75
        # Camera.dis+=2
        # Camera.num+=1
        w2c_mat = np.array(entry[6:], dtype=np.float32).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4, dtype=np.float32)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4 
        
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)
        # print(self.c2w_mat)
        # print(type(self.c2w_mat))
        # Saving the poses in the data file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_pose(self.c2w_mat, filename=f"./camera_poses/recorded_poses_{timestamp}.json")
        # input("Pose")
        
        #print("W2C", self.w2c_mat)
        #print("C2W", self.inv_mat)
        # Options to rotate, strafe, perturb the poses as they get generated
        # perturb_pose(w2c_mat_4x4, 0.85)
        # self.c2w_mat = rotation_pose(Camera.ca, clockwise=False, d = 'y')
        
        # Unlock this for normal
        # self.c2w_mat = strafe(self.c2w_mat, direction='R', distance=Camera.dis)
        # if Camera.pose_list is None:
        #     Camera.pose_list = []
        # Camera.pose_list.append(self.c2w_mat.tolist())
        # if Camera.num == 50:
        #     self.save_pose(Camera.pose_list)
        
    # My method
    # @staticmethod
    # def save_pose(pose_list, filename="./camera_poses/strafe_right.json"):
    #     os.makedirs(os.path.dirname(filename), exist_ok=True)
    #     data = {str(i+1): pose for i, pose in enumerate(pose_list)}
    #     with open(filename, 'w') as file:
    #         json.dump(data, file, indent=4)
            
    #     print(f"Poses saved at {filename}")
    
    @staticmethod
    def save_pose(pose, filename="./camera_poses/strafe_right.json"):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Initialize data dictionary
        data = {}

        # Try to load existing data if file exists
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            try:
                with open(filename, 'r') as file:
                    data = json.load(file)
            except json.JSONDecodeError:
                print(f"Warning: Existing file '{filename}' contains invalid JSON. It will be overwritten.")

        # Add new pose to the data
        new_index = str(len(data) + 1)
        data[new_index] = pose.tolist() if isinstance(pose, np.ndarray) else pose

        # Save updated data to JSON file
        with open(filename, 'w') as file:
            json.dump(data, file, indent=4)

        print(f"Poses saved at {filename}")
        
class RealEstate10kDatasetOM(Dataset):
    examples: List[Path]
    pose_file: Path
    stage: Stage
    to_tensor: tf.ToTensor
    overfit_to_index: Optional[int]
    num_target: int
    context_min_distance: int
    context_max_distance: int

    z_near: float = 0.7
    z_far: float = 10.0
    image_size: int = 64
    background_color: torch.tensor = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

    def __init__(
        self,
        root: Union[str, Path],
        # pose_file: Union[str, Path],
        num_context: int,
        num_target: int,
        context_min_distance: int,
        context_max_distance: int,
        stage: Stage = "train",
        overfit_to_index: Optional[int] = None,
        max_scenes: Optional[int] = None,
        pose_root: Optional[Union[str, Path]] = None,
        image_size: Optional[int] = 64,
    ) -> None:
        super().__init__()
        self.overfit_to_index = overfit_to_index
        self.num_context = num_context
        self.num_target = num_target
        self.context_min_distance = context_min_distance
        self.context_max_distance = context_max_distance
        self.image_size = image_size
        sub_dir = {"train": "train", "test": "test", "val": "test",}[stage]
        image_root = Path(root) / sub_dir
        # ek_paths = "../all_rgb_frames"
        # image_root = Path(ek_paths)
        scene_path_list = sorted(list(Path(image_root).glob("*/")))

        if max_scenes is not None:
            scene_path_list = scene_path_list[:max_scenes]
        self.stage = stage
        self.to_tensor = tf.ToTensor()
        self.rng = default_rng()
        self.normalize = normalize_to_neg_one_to_one

        if pose_root is None:
            pose_root = root
        # print("loading pose file")
        pose_file = Path(pose_root) / f"{stage}.mat"
        # print("Pose File:", pose_file)
        self.all_cam_params = scipy.io.loadmat(pose_file)
        
        # Logic to load only what's found and the early scenes only
        scene_path_list = scene_path_list[:300]
        scene_path_list = filtered(scene_path_list)
        dummy_img_path = str(next(scene_path_list[0].glob("*.jpg")))
        dummy_img = cv2.imread(dummy_img_path)
        h, w = dummy_img.shape[:2]

        assert w == 640 and h == 360
        # assert w == 256 and h == 256
        self.border = 140
        self.xy_pix = get_opencv_pixel_coordinates(
            x_resolution=self.image_size, y_resolution=self.image_size
        )
        self.xy_pix_128 = get_opencv_pixel_coordinates(
            x_resolution=128, y_resolution=128
        )

        self.len = 0
        all_rgb_files = []
        all_timestamps = []
        self.scene_path_list = []
        for i, scene_path in enumerate(scene_path_list):
            rgb_files = sorted(scene_path.glob("*.jpg"))
            self.len += len(rgb_files)
            # RealEstate Code
            timestamps = [int(rgb_file.name.split(".")[0]) for rgb_file in rgb_files]
            # timestamps = [int(rgb_file.name.split("_")[1].split(".")[0]) for rgb_file in rgb_files]
            sorted_ids = np.argsort(timestamps)
            all_rgb_files.append(np.array(rgb_files)[sorted_ids])
            self.scene_path_list.append(scene_path)
            all_timestamps.append(np.array(timestamps)[sorted_ids])
        self.all_rgb_files = np.concatenate(all_rgb_files)
        self.indices = torch.arange(0, len(self.scene_path_list))
        self.all_rgb_files = all_rgb_files
        self.all_timestamps = all_timestamps
        # print("NUM IMAGES", self.len)

    # @lru_cache(maxsize=None)
    def read_image(self, rgb_files, id):
        #print(len(rgb_files))
        # input("Checkpoint: If it doesn't fail in the next step, you are good to go. Press Enter to Continue!")
        rgb_file = rgb_files[id-1]
        # print(rgb_file)
        # rgb_file = "/home/projects/Rudra_Generative_Robotics/EK/all_rgb_frames/P01_01/frame_0000001919.jpg"
        # Three states
        # rgb_filer = "/home/projects/Rudra_Generative_Robotics/EK/DFM/test_env/inv.jpg"
        # rgb_filer = "/home/projects/Rudra_Generative_Robotics/EK/DFM/test_env/part.jpg"
        rgb_filer = "/home/projects/Rudra_Generative_Robotics/EK/DFM/test_env/vis.jpg"
        # Real Estate script
        print(f"reading {rgb_filer}")
        # rgb = torch.tensor(np.asarray(Image.open(rgb_file)).astype(np.float32)[:, self.border:-self.border, :]).permute(2, 0, 1) / 255.0

        rgb = torch.tensor(np.asarray(Image.open(rgb_filer)).astype(np.float32)).permute(2, 0, 1) / 255.0
        # print(rgb.shape, "SHAPE 1")
        rgb = F.interpolate(
           rgb.unsqueeze(0),
             size=(self.image_size, self.image_size),
             mode="bilinear",
             antialias=True,
         )[0]
        #print(rgb.shape, "Reshaped")
        # print("Printing IDs:", id)
        cam_param = self.all_cam_params[str(rgb_file.parent.name)][id][1:]
        cam_param = Camera(cam_param.flatten().tolist())
        return rgb, cam_param

    def __len__(self) -> int:
        return len(self.all_rgb_files)

    def __getitem__(self, index: int):
        scene_idx = random.randint(0, len(self.all_rgb_files) - 1)
        # start from reverse
        # scene_idx = len(self.all_rgb_files) - index - 1
        if self.overfit_to_index is not None:
            scene_idx = self.overfit_to_index

        def fallback():
            """Used if the desired index can't be loaded."""
            return self[random.randint(0, len(self.all_rgb_files) - 1)]
        print("This scene: ", scene_idx)
        rgb_files = self.all_rgb_files[scene_idx]
        timestamps = self.all_timestamps[scene_idx]
        assert (timestamps == sorted(timestamps)).all()
        num_frames = len(rgb_files)
        if num_frames < self.num_target + 1:
            return fallback()

        start_idx = self.rng.choice(len(rgb_files), 1)[0]
        # context_min_distance = self.context_min_distance  # * self.num_context
        # context_max_distance = self.context_max_distance  # * self.num_context

        # num_context = self.rng.choice(np.arange(1, self.num_context + 1), 1)[0]
        num_context = self.num_context
        context_min_distance = self.context_min_distance * num_context
        context_max_distance = self.context_max_distance * num_context

        end_idx = self.rng.choice(
            np.arange(
                start_idx + context_min_distance, start_idx + context_max_distance,
            ),
            1,
            replace=False,
        )[0]
        if end_idx >= len(rgb_files):
            return fallback()
        trgt_idx = self.rng.choice(
            np.arange(start_idx, end_idx), self.num_target, replace=False
        )

        flip = self.rng.choice([True, False])
        if flip:
            temp = start_idx
            start_idx = end_idx
            end_idx = temp

        ctxt_idx = [start_idx]
        trgt_idx[0] = end_idx
        if num_context != 1:
            distance = self.rng.choice(
                np.arange(context_min_distance, context_max_distance), 1,
            )
            if start_idx < end_idx:
                extra_ctxt_idx = self.rng.choice(
                    np.arange(
                        start_idx, max(start_idx + num_context - 1, end_idx - distance),
                    ),
                    num_context - 1,
                    replace=False,
                )
            else:
                extra_ctxt_idx = self.rng.choice(
                    np.arange(
                        min(start_idx - num_context + 1, end_idx + distance), start_idx,
                    ),
                    num_context - 1,
                    replace=False,
                )
            ctxt_idx.extend(extra_ctxt_idx)

        if flip:
            # sort the target indices increasingly
            trgt_idx = np.sort(trgt_idx)
        else:
            # sort the target indices decreasingly
            trgt_idx = np.sort(trgt_idx)[::-1]
        # if len(rgb_files) < trgt_idx.max() + 1:
        #     return fallback()

        trgt_rgbs = []
        trgt_c2w = []
        trgt_intrinsics = []
        for id in trgt_idx:
            print("Loading Target")
            rgb, cam_param = self.read_image(rgb_files, id)
            trgt_rgbs.append(rgb)
            trgt_intrinsics.append(cam_param.intrinsics)
            trgt_c2w.append(cam_param.c2w_mat)
        trgt_c2w = torch.tensor(np.array(trgt_c2w)).float()
        trgt_rgb = torch.stack(trgt_rgbs, axis=0)

        # load the ctxt
        ctxt_rgbs = []
        ctxt_c2w = []
        ctxt_intrinsics = []
        for id in ctxt_idx:
            print("Loading Context")
            rgb, cam_param = self.read_image(rgb_files, id)
            ctxt_rgbs.append(rgb)
            ctxt_intrinsics.append(torch.tensor(np.array(cam_param.intrinsics)).float())
            ctxt_c2w.append(cam_param.c2w_mat)
        ctxt_c2w = torch.tensor(np.array(ctxt_c2w)).float()
        ctxt_rgb = torch.stack(ctxt_rgbs, axis=0)
        ctxt_intrinsics = torch.stack(ctxt_intrinsics, axis=0)

        # ctxt_rgb, ctxt_cam_param = self.read_image(rgb_files, ctxt_idx)
        # ctxt_c2w = ctxt_cam_param.c2w_mat
        # ctxt_c2w = torch.tensor(np.array(ctxt_c2w)).float()
        # ctxt_intrinsics = ctxt_cam_param.intrinsics
        # ctxt_intrinsics = torch.tensor(np.array(ctxt_intrinsics)).float()
        inv_ctxt_c2w = torch.inverse(ctxt_c2w[0])
        ctxt_inv_ctxt_c2w_repeat = inv_ctxt_c2w.unsqueeze(0).repeat(num_context, 1, 1)
        trgt_inv_ctxt_c2w_repeat = inv_ctxt_c2w.unsqueeze(0).repeat(
            self.num_target, 1, 1
        )

        return (
            {
                "ctxt_c2w": torch.einsum(
                    "ijk, ikl -> ijl", ctxt_inv_ctxt_c2w_repeat, ctxt_c2w
                ),
                "trgt_c2w": torch.einsum(
                    "ijk, ikl -> ijl", trgt_inv_ctxt_c2w_repeat, trgt_c2w
                ),
                "ctxt_rgb": self.normalize(ctxt_rgb),
                "trgt_rgb": self.normalize(trgt_rgb),
                "ctxt_abs_camera_poses": None,
                "intrinsics": ctxt_intrinsics[0],
                "x_pix": rearrange(self.xy_pix, "h w c -> (h w) c"),
                "idx": torch.tensor([index]),
                "image_shape": torch.tensor([self.image_size, self.image_size, 3]),
                "num_context": torch.tensor([num_context]),
            },
            trgt_rgb,  # rearrange(rendered["image"], "c h w -> (h w) c"),
        )

    def data_for_video(self, video_idx, ctxt_idx, trgt_idx, num_frames_render=20):
        scene_idx = video_idx
        if len(self.all_rgb_files) < scene_idx:
            print('Not enough images loaded!')
            exit(0)
        
        rgb_files = self.all_rgb_files[scene_idx]
        timestamps = self.all_timestamps[scene_idx]
        assert (timestamps == sorted(timestamps)).all()
        num_frames = len(rgb_files)

        trgt_rgbs = []
        trgt_c2w = []
        trgt_intrinsics = []
        for id in trgt_idx:
            #print("Testing only")
            id = min(id, len(rgb_files) - 1)
            id = max(id, 0)
            # print("chck Loader")
            rgb, cam_param = self.read_image(rgb_files, id)
            # print("chck unLoader")
            trgt_rgbs.append(rgb)
            trgt_intrinsics.append(cam_param.intrinsics)
            trgt_c2w.append(cam_param.c2w_mat)
        trgt_c2w = torch.tensor(np.array(trgt_c2w)).float()
        trgt_rgb = torch.stack(trgt_rgbs, axis=0)
        # print("chck 1")
        # load the ctxt
        ctxt_rgbs = []
        ctxt_c2w = []
        ctxt_intrinsics = []
        for id in ctxt_idx:
            id = min(id, len(rgb_files) - 1)
            id = max(id, 0)
            rgb, cam_param = self.read_image(rgb_files, id)
            ctxt_rgbs.append(rgb)
            ctxt_intrinsics.append(torch.tensor(np.array(cam_param.intrinsics)).float())
            ctxt_c2w.append(cam_param.c2w_mat)
        ctxt_c2w = torch.tensor(np.array(ctxt_c2w)).float()
        ctxt_rgb = torch.stack(ctxt_rgbs, axis=0)
        ctxt_intrinsics = torch.stack(ctxt_intrinsics, axis=0)
        # print("chck 2")
        render_poses = []
        num_frames_render = min(ctxt_idx[0], len(rgb_files) - 1) - min(
            trgt_idx[0], len(rgb_files) - 1
        )
        noflip = False
        if num_frames_render < 0:
            noflip = True
            num_frames_render *= -1
        # print("chck 3")
        for i in range(1, num_frames_render + 1):
            # id = ctxt_idx[0] + i * (trgt_idx[0] - ctxt_idx[0]) // (num_frames_render)
            if noflip:
                id = ctxt_idx[0] + i
            else:
                id = trgt_idx[0] + i
            rgb_file = rgb_files[id]
            cam_param = self.all_cam_params[str(rgb_file.parent.name)][id][1:]
            cam_param = Camera(cam_param.flatten().tolist())
            render_poses.append(cam_param.c2w_mat)
        render_poses = torch.tensor(np.array(render_poses)).float()
        # print("chck 4")
        # print(
        #     f"ctxt_idx: {ctxt_idx}, trgt_idx: {trgt_idx}, num_frames_render: {num_frames_render}, len(rgb_files): {len(rgb_files)}"
        # )

        num_frames_render = render_poses.shape[0]
        inv_ctxt_c2w = torch.inverse(ctxt_c2w[0])
        ctxt_inv_ctxt_c2w_repeat = inv_ctxt_c2w.unsqueeze(0).repeat(
            self.num_context, 1, 1
        )
        trgt_inv_ctxt_c2w_repeat = inv_ctxt_c2w.unsqueeze(0).repeat(
            self.num_target, 1, 1
        )
        trgt_inv_ctxt_c2w_repeat_video = inv_ctxt_c2w.unsqueeze(0).repeat(
            num_frames_render, 1, 1
        )
        return (
            {
                "ctxt_c2w": torch.einsum(
                    "ijk, ikl -> ijl", ctxt_inv_ctxt_c2w_repeat, ctxt_c2w
                ),
                "trgt_c2w": torch.einsum(
                    "ijk, ikl -> ijl", trgt_inv_ctxt_c2w_repeat, trgt_c2w
                ),
                # "render_poses": torch.einsum(
                #     "ijk, ikl -> ijl", trgt_inv_ctxt_c2w_repeat_video, render_poses
                # ),
                "ctxt_rgb": self.normalize(ctxt_rgb),
                "trgt_rgb": self.normalize(trgt_rgb),
                "ctxt_abs_camera_poses": ctxt_c2w,
                "trgt_abs_camera_poses": trgt_c2w,
                "intrinsics": ctxt_intrinsics[0],
                "x_pix": rearrange(self.xy_pix, "h w c -> (h w) c"),
                "x_pix_128": rearrange(self.xy_pix_128, "h w c -> (h w) c"),
                # "idx": torch.tensor([index]),
                "image_shape": torch.tensor([self.image_size, self.image_size, 3]),
                # "folder_path": str(rgb_files[0].parent),
            },
            trgt_rgb,  # rearrange(rendered["image"], "c h w -> (h w) c"),
        )

    def data_for_video_GT(self, video_idx, ctxt_idx, trgt_idx, num_frames_render=20):
        scene_idx = video_idx
        rgb_files = self.all_rgb_files[scene_idx]
        timestamps = self.all_timestamps[scene_idx]
        assert (timestamps == sorted(timestamps)).all()
        num_frames = len(rgb_files)

        trgt_rgbs = []
        trgt_c2w = []
        trgt_intrinsics = []
        for id in trgt_idx:
            id = min(id, len(rgb_files) - 1)
            rgb, cam_param = self.read_image(rgb_files, id)
            trgt_rgbs.append(rgb)
            trgt_intrinsics.append(cam_param.intrinsics)
            trgt_c2w.append(cam_param.c2w_mat)
        trgt_c2w = torch.tensor(np.array(trgt_c2w)).float()
        trgt_rgb = torch.stack(trgt_rgbs, axis=0)

        # load the ctxt
        ctxt_rgbs = []
        ctxt_c2w = []
        ctxt_intrinsics = []
        for id in ctxt_idx:
            id = min(id, len(rgb_files) - 1)
            rgb, cam_param = self.read_image(rgb_files, id)
            ctxt_rgbs.append(rgb)
            ctxt_intrinsics.append(torch.tensor(np.array(cam_param.intrinsics)).float())
            ctxt_c2w.append(cam_param.c2w_mat)
        ctxt_c2w = torch.tensor(np.array(ctxt_c2w)).float()
        ctxt_rgb = torch.stack(ctxt_rgbs, axis=0)
        ctxt_intrinsics = torch.stack(ctxt_intrinsics, axis=0)

        render_poses = []
        num_frames_render = min(ctxt_idx[0], len(rgb_files) - 1) - min(
            trgt_idx[0], len(rgb_files) - 1
        )
        trgt_rgbs = []
        for i in range(1, num_frames_render + 1):
            # id = ctxt_idx[0] + i * (trgt_idx[0] - ctxt_idx[0]) // (num_frames_render)
            id = trgt_idx[0] + i
            rgb_file = rgb_files[id]
            cam_param = self.all_cam_params[str(rgb_file.parent.name)][id][1:]
            cam_param = Camera(cam_param.flatten().tolist())
            render_poses.append(cam_param.c2w_mat)
            rgb, cam_param = self.read_image(rgb_files, id)
            trgt_rgbs.append(rgb)
        trgt_rgb = torch.stack(trgt_rgbs, axis=0)
        render_poses = torch.tensor(np.array(render_poses)).float()
        inv_ctxt_c2w = torch.inverse(ctxt_c2w[0])
        ctxt_inv_ctxt_c2w_repeat = inv_ctxt_c2w.unsqueeze(0).repeat(
            self.num_context, 1, 1
        )
        trgt_inv_ctxt_c2w_repeat = inv_ctxt_c2w.unsqueeze(0).repeat(
            self.num_target, 1, 1
        )
        trgt_inv_ctxt_c2w_repeat_video = inv_ctxt_c2w.unsqueeze(0).repeat(
            num_frames_render, 1, 1
        )
        return (
            {
                "ctxt_c2w": torch.einsum(
                    "ijk, ikl -> ijl", ctxt_inv_ctxt_c2w_repeat, ctxt_c2w
                ),
                "trgt_c2w": torch.einsum(
                    "ijk, ikl -> ijl", trgt_inv_ctxt_c2w_repeat, trgt_c2w
                ),
                "render_poses": torch.einsum(
                    "ijk, ikl -> ijl", trgt_inv_ctxt_c2w_repeat_video, render_poses
                ),
                "ctxt_rgb": self.normalize(ctxt_rgb),
                "trgt_rgb": self.normalize(trgt_rgb),
                "intrinsics": ctxt_intrinsics[0],
                "x_pix": rearrange(self.xy_pix, "h w c -> (h w) c"),
                "x_pix_128": rearrange(self.xy_pix_128, "h w c -> (h w) c"),
                # "idx": torch.tensor([index]),
                "image_shape": torch.tensor([self.image_size, self.image_size, 3]),
                "folder_path": str(rgb_files[0].parent),
            },
            trgt_rgb,  # rearrange(rendered["image"], "c h w -> (h w) c"),
        )

def filtered(scene_path_list):
    new_scene_path_list = []  # List to hold paths with .jpg files
    deleted_paths_count = 0  # Counter for paths deleted

    # print(f'Total paths in input list: {len(scene_path_list)}')

    for scene_path in scene_path_list:
        # print(f'Traversing directory: {scene_path}')

        files_in_directory = os.listdir(scene_path)  # List all files in the directory
        jpg_files = [f for f in files_in_directory if f.endswith('.jpg')]  # Filter for .jpg files

        # print(f'Files found in directory: {files_in_directory}')
        # print(f'.jpg files found: {jpg_files}')

        if jpg_files:
            new_scene_path_list.append(scene_path)  # If .jpg files are found, keep the path
        else:
            deleted_paths_count += 1  # Increment the counter if no .jpg files are found
            # print(f'Deleting path from list: {scene_path}')

    # print(f'Total paths deleted: {deleted_paths_count}')
    #print(f'Total number of paths that will be loaded: {len(new_scene_path_list)}')

    return new_scene_path_list  # Return the updated list
