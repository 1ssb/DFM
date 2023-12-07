import os, sys, traceback
from PIL import Image
from einops import rearrange
from PixelNeRF import PixelNeRFModelCond
import torch
import imageio
import numpy as np
from utils import *
from torchvision.utils import make_grid
from accelerate import DistributedDataParallelKwargs
from accelerate import Accelerator
import matplotlib.pyplot as plt
import torch.nn.functional as F
from ek_loading.loader import ImagePairLoader
import pickle

def get_info(list_of_dicts, idx):
    frame = list_of_dicts[idx]
    ctxt_path = frame["input_path"]
    trgt_path = frame["target_path"]
    # ctxt_pose = frame["input_pose"]
    # trgt_pose = frame["target_pose"]
    ctxt_c2w = frame["input_pose"]
    trgt_c2w = frame["target_pose"]
    intrinsics = frame["input_intrinsics"]
    # trgt_c2w = frame["target_intrinsics"]
    return ctxt_path, trgt_path, ctxt_c2w, trgt_c2w, intrinsics

import os
import numpy as np
from PIL import Image
import imageio
import torch

def is_blank(image):
    return np.all(image == image[0, 0])

def check_image_properties(image, label):
    if image is None:
        print(f"{label} is None.")
        return
    print(f"{label} dtype: {image.dtype}, shape: {image.shape}, min: {image.min()}, max: {image.max()}")
    if is_blank(image):
        print(f"Warning: {label} is blank.")

def are_images_equal(image1, image2, tolerance=1e-6):
    difference = np.abs(image1 - image2)
    max_difference = np.max(difference)
    return max_difference < tolerance

def prepare_image(image):
    # Clip and convert dtype
    image = np.clip(image, 0, 255).astype(np.uint8)
    # Remove batch dimension and transpose to HWC
    if len(image.shape) == 4:
        image = image.squeeze(0)
    if image.shape[0] == 3:
        image = image.transpose((1, 2, 0))
    return image


def asset_saver(frames, depth_frames, conditioning_depth_img, inp, out, trainer, save_folder):
    # Make sure the folder exists
    os.makedirs(save_folder, exist_ok=True)

    # Save individual frames and depth frames
    for idx, (frame, depth_frame) in enumerate(zip(frames, depth_frames)):
        frame_path = os.path.join(save_folder, f'frame_{idx:04d}.png')
        depth_frame_path = os.path.join(save_folder, f'frame_depth_{idx:04d}.png')
        
        # Prepare and save frame
        prepared_frame = prepare_image(frame)
        check_image_properties(prepared_frame, f'Saving frame_{idx}')
        Image.fromarray(prepared_frame).save(frame_path)

        # Prepare and save depth frame
        prepared_depth_frame = prepare_image(depth_frame)
        check_image_properties(prepared_depth_frame, f'Saving depth_frame_{idx}')
        Image.fromarray(prepared_depth_frame).save(depth_frame_path)

    # Save videos
    video_path = os.path.join(save_folder, 'video.mp4')
    depth_video_path = os.path.join(save_folder, 'depth_video.mp4')
    
    prepared_frames = [prepare_image(frame) for frame in frames]
    prepared_depth_frames = [prepare_image(frame) for frame in depth_frames]
    
    imageio.mimwrite(video_path, prepared_frames, fps=10, quality=10)
    imageio.mimwrite(depth_video_path, prepared_depth_frames, fps=10, quality=10)

    # Unnormalize and clip context and target images
    # ctxt_img = trainer.model.unnormalize(inp["ctxt_rgb"][:, 0]).cpu().detach().numpy()
    # trgt_img = trainer.model.unnormalize(inp["trgt_rgb"][:, 0]).cpu().detach().numpy()
    # ctxt_img = prepare_image(ctxt_img)
    # trgt_img = prepare_image(trgt_img)
    
def prepare_video_viz(out):
    frames = out["videos"]
    check_tensor_properties(frames[0], "Initial frames")
    # print(frames[0])
    for f in range(len(frames)):
        frames[f] = rearrange(frames[f], "b h w c -> h (b w) c")
    check_tensor_properties(frames[0], "Rearranged frames")

    depth_frames = out["depth_videos"]
    check_tensor_properties(depth_frames[0], "Initial depth_frames")

    for f in range(len(depth_frames)):
        depth_frames[f] = rearrange(depth_frames[f], "(n b) h w -> n h (b w)", n=1)
    check_tensor_properties(depth_frames[0], "Rearranged depth_frames")

    conditioning_depth = out["conditioning_depth"].cpu()
    check_tensor_properties(conditioning_depth, "Initial conditioning_depth")

    # resize to depth_frames
    conditioning_depth = F.interpolate(
        conditioning_depth[:, None],
        size=depth_frames[0].shape[-2:],
        mode="bilinear",
        align_corners=True
    )[:, 0]
    check_tensor_properties(conditioning_depth, "Resized conditioning_depth")
    
    # Concatenate frames and depth frames
    concatenated_frames = []
    for frame, depth_frame in zip(frames, depth_frames):
        # Check and adjust the shape of depth_frame if necessary
        if depth_frame.shape[2] != 1 or depth_frame.shape[:2] != frame.shape[:2]:
            # Resize depth_frame to match the spatial dimensions of frame
            depth_frame = np.resize(depth_frame, (frame.shape[0], frame.shape[1], 1))

        # Concatenate along the channel dimension (assuming last dimension is channel)
        concatenated_frame = np.concatenate((frame, depth_frame), axis=2)
        concatenated_frames.append(concatenated_frame)

        # Print the shape of the concatenated frame
        check_tensor_properties(concatenated_frame, "Concatenated RGB-D")

        
    depth_frames.append(conditioning_depth)
    depth = torch.cat(depth_frames, dim=0)
    
    # Save raw depth frames as .npy files
    depth_frames_dir = "./only_depth_frames"
    os.makedirs(depth_frames_dir, exist_ok=True)
    for i, depth in enumerate(depth_frames):
        depth_numpy = depth.cpu().detach().numpy()
        np.save(os.path.join(depth_frames_dir, f"frame_{i}.npy"), depth_numpy)

    # Applying jet_depth and scaling
    depth = (
        torch.from_numpy(
            jet_depth(depth[:].cpu().detach().view(depth.shape[0], depth.shape[-1], -1))
        )
        * 255
    )

    # convert depth to list of images
    depth_frames = []
    for i in range(depth.shape[0]):
        depth_frames.append(depth[i].cpu().detach().numpy().astype(np.uint8))

    return concatenated_frames, frames, depth_frames[:-1], rearrange(torch.from_numpy(depth_frames[-1]), "h w c -> () c h w")


def are_tensors_almost_equal(tensor1, tensor2, epsilon=1e-7):
    if isinstance(tensor1, np.ndarray):
        tensor1 = torch.from_numpy(tensor1)
    if isinstance(tensor2, np.ndarray):
        tensor2 = torch.from_numpy(tensor2)
        
    difference = torch.abs(tensor1 - tensor2)
    max_difference = torch.max(difference)
    return max_difference < epsilon

def check_tensor_properties(tensor, name):
    print(f"{name} dtype: {tensor.dtype}, shape: {tensor.shape}, min: {tensor.min()}, max: {tensor.max()}")

import json
import os

def dict_writer(my_dict):
    file_path = "/home/projects/Rudra_Generative_Robotics/EK/DFM/output.json"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    def handle_tensor(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type '{type(obj).__name__}' is not JSON serializable")
    
    with open(file_path, 'w') as f:
        json.dump(my_dict, f, default=handle_tensor, indent=4)
