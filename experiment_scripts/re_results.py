import sys, os, re, shutil
from datetime import datetime
import wandb 
import hydra
import cProfile, GPUtil, time
from omegaconf import DictConfig
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from saver import asset_saver, dict_writer, prepare_video_viz
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import (
    GaussianDiffusion,
    Trainer,
)
import data_io
from einops import rearrange
from PixelNeRF import PixelNeRFModelCond
import torch
import imageio
import numpy as np
from utils import *
from torchvision.utils import make_grid
from accelerate import DistributedDataParallelKwargs
from accelerate import Accelerator
from results_configs import re_indices
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
import json, tqdm
from pathlib import Path
# from boxes import detector
# from upscaler import upscaler
# from observer import observer

##### My helper functions #####
def mover(render_number):
    base_dir = "/home/projects/Rudra_Generative_Robotics/EK/DFM/results/"
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    render_dir = os.path.join(base_dir, f"sample_{render_number}_{timestamp}")
    
    # Subdirectories for images, depth maps, and videos
    images_dir = os.path.join(render_dir, "images")
    depth_maps_dir = os.path.join(render_dir, "depth_images")
    videos_dir = os.path.join(render_dir, "videos")

    # Regular expression patterns to match files
    pattern1 = re.compile(r"frame_\d{4}\.png")
    pattern2 = re.compile(r"frame_depth_\d{4}\.png")

    # Create directories if they don't exist
    for directory in [images_dir, depth_maps_dir, videos_dir]:
        os.makedirs(directory, exist_ok=True)

    for file_name in os.listdir(base_dir):
        from_path = os.path.join(base_dir, file_name)

        if pattern1.match(file_name):
            to_path = os.path.join(images_dir, file_name)
        elif pattern2.match(file_name):
            to_path = os.path.join(depth_maps_dir, file_name)
        elif file_name in ["video.mp4", "depth_video.mp4"]:
            to_path = os.path.join(videos_dir, file_name)
        else:
            continue

        try:
            shutil.move(from_path, to_path)
        except Exception as e:
            print(f"Error moving file {file_name}: {e}")

    print("Results moved to timestamped subdirectory successfully!")

def save_rgbd(rgbd_list, render_number, base_dir='./RGBD/strafe-right'):
    items = 50
    if not isinstance(rgbd_list, list) or len(rgbd_list) != items:
        raise ValueError(f"rgbd_list must be a list of {items} items.")

    # Generate a timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join(base_dir, f"sample_{render_number}_{timestamp}")

    os.makedirs(save_dir, exist_ok=True)

    for idx, rgbd in enumerate(rgbd_list, start=1):
        # Check if the item is a PyTorch tensor, convert it to a NumPy array
        if isinstance(rgbd, torch.Tensor):
            rgbd = rgbd.detach().cpu().numpy()
        elif not isinstance(rgbd, np.ndarray):
            raise TypeError(f"Item at index {idx} is neither a NumPy array nor a PyTorch tensor.")

        # Save the NumPy array as an .npy file
        file_path = os.path.join(save_dir, f"{idx}.npy")
        np.save(file_path, rgbd)

    print(f"All RGB-D assets have been saved in {save_dir}.")


##### DFM Code ######

@hydra.main(
    version_base=None, config_path="../configurations/", config_name="config",
)
def train(cfg: DictConfig):
    # os.system("echo 'Initial CPU and Memory usage:'")
    # os.system("top -b -n1 | head -n 10")
    # os.system("echo 'Initial GPU usage:'")
    # os.system("nvidia-smi")
    run = wandb.init(**cfg.wandb)
    wandb.run.log_code(".")
    wandb.run.name = cfg.name
    print(f"run dir: {run.dir}")
    run_dir = run.dir
    wandb.save(os.path.join(run_dir, "checkpoint*"))
    wandb.save(os.path.join(run_dir, "video*"))
    # initialize the accelerator at the beginning
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        split_batches=True, mixed_precision="no", kwargs_handlers=[ddp_kwargs]
    )
    
    # dataset
    train_batch_size = cfg.batch_size
    start_time = time.time()
    dataset = data_io.get_dataset(cfg)
    dl = DataLoader(
        dataset,
        batch_size=train_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
    )
    data_loading_time = time.time() - start_time
    # print(f"Data loading time: {data_loading_time} seconds")
    torch.manual_seed(0)
    # torch.manual_seed(500)
    
    # model = PixelNeRFModel(near=1.2, far=4.0, dim=64, dim_mults=(1, 1, 2, 4)).cuda()
    render_settings = {
        "n_coarse": 64,
        "n_fine": 64,
        "n_coarse_coarse": 32,
        "n_coarse_fine": 0,
        "num_pixels": 64 ** 2,
        "n_feats_out": 64,
        "num_context": 1,
        "sampling": "patch",
        "cnn_refine": False,
        "self_condition": False,
        "lindisp": False,
        # "cnn_refine": True,
    }

    model = PixelNeRFModelCond(
        near=dataset.z_near,
        far=dataset.z_far,
        model=cfg.model_type,
        background_color=dataset.background_color,
        viz_type=cfg.dataset.viz_type,
        use_first_pool=cfg.use_first_pool,
        mode=cfg.mode,
        feats_cond=cfg.feats_cond,
        use_high_res_feats=True,
        render_settings=render_settings,
        use_viewdir=False,
        image_size=dataset.image_size,
        # use_viewdir=True,
    ).cuda()

    diffusion = GaussianDiffusion(
        model,
        image_size=dataset.image_size,
        timesteps=1000,  # number of steps
        sampling_timesteps=cfg.sampling_steps,
        loss_type="l2",  # L1 or L2
        objective="pred_x0",
        beta_schedule="cosine",
        use_guidance=cfg.use_guidance,
        guidance_scale=cfg.guidance_scale,
        temperature=cfg.temperature,
    ).cuda()

    print(f"using lr {cfg.lr}")
    trainer = Trainer(
        diffusion,
        dataloader=dl,
        train_batch_size=train_batch_size,
        train_lr=cfg.lr,
        train_num_steps=700000,  # total training steps
        gradient_accumulate_every=1,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=False,  # turn on mixed precision
        sample_every=1000,
        wandb_every=50,
        save_every=5000,
        num_samples=1,
        warmup_period=1_000,
        checkpoint_path=cfg.checkpoint_path,
        wandb_config=cfg.wandb,
        run_name=cfg.name,
        accelerator=accelerator,
        cfg=cfg,
    )
    sampling_type = cfg.sampling_type
    use_dataset_pose = True
    render_orig_traj = False
    # nsamples = 1
    
    if sampling_type == "simple":
        all_times= 0
        with torch.no_grad():
            for _ in range(1):
                step = 150
                for i in re_indices.interesting_indices:
                    #print(f"Starting rendering step {i}")
                    video_idx = i[0]
                    
                    start_idx = 0
                    end_idx = start_idx + step

                    ctxt_idx = [start_idx]
                    trgt_idx = np.array([end_idx], dtype=np.int64)

                    if i[1] == "flip":
                        trgt_idx = np.array([start_idx], dtype=np.int64)
                        ctxt_idx = [end_idx]

                    ctxt_idx_np = np.array(ctxt_idx, dtype=np.int64)
                    trgt_idx_np = np.array(trgt_idx, dtype=np.int64)

                    #print(f"Starting rendering step {ctxt_idx_np}, {trgt_idx_np}")
                    data = dataset.data_for_video(
                        video_idx=video_idx, ctxt_idx=ctxt_idx_np, trgt_idx=trgt_idx_np,
                    )
                    inp = to_gpu(data[0], "cuda")
                    for k in inp.keys():
                        inp[k] = inp[k].unsqueeze(0)
                    inp["num_frames_render"] = 50

                    if not use_dataset_pose:
                        poses = trainer.model.model.compute_poses(
                            "interpolation", inp, 5, max_angle=15
                        )
                        #print(f"poses shape: {poses.shape}")
                        inp["render_poses"] = poses
                        inp["trgt_c2w"] = poses[-1].unsqueeze(0).unsqueeze(0).cuda()

                    #if not render_orig_traj:
                    #    del inp["render_poses"]
                    #print(f"len of idx: {len(ctxt_idx)}")
                    
                    for j in range(1):  
                        #(nsamples): # Red Herring BEWARE! These are number of literal diff environment samples from path.
                        
                        # print(f"Starting sample {j}. Make sure you have enough samples in path!")
                        
                        discovery = 0
                        renders_per_frame = 1 # 50
                        render_index = 0
                        print("-----Initialisation complete, starting rendering-----")
                        print(f"Will be rendering {renders_per_frame} hypotheses")
                        
                        while(render_index<renders_per_frame):
                            
                            print(f"Rendering hypothesis {render_index+1}")
                            render_index+=1
                            
                            # Inefficient AF!
                            out = trainer.ema.ema_model.sample(batch_size=1, inp=inp)
                            
                            # Requires preprocessing
                            
                            (  
                                rgbd_frames,
                                rgb_frames,
                                depth_frames,
                                conditioning_depth_img,
                            ) = prepare_video_viz(out)
                            
                            # Save to file: Intermediate Processing followed by saving to respective directories to facilitate further post processing
                            
                            asset_saver(rgb_frames, depth_frames, conditioning_depth_img, inp, out, trainer, "/home/projects/Rudra_Generative_Robotics/EK/DFM/results")
                            mover(render_index)
                            save_rgbd(rgbd_frames, render_index)
                            
                            # Visualizer
                            
                            # json_file = '/home/projects/Rudra_Generative_Robotics/EK/DFM/camera_poses/strafe_right.json'
                            
                            # upscaler(method='Wave')
                            # upscaler(method='GAN')
                            
                            
                            # Call the object detector
                            
                            # object_info = detector('sink')
                            
                            # if object_info != None:
                            #     print("Object Found in Hallucination!")
                            #     rgbd_file = save_rgbd(rgbd_frames, save_dir = './RGBD/{timestamp}/')
                            #     observer(rgbd_file, object_info)
                            #     discovery+=1
                            #     all_times += count
                        
                        print(f"Rendering of {renders_per_frame} hypothesis worlds completed!")
                        # print(f"Object discovered {discovery} renders for a total of {all_times}.")
                        print("-------System Exiting-------")
                        
                        exit(0)
                    
                        
        
        
    else:
        print("Other things are stuff of dreams!")                   
        # We do not care about the complete autoregressive method in this one---impossible to scale
        
if __name__ == "__main__":
    train()


# Run code:
# python experiment_scripts/re_results.py dataset=realestate batch_size=1 num_target=1 num_context=1 model_type=dit feats_cond=true sampling_type=simple max_scenes=10000 stage=test use_guidance=true guidance_scale=2.0 temperature=0.85 sampling_steps=50 name=re10k_inference image_size=128 checkpoint_path=files/re10k_model.pt wandb=local
