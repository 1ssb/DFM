# import json
# import numpy as np
# import open3d as o3d
# import os
# import re

# def load_poses(filename):
#     with open(filename, 'r') as file:
#         poses = json.load(file)
#     return {int(key): np.array(value) for key, value in poses.items()}

# def load_rgbd_image(filename):
#     rgbd_data = np.load(filename)
#     rgb = rgbd_data[..., :3]
#     depth = rgbd_data[..., 3]
#     return rgb, depth

# def create_point_cloud(rgb, depth, camera_intrinsics, pose):

#     color_raw = o3d.geometry.Image((rgb * 255).astype(np.uint8))
#     depth_raw = o3d.geometry.Image(depth.astype(np.uint16))
#     rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, depth_scale=1000.0, convert_rgb_to_intensity=False)

#     pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsics)
#     pcd.transform(pose)

#     return pcd

# def main():
#     camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(128, 128, 128*0.48, 128*0.85, 64, 64)
#     poses = load_poses('./camera_poses/recorded_poses.json')

#     rgbd_files = [f for f in os.listdir('./RGBD/rgbd/computed/') if f.endswith('.npy')]
#     all_pcds = []

#     for file in rgbd_files:
#         file_number = int(re.search(r'(\d+)\.npy', file).group(1))
#         if file_number in poses:
#             rgb, depth = load_rgbd_image(f'./RGBD/rgbd/computed/{file}')
#             pcd = create_point_cloud(rgb, depth, camera_intrinsics, poses[file_number])
#             all_pcds.append(pcd)

#     combined_pcd = o3d.geometry.PointCloud()
#     for pcd in all_pcds:
#         combined_pcd += pcd

#     o3d.io.write_point_cloud("./RGBD/computed_depth.ply", combined_pcd)

# if __name__ == "__main__":
#     main()

############# Unnormalised NDC scale ##############

# import json
# import numpy as np
# import open3d as o3d
# import os
# import re

# def load_poses(filename, pose_scale=1):
#     with open(filename, 'r') as file:
#         poses = json.load(file)
#     return {int(key): np.array(value) * pose_scale for key, value in poses.items()}

# def load_rgbd_image(filename):
#     rgbd_data = np.load(filename)
#     rgb = rgbd_data[..., :3]
#     depth = rgbd_data[..., 3]
#     return rgb, depth

# def create_point_cloud(rgb, depth, pose, img_width, img_height):
#     points = []
#     colors = []

#     for v in range(img_height):
#         for u in range(img_width):
#             color = rgb[v, u] / 255.0
#             Z = depth[v, u]
#             if Z == 0: continue  # Skip if no depth information

#             # Convert to NDC and then to camera space
#             X = (u / img_width - 0.5) * 2 * Z
#             Y = (v / img_height - 0.5) * 2 * Z

#             points.append([X, Y, Z])
#             colors.append(color)

#     # Create point cloud
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(np.array(points))
#     pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
#     pcd.transform(pose)

#     return pcd

# def main():
#     img_width, img_height = 128, 128
#     poses = load_poses('./camera_poses/recorded_poses.json', pose_scale=1)

#     rgbd_files = sorted([f for f in os.listdir('./RGBD/rgbd/computed/') if f.endswith('.npy')])
#     all_pcds = []

#     for file in rgbd_files:
#         file_number = int(re.search(r'(\d+)\.npy', file).group(1))
#         if file_number in poses:
#             rgb, depth = load_rgbd_image(f'./RGBD/rgbd/computed/{file}')
#             pcd = create_point_cloud(rgb, depth, poses[file_number], img_width, img_height)
#             all_pcds.append(pcd)

#     combined_pcd = o3d.geometry.PointCloud()
#     for pcd in all_pcds:
#         combined_pcd += pcd

#     o3d.io.write_point_cloud("./RGBD/computed_depth.ply", combined_pcd)

# if __name__ == "__main__":
#     main()



#### If depth is actually distance ########

# import json
# import numpy as np
# import open3d as o3d
# import os
# import re

# def load_poses(filename, pose_scale=1):
#     with open(filename, 'r') as file:
#         poses = json.load(file)
#     return {int(key): np.array(value) * pose_scale for key, value in poses.items()}

# def load_rgbd_image(filename):
#     rgbd_data = np.load(filename)
#     rgb = rgbd_data[..., :3]
#     depth = rgbd_data[..., 3]
#     return rgb, depth

# def create_point_cloud(rgb, distance, pose, img_width, img_height):
#     points = []
#     colors = []

#     for v in range(img_height):
#         for u in range(img_width):
#             color = rgb[v, u] / 255.0
#             D = distance[v, u]  # Radial distance
#             if D == 0: continue  # Skip if no depth information

#             # Convert from pixel coordinates to normalized device coordinates
#             x_ndc = (u / img_width - 0.5) * 2
#             y_ndc = (v / img_height - 0.5) * 2

#             # Back-project to camera space (assuming pinhole camera model)
#             Z = D / np.sqrt(x_ndc**2 + y_ndc**2 + 1)  # Calculate true depth value
#             X = x_ndc * Z
#             Y = y_ndc * Z

#             points.append([X, Y, Z])
#             colors.append(color)

#     # Create point cloud
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(np.array(points))
#     pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
#     pcd.transform(pose)

#     return pcd

# def main():
#     img_width, img_height = 128, 128
#     poses = load_poses('./camera_poses/recorded_poses.json', pose_scale=1)

#     rgbd_files = sorted([f for f in os.listdir('./RGBD/rgbd/computed/') if f.endswith('.npy')])
#     all_pcds = []

#     for file in rgbd_files:
#         file_number = int(re.search(r'(\d+)\.npy', file).group(1))
#         if file_number in poses:
#             rgb, depth = load_rgbd_image(f'./RGBD/rgbd/computed/{file}')
#             pcd = create_point_cloud(rgb, depth, poses[file_number], img_width, img_height)
#             all_pcds.append(pcd)

#     combined_pcd = o3d.geometry.PointCloud()
#     for pcd in all_pcds:
#         combined_pcd += pcd

#     o3d.io.write_point_cloud("./RGBD/computed_clouds.ply", combined_pcd)

# if __name__ == "__main__":
#     main()


######### Correcting Pose dictionary ##########
# import json
# import os

# def correct_json_keys(file_path):
#     # Check if the file exists
#     if not os.path.exists(file_path):
#         print(f"File not found: {file_path}")
#         return

#     # Load the JSON data
#     with open(file_path, 'r') as file:
#         data = json.load(file)

#     # Correct the keys
#     new_data = {}
#     for key in sorted(data.keys(), key=int):
#         new_key = int(key)
#         if new_key >= 16:
#             new_key -= 1  # Decrement the key by 1 for keys >= 16
#         new_data[str(new_key)] = data[key]

#     # Save the updated data back to the file
#     with open(file_path, 'w') as file:
#         json.dump(new_data, file, indent=4)

# # Path to the JSON file
# file_path = '/home/projects/Rudra_Generative_Robotics/EK/DFM/camera_poses/recorded_poses_20231207_131523.json'
# correct_json_keys(file_path)


###### Conditional depths to Point clouds
import open3d as o3d
import numpy as np
import os
import json
import random
from datetime import datetime

def load_poses(pose_file):
    with open(pose_file, 'r') as file:
        poses = json.load(file)
    return {int(k): np.array(v) for k, v in poses.items()}

def create_point_cloud_from_depth(depth_image, cx, cy, fx, fy):
    if depth_image.ndim > 2:
        depth_image = depth_image[:, :, 0]  # Assuming depth is in the first channel
    h, w = depth_image.shape
    points = []

    for v in range(h):
        for u in range(w):
            Z = depth_image[v, u]
            if Z == 0:
                continue

            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            points.append([X, Y, Z])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    return pcd

def transform_point_cloud(pcd, pose):
    transform = np.linalg.inv(pose)
    return pcd.transform(transform)

def combine_point_clouds(depth_frames_dir, poses, fx, fy, cx, cy):
    selected_frames = random.sample(range(1, 6), 2)
    combined_pcd = o3d.geometry.PointCloud()

    for frame in selected_frames:
        depth_frame_path = os.path.join(depth_frames_dir, f"frame_{frame}.npy")
        if os.path.exists(depth_frame_path) and frame in poses:
            depth_image = np.load(depth_frame_path)
            pcd = create_point_cloud_from_depth(depth_image, cx, cy, fx, fy)
            transformed_pcd = transform_point_cloud(pcd, poses[frame])
            combined_pcd += transformed_pcd

    return combined_pcd

# Camera intrinsics
fx, fy = 0.48 * 128, 0.48 * 128  # Example values
cx, cy = 64, 64  # Center of the image

# Load poses
poses = load_poses('/home/projects/Rudra_Generative_Robotics/EK/DFM/camera_poses/recorded_poses_20231207_131523.json')

# Directory containing saved depth frames
depth_frames_dir = "./only_depth_frames"

# Combine point clouds from two random frames
combined_pcd = combine_point_clouds(depth_frames_dir, poses, fx, fy, cx, cy)

# Save the combined point cloud
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"./RGBD/combine/new_cloud_{timestamp}.ply"
o3d.io.write_point_cloud(output_filename, combined_pcd)



