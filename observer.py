import os, json
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from datetime import datetime
import open3d as o3d
from boxes import detector 
import copy

def depth_map_to_point_cloud(depth_map, centroids, enlargement_factor=1):
    height, width = depth_map.shape
    points = []
    colors = []

    for y in range(height):
        for x in range(width):
            z = depth_map[y, x]
            if z >= 0.7 and z <= 10.0:
                x_world = (x - 64.0) * z / 128.0
                y_world = (y - 64.0) * z / 128.0
                z_world = z
                points.append([x_world, y_world, z_world])
                colors.append([0, 1, 0])  # Default to green

    # Add enlarged red points for centroids
    for cx, cy in centroids:
        cx_int, cy_int = round(cx), round(cy)
        if 0 <= cx_int < width and 0 <= cy_int < height:
            z = depth_map[cy_int, cx_int]
            if z >= 0.7 and z <= 10.0:
                x_world = (cx_int - 64.0) * z / 128.0
                y_world = (cy_int - 64.0) * z / 128.0
                z_world = z
                # Add additional points around the centroid
                for i in range(-enlargement_factor, enlargement_factor + 1):
                    for j in range(-enlargement_factor, enlargement_factor + 1):
                        points.append([x_world + i * z / 128.0, y_world + j * z / 128.0, z_world])
                        colors.append([1, 0, 0])  # Red color

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

def transform_and_merge_point_clouds(all_clouds, translation_step=0.002, dir = 'R'):
    
    if dir == 'R':
        d = -1
    elif dir == 'L':
        d = 1
        
    merged_pcd = o3d.geometry.PointCloud()
    translation = 0.0
    for cloud in all_clouds:
        cloud_copy = copy.deepcopy(cloud)
        cloud_copy.points = o3d.utility.Vector3dVector(np.asarray(cloud_copy.points) + np.array([d*translation, 0, 0]))
        merged_pcd += cloud_copy
        translation += translation_step

    return merged_pcd
    
def call_rgbds_to_point_cloud(directory_path):
    pcd_all = o3d.geometry.PointCloud()
    depth_maps = {}
    all_clouds = []
    for i in range(1, 51):
        file_path = os.path.join(directory_path, f"{i}.npy")
        if os.path.exists(file_path):
            rgbd_data = np.load(file_path)
            depth_channel = rgbd_data[:, :, 3]
            depth_maps[f"frame_{i}"] = depth_channel
            cloud = depth_map_to_point_cloud(depth_channel)
            all_clouds.append(cloud)
    pcd_all = transform_and_merge_point_clouds(all_clouds) 
    return pcd_all, depth_maps

def transform_centroids(centroids, pose, scale=0.001):
    transformed_centroids = []
    for centroid in centroids:
        centroid_homog = np.append(centroid, 1)
        transformed = pose @ centroid_homog
        transformed[0] = transformed[0] * scale
        transformed[1] *= scale
        transformed[2] *= scale
        transformed_centroids.append(transformed[:3])
    return transformed_centroids

def load_poses(pose_file):
    with open(pose_file, 'r') as file:
        poses = json.load(file)
    return {int(k): np.array(v) for k, v in poses.items()}

def return_centroids(detection_dict):
    
    extracted_info = {}
    for key, value in detection_dict.items():
        if key.startswith("bbox_frame_"):
            frame_number = key.split("_")[2]
            centroids = [[(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2] for bbox in value]
            extracted_info[f"{frame_number}"] = centroids    
    return extracted_info

def observer(rgbd_directory, object_info):
    all_clouds = []
    depth_maps = {}
    
    # Load depth maps and generate point clouds for each frame
    for i in range(1, 51):
        file_path = os.path.join(rgbd_directory, f"{i}.npy")
        if os.path.exists(file_path):
            rgbd_data = np.load(file_path)
            depth_channel = rgbd_data[:, :, 3]
            depth_maps[f"frame_{i}"] = depth_channel
    
    frame_centroids = return_centroids(object_info)

    for frame, centroids in frame_centroids.items():
        depth_map = depth_maps.get(f"frame_{frame}")
        if depth_map is not None:
            # Round centroids to nearest integer
            rounded_centroids = [(round(cx), round(cy)) for cx, cy in centroids]
            # Generate point cloud with highlighted centroids
            cloud = depth_map_to_point_cloud(depth_map, rounded_centroids)
            all_clouds.append(cloud)

    # Merge all point clouds
    pcd_all = transform_and_merge_point_clouds(all_clouds)

    # Save the combined point cloud
    output_file = f"./centroids.ply"
    o3d.io.write_point_cloud(output_file, pcd_all)
    print(f"Combined point cloud data saved to {output_file}!")

    return 0

# Example usage
object_info = detector('chair')
if not object_info:
    print("Object not detected")
else:
    rgbd_directory = "/home/projects/Rudra_Generative_Robotics/EK/DFM/RGBD/test"
    observer(rgbd_directory, object_info)
