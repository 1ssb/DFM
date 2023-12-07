import json
import numpy as np
import open3d as o3d
import os
import re
import random

# Global option for selecting consecutive or random pairs
SELECT_CONSECUTIVE = False # True
F = 128 * 0.48
FX, FY = F , F

def load_poses(filename):
    with open(filename, 'r') as file:
        poses = json.load(file)
    return {int(key): np.array(value) for key, value in poses.items()}

def load_rgbd_image(filename):
    rgbd_data = np.load(filename)
    rgb = rgbd_data[..., :3]
    depth = rgbd_data[..., 3]
    return rgb, depth

def create_point_cloud(rgb, depth, pose, fx, fy, cx, cy):
    points = []
    colors = []

    for v in range(rgb.shape[0]):
        for u in range(rgb.shape[1]):
            Z = depth[v, u]
            if Z == 0: continue

            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy

            points.append([X, Y, Z])
            colors.append(rgb[v, u] / 255)  # Normalize RGB values

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    pcd.transform(pose)

    return pcd

def add_coordinate_axes():
    # Create coordinate axes at the world origin
    coord_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    return coord_axes

def main():
    fx, fy, cx, cy = FX, FY, 64, 64
    poses = load_poses('/home/projects/Rudra_Generative_Robotics/EK/DFM/camera_poses/recorded_poses_20231207_131523.json')

    rgbd_files = sorted([f for f in os.listdir('/home/projects/Rudra_Generative_Robotics/EK/DFM/RGBD/strafe-right/sample_1_20231204-204324/') if f.endswith('.npy')])
    if len(rgbd_files) < 2:
        print("Not enough RGBD files to compare.")
        return

    # Select two files based on the global option
    if SELECT_CONSECUTIVE:
        # Select two consecutive files
        idx = random.randint(0, len(rgbd_files) - 2)
        file1, file2 = rgbd_files[idx], rgbd_files[idx + 1]
    else:
        # Select two random files
        idx1, idx2 = random.sample(range(len(rgbd_files)), 2)
        file1, file2 = rgbd_files[idx1], rgbd_files[idx2]

    print(f"Files chosen are {file1} and {file2}.")
    rgb1, depth1 = load_rgbd_image(f'/home/projects/Rudra_Generative_Robotics/EK/DFM/RGBD/strafe-right/sample_1_20231204-204324/{file1}')
    rgb2, depth2 = load_rgbd_image(f'/home/projects/Rudra_Generative_Robotics/EK/DFM/RGBD/strafe-right/sample_1_20231204-204324/{file2}')

    pose_key1 = int(re.search(r'(\d+)\.npy', file1).group(1))
    pose_key2 = int(re.search(r'(\d+)\.npy', file2).group(1))
    print(f"Pose keys chosen are {pose_key1} and {pose_key2}.")  # Print the chosen pose keys

    pose1 = poses[pose_key1]
    pose2 = poses[pose_key2]
    print(f"{pose_key1}:", pose1)
    print(f"{pose_key2}:", pose2)
    pcd1 = create_point_cloud(rgb1, depth1, pose1, fx, fy, cx, cy)
    pcd2 = create_point_cloud(rgb2, depth2, pose2, fx, fy, cx, cy)

    combined_pcd = pcd1 + pcd2
    coord_axes = add_coordinate_axes()

    # # Calculate and print distances
    # geodesic_distance = np.linalg.norm(pose1[:3, 3] - pose2[:3, 3])
    # euclidean_distance = np.linalg.norm(pcd1.get_center() - pcd2.get_center())
    # print(f"Geodesic distance: {geodesic_distance}")
    # print(f"Euclidean distance: {euclidean_distance}")

    # Save the point cloud with RGB information
    output_filename = f"./RGBD/combine/combined_point_cloud_{pose_key1}_{pose_key2}.ply"
    o3d.io.write_point_cloud(output_filename, combined_pcd)
    print(f"Point cloud saved as {output_filename}")

if __name__ == "__main__":
    main()

####### Script to find tipping point #######

# import json
# import numpy as np
# import open3d as o3d
# import os
# import re
# from concurrent.futures import ThreadPoolExecutor
# from tqdm import tqdm

# RELAX = 20  # Percentage to relax the baseline by
# BASELINE_PAIRS_COUNT = 5  # Number of pairs to consider for baseline calculation

# def load_poses(filename):
#     with open(filename, 'r') as file:
#         poses = json.load(file)
#     return {int(key): np.array(value) for key, value in poses.items()}

# def load_rgbd_image(filename):
#     rgbd_data = np.load(filename)
#     rgb = rgbd_data[..., :3]
#     depth = rgbd_data[..., 3]
#     return rgb, depth

# def create_point_cloud(rgb, depth, pose, fx, fy, cx, cy):
#     points = []
#     colors = []

#     for v in range(rgb.shape[0]):
#         for u in range(rgb.shape[1]):
#             Z = depth[v, u]
#             if Z == 0: continue

#             X = (u - cx) * Z / fx
#             Y = (v - cy) * Z / fy

#             points.append([X, Y, Z])
#             colors.append(rgb[v, u] / 255)

#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(np.array(points))
#     pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
#     pcd.transform(pose)

#     return pcd

# def add_coordinate_axes():
#     coord_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
#     return coord_axes

# def extract_pose_key(filename):
#     match = re.search(r'(\d+)\.npy', filename)
#     return int(match.group(1)) if match else None

# def calculate_chamfer_distance(pcd1, pcd2):
#     pcd1_tree = o3d.geometry.KDTreeFlann(pcd1)
#     pcd2_tree = o3d.geometry.KDTreeFlann(pcd2)

#     distance = 0
#     for point in pcd1.points:
#         _, idx, _ = pcd2_tree.search_knn_vector_3d(point, 1)
#         nearest_point = np.asarray(pcd2.points)[idx[0]]
#         distance += np.linalg.norm(point - nearest_point)
    
#     for point in pcd2.points:
#         _, idx, _ = pcd1_tree.search_knn_vector_3d(point, 1)
#         nearest_point = np.asarray(pcd1.points)[idx[0]]
#         distance += np.linalg.norm(point - nearest_point)

#     return distance / (len(pcd1.points) + len(pcd2.points))

# def compare_poses(start, end, poses, rgbd_files, fx, fy, cx, cy, baseline_distance, inconsistencies):
#     file1, file2 = rgbd_files[start], rgbd_files[end]
#     pose_key1 = extract_pose_key(file1)
#     pose_key2 = extract_pose_key(file2)

#     print(f"Comparing poses {pose_key1} and {pose_key2}...")

#     rgb1, depth1 = load_rgbd_image(f'./RGBD/rgbd/computed/{file1}')
#     rgb2, depth2 = load_rgbd_image(f'./RGBD/rgbd/computed/{file2}')

#     pcd1 = create_point_cloud(rgb1, depth1, poses[pose_key1], fx, fy, cx, cy)
#     pcd2 = create_point_cloud(rgb2, depth2, poses[pose_key2], fx, fy, cx, cy)

#     chamfer_distance = calculate_chamfer_distance(pcd1, pcd2)
#     if chamfer_distance > baseline_distance * (1 + RELAX/100):
#         inconsistencies.append((pose_key1, pose_key2))

# def main():
#     fx, fy, cx, cy = 128 * 0.48, 128 * 0.85, 64, 64
#     poses = load_poses('./camera_poses/recorded_poses.json')

#     rgbd_files = sorted([f for f in os.listdir('./RGBD/rgbd/computed/') if f.endswith('.npy')])
#     if len(rgbd_files) < 2:
#         print("Not enough RGBD files to compare.")
#         return

#     # Calculate baseline Chamfer Distance for the first few pairs
#     baseline_distances = []
#     for i in range(1, min(BASELINE_PAIRS_COUNT, len(rgbd_files))):
#         file1, file2 = rgbd_files[0], rgbd_files[i]
#         pose_key1, pose_key2 = extract_pose_key(file1), extract_pose_key(file2)
#         rgb1, depth1 = load_rgbd_image(f'./RGBD/rgbd/computed/{file1}')
#         rgb2, depth2 = load_rgbd_image(f'./RGBD/rgbd/computed/{file2}')
#         pcd1 = create_point_cloud(rgb1, depth1, poses[pose_key1], fx, fy, cx, cy)
#         pcd2 = create_point_cloud(rgb2, depth2, poses[pose_key2], fx, fy, cx, cy)
#         baseline_distances.append(calculate_chamfer_distance(pcd1, pcd2))

#     baseline_average = sum(baseline_distances) / len(baseline_distances)
#     inconsistency_threshold = baseline_average * (1 + RELAX / 100)
#     inconsistencies = []

#     print(f"Baseline average distance: {baseline_average}")
#     print(f"Inconsistency threshold: {inconsistency_threshold}")

#     with ThreadPoolExecutor() as executor:
#         for start in range(len(rgbd_files) - 1):
#             futures = [executor.submit(compare_poses, start, end, poses, rgbd_files, fx, fy, cx, cy, inconsistency_threshold, inconsistencies)
#                        for end in range(start + 1, len(rgbd_files))]
#             for future in tqdm(futures):
#                 future.result()

#     if inconsistencies:
#         print("Inconsistencies detected at the following pose pairs:")
#         for pose_key1, pose_key2 in inconsistencies:
#             print(f"Poses {pose_key1} and {pose_key2}")
#     else:
#         print("No further inconsistencies detected within the threshold.")

# if __name__ == "__main__":
#     main()


