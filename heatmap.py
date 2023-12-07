import numpy as np
import open3d as o3d
import cv2
import os
from ultralytics import YOLO
import glob

COCO_OBJECT = 'sink'  # Specify the COCO object to detect

def create_point_cloud(rgb, depth, fx, fy, cx, cy, colors):
    points = []
    for v in range(rgb.shape[0]):
        for u in range(rgb.shape[1]):
            Z = depth[v, u]
            if Z == 0: continue
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            points.append([X, Y, Z])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    return pcd

def process_and_save_ply(rgbd_file_path, output_directory):
    model_path = 'yolov8n.pt'
    rgbd_array = np.load(rgbd_file_path)
    rgb_image = rgbd_array[..., :3].astype(np.uint8)
    depth_image = rgbd_array[..., 3]
    model = YOLO(model_path)
    results = model(cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR), stream=False, verbose=False)
    fx, fy = 128 * 0.48, 128 * 0.48
    cx, cy = 64, 64
    colors = np.full((rgb_image.shape[0], rgb_image.shape[1], 3), [0, 1, 0])  # Green color for other points

    max_confidence = 0
    detected = False
    for r in results:
        objects = [r.names[int(cls)] for cls in r.boxes.cls.cpu()]
        boxes = r.boxes.xyxy.cpu().tolist()
        confs = r.boxes.conf.cpu().tolist()
        for obj, box, conf in zip(objects, boxes, confs):
            if obj == COCO_OBJECT:
                x_min, y_min, x_max, y_max = map(int, box[:4])
                max_confidence = max(max_confidence, conf)
                colors[y_min:y_max, x_min:x_max, :] = [1, 0, 0]  # Red color for detected objects
                detected = True

    if detected:
        pcd = create_point_cloud(rgb_image, depth_image, fx, fy, cx, cy, colors.reshape(-1, 3))
        confidence_str = f"{max_confidence:.2f}"
        output_ply_path = os.path.join(output_directory, f"{os.path.basename(rgbd_file_path).split('.')[0]}_{confidence_str}.ply")
        o3d.io.write_point_cloud(output_ply_path, pcd)
        print(f"{COCO_OBJECT} detected with confidence {confidence_str}. Saved PLY file to {output_ply_path}")

def process_directory(directory_path):
    parent_directory = os.path.dirname(directory_path)
    base_output_directory = os.path.join(parent_directory, f'heatmaps_{COCO_OBJECT}')
    os.makedirs(base_output_directory, exist_ok=True)
    for rgbd_file_path in glob.glob(os.path.join(directory_path, '*.npy')):
        process_and_save_ply(rgbd_file_path, base_output_directory)

# Example usage
directory_path = '/home/projects/Rudra_Generative_Robotics/EK/DFM/data/strafe_right/sample_22/rgbd'
process_directory(directory_path)









########## all objects ##########




# import numpy as np
# import open3d as o3d
# import cv2
# import os
# from ultralytics import YOLO
# import glob

# DETECT_ALL = False  # Set to False to detect only a specific object
# COCO_OBJECT = 'tv'  # Object to detect if DETECT_ALL is False

# def create_point_cloud(rgb, depth, fx, fy, cx, cy, colors):
#     points = []
#     for v in range(rgb.shape[0]):
#         for u in range(rgb.shape[1]):
#             Z = depth[v, u]
#             if Z == 0: continue
#             X = (u - cx) * Z / fx
#             Y = (v - cy) * Z / fy
#             points.append([X, Y, Z])
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(np.array(points))
#     pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
#     return pcd

# def process_and_save_ply(rgbd_file_path, output_directory):
#     model_path = 'yolov8n.pt'
#     rgbd_array = np.load(rgbd_file_path)
#     rgb_image = rgbd_array[..., :3].astype(np.uint8)
#     depth_image = rgbd_array[..., 3]
#     model = YOLO(model_path)
#     results = model(cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR), stream=False, verbose=False)
#     fx, fy = 128 * 0.48, 128 * 0.48
#     cx, cy = 64, 64
#     colors = np.full((rgb_image.shape[0], rgb_image.shape[1], 3), [0, 1, 0])  # Green color for other points

#     detected_objects = set()
#     for r in results:
#         objects = [r.names[int(cls)] for cls in r.boxes.cls.cpu()]
#         boxes = r.boxes.xyxy.cpu().tolist()
#         for obj, box in zip(objects, boxes):
#             if DETECT_ALL or obj == COCO_OBJECT:
#                 x_min, y_min, x_max, y_max = map(int, box[:4])
#                 detected_objects.add(obj)
#                 colors[y_min:y_max, x_min:x_max, :] = [1, 0, 0]  # Red color for detected objects

#     if not detected_objects or (not DETECT_ALL and COCO_OBJECT not in detected_objects):
#         print(f"No specified objects detected in {rgbd_file_path}. Skipping PLY creation.")
#         return

#     pcd = create_point_cloud(rgb_image, depth_image, fx, fy, cx, cy, colors.reshape(-1, 3))
#     output_ply_path = os.path.join(output_directory, f"{os.path.basename(rgbd_file_path).split('.')[0]}_object_detected.ply")
#     o3d.io.write_point_cloud(output_ply_path, pcd)
#     print(f"Saved PLY file to {output_ply_path}")

# def process_directory(directory_path):
#     base_output_directory = './HeatMaps'
#     sample_name = os.path.basename(os.path.normpath(directory_path))
#     output_directory = os.path.join(base_output_directory, sample_name if DETECT_ALL else COCO_OBJECT + '_' + sample_name)
#     os.makedirs(output_directory, exist_ok=True)
#     for rgbd_file_path in glob.glob(os.path.join(directory_path, '*.npy')):
#         process_and_save_ply(rgbd_file_path, output_directory)

# # Example usage
# directory_path = '/home/projects/Rudra_Generative_Robotics/EK/DFM/data/strafe_right/sample_13/rgbd'
# process_directory(directory_path)
