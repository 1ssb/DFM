# from ultralytics import YOLO
# import os

# def detector(object_name, model_path='yolov8n.pt'):
#     print("Now Detecting")
#     image_directory = "/home/projects/Rudra_Generative_Robotics/EK/DFM/results/strafe-right/sample_1/images"

#     # Initialize the model
#     model = YOLO(model_path)  
#     results = model(image_directory, stream=False, verbose=False)

#     detection_info = {"frames": []}
#     found = False

#     for im, r in enumerate(results, start=1):
#         objects = [r.names[int(cls)] for cls in r.boxes.cls.cpu()]
#         boxes = r.boxes.xyxy.cpu().tolist()
#         confs = r.boxes.conf.cpu().tolist()
#         print(r)
#         if object_name in objects:
#             found = True
#             frame_key = f"frame_{im}"
#             detection_info["frames"].append(frame_key)
#             detection_info[f"bbox_{frame_key}"] = [box for obj, box in zip(objects, boxes) if obj == object_name]
#             detection_info[f"conf_{frame_key}"] = [conf for obj, conf in zip(objects, confs) if obj == object_name]

#     return detection_info if found else None

# # Example usage
# object_info = detector('sink')
# if object_info:
#     print("Object detected:")
#     # print(object_info)
# else:
#     print("Object not detected")




# from ultralytics import YOLO
# import os
# import json

# def detect_all_objects(model_path='yolov8n.pt'):
#     print("Starting Detection")
#     image_directory = "/home/projects/Rudra_Generative_Robotics/EK/DFM/data/strafe_right/sample_1/images"

#     # Initialize the model
#     model = YOLO(model_path)
#     results = model(image_directory, stream=False, verbose=False)

#     # Dictionary to store detection information for each frame
#     all_detections = {}

#     for im, r in enumerate(results, start=1):
#         frame_key = f"frame_{im}"
#         objects = [r.names[int(cls)] for cls in r.boxes.cls.cpu()]
#         boxes = r.boxes.xyxy.cpu().tolist()
#         confs = r.boxes.conf.cpu().tolist()

#         # Store detections for the current frame
#         frame_detections = []
#         for obj, box, conf in zip(objects, boxes, confs):
#             frame_detections.append({
#                 "class": obj,
#                 "bbox": box,  # [x1, y1, x2, y2]
#                 "confidence": conf
#             })

#         all_detections[frame_key] = frame_detections

#     return all_detections

# # Detect all objects and save the results to a JSON file
# detections = detect_all_objects()
# output_json_path = "/home/projects/Rudra_Generative_Robotics/EK/DFM/data/strafe_right/sample_1/detections_sample_1.json"
# with open(output_json_path, 'w') as json_file:
#     json.dump(detections, json_file, indent=4)

# print(f"Detections saved to {output_json_path}")


from ultralytics import YOLO
import os
import json

def detect_all_objects_in_sample(sample_number, model_path='yolov8n.pt'):
    print(f"Starting Detection for Sample {sample_number}")

    # Construct the directory paths
    image_directory = f"/home/projects/Rudra_Generative_Robotics/EK/DFM/data/strafe_right/sample_{sample_number}/images"
    output_json_path = f"/home/projects/Rudra_Generative_Robotics/EK/DFM/data/strafe_right/sample_{sample_number}/detections_sample_{sample_number}.json"

    # Initialize the model
    model = YOLO(model_path)
    results = model(image_directory, stream=False, verbose=False)

    # Dictionary to store detection information for each frame
    all_detections = {}

    for im, r in enumerate(results, start=1):
        frame_key = f"frame_{im}"
        objects = [r.names[int(cls)] for cls in r.boxes.cls.cpu()]
        boxes = r.boxes.xyxy.cpu().tolist()
        confs = r.boxes.conf.cpu().tolist()

        # Store detections for the current frame
        frame_detections = []
        for obj, box, conf in zip(objects, boxes, confs):
            frame_detections.append({
                "class": obj,
                "bbox": box,  # [x1, y1, x2, y2]
                "confidence": conf
            })

        all_detections[frame_key] = frame_detections

    # Save detections to JSON
    with open(output_json_path, 'w') as json_file:
        json.dump(all_detections, json_file, indent=4)

    print(f"Detections for Sample {sample_number} saved to {output_json_path}")

# Loop through sample directories from 1 to 50
for sample_number in range(1, 51):
    detect_all_objects_in_sample(sample_number)













# import shutil
# import re
# from datetime import datetime

# def mover():
#     from_directory = "/home/projects/Rudra_Generative_Robotics/EK/DFM/results/"
#     images = "/home/projects/Rudra_Generative_Robotics/EK/DFM/results/images/"
#     depth_maps = "/home/projects/Rudra_Generative_Robotics/EK/DFM/results/depth_maps/"
#     videos = "/home/projects/Rudra_Generative_Robotics/EK/DFM/results/videos/"

#     # Regular expression patterns to match files
#     pattern1 = re.compile(r"frame_\d{4}\.png")
#     pattern2 = re.compile(r"frame_depth_\d{4}\.png")

#     # Create directories if they don't exist
#     for directory in [images, depth_maps, videos]:
#         os.makedirs(directory, exist_ok=True)

#     # Generate timestamped subfolder for videos
#     timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     video_results_dir = os.path.join(videos, f"video_results_{timestamp}")
#     os.makedirs(video_results_dir, exist_ok=True)

#     for file_name in os.listdir(from_directory):
#         from_path = os.path.join(from_directory, file_name)

#         if pattern1.match(file_name):
#             to_path = os.path.join(images, file_name)
#         elif pattern2.match(file_name):
#             to_path = os.path.join(depth_maps, file_name)
#         elif file_name in ["video.mp4", "depth_video.mp4"]:
#             to_path = os.path.join(video_results_dir, file_name)
#         else:
#             continue

#         try:
#             shutil.move(from_path, to_path)
#         except Exception as e:
#             print(f"Error moving file {file_name}: {e}")

#     print("Results Cleaned!")
