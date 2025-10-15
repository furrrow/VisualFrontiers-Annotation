import os
import json
import math
import numpy as np
import csv

def get_existing_bags(path):
    existing_bags = []

    for root, dirs, files in os.walk(path):
        print(f"Number of files in {root}: {len(files)}")
        for filename in files:
            if filename.endswith(".bag"):
                existing_bags.append(filename)

    return existing_bags

def get_topics_and_info_from_bag(bag_path):
    try:
        import rosbag  # ROS1
        with rosbag.Bag(bag_path, "r") as bag:
            topic_info = bag.get_type_and_topic_info().topics
        
        topics_dict = {
            topic: info.msg_type
            for topic, info in topic_info.items()
        }
        return topics_dict
    
    except Exception as e:
        print(f"Error reading {bag_path}: {e}")
        return {}

def get_topics_from_bag(bag_path):
    try:
        import rosbag  # ROS1
        with rosbag.Bag(bag_path, "r") as bag:
            topic_info = bag.get_type_and_topic_info().topics
        
        topics = list(topic_info.keys())
        return topics
    
    except Exception as e:
        print(f"Error reading {bag_path}: {e}")
        return []
    
def write_dict_to_json(data_dict, output_path):
    with open(output_path, 'w') as json_file:
        json.dump(data_dict, json_file, indent=4)

def ensure_csv_has_header(path):
    exists = os.path.exists(path)
    if not exists or os.path.getsize(path) == 0:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["bag_name", "oov_count", "total_count"])

def append_csv_row(path, bag_name, oov_count, total_count):
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([bag_name, oov_count, total_count])