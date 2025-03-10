"""
Script for converting CognitiveDrone dataset to LeRobot format.

This script converts the CognitiveDrone dataset (stored in RLDS format) to the LeRobot format
which is required to fine-tune π0.

Usage:
uv run examples/cognitive_drone/convert_cognitive_drone_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/cognitive_drone/convert_cognitive_drone_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can clone the raw CognitiveDrone dataset from:
https://huggingface.co/datasets/ArtemLykov/CognitiveDrone_dataset
The resulting dataset will get saved to the $LEROBOT_HOME directory.
"""

import shutil
import os
import glob
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow as tf
import tyro

REPO_NAME = "cognitive_drone"  # Name of the output dataset, also used for the Hugging Face Hub

# Define the feature description for parsing TFRecord files
# You may need to adjust this based on the actual structure of your data
def get_feature_description():
    return {
        'steps': tf.io.FixedLenFeature([], tf.string),
    }

def parse_episode(serialized_example):
    # Parse the TFRecord example
    example = tf.io.parse_single_example(serialized_example, get_feature_description())
    
    # Parse the nested steps feature (adjust as needed)
    steps_feature = {
        'observation': tf.io.FixedLenFeature([], tf.string),
        'action': tf.io.FixedLenFeature([], tf.string),
        'language_instruction': tf.io.FixedLenFeature([], tf.string, default_value=b''),
    }
    
    # The steps tensor is a serialized sequence of step examples
    steps = tf.io.parse_sequence_example(
        example['steps'],
        sequence_features=steps_feature
    )
    
    return steps

def main(data_dir: str, *, push_to_hub: bool = False):
    # Clean up any existing dataset in the output directory
    output_path = LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)
    
    # Create LeRobot dataset, define features to store
    # Based on the CognitiveDrone dataset structure
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="drone",
        fps=10,  # Adjust based on the actual frame rate of the dataset
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),  # Adjust based on the actual image dimensions
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (12,),  # Adjust based on the drone state dimensions (position, orientation, etc.)
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (4,),  # Typical drone control (roll, pitch, yaw, throttle)
                "names": ["action"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Find all TFRecord files in the specified directory
    if os.path.isdir(os.path.join(data_dir, "data", "rlds", "train")):
        tfrecord_pattern = os.path.join(data_dir, "data", "rlds", "train", "*.tfrecord-*")
    else:
        tfrecord_pattern = os.path.join(data_dir, "*.tfrecord-*")
    
    tfrecord_files = glob.glob(tfrecord_pattern)
    
    if not tfrecord_files:
        raise ValueError(f"No TFRecord files found in {tfrecord_pattern}")
    
    print(f"Found {len(tfrecord_files)} TFRecord files.")
    
    # Create a dataset from the TFRecord files
    raw_dataset = tf.data.TFRecordDataset(tfrecord_files)
    for raw_record in raw_dataset.take(1):
        # Parse with a generic feature spec that captures all keys as strings
        example = tf.io.parse_single_example(raw_record, {"dummy": tf.io.VarLenFeature(tf.string)})
        print("Record keys:", list(example.keys()))
    
    episode_count = 0
    for serialized_example in raw_dataset:
        try:
            # Process each episode
            steps = parse_episode(serialized_example)
            
            # Extract steps data
            for i in range(len(steps['observation'])):
                # Add each frame to the dataset
                # Adjust the keys and parsing based on the actual structure of your data
                observation = tf.io.parse_tensor(steps['observation'][i], out_type=tf.float32)
                action = tf.io.parse_tensor(steps['action'][i], out_type=tf.float32)
                
                # Assuming observation contains both image and state
                image = observation[:256*256*3].reshape((256, 256, 3))
                state = observation[256*256*3:256*256*3+12]
                
                dataset.add_frame(
                    {
                        "image": image.numpy(),
                        "state": state.numpy(),
                        "action": action.numpy(),
                    }
                )
            
            # Save each episode with its associated task/instruction
            if steps['language_instruction']:
                instruction = steps['language_instruction'][0].decode()
            else:
                instruction = f"Drone navigation task {episode_count}"
            
            dataset.save_episode(task=instruction)
            episode_count += 1
            
            if episode_count % 10 == 0:
                print(f"Processed {episode_count} episodes")
                
        except Exception as e:
            print(f"Error processing episode: {e}")
            continue

    print(f"Successfully processed {episode_count} episodes.")
    
    # Consolidate the dataset
    dataset.consolidate(run_compute_stats=True)

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["cognitive_drone", "drone", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main) 