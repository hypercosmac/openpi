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
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow as tf
import tensorflow_datasets as tfds
import tyro

REPO_NAME = "cognitive_drone"  # Name of the output dataset, also used for the Hugging Face Hub

def main(data_dir: str, *, push_to_hub: bool = False):
    # Clean up any existing dataset in the output directory
    output_path = LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)
    
    print(f"Output will be saved to: {output_path}")
    
    # Create LeRobot dataset, define features to store
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

    # Load the CognitiveDrone dataset directly using tfds.load()
    try:
        raw_dataset = tfds.load("cognitive_drone", data_dir=data_dir, split="train")
        print(f"Loaded {len(raw_dataset)} episodes from the dataset.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Process each episode in the dataset
    for episode in raw_dataset:
        try:
            # Assuming episode contains the necessary fields
            for step in episode["steps"].as_numpy_iterator():
                dataset.add_frame(
                    {
                        "image": step["observation"]["image"],
                        "state": step["observation"]["state"],
                        "action": step["action"],
                    }
                )
            
            # Save the episode with its associated task/instruction
            instruction = step["language_instruction"].decode() if "language_instruction" in step else "Drone navigation task"
            dataset.save_episode(task=instruction)
            print("Processed one episode successfully.")
            
        except Exception as e:
            print(f"Error processing episode: {e}")
            continue

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