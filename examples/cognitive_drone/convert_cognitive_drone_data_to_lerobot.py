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

def parse_sequence(serialized_record):
    """Parse a SequenceExample record with proper feature specifications."""
    context_features = {
        # Add any context features if they exist in your data
    }
    
    # Define sequence features with defaults for missing ones
    sequence_features = {
        "steps/is_last": tf.io.FixedLenSequenceFeature([], tf.int64),
        "steps/language_instruction": tf.io.FixedLenSequenceFeature([], tf.string),
        "steps/observation": tf.io.FixedLenSequenceFeature([], tf.string),
        "steps/action": tf.io.FixedLenSequenceFeature([], tf.string),
    }
    
    # Define which features can be missing
    feature_list_dense_missing_assumed_empty = [
        "steps/action",  # Some episodes might not have actions
    ]

    try:
        context_data, seq_data = tf.io.parse_single_sequence_example(
            serialized_record,
            context_features=context_features,
            sequence_features=sequence_features,
            feature_list_dense_missing_assumed_empty=feature_list_dense_missing_assumed_empty
        )
        return context_data, seq_data
    except Exception as e:
        print(f"Error parsing sequence: {str(e)}")
        return None, None

def main(data_dir: str, *, push_to_hub: bool = False):
    # Clean up any existing dataset in the output directory
    output_path = LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)
    
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
                "shape": (12,),  # Adjust based on the drone state dimensions
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
    
    # Create a dataset from the TFRecord files with error handling
    raw_dataset = tf.data.TFRecordDataset(tfrecord_files)
    
    # Debug: Print the first record structure
    print("\nExamining first record structure:")
    for raw_record in raw_dataset.take(1):
        try:
            seq_ex = tf.train.SequenceExample.FromString(raw_record.numpy())
            print("\nContext features:", list(seq_ex.context.feature.keys()))
            print("Sequence features:", list(seq_ex.feature_lists.feature_list.keys()))
        except Exception as e:
            print(f"Error examining record structure: {str(e)}")
    
    # Reset the dataset after examination
    raw_dataset = tf.data.TFRecordDataset(tfrecord_files)
    
    episode_count = 0
    error_count = 0
    total_steps = 0
    
    for serialized_example in raw_dataset:
        try:
            # Parse the sequence example
            context_data, seq_data = parse_sequence(serialized_example)
            if seq_data is None:
                error_count += 1
                continue
            
            # Get the number of steps in this sequence
            num_steps = len(seq_data["steps/observation"])
            total_steps += num_steps
            
            # Process each step in the sequence
            for i in range(num_steps):
                try:
                    # Parse the observation and action tensors
                    observation = tf.io.parse_tensor(seq_data["steps/observation"][i], out_type=tf.float32)
                    
                    # Handle missing actions
                    if "steps/action" in seq_data and i < len(seq_data["steps/action"]):
                        action = tf.io.parse_tensor(seq_data["steps/action"][i], out_type=tf.float32)
                    else:
                        # Use zero action if missing
                        action = tf.zeros((4,), dtype=tf.float32)
                    
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
                except Exception as e:
                    print(f"Error processing step {i} in episode {episode_count}: {str(e)}")
                    continue
            
            # Get the language instruction if available
            if seq_data["steps/language_instruction"]:
                instruction = seq_data["steps/language_instruction"][0].decode()
            else:
                instruction = f"Drone navigation task {episode_count}"
            
            dataset.save_episode(task=instruction)
            episode_count += 1
            
            if episode_count % 10 == 0:
                print(f"Progress: Processed {episode_count} episodes, {total_steps} total steps, {error_count} errors")
                
        except Exception as e:
            print(f"Error processing episode {episode_count}: {str(e)}")
            error_count += 1
            continue

    print(f"\nFinal Statistics:")
    print(f"Successfully processed {episode_count} episodes")
    print(f"Total steps processed: {total_steps}")
    print(f"Total errors encountered: {error_count}")
    
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