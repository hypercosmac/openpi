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

def inspect_tfrecord(tfrecord_file):
    """Inspect the structure of a TFRecord file."""
    dataset = tf.data.TFRecordDataset([tfrecord_file])
    for serialized_example in dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(serialized_example.numpy())
        print("\nAvailable features in TFRecord:")
        for feature_name, feature in example.features.feature.items():
            print(f"- {feature_name}")
        return example.features.feature.keys()

def parse_example(serialized_record):
    """Parse an Example record with proper feature specifications."""
    try:
        # Parse raw example first to get feature types
        raw_example = tf.train.Example()
        raw_example.ParseFromString(serialized_record.numpy())
        
        # Define feature description based on actual data format
        feature_description = {}
        for key, feature in raw_example.features.feature.items():
            if key == 'steps/is_first' or key == 'steps/is_last' or key == 'steps/is_terminal':
                feature_description[key] = tf.io.FixedLenFeature([], tf.int64, default_value=0)
            elif key == 'steps/reward':
                feature_description[key] = tf.io.FixedLenFeature([], tf.float32, default_value=0.0)
            else:
                feature_description[key] = tf.io.FixedLenFeature([], tf.string, default_value='')
        
        # Parse example with correct feature description
        example = tf.io.parse_single_example(serialized_record, feature_description)
        return example
    except Exception as e:
        print(f"Error parsing example: {str(e)}")
        return None

def main(data_dir: str, *, push_to_hub: bool = False):
    # Clean up any existing dataset in the output directory
    output_path = LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)
    
    # Create LeRobot dataset, define features to store
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="drone",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (12,),
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (4,),
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
    
    # Process each TFRecord file separately to handle truncation errors
    episode_count = 0
    error_count = 0
    total_steps = 0
    current_episode_frames = []
    
    for tfrecord_file in tfrecord_files:
        try:
            # Create dataset for single file with error handling
            file_dataset = tf.data.TFRecordDataset([tfrecord_file], buffer_size=16*1024*1024)
            
            for serialized_example in file_dataset:
                try:
                    # Parse the example
                    example = parse_example(serialized_example)
                    if example is None:
                        error_count += 1
                        continue
                    
                    try:
                        # Parse the observation tensors
                        observation_state = tf.io.parse_tensor(example['steps/observation/state'], out_type=tf.float32)
                        observation_image = tf.io.parse_tensor(example['steps/observation/image'], out_type=tf.float32)
                        
                        # Parse action if available
                        if example['steps/action'] != b'':
                            action = tf.io.parse_tensor(example['steps/action'], out_type=tf.float32)
                        else:
                            action = tf.zeros((4,), dtype=tf.float32)
                        
                        # Create frame data
                        frame_data = {
                            "image": observation_image.numpy(),
                            "state": observation_state.numpy(),
                            "action": action.numpy(),
                        }
                        
                        # Add frame to current episode
                        current_episode_frames.append(frame_data)
                        
                        # If this is the last step in an episode, save it
                        if example['steps/is_last']:
                            # Add all frames from the episode
                            for frame in current_episode_frames:
                                dataset.add_frame(frame)
                            
                            # Get instruction if available, otherwise use default
                            try:
                                instruction = example['steps/language_instruction'].numpy().decode('utf-8')
                            except:
                                instruction = f"Drone navigation task {episode_count}"
                            
                            # Save episode
                            dataset.save_episode(task=instruction)
                            episode_count += 1
                            
                            # Clear current episode frames
                            current_episode_frames = []
                            
                            if episode_count % 10 == 0:
                                print(f"Progress: Processed {episode_count} episodes ({error_count} errors)")
                        
                        total_steps += 1
                        
                    except Exception as e:
                        print(f"Error processing tensors: {str(e)}")
                        error_count += 1
                        current_episode_frames = []  # Reset on error
                        continue
                    
                except Exception as e:
                    print(f"Error processing example: {str(e)}")
                    error_count += 1
                    current_episode_frames = []  # Reset on error
                    continue
                    
        except Exception as e:
            print(f"Error processing file {tfrecord_file}: {str(e)}")
            error_count += 1
            current_episode_frames = []  # Reset on error
            continue

    print(f"\nFinal Statistics:")
    print(f"Successfully processed {episode_count} episodes")
    print(f"Total steps processed: {total_steps}")
    print(f"Total errors encountered: {error_count}")
    
    if episode_count == 0:
        print("No episodes were processed successfully. Dataset creation failed.")
        return
    
    try:
        # Consolidate the dataset
        print("Consolidating dataset...")
        dataset.consolidate(run_compute_stats=True)

        # Optionally push to the Hugging Face Hub
        if push_to_hub:
            dataset.push_to_hub(
                tags=["cognitive_drone", "drone", "rlds"],
                private=False,
                push_videos=True,
                license="apache-2.0",
            )
    except Exception as e:
        print(f"Error during dataset consolidation: {str(e)}")
        return

if __name__ == "__main__":
    tyro.cli(main) 