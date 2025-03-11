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
    """Inspect the structure of a TFRecord file to determine if it contains Examples or SequenceExamples."""
    dataset = tf.data.TFRecordDataset([tfrecord_file])
    for serialized_example in dataset.take(1):
        try:
            # Try parsing as a SequenceExample first
            context_features = {}
            sequence_features = {
                "steps/observation/image": tf.io.FixedLenSequenceFeature([], tf.string),
                "steps/observation/state": tf.io.FixedLenSequenceFeature([], tf.string),
                "steps/action": tf.io.FixedLenSequenceFeature([], tf.string),
                "steps/is_first": tf.io.FixedLenSequenceFeature([], tf.int64),
                "steps/is_last": tf.io.FixedLenSequenceFeature([], tf.int64),
                "steps/is_terminal": tf.io.FixedLenSequenceFeature([], tf.int64),
                "steps/reward": tf.io.FixedLenSequenceFeature([], tf.float32),
            }
            
            try:
                context, sequence = tf.io.parse_single_sequence_example(
                    serialized_example, context_features, sequence_features
                )
                print("\nDetected SequenceExample format")
                print("Context features:")
                for key in context:
                    print(f"- {key}")
                print("Sequence features:")
                for key in sequence:
                    print(f"- {key}")
                return "sequence_example", context, sequence
            except Exception as e:
                print(f"Not a SequenceExample: {str(e)}")
                
            # Try parsing as an Example
            example = tf.train.Example()
            example.ParseFromString(serialized_example.numpy())
            print("\nDetected Example format")
            print("Features:")
            for feature_name in example.features.feature:
                print(f"- {feature_name}")
            return "example", example.features.feature, None
                
        except Exception as e:
            print(f"Error inspecting record: {str(e)}")
            return "unknown", None, None

def parse_sequence_example(serialized_record):
    """Parse a SequenceExample record with proper feature specifications."""
    try:
        # Define context features (if any)
        context_features = {
            # Add any context features here if needed
        }
        
        # Define sequence features
        sequence_features = {
            "steps/observation/image": tf.io.FixedLenSequenceFeature([], tf.string),
            "steps/observation/state": tf.io.FixedLenSequenceFeature([], tf.string),
            "steps/action": tf.io.FixedLenSequenceFeature([], tf.string),
            "steps/is_first": tf.io.FixedLenSequenceFeature([], tf.int64),
            "steps/is_last": tf.io.FixedLenSequenceFeature([], tf.int64),
            "steps/is_terminal": tf.io.FixedLenSequenceFeature([], tf.int64, default_value=0),
            "steps/reward": tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0.0),
        }
        
        # Try to parse language instruction if available
        try:
            sequence_features["steps/language_instruction"] = tf.io.FixedLenSequenceFeature(
                [], tf.string, allow_missing=True
            )
        except:
            pass
        
        # Parse the sequence example
        context, sequence_data = tf.io.parse_single_sequence_example(
            serialized_record, context_features, sequence_features
        )
        
        return context, sequence_data
    except Exception as e:
        print(f"Error parsing sequence example: {str(e)}")
        return None, None

def parse_example(serialized_record):
    """Parse a standard Example record with proper feature specifications."""
    try:
        # Define required features with proper bytes literals for string defaults
        feature_description = {
            'steps/is_first': tf.io.FixedLenFeature([], tf.int64, default_value=0),
            'steps/is_last': tf.io.VarLenFeature(tf.int64),
            #'steps/is_last': tf.io.FixedLenFeature([], tf.int64, default_value=0),
            'steps/is_terminal': tf.io.FixedLenFeature([], tf.int64, default_value=0),
            'steps/reward': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
            'steps/action': tf.io.FixedLenFeature([], tf.string, default_value=b''),
            'steps/observation/state': tf.io.FixedLenFeature([], tf.string, default_value=b''),
            'steps/observation/image': tf.io.FixedLenFeature([], tf.string, default_value=b''),
            # Make language instruction optional using VarLenFeature
            'steps/language_instruction': tf.io.VarLenFeature(tf.string)
        }
        
        # Parse example
        example = tf.io.parse_single_example(serialized_record, feature_description)
        
        # Convert sparse tensor to dense for language instruction if present
        if isinstance(example['steps/language_instruction'], tf.sparse.SparseTensor):
            example['steps/language_instruction'] = tf.sparse.to_dense(
                example['steps/language_instruction'], default_value=b''
            )[0]  # Take first value if multiple exist
        
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
    
    # Process each TFRecord file
    episode_count = 0
    error_count = 0
    total_steps = 0
    current_episode_frames = []
    current_episode_instruction = None
    
    for file_idx, tfrecord_file in enumerate(tfrecord_files):
        print(f"Processing file {file_idx+1}/{len(tfrecord_files)}: {os.path.basename(tfrecord_file)}")
        
        try:
            # Create dataset for single file
            file_dataset = tf.data.TFRecordDataset([tfrecord_file], buffer_size=16*1024*1024)
            
            for example_idx, serialized_example in enumerate(file_dataset):
                try:
                    # Parse example
                    example = parse_example(serialized_example)
                    if example is None:
                        error_count += 1
                        continue
                    
                    try:
                        # Process Example format (one step per example)
                        is_first = example['steps/is_first'].numpy()
                        is_last = example['steps/is_last'].numpy()
                        is_terminal = example['steps/is_terminal'].numpy()
                        
                        # Parse observation state
                        observation_state = tf.io.parse_tensor(
                            example['steps/observation/state'], out_type=tf.float32
                        )
                        
                        # Parse observation image
                        observation_image = tf.io.parse_tensor(
                            example['steps/observation/image'], out_type=tf.float32
                        )
                        
                        # Parse action
                        try:
                            action_data = example['steps/action']
                            if action_data != b'':
                                action = tf.io.parse_tensor(action_data, out_type=tf.float32)
                            else:
                                action = tf.zeros((4,), dtype=tf.float32)
                        except:
                            action = tf.zeros((4,), dtype=tf.float32)
                        
                        # Get language instruction if available
                        if is_first:
                            try:
                                language_instruction = example['steps/language_instruction'].numpy().decode('utf-8')
                                if language_instruction and language_instruction != '':
                                    current_episode_instruction = language_instruction
                            except:
                                pass
                        
                        # Validate tensor shapes before adding to dataset
                        if (observation_image.shape != (256, 256, 3) or
                            observation_state.shape[0] < 12 or
                            action.shape[0] < 4):
                            print(f"Warning: Unexpected tensor shapes - image: {observation_image.shape}, "
                                  f"state: {observation_state.shape}, action: {action.shape}")
                            continue
                        
                        # Create frame data
                        frame_data = {
                            "image": observation_image.numpy(),
                            "state": observation_state.numpy()[:12],  # Ensure consistent state size
                            "action": action.numpy()[:4],             # Ensure consistent action size
                        }
                        
                        # Add frame to current episode
                        current_episode_frames.append(frame_data)
                        total_steps += 1
                        
                        # Handle episode completion
                        if is_last or is_terminal:
                            if len(current_episode_frames) > 0:
                                # Add all frames to the dataset
                                for frame in current_episode_frames:
                                    dataset.add_frame(frame)
                                
                                # Use stored instruction or default
                                if current_episode_instruction is None or current_episode_instruction == "":
                                    instruction = f"Drone navigation task {episode_count}"
                                else:
                                    instruction = current_episode_instruction
                                
                                # Save episode
                                dataset.save_episode(task=instruction)
                                episode_count += 1
                                
                                if episode_count % 5 == 0:
                                    print(f"Progress: Processed {episode_count} episodes, {total_steps} steps")
                            
                            # Reset for next episode
                            current_episode_frames = []
                            current_episode_instruction = None
                    
                    except Exception as e:
                        print(f"Error processing example {example_idx}: {str(e)}")
                        error_count += 1
                        # Don't reset frames to allow for recovery from minor errors
                        
                except Exception as e:
                    print(f"Error processing example {example_idx}: {str(e)}")
                    error_count += 1
                    # Don't reset on minor errors to allow for recovery
            
            # Handle any remaining episode at the end of the file
            if len(current_episode_frames) > 0:
                print(f"File ended with {len(current_episode_frames)} pending frames. Saving as a partial episode.")
                
                # Add all frames to the dataset
                for frame in current_episode_frames:
                    dataset.add_frame(frame)
                
                # Use stored instruction or default
                if current_episode_instruction is None or current_episode_instruction == "":
                    instruction = f"Drone navigation task {episode_count} (partial)"
                else:
                    instruction = f"{current_episode_instruction} (partial)"
                
                # Save episode
                dataset.save_episode(task=instruction)
                episode_count += 1
                
                # Reset for next file
                current_episode_frames = []
                current_episode_instruction = None
                
        except Exception as e:
            print(f"Error processing file {tfrecord_file}: {str(e)}")
            error_count += 1
            current_episode_frames = []  # Reset on major error
            current_episode_instruction = None

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