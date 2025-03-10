import modal
import os
import shutil
from pathlib import Path
import json
import subprocess

# Define the output path for the dataset
LEROBOT_HOME = "/cognitive-drone-volume"  # This should be the path within the volume
REPO_NAME = "cognitive_drone"

# Define a simplified version of LeRobotDataset for our needs
class SimpleLeRobotDataset:
    def __init__(self, repo_id, robot_type, fps, features):
        self.repo_id = repo_id
        self.robot_type = robot_type
        self.fps = fps
        self.features = features
        self.output_path = Path(LEROBOT_HOME) / repo_id  # Ensure this is within the writable path
        self.current_episode = []
        self.episode_count = 0
        self.frame_count = 0
        
        # Create output directories
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.output_path / "episodes", exist_ok=True)
        os.makedirs(self.output_path / "images", exist_ok=True)
        os.makedirs(self.output_path / "meta", exist_ok=True)
    
    @classmethod
    def create(cls, repo_id, robot_type, fps, features, **kwargs):
        return cls(repo_id, robot_type, fps, features)
    
    def add_frame(self, frame_data):
        # Save image to file
        image_path = self.output_path / "images" / f"frame_{self.frame_count:06d}.jpg"
        Image.fromarray((frame_data["image"] * 255).astype(np.uint8)).save(image_path)
        
        # Store frame data
        self.current_episode.append({
            "image": str(image_path),
            "state": frame_data["state"].tolist(),
            "action": frame_data["action"].tolist(),
            "frame_id": self.frame_count
        })
        
        self.frame_count += 1
        return self.frame_count - 1
    
    def save_episode(self, task):
        if not self.current_episode:
            return
        
        # Save episode metadata
        episode_path = self.output_path / "episodes" / f"episode_{self.episode_count:06d}.json"
        episode_data = {
            "frames": self.current_episode,
            "task": task,
            "episode_id": self.episode_count
        }
        
        with open(episode_path, 'w') as f:
            json.dump(episode_data, f)
        
        self.episode_count += 1
        self.current_episode = []
        return self.episode_count - 1
    
    def consolidate(self, run_compute_stats=True):
        # Save dataset metadata
        info = {
            "repo_id": self.repo_id,
            "robot_type": self.robot_type,
            "fps": self.fps,
            "features": self.features,
            "episode_count": self.episode_count,
            "frame_count": self.frame_count,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(self.output_path / "meta" / "info.json", 'w') as f:
            json.dump(info, f)
        
        print(f"Dataset consolidated with {self.episode_count} episodes and {self.frame_count} frames.")

# Define a Modal app
app = modal.App("cognitive-drone-conversion")

# Create a Modal volume to store the dataset
volume = modal.Volume.from_name("cognitive-drone-volume", create_if_missing=True)

# Define the image with all necessary dependencies, including git
image = modal.Image.debian_slim().apt_install(
    ["git", "libosmesa6-dev", "libgl1-mesa-glx", "libglew-dev", "libglfw3-dev", "libgles2-mesa-dev"]
).pip_install(
    [
        "tensorflow==2.15.0",
        "tensorflow-datasets==4.9.3",
        "tyro==0.7.3",
        "numpy==1.26.3",
        "pillow==10.2.0",
        "tqdm==4.66.1",
    ]
).run_commands([
    "python -m pip install --upgrade pip",
])

# Define the function to run the conversion
@app.function(
    gpu="T4",  # Specify T4 GPU
    image=image,
    volumes={"/cognitive-drone-volume": volume},  # Ensure this is correctly set
    timeout=3600,  # Set a timeout of 1 hour (adjust as needed)
)
def convert_cognitive_drone_data(data_dir: str):
    import tensorflow as tf
    import numpy as np
    from pathlib import Path
    import glob
    import os
    import json
    from PIL import Image
    import time
    
    # Set up environment variables
    os.environ["LEROBOT_HOME"] = LEROBOT_HOME  # Set the LEROBOT_HOME environment variable
    os.makedirs(LEROBOT_HOME, exist_ok=True)  # Ensure the directory exists
    
    # Create output directory
    output_path = Path(LEROBOT_HOME) / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)
    
    print(f"Output will be saved to: {output_path}")
    
    # Create LeRobot dataset
    dataset = SimpleLeRobotDataset.create(
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
        }
    )
    
    # Find all TFRecord files in the specified directory
    tfrecord_pattern = os.path.join(data_dir, "*.tfrecord-*")
    tfrecord_files = glob.glob(tfrecord_pattern)
    
    if not tfrecord_files:
        raise ValueError(f"No TFRecord files found in {tfrecord_pattern}")
    
    print(f"Found {len(tfrecord_files)} TFRecord files.")
    
    # Create a dataset from the TFRecord files
    raw_dataset = tf.data.TFRecordDataset(tfrecord_files)
    
    # Define the feature description for parsing TFRecord files
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
    
    episode_count = 0
    for serialized_example in raw_dataset:
        try:
            # Process each episode
            steps = parse_episode(serialized_example)
            
            # Extract steps data
            for i in range(len(steps[1]['observation'])):
                # Add each frame to the dataset
                observation = tf.io.parse_tensor(steps[1]['observation'][i], out_type=tf.float32)
                action = tf.io.parse_tensor(steps[1]['action'][i], out_type=tf.float32)
                
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
            if len(steps[1]['language_instruction']) > 0:
                instruction = steps[1]['language_instruction'][0].decode()
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
    
    return {
        "output_path": str(output_path),
        "episodes_processed": episode_count
    }

# Define a function to download the dataset from Hugging Face
@app.function(
    image=image,
    volumes={"/cognitive-drone-volume": volume},
)
def download_dataset():
    import os
    import subprocess

    # Define the path for the dataset
    dataset_path = "/data/CognitiveDrone_dataset/"

    # Check if the dataset directory already exists
    if os.path.exists(dataset_path):
        print(f"Dataset directory '{dataset_path}' already exists. Removing and replacing it.")
        shutil.rmtree(dataset_path)  # Remove the existing directory

    print("Downloading the CognitiveDrone dataset...")
    try:
        # Clone the dataset from Hugging Face
        subprocess.run([
            "git", "clone", "https://huggingface.co/datasets/ArtemLykov/CognitiveDrone_dataset", dataset_path
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")
        raise

    return dataset_path

# Local entrypoint to start the server when running the Modal app
@app.local_entrypoint()
def main():
    # This will invoke the run_server function on Modal (with GPU and all settings)
    print("Downloading the CognitiveDrone dataset...")
    data_dir = download_dataset.remote()
    data_dir = os.path.join(data_dir, "data", "rlds", "train")

    # Clean up any existing dataset in the output directory
    output_path = Path(LEROBOT_HOME) / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    dataset = SimpleLeRobotDataset.create(
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
    )

    # Find all TFRecord files in the specified directory
    tfrecord_pattern = os.path.join(data_dir, "*.tfrecord-*")
    tfrecord_files = glob.glob(tfrecord_pattern)

    if not tfrecord_files:
        raise ValueError(f"No TFRecord files found in {tfrecord_pattern}")

    print(f"Found {len(tfrecord_files)} TFRecord files.")

    # Create a dataset from the TFRecord files
    raw_dataset = tf.data.TFRecordDataset(tfrecord_files)

    episode_count = 0
    for serialized_example in raw_dataset:
        try:
            # Process each episode
            steps = parse_episode(serialized_example)

            # Extract steps data
            for i in range(len(steps['observation'])):
                observation = tf.io.parse_tensor(steps['observation'][i], out_type=tf.float32)
                action = tf.io.parse_tensor(steps['action'][i], out_type=tf.float32)

                image = observation[:256*256*3].reshape((256, 256, 3))
                state = observation[256*256*3:256*256*3+12]

                dataset.add_frame(
                    {
                        "image": image.numpy(),
                        "state": state.numpy(),
                        "action": action.numpy(),
                    }
                )

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

    return {
        "output_path": str(output_path),
        "episodes_processed": episode_count
    }
