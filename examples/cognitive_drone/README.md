# Cognitive Drone Example

This example demonstrates how to fine-tune π0 (Pi Zero) with the Cognitive Drone dataset for autonomous drone navigation and decision-making capabilities.

## Overview

The Cognitive Drone dataset contains demonstrations of drone navigation and decision-making tasks. It is organized in RLDS (Reinforcement Learning Datasets) format and needs to be converted to LeRobot format for fine-tuning π0.

The paper [Cognitive Drone: Context-Aware and Decision-Making Capabilities for Autonomous Drones](https://arxiv.org/abs/2503.01378) describes the dataset and its applications.

## Dataset

The Cognitive Drone dataset can be downloaded from Hugging Face:

```bash
git clone https://huggingface.co/datasets/ArtemLykov/CognitiveDrone_dataset
```

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Convert the Cognitive Drone dataset to LeRobot format:

```bash
uv run convert_cognitive_drone_data_to_lerobot.py --data_dir /path/to/CognitiveDrone_dataset
```

This will convert the dataset and save it to the `$LEROBOT_HOME/cognitive_drone` directory.

3. (Optional) Push the converted dataset to Hugging Face:

```bash
uv run convert_cognitive_drone_data_to_lerobot.py --data_dir /path/to/CognitiveDrone_dataset --push_to_hub
```

## Fine-tuning π0

To fine-tune π0 with the Cognitive Drone dataset, you need to:

1. Ensure that the converted dataset is in the correct format and location
2. Use the OpenPI training pipeline to fine-tune π0 with the dataset

Example fine-tuning command:

```bash
python -m openpi.training.run \
    --config-path path/to/training_config.yaml \
    --data.repo_id cognitive_drone
```

## Evaluation

The `main.py` script provides a simulation environment to evaluate the fine-tuned model. To run the evaluation:

1. First, start the π0 model server:

```bash
python -m openpi.server.run --model-path /path/to/finetuned/model
```

2. Then, run the evaluation script:

```bash
uv run main.py
```

This will evaluate the model on simulated drone tasks and save videos of the executions to `data/cognitive_drone/videos`.

## Customization

You can customize the drone environment, tasks, and evaluation parameters in `main.py`. The drone environment is currently a simple placeholder - in a real application, you would integrate with a proper drone simulator or a real drone.

## Dataset Structure

The Cognitive Drone dataset contains:

- Images from the drone's camera
- Drone state information (position, orientation, etc.)
- Control actions (roll, pitch, yaw, throttle)
- Language instructions describing the task

When converted to LeRobot format, it follows a similar structure to the Libero dataset, making it compatible with the π0 fine-tuning pipeline. 