"""
Main script for fine-tuning and evaluating π0 model on the CognitiveDrone dataset.

This script provides functionality for:
1. Fine-tuning the π0 model on the CognitiveDrone dataset
2. Evaluating the fine-tuned model in a drone simulation environment

For more details on the CognitiveDrone dataset, see the paper:
"Cognitive Drone: Context-Aware and Decision-Making Capabilities for Autonomous Drones"
"""

import collections
import dataclasses
import logging
import math
import pathlib
import time
from typing import Dict, List, Optional, Tuple

import imageio
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

# Dummy action for initialization steps (adjust based on drone controls)
DRONE_DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0]  # [roll, pitch, yaw, throttle]
DRONE_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # Drone environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "cognitive_drone"  # Task suite name
    num_steps_wait: int = 5  # Number of steps to wait for stabilization
    num_trials_per_task: int = 10  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/cognitive_drone/videos"  # Path to save videos
    seed: int = 42  # Random Seed (for reproducibility)


def setup_drone_env():
    """
    Set up the drone simulation environment.
    
    In a real implementation, this would initialize the drone simulator
    with appropriate physics and observation space.
    """
    # This is a placeholder - in a real implementation, you would
    # initialize your drone simulator here
    class DroneEnv:
        def __init__(self):
            self.state = {"position": np.zeros(3), "orientation": np.zeros(4)}
            
        def reset(self):
            self.state = {"position": np.zeros(3), "orientation": np.zeros(4)}
            return self._get_observation()
            
        def step(self, action):
            # Simple simulation of drone dynamics
            # In a real implementation, this would use proper physics
            self.state["position"] += np.array([0.1 * action[0], 0.1 * action[1], 0.05 * action[3]])
            reward = 0.0
            done = False
            info = {}
            return self._get_observation(), reward, done, info
            
        def _get_observation(self):
            # Generate a simple observation
            # In a real implementation, this would render the drone's camera view
            obs = {
                "image": np.zeros((DRONE_ENV_RESOLUTION, DRONE_ENV_RESOLUTION, 3), dtype=np.uint8),
                "state": np.concatenate([self.state["position"], self.state["orientation"], np.zeros(5)])
            }
            return obs
            
        def set_init_state(self, init_state):
            self.state = init_state
            return self._get_observation()
            
        def seed(self, seed):
            np.random.seed(seed)
            
    return DroneEnv()


def get_drone_tasks(task_suite_name):
    """
    Get the drone tasks from the task suite.
    
    In a real implementation, this would load the tasks from a configuration file
    or a database based on the task suite name.
    """
    # This is a placeholder - in a real implementation, you would
    # load real tasks and their initial states
    tasks = [
        {"id": 0, "description": "Navigate to the target location"},
        {"id": 1, "description": "Follow the specified trajectory"},
        {"id": 2, "description": "Inspect the structure and report anomalies"},
    ]
    
    # Generate some random initial states for each task
    initial_states = {}
    for task in tasks:
        initial_states[task["id"]] = [
            {"position": np.random.randn(3) * 0.1, "orientation": np.random.randn(4) * 0.1}
            for _ in range(10)  # Generate 10 initial states per task
        ]
    
    return tasks, initial_states


def eval_drone(args: Args) -> None:
    """
    Evaluate the fine-tuned π0 model on drone tasks.
    """
    # Set random seed
    np.random.seed(args.seed)

    # Initialize drone environment and tasks
    env = setup_drone_env()
    tasks, initial_states = get_drone_tasks(args.task_suite_name)
    
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    # Maximum steps per episode
    max_steps = 300  # Adjust based on the complexity of drone tasks
    
    # Initialize the client to connect to the π0 model server
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task in tqdm.tqdm(tasks):
        task_id = task["id"]
        task_description = task["description"]
        
        # Get initial states for this task
        task_initial_states = initial_states[task_id]

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(min(args.num_trials_per_task, len(task_initial_states)))):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial state
            obs = env.set_init_state(task_initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # Do nothing for the first few timesteps to stabilize
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(DRONE_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    img = np.ascontiguousarray(obs["image"])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/state": obs["state"],
                            "prompt": str(task_description),
                        }

                        # Query model to get action
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_drone) 