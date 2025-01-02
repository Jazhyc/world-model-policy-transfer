# World Model Agents with Change-Based Intrinsic Motivation

This repository contains the code for the paper "World Model Agents with Change-Based Intrinsic Motivation" accepted at the Northern Lights Deep Learning Conference (NLDL), 2025.

## Abstract

Sparse reward environments pose a significant challenge for reinforcement learning. This paper explores the effectiveness of Change-Based Exploration Transfer (CBET), a technique combining intrinsic motivation and transfer learning, for world model agents. We adapt CBET for DreamerV3 and compare its performance against IMPALA in the sparse reward environments of Crafter and Minigrid. Our results show that CBET can improve DreamerV3's returns in complex environments like Crafter but may be detrimental in simpler environments like Minigrid, highlighting the importance of considering environment complexity when applying exploration strategies.

## Repository Structure

This repository is organized into two main folders:

-   **`DreamerV3/`**: Contains the adapted DreamerV3 implementation for our experiments.
-   **`IMPALA/`**: Contains the adapted IMPALA implementation (based on TorchBeast) for our experiments.

Each folder contains the necessary code, along with modified environments and utility scripts to run the experiments presented in the paper.

## Requirements

The two algorithms have different requirements:

- **IMPALA**: Python 3.8
- **DreamerV3**: Python 3.11

Each folder contains a `requirements.txt` file that can be used to install most dependencies:

```bash
pip install -r requirements.txt
```

Note: The torch and jax frameworks might require custom installation commands depending on your system and whether you intend to use a GPU or CPU. Please refer to the official documentation for installation instructions.

## Running the Experiments

Example commands for running the experiments are provided in the `hpc.sh` files within each algorithm's folder. These scripts were used on a high-performance computing cluster but can be adapted for local execution.

- **`DreamerV3/hpc.sh`**: Example commands for running DreamerV3 experiments.
- **`IMPALA/hpc.sh`**: Example commands for running IMPALA experiments.

You may need to modify the scripts based on your specific environment setup and resource availability.

## Citation

If you use our findings in your research, please cite our paper:

```bibtex
@inproceedings{ferrao2024world,
title={World Model Agents with Change-Based Intrinsic Motivation},
author={Jeremias Lino Ferrao and Rafael F. Cunha},
booktitle={Northern Lights Deep Learning Conference 2025},
year={2024},
url={https://openreview.net/forum?id=0io7gvXniL}
}
```
