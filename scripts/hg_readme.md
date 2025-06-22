
# Gestalt Pattern Reasoning Benchmark

![alt text](intro.png "Title")

This repository contains a dataset and benchmarking framework for **Gestalt pattern reasoning**. 
The dataset consists of thousands of procedurally generated visual patterns based on **Gestalt principles**, 
including proximity, similarity, closure, symmetry, and continuity. 
The benchmark is designed to evaluate both human and AI performance in recognizing and reasoning about these patterns.


## Github
The Dataset Generator and Benchmark Evaluation Code is on the [Github](https://github.com/akweury/grb).

## Dataset

principle_all_resolution_224_num_100.zip: includes 5 Gestalt Principles, each task has 100 examples for each labels. The image resolution is 224x224.

## File Structure
```
Gestalt Reasoning Benchmark/
│── data/
│   │── raw_patterns/         # Unprocessed/generated raw patterns
│   │   │── proximity/
│   │   │   │── train/
│   │   │   │   │── 0001_red_triangle/
│   │   │   │   │   │── positive/
│   │   │   │   │   │   │── 00000.png
│   │   │   │   │   │   │── 00000.json
│   │   │   │   │   │   │── 00001.png
│   │   │   │   │   │   │── 00001.json
│   │   │   │   │   │   │── 00002.png
│   │   │   │   │   │   │── 00002.json
│   │   │   │   │   │── negative/
│   │   │   │   │   │   │── 00000.png
│   │   │   │   │   │   │── 00000.json
│   │   │   │   │   │   │── 00001.png
│   │   │   │   │   │   │── 00001.json
│   │   │   │   │   │   │── 00002.png
│   │   │   │   │   │   │── 00002.json
│   │   │   │── test/
│   │   │   │   │── 0001_red_triangle/
│   │   │   │   │   │── positive/
│   │   │   │   │   │   │── 00000.png
│   │   │   │   │   │   │── 00000.json
│   │   │   │   │   │   │── 00001.png
│   │   │   │   │   │   │── 00001.json
│   │   │   │   │   │   │── 00002.png
│   │   │   │   │   │   │── 00002.json
│   │   │   │   │   │── negative/
│   │   │   │   │   │   │── 00000.png
│   │   │   │   │   │   │── 00000.json
│   │   │   │   │   │   │── 00001.png
│   │   │   │   │   │   │── 00001.json
│   │   │   │   │   │   │── 00002.png
│   │   │   │   │   │   │── 00002.json

```


## Citation

If you use the Gestalt Vision dataset in your research or work, please cite our paper:

```
@inproceedings{sha2025gestalt,
  author    = {Jingyuan Sha and Hikaru Shindo and Kristian Kersting and Devendra Singh Dhami},
  title     = {Gestalt Vision: A Dataset for Evaluating Gestalt Principles in Visual Perception},
  booktitle = {Proceedings of the 19th International Conference on Neurosymbolic Learning and Reasoning (NeSy)},
  year      = {2025}
}
```