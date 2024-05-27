from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.utils.data import Subset, Dataset


def compute_summary_statistics(class_distribution):
    num_classes = len(class_distribution)
    total_images = sum(class_distribution.values())
    mean_images_per_class = total_images / num_classes
    std_images_per_class = np.std(list(class_distribution.values()))
    
    summary_stats = {
        "Number of Classes": num_classes,
        "Total Number of Images": total_images,
        "Mean Images per Class": mean_images_per_class,
        "Standard Deviation of Images per Class": std_images_per_class
    }
    
    return summary_stats


def get_class_distribution(dataset):
    class_counts = {k: 0 for k, v in dataset.class_to_idx.items()}
    for _, label in dataset:
        class_name = dataset.classes[label]
        class_counts[class_name] += 1
    return class_counts


def display_summary_statistics(summary_statistics, class_distribution):
    print("Summary Statistics:")
    for key, value in summary_statistics.items():
        print(f"{key}: {value}")
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=list(class_distribution.keys()), y=list(class_distribution.values()))
    plt.xticks(rotation=90)
    plt.xlabel("Class")
    plt.ylabel("Number of Samples")
    plt.title("Class Distribution")
    plt.show()
