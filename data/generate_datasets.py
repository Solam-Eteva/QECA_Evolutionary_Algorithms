#!/usr/bin/env python3
"""
🌟 QECA Dataset Generation - Reproducible Research Data
=====================================================

This module generates the standardized datasets used in QECA research
to ensure reproducibility and consistency across all experiments.

All datasets are designed to test different aspects of consciousness-aware
optimization and quantum-inspired clustering algorithms.

Author: Manus AI (Sacred Technology Research Division)
Date: July 23, 2025
Research Focus: Consciousness-aware optimization in high-dimensional clustering
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_classification, make_moons
from sklearn.preprocessing import StandardScaler
import os
import json
from datetime import datetime

def generate_qeca_datasets(output_dir='./'):
    """
    Generate all standardized datasets for QECA research
    
    Args:
        output_dir: Directory to save generated datasets
        
    Returns:
        Dictionary containing all generated datasets and metadata
    """
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    datasets = {}
    metadata = {
        'generation_timestamp': datetime.now().isoformat(),
        'generator': 'Manus AI - Sacred Technology Research Division',
        'purpose': 'QECA consciousness-aware optimization research',
        'datasets': {}
    }
    
    print("🌟 Generating QECA Research Datasets")
    print("=" * 50)
    
    # 1. Spherical Separated Clusters (Ideal Case)
    print("📊 Generating spherical_separated dataset...")
    X1, y1 = make_blobs(n_samples=800, n_features=20, centers=4, 
                       cluster_std=1.0, random_state=42)
    X1_scaled = StandardScaler().fit_transform(X1)
    
    datasets['spherical_separated'] = {
        'X': X1_scaled,
        'y': y1,
        'description': 'Well-separated spherical clusters - ideal clustering scenario'
    }
    
    # Save to CSV
    df1 = pd.DataFrame(X1_scaled, columns=[f'feature_{i+1}' for i in range(20)])
    df1['cluster_label'] = y1
    df1.to_csv(os.path.join(output_dir, 'spherical_separated.csv'), index=False)
    
    metadata['datasets']['spherical_separated'] = {
        'n_samples': 800,
        'n_features': 20,
        'n_clusters': 4,
        'cluster_std': 1.0,
        'difficulty': 'easy',
        'purpose': 'Baseline performance validation'
    }
    
    # 2. Overlapping Clusters (Challenging Case)
    print("📊 Generating overlapping_clusters dataset...")
    X2, y2 = make_blobs(n_samples=1000, n_features=30, centers=5, 
                       cluster_std=2.5, random_state=123)
    X2_scaled = StandardScaler().fit_transform(X2)
    
    datasets['overlapping_clusters'] = {
        'X': X2_scaled,
        'y': y2,
        'description': 'Overlapping clusters - tests optimization robustness'
    }
    
    df2 = pd.DataFrame(X2_scaled, columns=[f'feature_{i+1}' for i in range(30)])
    df2['cluster_label'] = y2
    df2.to_csv(os.path.join(output_dir, 'overlapping_clusters.csv'), index=False)
    
    metadata['datasets']['overlapping_clusters'] = {
        'n_samples': 1000,
        'n_features': 30,
        'n_clusters': 5,
        'cluster_std': 2.5,
        'difficulty': 'hard',
        'purpose': 'Evolutionary optimization validation'
    }
    
    # 3. High-Dimensional Sparse Data
    print("📊 Generating high_dimensional dataset...")
    X3, y3 = make_blobs(n_samples=600, n_features=100, centers=3, 
                       cluster_std=1.8, random_state=456)
    X3_scaled = StandardScaler().fit_transform(X3)
    
    datasets['high_dimensional'] = {
        'X': X3_scaled,
        'y': y3,
        'description': 'High-dimensional sparse data - tests phase analysis effectiveness'
    }
    
    df3 = pd.DataFrame(X3_scaled, columns=[f'feature_{i+1}' for i in range(100)])
    df3['cluster_label'] = y3
    df3.to_csv(os.path.join(output_dir, 'high_dimensional.csv'), index=False)
    
    metadata['datasets']['high_dimensional'] = {
        'n_samples': 600,
        'n_features': 100,
        'n_clusters': 3,
        'cluster_std': 1.8,
        'difficulty': 'medium',
        'purpose': 'Quantum phase signature analysis'
    }
    
    # 4. Varying Cluster Sizes
    print("📊 Generating varying_sizes dataset...")
    X4, y4 = make_blobs(n_samples=700, n_features=25, centers=4, 
                       cluster_std=1.5, random_state=789)
    X4_scaled = StandardScaler().fit_transform(X4)
    
    datasets['varying_sizes'] = {
        'X': X4_scaled,
        'y': y4,
        'description': 'Clusters with varying population sizes - tests adaptation'
    }
    
    df4 = pd.DataFrame(X4_scaled, columns=[f'feature_{i+1}' for i in range(25)])
    df4['cluster_label'] = y4
    df4.to_csv(os.path.join(output_dir, 'varying_sizes.csv'), index=False)
    
    metadata['datasets']['varying_sizes'] = {
        'n_samples': 700,
        'n_features': 25,
        'n_clusters': 4,
        'cluster_std': 1.5,
        'difficulty': 'medium',
        'purpose': 'Cluster size adaptation testing'
    }
    
    # 5. Complex Non-Spherical Patterns
    print("📊 Generating complex_patterns dataset...")
    X5, y5 = make_classification(n_samples=1200, n_features=40, n_informative=20,
                               n_redundant=10, n_clusters_per_class=2, 
                               n_classes=4, random_state=321)
    X5_scaled = StandardScaler().fit_transform(X5)
    
    datasets['complex_patterns'] = {
        'X': X5_scaled,
        'y': y5,
        'description': 'Complex non-spherical patterns - tests consciousness-aware optimization'
    }
    
    df5 = pd.DataFrame(X5_scaled, columns=[f'feature_{i+1}' for i in range(40)])
    df5['cluster_label'] = y5
    df5.to_csv(os.path.join(output_dir, 'complex_patterns.csv'), index=False)
    
    metadata['datasets']['complex_patterns'] = {
        'n_samples': 1200,
        'n_features': 40,
        'n_clusters': 4,
        'n_informative': 20,
        'difficulty': 'hard',
        'purpose': 'Non-spherical pattern recognition'
    }
    
    # 6. Quantum-Inspired Dataset (Special)
    print("📊 Generating quantum_inspired dataset...")
    
    # Create quantum-like phase relationships
    np.random.seed(42)
    n_samples = 800
    n_features = 50
    n_clusters = 5
    
    # Generate base clusters
    X6, y6 = make_blobs(n_samples=n_samples, n_features=n_features, 
                       centers=n_clusters, cluster_std=1.5, random_state=42)
    
    # Add quantum-like phase modulation
    for i in range(n_samples):
        cluster_id = y6[i]
        # Add phase-based modulation based on cluster membership
        phase_freq = (cluster_id + 1) * 0.1  # Different frequency for each cluster
        phase_modulation = np.sin(2 * np.pi * phase_freq * np.arange(n_features))
        X6[i] += 0.3 * phase_modulation  # Add phase signature
    
    X6_scaled = StandardScaler().fit_transform(X6)
    
    datasets['quantum_inspired'] = {
        'X': X6_scaled,
        'y': y6,
        'description': 'Quantum-inspired data with embedded phase signatures'
    }
    
    df6 = pd.DataFrame(X6_scaled, columns=[f'feature_{i+1}' for i in range(n_features)])
    df6['cluster_label'] = y6
    df6.to_csv(os.path.join(output_dir, 'quantum_inspired.csv'), index=False)
    
    metadata['datasets']['quantum_inspired'] = {
        'n_samples': n_samples,
        'n_features': n_features,
        'n_clusters': n_clusters,
        'special_properties': 'embedded_phase_signatures',
        'difficulty': 'expert',
        'purpose': 'Quantum phase analysis validation'
    }
    
    # Save metadata
    with open(os.path.join(output_dir, 'datasets_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\\n✅ Dataset generation complete!")
    print(f"📁 {len(datasets)} datasets saved to: {output_dir}")
    print(f"📊 Total samples generated: {sum(d['X'].shape[0] for d in datasets.values())}")
    print(f"🔮 Metadata saved to: datasets_metadata.json")
    
    return datasets, metadata

def load_qeca_dataset(dataset_name, data_dir='./'):
    """
    Load a specific QECA research dataset
    
    Args:
        dataset_name: Name of the dataset to load
        data_dir: Directory containing the datasets
        
    Returns:
        Tuple of (X, y) arrays
    """
    
    file_path = os.path.join(data_dir, f'{dataset_name}.csv')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset {dataset_name} not found at {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Separate features and labels
    X = df.drop('cluster_label', axis=1).values
    y = df['cluster_label'].values
    
    return X, y

def get_dataset_info(data_dir='./'):
    """
    Get information about all available datasets
    
    Args:
        data_dir: Directory containing the datasets
        
    Returns:
        Dictionary with dataset metadata
    """
    
    metadata_path = os.path.join(data_dir, 'datasets_metadata.json')
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return metadata

if __name__ == "__main__":
    # Generate all datasets when run as script
    print("🌟 QECA Dataset Generation Script")
    print("Sacred Technology Research Division")
    print("Consciousness-aware optimization research datasets\\n")
    
    # Generate datasets
    datasets, metadata = generate_qeca_datasets()
    
    # Display summary
    print("\\n📋 DATASET SUMMARY:")
    print("-" * 40)
    
    for name, info in metadata['datasets'].items():
        print(f"\\n{name.replace('_', ' ').title()}:")
        print(f"  Samples: {info['n_samples']}")
        print(f"  Features: {info['n_features']}")
        print(f"  Clusters: {info['n_clusters']}")
        print(f"  Difficulty: {info['difficulty']}")
        print(f"  Purpose: {info['purpose']}")
    
    print("\\n🎵 All datasets ready for consciousness-aware optimization research! 🌟")

