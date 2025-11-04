"""
Utility functions for Medical Embedding Analysis Tutorial

This module contains complex helper functions used throughout the tutorial.
Functions are externalized here to keep the main notebook clean and focused
on educational content.
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
import imagehash
import re
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF
from skimage.metrics import structural_similarity as ssim
import time
import cv2


def load_location_labels(dataset_path, sample_data):
    """
    Extract anatomical location labels from SLAKE train/val/test JSON files.

    Args:
        dataset_path: Path to SLAKE dataset directory
        sample_data: List of sample dictionaries with 'sample_id' field

    Returns:
        Dictionary mapping sample_id to location label
        List of location labels in same order as sample_data
    """
    dataset_path = Path(dataset_path)

    # Load all JSON files (train, test, validate)
    json_files = ['train.json', 'test.json', 'validate.json']
    img_id_to_location = {}
    img_name_to_location = {}

    for json_file in json_files:
        json_path = dataset_path / json_file
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)

            # Extract img_id -> location mapping
            for entry in data:
                img_id = entry['img_id']
                img_name = entry['img_name']
                location = entry['location']
                img_id_to_location[img_id] = location
                img_name_to_location[img_name] = location

    # Map sample_ids to locations
    # Sample IDs are like 'xmlab0', 'xmlab1', etc.
    # img_names are like 'xmlab1/source.jpg'
    labels_dict = {}
    labels_list = []

    for sample in sample_data:
        sample_id = sample['sample_id']
        # Try to find location by constructing img_name
        img_name = f"{sample_id}/source.jpg"

        if img_name in img_name_to_location:
            location = img_name_to_location[img_name]
        else:
            # Fallback: try to extract numeric ID and look up
            # e.g., 'xmlab1' -> img_id 1
            try:
                numeric_id = int(sample_id.replace('xmlab', ''))
                location = img_id_to_location.get(numeric_id, 'Unknown')
            except ValueError:
                location = 'Unknown'

        labels_dict[sample_id] = location
        labels_list.append(location)

    return labels_dict, labels_list


def perceptual_hash_duplicates(image_paths, hash_size=8, threshold=5):
    """
    Detect duplicate/near-duplicate images using perceptual hashing.

    Args:
        image_paths: List of image file paths
        hash_size: Size of hash (default 8 for average hash)
        threshold: Hamming distance threshold for similarity (0-10)

    Returns:
        List of duplicate groups (each group is a list of indices)
    """
    # Compute hashes for all images
    hashes = []
    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert('RGB')
            img_hash = imagehash.average_hash(img, hash_size=hash_size)
            hashes.append(img_hash)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            hashes.append(None)

    # Find duplicates by comparing hashes
    duplicate_groups = []
    used_indices = set()

    for i in range(len(hashes)):
        if i in used_indices or hashes[i] is None:
            continue

        group = [i]
        for j in range(i + 1, len(hashes)):
            if j in used_indices or hashes[j] is None:
                continue

            # Hamming distance between hashes
            distance = hashes[i] - hashes[j]
            if distance <= threshold:
                group.append(j)
                used_indices.add(j)

        if len(group) > 1:
            duplicate_groups.append(group)
            used_indices.add(i)

    return duplicate_groups


def embedding_similarity_duplicates(embeddings, threshold=0.95):
    """
    Detect duplicate/near-duplicate images using embedding similarity.

    Args:
        embeddings: numpy array of shape (n_samples, embedding_dim)
        threshold: Cosine similarity threshold (0-1, higher = more similar)

    Returns:
        List of duplicate groups (each group is a list of indices)
    """
    # Compute pairwise cosine similarity
    similarity_matrix = cosine_similarity(embeddings)

    # Set diagonal to 0 (ignore self-similarity)
    np.fill_diagonal(similarity_matrix, 0)

    # Find duplicates
    duplicate_groups = []
    used_indices = set()

    for i in range(len(embeddings)):
        if i in used_indices:
            continue

        # Find all samples similar to sample i
        similar_indices = np.where(similarity_matrix[i] >= threshold)[0]

        if len(similar_indices) > 0:
            group = [i] + similar_indices.tolist()
            duplicate_groups.append(group)
            used_indices.update(group)

    return duplicate_groups


def lda_topic_modeling(qa_data_per_cluster, text_to_image_map, cluster_labels,
                       n_topics=5, n_top_words=10, max_features=1000):
    """
    Perform LDA topic modeling on QA pairs grouped by visual clusters.

    Args:
        qa_data_per_cluster: Dict mapping cluster_id to list of QA text strings
        text_to_image_map: List mapping text embedding indices to image indices
        cluster_labels: Array of cluster labels for images
        n_topics: Number of topics to extract per cluster
        n_top_words: Number of top words to return per topic
        max_features: Maximum vocabulary size for TF-IDF

    Returns:
        Dictionary mapping cluster_id to:
            - 'topics': List of topic word lists
            - 'topic_weights': Distribution of topics
            - 'vectorizer': Fitted vectorizer
    """
    results = {}

    # Get unique cluster labels (excluding noise if present)
    unique_clusters = sorted([c for c in set(cluster_labels) if c != -1])

    for cluster_id in unique_clusters:
        # Get all QA texts for this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_img_indices = np.where(cluster_mask)[0]

        # Gather QA texts for images in this cluster
        qa_texts = []
        for img_idx in cluster_img_indices:
            # Find text embeddings for this image
            text_indices = [i for i, img in enumerate(text_to_image_map) if img == img_idx]
            # Note: This assumes qa_data_per_cluster contains actual QA texts
            # In practice, you'd need to pass the actual QA data

        # Skip if too few documents
        if len(qa_texts) < n_topics:
            continue

        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            min_df=2,
            max_df=0.8
        )

        try:
            tfidf_matrix = vectorizer.fit_transform(qa_texts)

            # LDA topic modeling
            lda = LatentDirichletAllocation(
                n_components=min(n_topics, len(qa_texts)),
                random_state=42,
                max_iter=20
            )
            lda.fit(tfidf_matrix)

            # Extract top words for each topic
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            for topic_idx, topic in enumerate(lda.components_):
                top_indices = topic.argsort()[-n_top_words:][::-1]
                top_words = [feature_names[i] for i in top_indices]
                topics.append(top_words)

            results[cluster_id] = {
                'topics': topics,
                'topic_distribution': lda.transform(tfidf_matrix).mean(axis=0),
                'n_documents': len(qa_texts)
            }

        except Exception as e:
            print(f"Error processing cluster {cluster_id}: {e}")
            continue

    return results


def select_representative_samples(cluster_labels, embeddings, samples_per_cluster=5):
    """
    Select representative samples from each cluster using diversity-based selection.

    Strategy: For each cluster, select samples that are:
    1. Close to the cluster centroid (representative)
    2. Diverse from each other (coverage)

    Args:
        cluster_labels: Array of cluster labels
        embeddings: numpy array of embeddings
        samples_per_cluster: Number of samples to select per cluster

    Returns:
        List of selected sample indices
    """
    selected_indices = []

    unique_clusters = sorted([c for c in set(cluster_labels) if c != -1])

    for cluster_id in unique_clusters:
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        cluster_embeddings = embeddings[cluster_mask]

        if len(cluster_indices) <= samples_per_cluster:
            # If cluster is small, take all samples
            selected_indices.extend(cluster_indices.tolist())
            continue

        # Compute cluster centroid
        centroid = cluster_embeddings.mean(axis=0, keepdims=True)

        # Compute distance to centroid
        distances_to_centroid = np.linalg.norm(cluster_embeddings - centroid, axis=1)

        # Select most central sample first
        selected_in_cluster = []
        central_idx = np.argmin(distances_to_centroid)
        selected_in_cluster.append(cluster_indices[central_idx])

        # Iteratively select diverse samples
        selected_embeddings = [cluster_embeddings[central_idx]]
        remaining_indices = [i for i in range(len(cluster_indices)) if i != central_idx]

        for _ in range(min(samples_per_cluster - 1, len(remaining_indices))):
            # For each remaining sample, compute min distance to already selected
            max_min_distance = -1
            best_idx = None

            for rem_idx in remaining_indices:
                rem_embedding = cluster_embeddings[rem_idx:rem_idx+1]
                min_distance = min([
                    np.linalg.norm(rem_embedding - sel_emb)
                    for sel_emb in selected_embeddings
                ])

                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_idx = rem_idx

            if best_idx is not None:
                selected_in_cluster.append(cluster_indices[best_idx])
                selected_embeddings.append(cluster_embeddings[best_idx])
                remaining_indices.remove(best_idx)

        selected_indices.extend(selected_in_cluster)

    return sorted(selected_indices)


def create_glasbey_colormap(n_colors):
    """
    Create a Glasbey-like categorical colormap for maximum color distinction.

    Args:
        n_colors: Number of distinct colors needed

    Returns:
        List of RGB tuples normalized to [0, 1]
    """
    # Use matplotlib's built-in colormaps strategically
    # For small n, use tab10/tab20
    # For larger n, use gist_rainbow with smart sampling

    if n_colors <= 10:
        import matplotlib.pyplot as plt
        cmap = plt.cm.tab10
        colors = [cmap(i) for i in range(n_colors)]
    elif n_colors <= 20:
        import matplotlib.pyplot as plt
        cmap = plt.cm.tab20
        colors = [cmap(i) for i in range(n_colors)]
    else:
        # For many colors, use perceptually distinct sampling
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        # Combine multiple colormaps for diversity
        base_colors = list(mcolors.TABLEAU_COLORS.values())
        base_colors.extend(list(mcolors.CSS4_COLORS.values()))

        # Sample evenly from the combined set
        step = len(base_colors) // n_colors
        colors = []
        for i in range(n_colors):
            idx = (i * step) % len(base_colors)
            color_hex = base_colors[idx]
            # Convert hex to RGB
            if color_hex.startswith('#'):
                rgb = mcolors.hex2color(color_hex)
            else:
                rgb = mcolors.to_rgb(color_hex)
            colors.append(rgb)

    return colors


def filter_roman_text(text):
    """
    Filter text to keep only Roman alphabet characters, numbers, and basic punctuation.

    Args:
        text: Input text string

    Returns:
        Filtered text with only Roman characters
    """
    # Keep only ASCII letters, numbers, spaces, and basic punctuation
    return re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', text)


def benchmark_dr_scaling(dr_methods_dict, sizes=[100, 500, 1000, 2000, 5000], dim=512, n_components=2, n_trials=3):
    """
    Benchmark dimensionality reduction methods with increasing dataset sizes.

    Args:
        dr_methods_dict: Dict of {method_name: method_class} to benchmark
        sizes: List of dataset sizes to test
        dim: Input dimensionality
        n_components: Output dimensionality
        n_trials: Number of trials per size (for averaging)

    Returns:
        Dictionary mapping method name to list of runtimes for each size
    """
    results = {name: [] for name in dr_methods_dict.keys()}

    for size in sizes:
        for name, method_class in dr_methods_dict.items():
            runtimes = []

            for _ in range(n_trials):
                # Generate random data
                X_random = np.random.randn(size, dim).astype(np.float32)
                X_random = X_random / np.linalg.norm(X_random, axis=1, keepdims=True)

                # Benchmark
                start = time.time()
                try:
                    if name == 'PCA':
                        method = method_class(n_components=n_components, random_state=42)
                    elif name == 't-SNE':
                        method = method_class(n_components=n_components, random_state=42, perplexity=min(30, size//4))
                    elif name == 'UMAP':
                        method = method_class(n_components=n_components, random_state=42, n_neighbors=min(15, size//2))
                    elif name == 'h-NNE':
                        method = method_class(n_components=n_components)
                    else:
                        method = method_class(n_components=n_components, random_state=42)

                    _ = method.fit_transform(X_random)
                    runtime = time.time() - start
                    runtimes.append(runtime)
                except Exception as e:
                    print(f"Error with {name} at size {size}: {e}")
                    runtimes.append(np.nan)

            # Average runtime across trials
            avg_runtime = np.nanmean(runtimes)
            results[name].append(avg_runtime)

    return results, sizes


def benchmark_clustering_methods(clustering_methods_dict, sizes=[100, 500, 1000, 2000, 5000], dim=512, n_trials=3):
    """
    Benchmark clustering methods with increasing dataset sizes.

    Args:
        clustering_methods_dict: Dict of {method_name: (method_class, params)} to benchmark
        sizes: List of dataset sizes to test
        dim: Input dimensionality
        n_trials: Number of trials per size

    Returns:
        Dictionary mapping method name to list of runtimes for each size
    """
    results = {name: [] for name in clustering_methods_dict.keys()}

    for size in sizes:
        for name, (method_class, params) in clustering_methods_dict.items():
            runtimes = []

            for _ in range(n_trials):
                # Generate random data
                X_random = np.random.randn(size, dim).astype(np.float32)
                X_random = X_random / np.linalg.norm(X_random, axis=1, keepdims=True)

                # Benchmark
                start = time.time()
                try:
                    if name == 'FINCH':
                        # FINCH doesn't use sklearn API
                        from finch import FINCH
                        _ = FINCH(X_random, verbose=False)
                    else:
                        method = method_class(**params)
                        _ = method.fit_predict(X_random)

                    runtime = time.time() - start
                    runtimes.append(runtime)
                except Exception as e:
                    print(f"Error with {name} at size {size}: {e}")
                    runtimes.append(np.nan)

            avg_runtime = np.nanmean(runtimes)
            results[name].append(avg_runtime)

    return results, sizes


def find_most_similar_pairs(embeddings, top_k=5):
    """
    Find the top-k most similar pairs of samples based on embedding similarity.

    Useful for visualizing what the duplicate detection methods consider similar,
    even if they're not true duplicates.

    Args:
        embeddings: numpy array of embeddings (n_samples, dim)
        top_k: Number of most similar pairs to return

    Returns:
        List of tuples (idx1, idx2, similarity_score)
    """
    # Compute pairwise similarity
    similarity_matrix = cosine_similarity(embeddings)

    # Set diagonal to -1 (ignore self-similarity)
    np.fill_diagonal(similarity_matrix, -1)

    # Find top-k similar pairs
    # Only consider upper triangle to avoid duplicates (i,j) and (j,i)
    pairs = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            pairs.append((i, j, similarity_matrix[i, j]))

    # Sort by similarity descending
    pairs.sort(key=lambda x: x[2], reverse=True)

    return pairs[:top_k]


def select_samples_random(n_samples, total_samples, random_state=42):
    """
    Random sampling strategy.

    Args:
        n_samples: Number of samples to select
        total_samples: Total number of samples available
        random_state: Random seed

    Returns:
        List of selected indices
    """
    np.random.seed(random_state)
    return np.random.choice(total_samples, size=min(n_samples, total_samples), replace=False).tolist()


def select_samples_stratified(cluster_labels, samples_per_cluster=5):
    """
    Stratified sampling: Select equal number from each cluster.

    Args:
        cluster_labels: Array of cluster labels
        samples_per_cluster: Number of samples per cluster

    Returns:
        List of selected indices
    """
    selected = []
    unique_clusters = sorted([c for c in set(cluster_labels) if c != -1])

    for cluster_id in unique_clusters:
        cluster_indices = np.where(cluster_labels == cluster_id)[0]

        # Sample randomly from this cluster
        n_select = min(samples_per_cluster, len(cluster_indices))
        selected_from_cluster = np.random.choice(cluster_indices, size=n_select, replace=False)
        selected.extend(selected_from_cluster.tolist())

    return sorted(selected)


def select_samples_boundary(cluster_labels, embeddings, samples_per_cluster=5):
    """
    Boundary-based sampling: Select samples near cluster boundaries.

    These samples are more informative for learning decision boundaries.

    Args:
        cluster_labels: Array of cluster labels
        embeddings: numpy array of embeddings
        samples_per_cluster: Number of samples per cluster

    Returns:
        List of selected indices
    """
    selected = []
    unique_clusters = sorted([c for c in set(cluster_labels) if c != -1])

    for cluster_id in unique_clusters:
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        cluster_embeddings = embeddings[cluster_mask]

        if len(cluster_indices) <= samples_per_cluster:
            selected.extend(cluster_indices.tolist())
            continue

        # Compute cluster centroid
        centroid = cluster_embeddings.mean(axis=0, keepdims=True)

        # Select samples FARTHEST from centroid (near boundaries)
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        farthest_indices = np.argsort(distances)[-samples_per_cluster:]

        selected.extend(cluster_indices[farthest_indices].tolist())

    return sorted(selected)


def create_wordcloud_data(topics_dict, cluster_id):
    """
    Create word frequency dictionary for wordcloud generation.

    Args:
        topics_dict: Dictionary from topic modeling with cluster topics
        cluster_id: Cluster ID to generate wordcloud for

    Returns:
        Dictionary of {word: frequency} for wordcloud
    """
    if cluster_id not in topics_dict:
        return {}

    topics = topics_dict[cluster_id]['topics']

    # Combine all topics and weight by position (earlier words = higher weight)
    word_freq = {}
    for topic_idx, topic_words in enumerate(topics):
        for word_idx, word in enumerate(topic_words):
            # Weight decreases with position in list
            weight = len(topic_words) - word_idx
            word_freq[word] = word_freq.get(word, 0) + weight

    return word_freq


def nmf_topic_modeling(cluster_qa_texts, n_topics=3, n_top_words=8, max_features=500):
    """
    Alternative topic modeling using NMF (Non-negative Matrix Factorization).

    NMF often produces more interpretable topics than LDA for short documents.

    Args:
        cluster_qa_texts: Dict mapping cluster_id to list of text strings
        n_topics: Number of topics to extract
        n_top_words: Number of top words per topic
        max_features: Maximum vocabulary size

    Returns:
        Dictionary mapping cluster_id to topic information
    """
    results = {}

    for cluster_id, qa_texts in cluster_qa_texts.items():
        if len(qa_texts) < n_topics:
            continue

        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            min_df=2,
            max_df=0.8
        )

        try:
            tfidf_matrix = vectorizer.fit_transform(qa_texts)

            # NMF
            nmf = NMF(n_components=min(n_topics, len(qa_texts)), random_state=42, max_iter=200)
            nmf.fit(tfidf_matrix)

            # Extract top words
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            for topic_idx, topic in enumerate(nmf.components_):
                top_indices = topic.argsort()[-n_top_words:][::-1]
                top_words = [feature_names[i] for i in top_indices]
                topics.append(top_words)

            results[cluster_id] = {
                'topics': topics,
                'n_documents': len(qa_texts),
                'method': 'NMF'
            }

        except Exception as e:
            print(f"Error processing cluster {cluster_id}: {e}")
            continue

    return results


def find_optimal_hierarchy_level(finch_hierarchy, num_clust, target_n_clusters):
    """
    Find the FINCH hierarchy level closest to a target number of clusters.

    Args:
        finch_hierarchy: FINCH hierarchy array
        num_clust: List of cluster counts per level
        target_n_clusters: Target number of clusters

    Returns:
        Optimal level index and its cluster count
    """
    differences = [abs(n - target_n_clusters) for n in num_clust]
    optimal_level = np.argmin(differences)
    return optimal_level, num_clust[optimal_level]


def compute_ssim(image_path1, image_path2, target_size=(256, 256)):
    """
    Compute SSIM (Structural Similarity Index) between two images.

    Args:
        image_path1: Path to first image
        image_path2: Path to second image
        target_size: Resize images to this size for comparison

    Returns:
        SSIM score (0 to 1, higher = more similar)
    """
    try:
        # Load images in grayscale
        img1 = cv2.imread(str(image_path1), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(image_path2), cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            return 0.0

        # Resize to same size
        img1 = cv2.resize(img1, target_size)
        img2 = cv2.resize(img2, target_size)

        # Compute SSIM
        score = ssim(img1, img2)
        return score
    except Exception as e:
        print(f"Error computing SSIM: {e}")
        return 0.0


def load_modality_labels(dataset_path, sample_data):
    """
    Extract modality labels from SLAKE train/val/test JSON files.

    Args:
        dataset_path: Path to SLAKE dataset directory
        sample_data: List of sample dictionaries with 'sample_id' field

    Returns:
        Dictionary mapping sample_id to modality label
        List of modality labels in same order as sample_data
    """
    dataset_path = Path(dataset_path)

    # Load all JSON files
    json_files = ['train.json', 'test.json', 'validate.json']
    img_name_to_modality = {}

    for json_file in json_files:
        json_path = dataset_path / json_file
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)

            # Extract img_name -> modality mapping
            for entry in data:
                img_name = entry['img_name']
                modality = entry.get('modality', 'Unknown')
                img_name_to_modality[img_name] = modality

    # Map sample_ids to modalities
    labels_dict = {}
    labels_list = []

    for sample in sample_data:
        sample_id = sample['sample_id']
        img_name = f"{sample_id}/source.jpg"

        modality = img_name_to_modality.get(img_name, 'Unknown')
        labels_dict[sample_id] = modality
        labels_list.append(modality)

    return labels_dict, labels_list


def filter_cluster_specific_words(cluster_topics, min_cluster_specificity=0.3):
    """
    Filter out words that appear in too many clusters (not cluster-specific).

    Strategy: Remove words that appear in >30% of clusters, as they're
    likely dataset-level topics rather than cluster-specific.

    Args:
        cluster_topics: Dict mapping cluster_id to topic info
        min_cluster_specificity: Only keep words appearing in < this fraction of clusters

    Returns:
        Filtered cluster_topics dictionary
    """
    # Count how many clusters each word appears in
    word_to_clusters = defaultdict(set)

    for cluster_id, topic_info in cluster_topics.items():
        topics = topic_info.get('topics', [])
        for topic in topics:
            for word in topic:
                word_to_clusters[word].add(cluster_id)

    n_clusters = len(cluster_topics)

    # Determine which words to filter (appear in too many clusters)
    common_words = set()
    for word, clusters in word_to_clusters.items():
        if len(clusters) / n_clusters > min_cluster_specificity:
            common_words.add(word)

    # Filter topics
    filtered_topics = {}
    for cluster_id, topic_info in cluster_topics.items():
        topics = topic_info.get('topics', [])
        filtered_topic_words = []

        for topic in topics:
            # Filter out common words
            filtered_words = [w for w in topic if w not in common_words]
            # If we filtered too many, keep some common words to maintain topic size
            if len(filtered_words) < 3 and len(topic) > 0:
                filtered_words = topic[:min(5, len(topic))]
            filtered_topic_words.append(filtered_words)

        filtered_topics[cluster_id] = {
            **topic_info,
            'topics': filtered_topic_words,
            'filtered_common_words': list(common_words & set([w for t in topics for w in t]))
        }

    return filtered_topics
