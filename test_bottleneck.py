import numpy as np
import h5py
import os
import torch
import time
import logging
from viztracer import VizTracer
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
from torch.cuda.amp import autocast, GradScaler
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tracer = VizTracer()


class Timer:
    def __init__(self):
        self.start_time = 0
        self.end_time = 0

    def start(self):
        self.start_time = time.time()

    def end(self):
        self.end_time = time.time()
        return self.end_time - self.start_time

def merge_sorted_unique(a, b):
    """优化版本：合并两个已排序的 1D 张量并去重，纯 PyTorch 操作，避免 CPU-GPU 数据拷贝"""
    if a.numel() == 0:
        return b
    if b.numel() == 0:
        return a

    combined = torch.cat((a, b))  # O(N)
    merged_sorted, _ = torch.sort(combined)
    return torch.unique_consecutive(merged_sorted)

def compute_eta(data, T):
    """
    Compute eta based on the anisotropic cost formula.

    Args:
        data (torch.Tensor): Shape (n, d), dataset of n points with dimension d.
        T (float): Threshold parameter.

    Returns:
        torch.Tensor: Shape (n,), computed eta for each data point.
    """
    # Step 1: Compute squared L2 norms (|x|^2) for each data point
    squared_l2_norms = torch.norm(data, p=2, dim=1) ** 2  # Shape: (n,)

    # Step 2: Calculate parallel cost
    parallel_cost = (T ** 2) / squared_l2_norms  # Shape: (n,)

    # Step 3: Calculate perpendicular cost
    dims = data.size(1)  # Number of dimensions (d)
    perpendicular_cost = (1.0 - (T ** 2) / squared_l2_norms) / (dims - 1)  # Shape: (n,)

    # Step 4: Compute eta (ratio of parallel_cost to perpendicular_cost)
    eta = parallel_cost / perpendicular_cost  # Shape: (n,)

    eta = eta.unsqueeze(1)  # Shape: (n, 1)
    return eta


def compute_anisotropic_loss(data, centers, T, eta_weight=None):
    """
    Compute anisotropic loss for data points and cluster centers.

    Args:
        data (torch.Tensor): Shape (n, d), n data points with dimension d.
        centers (torch.Tensor): Shape (k, d), k cluster centers with dimension d.
        T (float): Threshold parameter for computing η.
        eta_weight (torch.Tensor, optional): Precomputed η values, shape (n,1).

    Returns:
        torch.Tensor: Shape (n, k), anisotropic loss for each data point-center pair.
    """
    # Step 1: Compute residuals (x_i - c_j) for all centers
    residuals = data.unsqueeze(1) - centers.unsqueeze(0)  # Shape: (n, k, d)

    # Step 2: Compute norms of data points
    data_norms = torch.norm(data, p=2, dim=1, keepdim=True)  # Shape: (n, 1)

    # Step 3: Compute parallel residual error
    dot_products = torch.einsum("nd,nkd->nk", data, residuals)  # Shape: (n, k)
    k = centers.size(0)
    r_parallel = (dot_products / (data_norms ** 2)).unsqueeze(2) * data.unsqueeze(1).expand(-1, k,
                                                                                            -1)  # Shape: (n, k, d)

    # Step 4: Compute orthogonal residual error
    r_orthogonal = residuals - r_parallel  # Shape: (n, k, d)

    eta = torch.ones_like(data_norms)  # shape (n,1).

    # Step 5: Compute η if not precomputed
    if eta_weight is None:
        eta = compute_eta(data, T)
    else:
        eta = eta_weight
    # Step 6: Compute losses
    parallel_loss = torch.norm(r_parallel, p=2, dim=2) ** 2  # Shape: (n, k)
    orthogonal_loss = torch.norm(r_orthogonal, p=2, dim=2) ** 2  # Shape: (n, k)

    # eta_tensor = torch.full_like(data_norms, eta) #when eta is a float
    anisotropic_loss = eta * parallel_loss + orthogonal_loss  # Shape: (n, k)

    return anisotropic_loss


def compute_anisotropic_loss_batch(data, centers, T, eta_weight=None, batch_size=1000):
    """
    Compute anisotropic loss for data points and cluster centers using batch processing.

    Args:
        data (torch.Tensor): Shape (n, d), n data points with dimension d.
        centers (torch.Tensor): Shape (k, d), k cluster centers with dimension d.
        T (float): Threshold parameter for computing η.
        eta_weight (torch.Tensor, optional): Precomputed η values, shape (n, 1).
        batch_size (int): Number of data points to process in each batch.

    Returns:
        torch.Tensor: Shape (n, k), anisotropic loss for each data point-center pair.
    """
    num_data_points = data.size(0)
    num_clusters = centers.size(0)
    losses = torch.zeros(num_data_points, num_clusters, device=data.device)

    # Compute η if not precomputed
    if eta_weight is None:
        eta = compute_eta(data, T)  # Shape: (n, 1)
    else:
        if eta_weight.device != data.device:
            eta_weight = eta_weight.to(data.device)
        eta = eta_weight  # Shape: (n, 1)

    # Process data in batches
    for batch_start in range(0, num_data_points, batch_size):
        batch_end = min(batch_start + batch_size, num_data_points)
        batch_data = data[batch_start:batch_end]  # Shape: (batch_size, d)
        batch_eta = eta[batch_start:batch_end]  # Shape: (batch_size, 1)

        # Step 1: Compute residuals (batch)
        batch_residuals = batch_data.unsqueeze(1) - centers.unsqueeze(0)  # Shape: (batch_size, k, d)

        # Step 2: Compute norms of batch data points
        batch_data_norms = torch.norm(batch_data, p=2, dim=1, keepdim=True)  # Shape: (batch_size, 1)

        # Step 3: Compute parallel residual error
        batch_dot_products = torch.einsum("nd,nkd->nk", batch_data, batch_residuals)  # Shape: (batch_size, k)
        r_parallel = ((batch_dot_products / (batch_data_norms ** 2)).unsqueeze(2)
                      * batch_data.unsqueeze(1).expand(-1, num_clusters, -1))  # Shape: (batch_size, k, d)

        # Step 4: Compute orthogonal residual error
        r_orthogonal = batch_residuals - r_parallel  # Shape: (batch_size, k, d)

        # Step 5: Compute losses
        parallel_loss = torch.norm(r_parallel, p=2, dim=2) ** 2  # Shape: (batch_size, k)
        orthogonal_loss = torch.norm(r_orthogonal, p=2, dim=2) ** 2  # Shape: (batch_size, k)

        batch_loss = batch_eta * parallel_loss + orthogonal_loss  # Shape: (batch_size, k)

        # Step 6: Store the computed losses
        losses[batch_start:batch_end] = batch_loss

    return losses


def update_cluster_centers(data, assignments, num_clusters, cluster_centers, T, eta_weight=None):
    """
    Update cluster centers using the closed-form solution (Theorem 4.2).

    Args:
        data (torch.Tensor): Shape (n, d), dataset of n points with dimension d.
        assignments (torch.Tensor): Shape (n,), cluster assignments for each datapoint.
        num_clusters (int): Number of clusters.
        T (float): Threshold parameter for computing h_parallel and h_perpendicular.

    Returns:
        torch.Tensor: Updated cluster centers of shape (k, d).
    """
    n, d = data.size()  # Number of data points, dimension

    if assignments.device != data.device:
        assignments = assignments.to(data.device)

    # Sort assignments and compute boundaries for clusters
    sorted_assignments, indices = torch.sort(assignments)
    boundaries = torch.searchsorted(sorted_assignments, torch.arange(num_clusters + 1, device=assignments.device))

    # # Compute eta for each data point
    # eta_tensor = compute_eta(data, T)  # Shape: (n, 1)
    if eta_weight is None:
        eta_tensor = compute_eta(data, T)  # Shape: (n, 1)
    else:
        if eta_weight.device != data.device:
            eta_weight = eta_weight.to(data.device)
        eta_tensor = eta_weight  # Shape: (n, 1)
    eta = eta_tensor[0]  # Now eta is a scalar value

    # Precompute norms and related variables
    norms = torch.norm(data, p=2, dim=1)  # Norms of all data points
    norms_pow = norms ** (0.5 * (eta - 3))
    fillzero = 1 if eta == 1 else 0
    norms_pow = torch.where(norms < 1e-20, torch.tensor(fillzero, device=data.device), norms_pow)
    norms_eta1 = norms ** (eta - 1)
    dp_norm_sum = norms_eta1.sum()  # scalar, Sum of norm^(eta - 1)

    # Loop through each cluster
    for k in range(num_clusters):
        # Get the indices of the points assigned to cluster k
        cluster_indices = indices[boundaries[k]:boundaries[k + 1]]
        # Extract the data points assigned to cluster k
        cluster_data = data[cluster_indices]  # Shape: (num_points_in_cluster, d)

        if cluster_data.size(0) == 0:
            continue  # Skip empty clusters

        """Todo: norms_pow"""
        norms_pow_k = norms_pow[cluster_indices]  # Shape: (num_points_in_cluster,)
        X = cluster_data * norms_pow_k.view(-1, 1)  # Shape: (num_points_in_cluster, d)
        # Compute the matrix to invert
        xtx = torch.mm(X.T, X)  # X^T X (shape: d, d)

        norms_eta1_k = norms_eta1[cluster_indices]  # Shape: (num_points_in_cluster,)

        norm_weighted_dp_sum = (cluster_data * norms_eta1_k.view(-1, 1)).sum(dim=0)  # Shape: (d,)

        to_invert = dp_norm_sum * torch.eye(d, device=data.device) + (eta - 1) * xtx  # Shape: (d,d)

        # Solve for the new center (cj*) using the closed-form solution
        inv_matrix = torch.linalg.inv(to_invert)  # Inverse of the matrix to invert
        cluster_center = eta * torch.matmul(inv_matrix, norm_weighted_dp_sum)  # Shape: (d,)

        # Store the updated center for cluster k
        cluster_centers[k] = cluster_center

    return cluster_centers


def kmeans_anisotropic(data, num_clusters, max_iters=100, T=0.2):
    n, d = data.size()
    random_indices = torch.randperm(n)[:num_clusters]
    cluster_centers = data[random_indices]

    assignments = torch.zeros(n, dtype=torch.long, device=data.device)
    eta_weight = compute_eta(data, T)
    for i in range(max_iters):
        losses = compute_anisotropic_loss_batch(data, cluster_centers, T, eta_weight)  # Shape (n, k)
        new_assignments = losses.argmin(dim=1)  # Shape (n, )

        if torch.equal(new_assignments, assignments):
            print(f"Converged in {i + 1} iterations")
            break
        assignments = new_assignments
        cluster_centers = update_cluster_centers(data, assignments, num_clusters, cluster_centers, T, eta_weight)

    sorted_assignments, indices = torch.sort(assignments)
    # print(indices)
    boundaries = torch.searchsorted(sorted_assignments, torch.arange(num_clusters + 1, device=assignments.device))
    # print(boundaries)
    inverted_index = {
        k: indices[boundaries[k]:boundaries[k + 1]] for k in range(num_clusters)
    }

    return cluster_centers, assignments, inverted_index


def compute_normalized_residual(data_points, centroids):
    """
    Compute normalized residuals for data points and their corresponding centroids.

    Parameters:
        data_points (torch.Tensor): Tensor of shape (n, d) or (d,), representing data points.
        centroids (torch.Tensor): Tensor of shape (n, d) or (d,), representing corresponding centroids.

    Returns:
        torch.Tensor: Normalized residuals of shape (n, d) or (d,).
    """
    # Ensure data_points and centroids have the same shape
    assert data_points.shape == centroids.shape, "Shapes of data points and centroids must match"
    # print(f"Dataset device: {data_points.device}")
    # print(f"Cluster centers device: {centroids.device}")
    # If inputs are 1D (d,), reshape them to (1, d) for consistent handling
    if data_points.dim() == 1:
        data_points = data_points.unsqueeze(0)
        centroids = centroids.unsqueeze(0)

    # Calculate the residuals
    residuals = data_points - centroids  # Shape (n, d)

    # Compute the squared norm along dimension d for each data point
    sqnorm = torch.sum(residuals ** 2, dim=1, keepdim=True)  # Shape (n, 1)

    # Avoid division by zero by setting very small norms to 1 (we'll set their residuals to 0 later)
    sqnorm = torch.where(sqnorm < 1e-7, torch.tensor(1.0, device=sqnorm.device), sqnorm)

    # Normalize the residuals by their norms
    normalized_residuals = residuals / torch.sqrt(sqnorm)  # Shape (n, d)

    # Create a mask for very small norms and expand it to (n, d)
    small_norm_mask = (sqnorm < 1e-7).expand_as(normalized_residuals)

    # Set residuals to zero where sqnorm was too small
    normalized_residuals[small_norm_mask] = 0.0

    # If the input was originally 1D, return a 1D result
    if normalized_residuals.shape[0] == 1:
        return normalized_residuals.squeeze(0)

    return normalized_residuals


# batched matrix operation

def compute_secondary_assignments_optimized(data, cluster_centers, primary_assignments, residuals, inverted_index,
                                            lambda_param=1.0, batch_size=10000):
    num_clusters = cluster_centers.size(0)
    num_data_points = data.size(0)

    # Initialize secondary assignments tensor
    secondary_assignments = torch.zeros(num_data_points, dtype=torch.long, device=data.device)

    # Process data in batches to reduce memory usage
    for batch_start in range(0, num_data_points, batch_size):
        batch_end = min(batch_start + batch_size, num_data_points)
        batch_data = data[batch_start:batch_end]  # (batch_size, data_dim)
        batch_primary_assignments = primary_assignments[batch_start:batch_end]  # (batch_size,)
        batch_residuals = residuals[batch_start:batch_end]  # (batch_size, data_dim)

        batch_size_actual = batch_data.size(0)

        # Step 1: Mask primary assignments
        mask = torch.ones((batch_size_actual, num_clusters), dtype=torch.bool, device=data.device)
        mask.scatter_(1, batch_primary_assignments.unsqueeze(1), False)  # (batch_size, num_clusters)

        # Step 2: Compute residuals for all non-primary clusters
        candidate_residuals = (batch_data.unsqueeze(1) - cluster_centers.unsqueeze(0)) * mask.unsqueeze(-1)
        # Shape: (batch_size, num_clusters, data_dim)

        # Step 3: Compute distance terms ||r'||^2
        dist_terms = torch.norm(candidate_residuals, dim=2).pow(2)  # (batch_size, num_clusters)

        # Step 4: Compute penalty terms for alignment between primary and candidate residuals
        dot_products = (batch_residuals.unsqueeze(1) * candidate_residuals).sum(dim=2)  # (batch_size, num_clusters)
        parallel_penalty = lambda_param * dot_products.pow(2)  # (batch_size, num_clusters)

        # Step 5: Compute total penalized distance
        penalized_distances = dist_terms + parallel_penalty  # (batch_size, num_clusters)
        penalized_distances[~mask] = float('inf')

        # Step 6: Find the secondary assignments for the batch
        batch_secondary_assignments = penalized_distances.argmin(dim=1)  # (batch_size,)

        # Step 7: Store results
        secondary_assignments[batch_start:batch_end] = batch_secondary_assignments

    # Step 8: Sort secondary assignments and update inverted index
    sorted_secondary, secondary_indices = torch.sort(secondary_assignments)  # (num_data_points,)

    boundaries = torch.searchsorted(sorted_secondary,
                                    torch.arange(num_clusters + 1, device=data.device))  # (num_clusters+1,)

    for k in range(num_clusters):
        new_indices = secondary_indices[boundaries[k]:boundaries[k + 1]]  # Indices of points assigned to cluster k
        inverted_index[k] = torch.cat((inverted_index[k], new_indices))
    return secondary_assignments



def compute_query_secondary_assignments_optimized(queries, cluster_centers, primary_assignments, residuals,
                                                  lambda_param=1.0):
    if queries.dim() == 1:
        queries = queries.unsqueeze(0)
        residuals = residuals.unsqueeze(0)
        primary_assignments = primary_assignments.unsqueeze(0)

    num_queries = queries.size(0)
    num_clusters = cluster_centers.size(0)

    device = queries.device
    mask = torch.ones((num_queries, num_clusters), dtype=torch.bool, device=device)
    mask.scatter_(1, primary_assignments.unsqueeze(1), False)

    candidate_centers = cluster_centers.unsqueeze(0)
    queries_expanded = queries.unsqueeze(1)
    candidate_residuals = queries_expanded - candidate_centers
    candidate_residuals *= mask.unsqueeze(-1)

    dist_terms = torch.norm(candidate_residuals, dim=2).pow(2)
    primary_residuals = residuals.unsqueeze(1)
    dot_products = (primary_residuals * candidate_residuals).sum(dim=2)
    parallel_penalty = lambda_param * dot_products.pow(2)

    penalized_distances = dist_terms + parallel_penalty
    penalized_distances[~mask] = float('inf')

    secondary_assignments = penalized_distances.argmin(dim=1)

    return secondary_assignments


def merge_sorted_unique(a, b):
    """合并两个已排序的 1D 张量并去重，纯 PyTorch 操作"""
    if a.numel() == 0:
        return b
    if b.numel() == 0:
        return a
    combined = torch.cat((a, b))
    merged_sorted, _ = torch.sort(combined)
    return torch.unique_consecutive(merged_sorted)


"""no batch just grouping"""


def search_batched_chunked_matmul_mixed_precision(queries, cluster_centers, data_inverted_index, data, k=100,
                                                  batch_size=20,
                                                  candidate_chunk_size=100000):
    """
    使用混合精度的批量搜索：
    1. 在关键计算步骤中使用混合精度（FP16）以提高计算效率和减少内存使用。
    2. 其他优化策略可以在此基础上继续进行。
    """
    device = queries.device
    nq, d = queries.shape

    # 创建 GradScaler 用于缩放梯度
    scaler = GradScaler()

    # Step 1: 计算主中心
    with autocast():  # 使用混合精度计算
        inner_products = torch.matmul(queries, cluster_centers.T)
        primary_center_indices = torch.argmax(inner_products, dim=1)

    # Step 2: 计算残差
    primary_centers = cluster_centers[primary_center_indices]
    with autocast():  # 使用混合精度计算
        residuals = compute_normalized_residual(queries, primary_centers)

    # Step 3: 计算次中心
    # start = time.time()
    with autocast():  # 使用混合精度计算
        secondary_center_indices = compute_query_secondary_assignments_optimized(
            queries, cluster_centers, primary_center_indices, residuals, lambda_param=1.0
        )
    # end = time.time()
    # print(f"compute_query_secondary_assignments_optimized time:{end - start}s")

    # Step 4: 按 (primary, secondary) 组合对查询进行分组
    combinations = torch.stack([primary_center_indices, secondary_center_indices], dim=1)
    unique_combos, inverse_indices = torch.unique(combinations, return_inverse=True, dim=0)

    # 初始化输出张量
    all_nearest_neighbors = torch.full((nq, k), -1, dtype=torch.long, device=device)
    all_top_k_distances = torch.full((nq, k), float('-inf'), device=device)

    # 批量处理各 (primary, secondary) 组合
    num_batches = (unique_combos.size(0) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        batch_combos = unique_combos[batch_idx * batch_size: (batch_idx + 1) * batch_size]

        # 4.1 收集当前批次的候选点集合
        batch_candidates = []
        for c1, c2 in batch_combos:
            c1_candidates = data_inverted_index.get(c1.item(), torch.tensor([], device=device, dtype=torch.long))
            c2_candidates = data_inverted_index.get(c2.item(), torch.tensor([], device=device, dtype=torch.long))
            combined_candidates = torch.cat([c1_candidates, c2_candidates])
            unique_candidates = torch.unique(combined_candidates)  # 去重
            batch_candidates.append(unique_candidates)

        # 4.2 对当前批次中的每个 (primary, secondary) 组合处理
        for i, (c1, c2) in enumerate(batch_combos):
            query_mask = (inverse_indices == (batch_idx * batch_size + i))
            group_queries = queries[query_mask]
            unique_candidates = batch_candidates[i]  # 当前组合下的候选集

            # 获取候选点对应的向量
            candidate_vectors = data[unique_candidates]  # shape: (nc, d)
            nc = unique_candidates.size(0)

            # 分块处理候选向量，避免一次性构造太大矩阵
            partial_scores = []
            partial_indices = []
            for start in range(0, nc, candidate_chunk_size):
                end = min(start + candidate_chunk_size, nc)
                candidate_chunk = candidate_vectors[start:end]  # shape: (chunk_size, d)
                with autocast():  # 使用混合精度计算
                    scores_chunk = torch.matmul(group_queries, candidate_chunk.T)  # shape: (num_queries, chunk_size)
                # 取得当前块的局部 top-k（每个查询）
                chunk_scores, chunk_indices = torch.topk(scores_chunk, k, dim=1)
                partial_scores.append(chunk_scores)
                partial_indices.append(chunk_indices + start)  # 调整为候选集中的索引

            # 合并所有块的局部结果
            all_partial_scores = torch.cat(partial_scores, dim=1)  # shape: (num_queries, total_candidates_in_chunks)
            all_partial_indices = torch.cat(partial_indices, dim=1)

            # 对合并后的结果再取全局 top-k
            group_topk_scores, topk_order = torch.topk(all_partial_scores, k, dim=1)
            group_topk_local_indices = torch.gather(all_partial_indices, 1, topk_order)

            # 当候选数不足 k 时，做适当 padding
            if nc < k:
                pad_size = k - nc
                pad_scores = torch.full((group_queries.shape[0], pad_size), float('-inf'), device=device)
                pad_indices = torch.full((group_queries.shape[0], pad_size), -1, dtype=torch.long, device=device)
                group_topk_scores = torch.cat([group_topk_scores, pad_scores], dim=1)
                group_topk_local_indices = torch.cat([group_topk_local_indices, pad_indices], dim=1)

            # 将局部候选索引转换为全局索引
            global_indices = unique_candidates[group_topk_local_indices]
            all_nearest_neighbors[query_mask] = global_indices
            # all_top_k_distances[query_mask] = group_topk_scores
            group_topk_scores = group_topk_scores.to(torch.float32)
            all_top_k_distances[query_mask] = group_topk_scores

    return all_nearest_neighbors, all_top_k_distances


def search_batched_chunked_matmul_mixed_precision_2unique(queries, cluster_centers, data_inverted_index, data, k=100,
                                                          batch_size=20, candidate_chunk_size=100000):
    """
    使用混合精度的批量搜索，并对以下两个 unique 操作进行优化：
      1. 将 (primary, secondary) 组合转换为 1D 键，避免二维 unique 操作；
      2. 对每个组合中候选点集合的 unique 操作，采用双指针归并去重（假定输入候选数组已排序）。
    """
    device = queries.device
    nq, d = queries.shape
    num_clusters = cluster_centers.size(0)

    scaler = GradScaler()

    # Step 1: 计算主中心
    with autocast():
        inner_products = torch.matmul(queries, cluster_centers.T)
        primary_center_indices = torch.argmax(inner_products, dim=1)

    # Step 2: 计算残差
    primary_centers = cluster_centers[primary_center_indices]
    with autocast():
        residuals = compute_normalized_residual(queries, primary_centers)

    # Step 3: 计算次中心
    with autocast():
        secondary_center_indices = compute_query_secondary_assignments_optimized(
            queries, cluster_centers, primary_center_indices, residuals, lambda_param=1.0
        )

    # Step 4: 按 (primary, secondary) 组合对查询进行分组
    # 【优化1】：使用 1D 键替代二维 unique 操作
    combined_keys = primary_center_indices * num_clusters + secondary_center_indices
    unique_keys, inverse_indices = torch.unique(combined_keys, return_inverse=True)
    unique_primary = unique_keys // num_clusters
    unique_secondary = unique_keys % num_clusters
    unique_combos = torch.stack([unique_primary, unique_secondary], dim=1)

    # 初始化输出张量
    all_nearest_neighbors = torch.full((nq, k), -1, dtype=torch.long, device=device)
    all_top_k_distances = torch.full((nq, k), float('-inf'), device=device)

    num_batches = (unique_combos.size(0) + batch_size - 1) // batch_size

    # 【工具函数】：归并两个已排序张量并去重
    def merge_sorted_unique(a, b):
        """优化版本：合并两个已排序的 1D 张量并去重，纯 PyTorch 操作，避免 CPU-GPU 数据拷贝"""
        if a.numel() == 0:
            return b
        if b.numel() == 0:
            return a

        combined = torch.cat((a, b))  # O(N)
        merged_sorted, _ = torch.sort(combined)
        return torch.unique_consecutive(merged_sorted)

    # 批量处理各 (primary, secondary) 组合
    for batch_idx in range(num_batches):
        batch_combos = unique_combos[batch_idx * batch_size: (batch_idx + 1) * batch_size]

        #  批量load当前批次的候选点集合
        batch_candidates = []
        for c1, c2 in batch_combos:
            c1_candidates = data_inverted_index.get(c1.item(), torch.tensor([], device=device, dtype=torch.long))
            c2_candidates = data_inverted_index.get(c2.item(), torch.tensor([], device=device, dtype=torch.long))
            # 【优化2】：对两个有序候选点数组归并去重
            unique_candidates = merge_sorted_unique(c1_candidates, c2_candidates)
            batch_candidates.append(unique_candidates)

        # 4.2 对当前批次中的每个 (primary, secondary) 组合处理
        for i, (c1, c2) in enumerate(batch_combos):
            query_mask = (inverse_indices == (batch_idx * batch_size + i))
            group_queries = queries[query_mask]
            unique_candidates = batch_candidates[i]  # 当前组合下的候选集
            candidate_vectors = data[unique_candidates]  # shape: (nc, d)
            nc = unique_candidates.size(0)

            partial_scores = []
            partial_indices = []
            for start in range(0, nc, candidate_chunk_size):
                end = min(start + candidate_chunk_size, nc)
                candidate_chunk = candidate_vectors[start:end]  # shape: (chunk_size, d)
                chunk_size = candidate_chunk.shape[0]
                with autocast():
                    scores_chunk = torch.matmul(group_queries, candidate_chunk.T)  # shape: (num_queries, chunk_size)
                if chunk_size < k:
                    # 如果 chunk_size < k，手动填充 `-inf` 以确保 torch.topk 可执行
                    pad_size = k - chunk_size
                    pad_scores = torch.full((scores_chunk.shape[0], pad_size), float('-inf'),
                                            device=scores_chunk.device)
                    scores_chunk = torch.cat([scores_chunk, pad_scores], dim=1)

                chunk_scores, chunk_indices = torch.topk(scores_chunk, k, dim=1)
                partial_scores.append(chunk_scores)
                partial_indices.append(chunk_indices + start)  # 调整为候选集中的索引

            all_partial_scores = torch.cat(partial_scores, dim=1)  # shape: (num_queries, total_candidates_in_chunks)
            all_partial_indices = torch.cat(partial_indices, dim=1)

            group_topk_scores, topk_order = torch.topk(all_partial_scores, k, dim=1)
            group_topk_local_indices = torch.gather(all_partial_indices, 1, topk_order)
            if nc < k:
                pad_size = k - nc
                pad_scores = torch.full((group_queries.shape[0], pad_size), float('-inf'), device=device)
                pad_indices = torch.full((group_queries.shape[0], pad_size), -1, dtype=torch.long, device=device)
                group_topk_scores = torch.cat([group_topk_scores, pad_scores], dim=1)
                group_topk_local_indices = torch.cat([group_topk_local_indices, pad_indices], dim=1)

            global_indices = unique_candidates[group_topk_local_indices]
            all_nearest_neighbors[query_mask] = global_indices
            group_topk_scores = group_topk_scores.to(torch.float32)
            all_top_k_distances[query_mask] = group_topk_scores

    return all_nearest_neighbors, all_top_k_distances


def multi_arange(starts, ends):
    """
    merge multiple ranges into one tensor efficiently
    """
    sizes = ends - starts
    begin_idx = sizes.cumsum(0)
    ptr = torch.cat([torch.zeros(1, dtype=torch.int64, device=device), begin_idx])
    begin_idx = ptr[:-1]
    result = torch.arange(ptr[-1], device=device) - (begin_idx - starts).repeat_interleave(sizes)

    return result



def search_batched_chunked_matmul_mixed_precision_csr_newly_optimized(queries, cluster_centers, offset, data_points, data, k=10,
                                                             batch_size=20, candidate_chunk_size=100000):
    """
    使用CSR格式优化IVF索引的批量搜索，并行获取候选点，避免遍历查找。
    """
    device = queries.device
    nq, d = queries.shape
    num_clusters = cluster_centers.size(0)

    scaler = GradScaler()

    # Step 1: 计算主中心
    with autocast():
        inner_products = torch.matmul(queries, cluster_centers.T)
        primary_center_indices = torch.argmax(inner_products, dim=1)

    # Step 2: 计算残差
    primary_centers = cluster_centers[primary_center_indices]
    with autocast():
        residuals = compute_normalized_residual(queries, primary_centers)

    # Step 3: 计算次中心
    with autocast():
        secondary_center_indices = compute_query_secondary_assignments_optimized(
            queries, cluster_centers, primary_center_indices, residuals, lambda_param=1.0
        )

    # Step 4: 使用 1D 索引优化 unique 操作
    combined_keys = primary_center_indices * num_clusters + secondary_center_indices
    unique_keys, inverse_indices = torch.unique(combined_keys, return_inverse=True)
    unique_primary = unique_keys // num_clusters
    unique_secondary = unique_keys % num_clusters
    unique_combos = torch.stack([unique_primary, unique_secondary], dim=1)

    # 初始化输出张量
    all_nearest_neighbors = torch.full((nq, k), -1, dtype=torch.long, device=device)
    all_top_k_distances = torch.full((nq, k), float('-inf'), device=device)

    num_batches = (unique_combos.size(0) + batch_size - 1) // batch_size

    # 批量处理各 (primary, secondary) 组合
    for batch_idx in range(num_batches):
        batch_combos = unique_combos[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        batch_size_current = batch_combos.size(0)  # 实际当前批次的组合数量
        # 批量获取候选点（核心优化部分）
        batch_candidates = []

        query_clusters = torch.stack([batch_combos[:, 0], batch_combos[:, 1]], dim=1).flatten()
        starts = offset[query_clusters]
        ends = offset[query_clusters + 1]
        all_indices = multi_arange(starts, ends)

        split_sizes = (ends - starts).tolist()  # 形状为 [batch_size*2]
        split_indices = torch.split(all_indices, split_sizes)

        for i in range(batch_size_current):
            p_indices = split_indices[2 * i]
            s_indices = split_indices[2 * i + 1]
            merged = torch.cat([p_indices, s_indices])  # 合并索引
            candidates = data_points[merged]  # 获取候选数据
            sorted_candidates, _ = torch.sort(candidates)  # 全局排序
            unique_candidates = torch.unique_consecutive(sorted_candidates)  # 仅去除相邻重复项

            batch_candidates.append(unique_candidates)

        # **并行处理查询**
        for i, (c1, c2) in enumerate(batch_combos):
            query_mask = (inverse_indices == (batch_idx * batch_size + i))
            group_queries = queries[query_mask]
            unique_candidates = batch_candidates[i]  # 直接获取当前组合的候选集
            candidate_vectors = data[unique_candidates]  # shape: (nc, d)
            nc = unique_candidates.size(0)

            partial_scores = []
            partial_indices = []
            for start in range(0, nc, candidate_chunk_size):
                end = min(start + candidate_chunk_size, nc)
                candidate_chunk = candidate_vectors[start:end]  # shape: (chunk_size, d)
                chunk_size = candidate_chunk.shape[0]
                with autocast():
                    scores_chunk = torch.matmul(group_queries, candidate_chunk.T)  # shape: (num_queries, chunk_size)
                if chunk_size < k:
                    pad_size = k - chunk_size
                    pad_scores = torch.full((scores_chunk.shape[0], pad_size), float('-inf'),
                                            device=scores_chunk.device)
                    scores_chunk = torch.cat([scores_chunk, pad_scores], dim=1)

                chunk_scores, chunk_indices = torch.topk(scores_chunk, k, dim=1)
                partial_scores.append(chunk_scores)
                partial_indices.append(chunk_indices + start)  # 调整为候选集中的索引

            all_partial_scores = torch.cat(partial_scores, dim=1)  # shape: (num_queries, total_candidates_in_chunks)
            all_partial_indices = torch.cat(partial_indices, dim=1)

            group_topk_scores, topk_order = torch.topk(all_partial_scores, k, dim=1)
            group_topk_local_indices = torch.gather(all_partial_indices, 1, topk_order)
            if nc < k:
                pad_size = k - nc
                pad_scores = torch.full((group_queries.shape[0], pad_size), float('-inf'), device=device)
                pad_indices = torch.full((group_queries.shape[0], pad_size), -1, dtype=torch.long, device=device)
                group_topk_scores = torch.cat([group_topk_scores, pad_scores], dim=1)
                group_topk_local_indices = torch.cat([group_topk_local_indices, pad_indices], dim=1)

            global_indices = unique_candidates[group_topk_local_indices]
            all_nearest_neighbors[query_mask] = global_indices
            group_topk_scores = group_topk_scores.to(torch.float32)
            all_top_k_distances[query_mask] = group_topk_scores

    return all_nearest_neighbors, all_top_k_distances




def compute_recall_torch(neighbors, true_neighbors):
    # Ensure input tensors have the correct dimensions
    if neighbors.ndim == 1:
        neighbors = neighbors.unsqueeze(0)
    if true_neighbors.ndim == 1:
        true_neighbors = true_neighbors.unsqueeze(0)

    # Initialize a list to store recall values for each row
    recalls = []

    # Calculate recall for each row
    for gt_row, row in zip(true_neighbors, neighbors):
        # Use broadcasting to find intersections
        intersection = (gt_row.unsqueeze(1) == row.unsqueeze(0)).any(dim=1)
        recall = intersection.sum().item() / gt_row.size(0)  # Recall for this row
        recalls.append(recall)
    average_recall = sum(recalls) / len(recalls)
    # Convert the list of recalls to a tensor
    return average_recall, torch.tensor(recalls)


def compute_recall_tensor(neighbors, true_neighbors):
    """
    Computes recall as the fraction of true neighbors found in the nearest neighbors.
    Arguments:
        neighbors (torch.Tensor): Tensor of nearest neighbor indices, shape (k,)
        true_neighbors (torch.Tensor): Tensor of true neighbor indices, shape (m,)
    Returns:
        float: Recall value
    """
    # Step 1: Remove duplicates from neighbors
    unique_neighbors = torch.unique(neighbors)  # Ensure no duplicate neighbors

    # Step 2: Find common elements between neighbors and true_neighbors
    common_elements = torch.isin(unique_neighbors, true_neighbors)  # Boolean mask
    num_common_elements = common_elements.sum().item()  # Count of common elements

    # Step 2: Compute recall
    return num_common_elements / true_neighbors.size(0)


download_path = "glove-100-angular.hdf5"
# Open the downloaded file with h5py
glove_h5py = h5py.File(download_path, "r")

dataset = glove_h5py['train']
queries = glove_h5py['test']
true_neighbors = glove_h5py['neighbors']

dataset10w = torch.tensor(dataset[:]).to(device)
queries = torch.tensor(queries[:]).to(device)
true_neighbors = torch.tensor(true_neighbors[:]).to(device)
dataset_norms = torch.norm(dataset10w, p=2, dim=1, keepdim=True)  # Compute L2 norm of each data point
dataset_normalized = dataset10w / dataset_norms  # Normalize each data point

start = time.time()
cluster_centers, primary_assignments, primary_inverted_index = (
    kmeans_anisotropic(dataset_normalized, num_clusters=2000, max_iters=200))
end = time.time()
logging.info(f"Kmeans anisotropic time: {end - start}")

primary_centroids = cluster_centers[primary_assignments]

start = time.time()
# tracer.start()
normalized_residuals = compute_normalized_residual(dataset_normalized, primary_centroids)
# tracer.save('compute_normalized_residuals.json')
end = time.time()
logging.info("Compute normalized residuals completed !! Elapsed time: {} s".format(end - start))
#
start = time.time()
# tracer.start()
secondary_assignments = compute_secondary_assignments_optimized(dataset_normalized, cluster_centers,
                                                                primary_assignments, normalized_residuals,
                                                                primary_inverted_index, lambda_param=1.0)
# tracer.save('compute_secondary_assignments.json')
end = time.time()
logging.info("Secondary assignments completed !! Elapsed time: {} s".format(end - start))

queries = queries[:]
true_neighbors1 = true_neighbors[:, :10]

# 按 cluster ID 排序，保证索引顺序
sorted_keys = sorted(primary_inverted_index.keys())

# 构造 data_points 和 offset
data_points_list = []
offset_list = [0]

for cluster_id in sorted_keys:
    points = primary_inverted_index[cluster_id]
    data_points_list.append(points)
    offset_list.append(offset_list[-1] + len(points))

# 拼接成 GPU Tensor
data_points = torch.cat(data_points_list).to(device)
offset = torch.tensor(offset_list, dtype=torch.int64, device=device)



"""use this to compare precision"""
timer = Timer()
timer.start()
all_nearest_neighbors1, all_top_k_distances = (
    search_batched_chunked_matmul_mixed_precision(queries, cluster_centers, primary_inverted_index, dataset_normalized,
                                      k=10, batch_size=20))
tm = timer.end()
logging.info("search_batched_chunked_matmul_mixed_precision batch_size=20 completed !! Elapsed time: {} s".format(tm))
average_recall, recalls = compute_recall_torch(all_nearest_neighbors1, true_neighbors1)
print(f"Average chunked_matmul Recall: {average_recall}")
print(recalls)

"""trial FP16:"""
timer = Timer()
timer.start()
all_nearest_neighbors1, all_top_k_distances = (
    search_batched_chunked_matmul_mixed_precision_2unique(queries, cluster_centers, primary_inverted_index,
                                                          dataset_normalized, k=10, batch_size=20))

tm = timer.end()
logging.info("search_batched_chunked_matmul_mixed_precision_2unique completed !! Elapsed time: {} s".format(tm))
average_recall, recalls = compute_recall_torch(all_nearest_neighbors1, true_neighbors1)

print(f"Average Recall1:  {average_recall}")
print(recalls)


timer = Timer()
timer.start()
all_nearest_neighbors1, all_top_k_distances = (
    search_batched_chunked_matmul_mixed_precision_csr_newly_optimized(queries, cluster_centers, offset, data_points,
                                                          dataset_normalized, k=10, batch_size=20))

tm = timer.end()
logging.info("search_batched_chunked_matmul_mixed_precision_csr_newly_optimized  batch_size=20 completed !! Elapsed time: {} s".format(tm))
average_recall, recalls = compute_recall_torch(all_nearest_neighbors1, true_neighbors1)

print(f"Average Recall1:  {average_recall}")
print(recalls)

timer = Timer()
timer.start()
all_nearest_neighbors1, all_top_k_distances = (
    search_batched_chunked_matmul_mixed_precision_csr_newly_optimized(queries, cluster_centers, offset, data_points,
                                                          dataset_normalized, k=10, batch_size=50))

tm = timer.end()
logging.info("search_batched_chunked_matmul_mixed_precision_csr_newly_optimized completed batch_size=50 !! Elapsed time: {} s".format(tm))
average_recall, recalls = compute_recall_torch(all_nearest_neighbors1, true_neighbors1)

print(f"Average Recall1:  {average_recall}")
print(recalls)

timer = Timer()
timer.start()
all_nearest_neighbors1, all_top_k_distances = (
    search_batched_chunked_matmul_mixed_precision_csr_newly_optimized(queries, cluster_centers, offset, data_points,
                                                          dataset_normalized, k=10, batch_size=100))

tm = timer.end()
logging.info("search_batched_chunked_matmul_mixed_precision_csr_newly_optimized completed batch_size=100 !! Elapsed time: {} s".format(tm))
average_recall, recalls = compute_recall_torch(all_nearest_neighbors1, true_neighbors1)

print(f"Average Recall1:  {average_recall}")
print(recalls)




