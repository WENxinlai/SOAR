import numpy as np
import h5py
import os
import torch
import time
from torch.profiler import profile, ProfilerActivity, record_function, tensorboard_trace_handler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def update_cluster_centers(data, assignments, num_clusters, T):
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

    # No need to reinitialize cluster_centers, use the current ones
    cluster_centers = torch.zeros(num_clusters, d, device=data.device)

    # Compute eta for each data point
    eta_tensor = compute_eta(data, T)  # Shape: (n, 1)
    eta = eta_tensor[0]  # Now eta is a scalar value
    # Loop through each cluster
    for k in range(num_clusters):
        # Get the indices of the points assigned to cluster k
        cluster_indices = (assignments == k)

        # Extract the data points assigned to cluster k
        cluster_data = data[cluster_indices]  # Shape: (num_points_in_cluster, d)

        if cluster_data.size(0) == 0:
            continue  # Skip empty clusters

        # Compute the norms for the data points in this cluster
        norms = torch.norm(cluster_data, p=2, dim=1)  # Shape: (num_points_in_cluster,)
        """下两句要再check"""
        norms_pow = norms ** (0.5 * (eta - 3))  # Shape: (num_points_in_cluster,)
        # Step 3: Apply fillzero for very small norms (less than threshold)
        fillzero = 1 if eta == 1 else 0  # Determine fillzero based on eta value
        norms_pow = torch.where(norms < 1e-20, torch.tensor(fillzero, device=data.device), norms_pow)
        # Handle very small norms

        X = cluster_data * norms_pow.view(-1, 1)  # Shape: (num_points_in_cluster, d)
        # Compute the matrix to invert
        xtx = torch.mm(X.T, X)  # X^T X (shape: d, d)

        # Calculate the weighted norms (norm^(eta - 1))
        norms_eta1 = norms ** (eta - 1)  # Shape: (num_points_in_cluster,)

        norm_weighted_dp_sum = (cluster_data * norms_eta1.view(-1, 1)).sum(dim=0)  # Shape: (d,)

        # Calculate dp_norm_sum
        dp_norm_sum = norms_eta1.sum()  # Sum of norm^(eta - 1) for this cluster, scalar
        """dp_norm_sum is right"""

        """以下是ok�?"""
        to_invert = dp_norm_sum * torch.eye(d, device=data.device) + (eta - 1) * xtx  #这句ok

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

    for i in range(max_iters):
        losses = compute_anisotropic_loss_batch(data, cluster_centers, T)
        new_assignments = losses.argmin(dim=1)

        if torch.equal(new_assignments, assignments):  #用assignment判断是否收敛而不是新旧中心的距离
            print(f"Converged in {i + 1} iterations")
            break
        assignments = new_assignments
        cluster_centers = update_cluster_centers(data, assignments, num_clusters, T)

    inverted_index = {j: (assignments == j).nonzero(as_tuple=True)[0] for j in range(num_clusters)}
    return cluster_centers, assignments, inverted_index


download_path = "glove-100-angular.hdf5"
# Open the downloaded file with h5py
glove_h5py = h5py.File(download_path, "r")

dataset = glove_h5py['train']
queries = glove_h5py['test']
# Press the green button in the gutter to run the script.
true_neighbors = glove_h5py['neighbors']

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

# �? h5py 数据集转换为 Tensor
dataset10w = torch.tensor(dataset[:100000]).to(device)
dataset_norms = torch.norm(dataset10w, p=2, dim=1, keepdim=True)  # Compute L2 norm of each data point
dataset_normalized = dataset10w / dataset_norms  # Normalize each data point

cluster_centers, primary_assignments, primary_inverted_index = (
    kmeans_anisotropic(dataset_normalized, num_clusters=200, max_iters=200)) # here use  dataset_normalized!! instead of dataset10w 

