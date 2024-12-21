import numpy as np
import h5py
import os
import torch
import time

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

        """Todo:norms_eta1"""
        norms_eta1_k = norms_eta1[cluster_indices]  # Shape: (num_points_in_cluster,)

        norm_weighted_dp_sum = (cluster_data * norms_eta1_k.view(-1, 1)).sum(dim=0)  # Shape: (d,)

        to_invert = dp_norm_sum * torch.eye(d, device=data.device) + (eta - 1) * xtx  

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
        losses = compute_anisotropic_loss_batch(data, cluster_centers, T, eta_weight)
        new_assignments = losses.argmin(dim=1)

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

def compute_secondary_assignments(data, cluster_centers, primary_assignments, residuals, inverted_index,
                                  lambda_param=1.0, batch_size=10000):
    num_clusters = cluster_centers.size(0)
    num_data_points = data.size(0)
    data_dim = data.size(1)

    # Initialize secondary assignments tensor
    secondary_assignments = torch.zeros(num_data_points, dtype=torch.long, device=data.device)

    # Process data in batches to reduce memory usage
    for batch_start in range(0, num_data_points, batch_size):
        batch_end = min(batch_start + batch_size, num_data_points)
        batch_data = data[batch_start:batch_end]
        batch_primary_assignments = primary_assignments[batch_start:batch_end]
        batch_residuals = residuals[batch_start:batch_end]

        batch_size_actual = batch_data.size(0)

        # Step 1: Mask primary assignments for each data point in the batch
        mask = (torch.arange(num_clusters, device=data.device).unsqueeze(0).expand(batch_size_actual, -1)
                != batch_primary_assignments.unsqueeze(1))

        # Step 2: Compute residuals r' for all non-primary clusters
        candidate_centers = cluster_centers.unsqueeze(0).expand(batch_size_actual, -1, -1)
        # Shape: (batch_size, num_clusters, data_dim)
        batch_data_expanded = batch_data.unsqueeze(1).expand(-1, num_clusters,
                                                             -1)  # Shape: (batch_size, num_clusters, data_dim)
        candidate_residuals = (batch_data_expanded - candidate_centers) * mask.unsqueeze(
            -1)  # Shape: (batch_size, num_clusters, data_dim)

        # Step 3: Compute distance terms ||r'||^2 for each candidate residual
        dist_terms = torch.norm(candidate_residuals, dim=2) ** 2  # Shape: (batch_size, num_clusters)

        # Step 4: Compute penalty terms for alignment between primary and candidate residuals
        primary_residuals = batch_residuals.unsqueeze(1).expand(-1, num_clusters,
                                                                -1)  # Shape: (batch_size, num_clusters, data_dim)
        dot_products = torch.einsum('ijk,ijk->ij', primary_residuals,
                                    candidate_residuals)  # Shape: (batch_size, num_clusters)
        # primary_norms_squared = torch.norm(primary_residuals, dim=2) ** 2  # Shape: (batch_size, num_clusters)

        # Compute parallel penalty: λ * (dot_products^2 )
        parallel_penalty = lambda_param * (dot_products ** 2)  # Shape: (batch_size, num_clusters)

        # Step 5: Add distance and penalty terms to get the total penalized distance
        penalized_distances = dist_terms + parallel_penalty  # Shape: (batch_size, num_clusters)

        # Step 6: Set penalized distance to infinity for primary assignments
        # penalized_distances[~mask] = float('inf')
        penalized_distances[~mask] = torch.tensor(float('inf'), device=penalized_distances.device)

        # Step 7: Find the secondary assignments for the batch
        batch_secondary_assignments = penalized_distances.argmin(dim=1)  # Shape: (batch_size,)

        # Step 8: Store results back into secondary_assignments tensor
        secondary_assignments[batch_start:batch_end] = batch_secondary_assignments

    
    sorted_secondary, secondary_indices = torch.sort(secondary_assignments)

    # Step 2: to find the boundaries for each cluster.
    boundaries = torch.searchsorted(sorted_secondary,
                                    torch.arange(num_clusters + 1, device=data.device))

    # Step 3: get secondary_indices and merge to inverted_index
    for k in range(num_clusters):
        # get secondary_indices belongs to kth cluster 
        new_indices = secondary_indices[boundaries[k]:boundaries[k + 1]]

        inverted_index[k] = torch.cat((inverted_index[k].to(data.device), new_indices))
    return secondary_assignments


def compute_query_secondary_assignments(queries, cluster_centers, primary_assignments, residuals, lambda_param=1.0):
    """
  Compute secondary assignments for queries based on maximum inner product with cluster centers.

  queries: torch.Tensor of shape (n, d) or (d,), where n is the number of queries
  cluster_centers: torch.Tensor of shape (num_clusters, d)
  primary_assignments: torch.Tensor of shape (n,) with primary center indices for each query
  residuals: torch.Tensor of shape (n, d) or (d,) with normalized residuals of each query
  lambda_param: float, penalty parameter for alignment penalty

  Returns:
      torch.Tensor of shape (n,) with secondary center indices for each query
  """
    # Ensure queries are of shape (n, d)
    if queries.dim() == 1:
        queries = queries.unsqueeze(0)
        residuals = residuals.unsqueeze(0)
        primary_assignments = primary_assignments.unsqueeze(0)

    num_queries = queries.size(0)
    num_clusters = cluster_centers.size(0)

    # Step 1: Mask primary assignments for each query
    mask = torch.arange(num_clusters, device=queries.device).unsqueeze(0).expand(num_queries,
                                                                                 -1) != primary_assignments.unsqueeze(1)

    # Step 2: Compute residuals r' for all non-primary clusters
    candidate_centers = cluster_centers.unsqueeze(0).expand(num_queries, -1,
                                                            -1)  # Shape: (num_queries, num_clusters, d)
    queries_expanded = queries.unsqueeze(1).expand(-1, num_clusters, -1)  # Shape: (num_queries, num_clusters, d)
    candidate_residuals = (queries_expanded - candidate_centers) * mask.unsqueeze(
        -1)  # Shape: (num_queries, num_clusters, d)

    # Step 3: Compute distance terms ||r'||^2 for each candidate residual
    dist_terms = torch.norm(candidate_residuals, dim=2) ** 2  # Shape: (num_queries, num_clusters)

    # Step 4: Compute penalty terms for alignment between primary and candidate residuals
    primary_residuals = residuals.unsqueeze(1).expand(-1, num_clusters, -1)  # Shape: (num_queries, num_clusters, d)
    dot_products = torch.einsum('ijk,ijk->ij', primary_residuals,
                                candidate_residuals)  # Shape: (num_queries, num_clusters)

    # Compute parallel penalty: λ * (dot_products^2)
    parallel_penalty = lambda_param * (dot_products ** 2)  # Shape: (num_queries, num_clusters)

    # Step 5: Add distance and penalty terms to get the total penalized distance
    penalized_distances = dist_terms + parallel_penalty  # Shape: (num_queries, num_clusters)

    # Step 6: Set penalized distance to infinity for primary assignments
    penalized_distances[~mask] = torch.tensor(float('inf'), device=penalized_distances.device)

    # Step 7: Find the secondary assignments for each query
    secondary_assignments = penalized_distances.argmin(dim=1)  # Shape: (num_queries,)

    return secondary_assignments


def search_single(query, cluster_centers, data_inverted_index, data, k=100):
    """
  Perform a search with MIPS by selecting top clusters and then finding the nearest neighbors in those clusters.

  query: torch.Tensor of shape (d,)
  cluster_centers: torch.Tensor of shape (c, d)
  data_inverted_index: dict mapping center indices to data indices
  data: torch.Tensor containing all data points
  k: int, number of nearest neighbors to retrieve
  """

    # Step 1: Find primary center assignment
    inner_products = torch.matmul(cluster_centers, query)
    primary_center_index = torch.argmax(inner_products)  # Convert to scalar
    primary_center = cluster_centers[primary_center_index]

    # Step 2: Compute residual vector r
    residuals = compute_normalized_residual(query, primary_center)

    # Step 3: Compute secondary assignment for query
    secondary_center_index = compute_query_secondary_assignments(
        query, cluster_centers, primary_center_index, residuals, lambda_param=1.0
    )
    
    # Step 4: Collect candidate indices from both primary and secondary centers
    primary_candidate_indices = data_inverted_index[primary_center_index.item()]  # Tensor
    secondary_candidate_indices = data_inverted_index[secondary_center_index.item()]  # Tensor

    # Step 5: merge candidate indices from both primary and secondary centers
    candidate_indices = torch.cat((primary_candidate_indices, secondary_candidate_indices))

    # Step 6: remove reduplicate indices
    candidate_indices = torch.unique(candidate_indices)

    # Step 7: get candidate_vectors using indices
    candidate_vectors = data[candidate_indices]  # Shape: (len(candidate_indices), d)
    # Step 8: Identify top-K nearest neighbors among candidates
    if candidate_vectors.size(0) < k:
        # if num of candidate_vectors < k，return all candidate_vectors
        top_k_distances = torch.matmul(candidate_vectors, query)
        top_k_indices = torch.arange(candidate_vectors.size(0), device=query.device)  # 返回全部候选向量的索引
    else:
        candidate_inner_products = torch.matmul(candidate_vectors, query)
        top_k_distances, top_k_indices = candidate_inner_products.topk(k, largest=True)

    # Use tensor indexing to gather the nearest neighbor indices
    nearest_neighbors = candidate_indices[top_k_indices]

    return nearest_neighbors, top_k_distances

# def search_batched(queries, cluster_centers, data_inverted_index, data, k=100):
#     """
#     Perform batched search with MIPS for multiple query points.
#
#     Args:
#         queries: torch.Tensor of shape (nq, d), where nq is the number of queries.
#         cluster_centers: torch.Tensor of shape (c, d), cluster center vectors.
#         data_inverted_index: dict mapping cluster indices to data indices.
#         data: torch.Tensor of shape (n, d), dataset of all data points.
#         k: int, number of nearest neighbors to retrieve for each query.
#
#     Returns:
#         torch.Tensor: Nearest neighbor indices of shape (nq, k).
#     """
#     nq, d = queries.shape
#     c = cluster_centers.shape[0]
#
#     # Step 1: Compute inner products and find primary centers
#     inner_products = torch.matmul(queries, cluster_centers.T)  # Shape: (nq, c)
#     primary_center_indices = torch.argmax(inner_products, dim=1)  # Shape: (nq,)
#
#     # Step 2: Compute residuals
#     primary_centers = cluster_centers[primary_center_indices]  # Shape: (nq, d)
#     residuals = compute_normalized_residual(queries, primary_centers)  # Shape: (nq, d)
#
#     # Step 3: Compute secondary center assignments
#     secondary_center_indices = compute_query_secondary_assignments(
#         queries, cluster_centers, primary_center_indices, residuals, lambda_param=1.0
#     )  # Shape: (nq,)
#
#     # Step 4: Collect candidate indices for each query
#     candidate_indices = collect_candidate_indices_batched(
#         primary_center_indices, secondary_center_indices, data_inverted_index
#     )  # Shape: (nq, ?), dynamic candidate size per query
#
#     # Step 5: Compute top-K nearest neighbors
#     nearest_neighbors, top_k_distances = compute_topk_candidates_batched(
#         queries, candidate_indices, data, k
#     )  # Shape: (nq, k),
#
#     return nearest_neighbors, top_k_distances

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


def evaluate_precision_tensor(queries, true_neighbors, cluster_centers, primary_inverted_index, dataset_normalized, s=400000,
                              k=100):
    """
    Evaluates recall (precision) for a set of queries using GPU tensors.
    Arguments:
        queries (torch.Tensor): Query vectors, shape (n, d)
        true_neighbors (torch.Tensor): True neighbor indices, shape (n, m)
        cluster_centers (torch.Tensor): Cluster center vectors, shape (c, d)
        primary_inverted_index (dict): Mapping from cluster indices to data indices
        dataset10w (torch.Tensor): Dataset of all data points, shape (num_points, d)
        s (int): Maximum index to filter true neighbors
        k (int): Number of nearest neighbors to retrieve
    Returns:
        float: Average recall over all queries
        list[float]: Recall for each query
    """
    recalls = []

    for i, query in enumerate(queries):
        # Step 1: Filter true neighbors with index < s
        filtered_neighbors = true_neighbors[i][true_neighbors[i] < s]

        # If no valid neighbors, skip this query
        if filtered_neighbors.size(0) == 0:
            recalls.append(0.0)
            continue

        # Step 2: Call search_single to get nearest neighbors
        nearest_neighbors, top_k_distances = search_single(query, cluster_centers, primary_inverted_index, dataset_normalized,
                                                           k)

        # Step 3: Compute recall
        recall = compute_recall_tensor(nearest_neighbors, filtered_neighbors)
        recalls.append(recall)

    # Step 4: Compute average recall
    average_recall = sum(recalls) / len(recalls)
    return average_recall, recalls


download_path = "glove-100-angular.hdf5"
# Open the downloaded file with h5py
glove_h5py = h5py.File(download_path, "r")

dataset = glove_h5py['train']
queries = glove_h5py['test']
# Press the green button in the gutter to run the script.
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
print(f"Kmeans anisotropic time: {end - start}")

primary_centroids = cluster_centers[primary_assignments]

start = time.time()
# tracer.start()
normalized_residuals = compute_normalized_residual(dataset_normalized, primary_centroids)
# tracer.save('compute_normalized_residuals.json')
end = time.time()
print("Compute normalized residuals completed !! Elapsed time: {} s".format(end - start))
#
start = time.time()
# tracer.start()
secondary_assignments = compute_secondary_assignments(dataset_normalized, cluster_centers,
                                                      primary_assignments, normalized_residuals,
                                                      primary_inverted_index, lambda_param=1.0)
# tracer.save('compute_secondary_assignments.json')
end = time.time()
print("Secondary assignments completed !! Elapsed time: {} s".format(end - start))

queries = queries[:100]
true_neighbors1 = true_neighbors[:100]
average_recall, recalls = evaluate_precision_tensor(queries, true_neighbors1, cluster_centers,
                                                    primary_inverted_index,
                                                    dataset_normalized, s=1200000, k=100)

print(f"Average Recall: {average_recall}")
print(recalls)
