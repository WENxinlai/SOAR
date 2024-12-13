import numpy as np
import h5py
import os
import torch
import time
from torch.profiler import profile, record_function, ProfilerActivity
import logging
from viztracer import VizTracer
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch_scatter import segment_csr
class Timer:
    def __init__(self):
        self.start_time = 0
        self.end_time = 0
    
    def start(self):
        torch.cuda.synchronize()
        self.start_time = time.time()

    def end(self):
        torch.cuda.synchronize()
        self.end_time = time.time()
        return self.end_time - self.start_time
    
# standard VQ
def kmeans(data, num_clusters, max_iters=100, tolerance=1e-4):
    # Step 1: Initialize cluster centers randomly from the dataset
    # center_idx = torch.randperm(data.size(0))[:num_clusters]
    center_idx = torch.randperm(data.size(0), device=data.device)[:num_clusters]
    cluster_centers = data[center_idx]
    # assignments = torch.zeros(data.size(0), dtype=torch.long)
    assignments = torch.zeros(data.size(0), dtype=torch.long, device=data.device)

    iters = 0
    while iters < max_iters:
        # Step 2: Assign each vector to the nearest cluster center using Euclidean
        distances = torch.cdist(data, cluster_centers, p=2)  # 计算欧几里得距离
        assignments = distances.argmin(dim=1)

        # Step 3: Update cluster centers
        """
        if (assignments == k).any()
        checks if there are any points assigned to cluster k
        This line handles Empty Clusters
        """
        new_centers = torch.stack([
            data[assignments == k].mean(dim=0) if (assignments == k).any() else cluster_centers[k]
            for k in range(num_clusters)])
        
        # print('new_center sum:', torch.sum(new_centers))
        # print('new_center:', new_centers[0])

        # New Step 3: Update cluster centers by tensor operators, avoid for-loop. e.g. 'for k in range(num_clusters)'
        new_centers = cluster_centers.clone()

        # logging.info('Assignment: ', assignments)
        aggsignmentsSorted = torch.sort(assignments)
        uniqueAssignments = torch.unique(aggsignmentsSorted.values)

        # the number of data points in each cluster represented in the format of accumulative sum
        dataPointNumInEachClusterAccumlate = torch.cat([torch.tensor([0], device=device), torch.searchsorted(aggsignmentsSorted.values, uniqueAssignments, right=True)])
        # the cluster is the cluster that has data points
        dataPointNumInEachCluster = torch.diff(dataPointNumInEachClusterAccumlate)

        dataPointsTobeAggregated = data[aggsignmentsSorted.indices]
        # print('dataPointsTobeAggregated:', dataPointsTobeAggregated[0])
        dataPointAggregated = segment_csr(dataPointsTobeAggregated, dataPointNumInEachClusterAccumlate, reduce="sum")
        # print('dataPointAggregated Sum:', dataPointAggregated[0])
        dataPointAggregated = dataPointAggregated / dataPointNumInEachCluster.unsqueeze(1)
        # print('dataPointAggregated:', dataPointAggregated[0])
        new_centers[uniqueAssignments] = dataPointAggregated
        # print('uniqueAssignments:', uniqueAssignments)
        # print('new_center1:', new_centers[0])
        # print('new_center1 sum:', torch.sum(new_centers))

        # new_centers 的形状是 (num_clusters, data_dim)
        # Step 4: Check for convergence by comparing new and old centers
        center_shift = torch.norm(new_centers - cluster_centers, dim=1).max()
        cluster_centers = new_centers
        if center_shift < tolerance:
            # logging.info(f"Converged in {iters + 1} iterations")
            break
        iters += 1
    logging.info(f"KMeans converged in {iters + 1} iterations")
    # Construct inverted index over π; for each partition i, store the indices of datapoints in that partition
    inverted_index = {k: (assignments == k).nonzero(as_tuple=True)[0].to(data.device) for k in range(num_clusters)}
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
                                  lambda_param=1.0, batch_size=100):
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
        candidate_centers = cluster_centers.unsqueeze(0).expand(batch_size_actual, -1,-1)
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
        #primary_norms_squared = torch.norm(primary_residuals, dim=2) ** 2  # Shape: (batch_size, num_clusters)

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

    # Step 9:  Add datapoint idx into inverted index for secondary assignments
    for center_id in range(num_clusters):
        # 找出 secondary_assignments 中属于当前 center_id 的数据点索引
        secondary_indices = (secondary_assignments == center_id).nonzero(as_tuple=True)[0]
        inverted_index[center_id] = torch.cat((inverted_index[center_id].to(data.device), secondary_indices))

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
    mask = torch.arange(num_clusters,device=queries.device).unsqueeze(0).expand(num_queries, -1) != primary_assignments.unsqueeze(1)

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
    candidate_indices = set()
    candidate_indices.update(data_inverted_index[primary_center_index.item()])
    candidate_indices.update(data_inverted_index[secondary_center_index.item()])

    candidate_indices = torch.tensor(list(candidate_indices), dtype=torch.long,device=data.device)  # Convert to list for indexing
    candidate_vectors = data[candidate_indices]  # Shape: (len(candidate_indices), d)

    # Step 5: Identify top-K nearest neighbors among candidates
    if candidate_vectors.size(0) < k:
        # 若候选向量个数小于 k，直接返回所有候选向量
        top_k_distances = torch.matmul(candidate_vectors, query)
        top_k_indices = torch.arange(candidate_vectors.size(0), device=query.device)  # 返回全部候选向量的索引
    else:
        candidate_inner_products = torch.matmul(candidate_vectors, query)
        top_k_distances, top_k_indices = candidate_inner_products.topk(k, largest=True)

    # Use tensor indexing to gather the nearest neighbor indices
    nearest_neighbors = candidate_indices[top_k_indices]

    return nearest_neighbors, top_k_distances


def compute_recall(neighbors, true_neighbors):
    nearest_neighbors_np = neighbors.numpy()
    # Step 2: Find the common elements
    common_elements = np.intersect1d(nearest_neighbors_np, true_neighbors)
    # Step 3: Count the number of common elements
    num_common_elements = len(common_elements)
    return num_common_elements / true_neighbors.size


def evaluate_precision(queries, true_neighbors, cluster_centers, primary_inverted_index, dataset10w, s=400000, k=100):
    recalls = []

    for i, query in enumerate(queries):
        # Step 1: 获取真实最近邻且索引小于 s 的元素
        filtered_neighbors = true_neighbors[i][true_neighbors[i] < s]
        count = filtered_neighbors.size  # 统计 filtered_neighbors 的数量

        # Step 2: 调用 search_single 获取近邻结果
        nearest_neighbors, top_k_distances = search_single(query, cluster_centers, primary_inverted_index, dataset10w,
                                                           k)

        # Step 3: 计算精度
        recall = compute_recall(nearest_neighbors, filtered_neighbors)

        # Step 4: 将精度存入 recalls 数组中
        recalls.append(recall)

    # 计算平均精度
    average_recall = np.mean(recalls)
    return average_recall, recalls


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



def evaluate_precision_tensor(queries, true_neighbors, cluster_centers, primary_inverted_index, dataset10w, s=400000,
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
        nearest_neighbors, top_k_distances = search_single(query, cluster_centers, primary_inverted_index, dataset10w,
                                                           k)

        # Step 3: Compute recall
        recall = compute_recall_tensor(nearest_neighbors, filtered_neighbors)
        recalls.append(recall)

    # Step 4: Compute average recall
    average_recall = sum(recalls) / len(recalls)
    return average_recall, recalls





if __name__ == '__main__':
    timer = Timer()
    timer.start()
    download_path = "glove-100-angular.hdf5"
    # download_path = "/data/coding/glove-100-angular.hdf5"
    # Open the downloaded file with h5py
    glove_h5py = h5py.File(download_path, "r")

    dataset = glove_h5py['train']
    queries = glove_h5py['test']
    # Press the green button in the gutter to run the script.
    true_neighbors = glove_h5py['neighbors']

    # See PyCharm help at https://www.jetbrains.com/help/pycharm/

    # 将 h5py 数据集转换为 Tensor


    dataset10w = torch.tensor(dataset[:100000]).to(device)
    queries = torch.tensor(queries[:]).to(device)
    true_neighbors = torch.tensor(true_neighbors[:]).to(device)
    tm = timer.end()
    logging.info("Data loaded successfully !! Elapsed time: {:.2f} s".format(tm))


    
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #             on_trace_ready=torch.profiler.tensorboard_trace_handler('./log1')) as prof:
    #     with record_function("KMeans Clustering"):
    #         timer.start()
    #         cluster_centers, primary_assignments, primary_inverted_index = kmeans(dataset10w, num_clusters=200,
    #                                                                             max_iters=200)
    #         tm = timer.end()
    #         logging.info("KMeans clustering completed !! Elapsed time: {} s".format(tm))

    #     with record_function("Compute Residuals"):
    #         timer.start()
    #         primary_centroids = cluster_centers[primary_assignments]
    #         normalized_residuals = compute_normalized_residual(dataset10w, primary_centroids)
    #         tm = timer.end()
    #         logging.info("Compute residuals completed !! Elapsed time: {} s".format(tm))


    #     with record_function("Secondary Assignments"):
    #         timer.start()
    #         secondary_assignments = compute_secondary_assignments(dataset10w, cluster_centers,
    #                                                             primary_assignments, normalized_residuals,
    #                                                             primary_inverted_index, lambda_param=1.0)
    #         tm = timer.end()
    #         logging.info("Secondary assignments completed !! Elapsed time: {} s".format(tm))
    
    tracer = VizTracer()
    timer.start()
    tracer.start()
    cluster_centers, primary_assignments, primary_inverted_index = kmeans(dataset10w, num_clusters=200, max_iters=200)
    tracer.save('kemans.json')
    tm = timer.end()
    logging.info("KMeans clustering completed !! Elapsed time: {} s".format(tm))

    timer.start()
    primary_centroids = cluster_centers[primary_assignments]
    tm = timer.end()
    logging.info("Compute residuals completed !! Elapsed time: {} s".format(tm))

    timer.start()
    tracer.start()
    normalized_residuals = compute_normalized_residual(dataset10w, primary_centroids)
    tracer.save('compute_normalized_residuals.json')
    tm = timer.end()
    logging.info("Compute normalized residuals completed !! Elapsed time: {} s".format(tm))

    timer.start()
    tracer.start()
    secondary_assignments = compute_secondary_assignments(dataset10w, cluster_centers,
                                                                primary_assignments, normalized_residuals,
                                                                primary_inverted_index, lambda_param=1.0)
    tracer.save('compute_secondary_assignments.json')
    tm = timer.end()
    logging.info("Secondary assignments completed !! Elapsed time: {} s".format(tm))

    timer.start()
    # search
    queries = queries[:100]
    true_neighbors1 = true_neighbors[:100]
    average_recall, recalls = evaluate_precision_tensor(queries, true_neighbors1, cluster_centers, primary_inverted_index,
                                                dataset10w, s=100000, k=100)
    tm = timer.end()
    logging.info("Search completed !! Elapsed time: {} s".format(tm))
    logging.info(f"Average Recall: {average_recall}, Recalls: {recalls}")
    # print(average_recall)
    # print(recalls)
    
# def compute_recall(neighbors, true_neighbors):
#     total = 0
#     for gt_row, row in zip(true_neighbors, neighbors):
#         total += np.intersect1d(gt_row, row).shape[0]
#     return total / true_neighbors.size