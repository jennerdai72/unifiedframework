import torch
import numpy as np

def compute_frobenius_norm(matrix1, matrix2):
    """
    Computes the Frobenius norm between two matrices.
    
    Args:
        matrix1 (torch.Tensor): First input matrix.
        matrix2 (torch.Tensor): Second input matrix.
        
    Returns:
        float: Frobenius norm of the difference between the matrices.
    """
    difference = matrix1 - matrix2
    frobenius_norm = torch.norm(difference, 'fro').item()
    return frobenius_norm

def compute_cosine_similarity(matrix1, matrix2):
    """
    Computes the cosine similarity between two matrices.
    
    Args:
        matrix1 (torch.Tensor): First input matrix.
        matrix2 (torch.Tensor): Second input matrix.
        
    Returns:
        float: Cosine similarity between the matrices.
    """
    matrix1_flat = matrix1.view(-1)
    matrix2_flat = matrix2.view(-1)
    cosine_similarity = torch.nn.functional.cosine_similarity(matrix1_flat, matrix2_flat, dim=0).item()
    return cosine_similarity

def compare_matrices(sourceLayerMatrix, targetLayerMatrix):
    """
    Compares the similarity of two matrices using different similarity measures.
    
    Args:
        sourceLayerMatrix (torch.Tensor): Source layer distance matrix.
        targetLayerMatrix (torch.Tensor): Target layer distance matrix.
        
    Returns:
        dict: Dictionary containing similarity measures.
    """
    frobenius_norm = compute_frobenius_norm(sourceLayerMatrix, targetLayerMatrix)
    cosine_similarity = compute_cosine_similarity(sourceLayerMatrix, targetLayerMatrix)
    
    return {
        'frobenius_norm': frobenius_norm,
        'cosine_similarity': cosine_similarity
    }

# Example usage
sourceLayerMatrix = torch.randn(10, 10)
targetLayerMatrix = torch.randn(10, 10)

similarity_measures = compare_matrices(sourceLayerMatrix, targetLayerMatrix)
print(f"Frobenius Norm: {similarity_measures['frobenius_norm']}")
print(f"Cosine Similarity: {similarity_measures['cosine_similarity']}")
