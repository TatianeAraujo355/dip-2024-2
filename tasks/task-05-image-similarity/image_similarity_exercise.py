# image_similarity_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `compare_images(i1, i2)` that receives two grayscale images
represented as NumPy arrays (2D arrays of shape (H, W)) and returns a dictionary with the following metrics:

1. Mean Squared Error (MSE)
2. Peak Signal-to-Noise Ratio (PSNR)
3. Structural Similarity Index (SSIM) - simplified version without using external libraries
4. Normalized Pearson Correlation Coefficient (NPCC)

You must implement these functions yourself using only NumPy (no OpenCV, skimage, etc).

Each function should be implemented as a helper function and called inside `compare_images(i1, i2)`.

Function signature:
    def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:

The return value should be like:
{
    "mse": float,
    "psnr": float,
    "ssim": float,
    "npcc": float
}

Assume that i1 and i2 are normalized grayscale images (values between 0 and 1).
"""

import numpy as np

def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:
    # Your implementation here
    def mse(i1: np.ndarray, i2: np.ndarray) -> float:
        return np.mean((i1 - i2) ** 2)

    def psnr(i1: np.ndarray, i2: np.ndarray) -> float:
        mse_value = mse(i1, i2)
        if mse_value == 0:
            return float('inf')  # Se as imagens forem idênticas, PSNR é infinito
        return 10 * np.log10(1.0 / mse_value)

    def ssim(i1: np.ndarray, i2: np.ndarray) -> float:
        mean1, mean2 = np.mean(i1), np.mean(i2)
        var1, var2 = np.var(i1), np.var(i2)
        covar = np.mean((i1 - mean1) * (i2 - mean2))
        
        c1, c2 = 0.01 ** 2, 0.03 ** 2  # Pequenas constantes para estabilidade
        num = (2 * mean1 * mean2 + c1) * (2 * covar + c2)
        den = (mean1 ** 2 + mean2 ** 2 + c1) * (var1 + var2 + c2)
        
        return num / den

    def npcc(i1: np.ndarray, i2: np.ndarray) -> float:
        mean1, mean2 = np.mean(i1), np.mean(i2)
        num = np.sum((i1 - mean1) * (i2 - mean2))
        den = np.sqrt(np.sum((i1 - mean1) ** 2) * np.sum((i2 - mean2) ** 2))
        return num / den if den != 0 else 0

    return {
        "mse": mse(i1, i2),
        "psnr": psnr(i1, i2),
        "ssim": ssim(i1, i2),
        "npcc": npcc(i1, i2)
    }
    