# image_geometry_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `apply_geometric_transformations(img)` that receives a grayscale image
represented as a NumPy array (2D array) and returns a dictionary with the following transformations:

1. Translated image (shift right and down)
2. Rotated image (90 degrees clockwise)
3. Horizontally stretched image (scale width by 1.5)
4. Horizontally mirrored image (flip along vertical axis)
5. Barrel distorted image (simple distortion using a radial function)

You must use only NumPy to implement these transformations. Do NOT use OpenCV, PIL, skimage or similar libraries.

Function signature:
    def apply_geometric_transformations(img: np.ndarray) -> dict:

The return value should be like:
{
    "translated": np.ndarray,
    "rotated": np.ndarray,
    "stretched": np.ndarray,
    "mirrored": np.ndarray,
    "distorted": np.ndarray
}
"""

import numpy as np

def apply_geometric_transformations(img: np.ndarray) -> dict:
    # Your implementation here
    def translate(img: np.ndarray, shift_x: int, shift_y: int) -> np.ndarray:
        """Desloca a imagem para a direita (x) e para baixo (y)."""
        translated = np.zeros_like(img)
        translated[shift_y:, shift_x:] = img[:-shift_y, :-shift_x]
        return translated
    
    def rotate_90_clockwise(img: np.ndarray) -> np.ndarray:
        """Rotaciona 90 graus no sentido horário."""
        return np.flipud(np.transpose(img))
    
    def stretch_horizontal(img: np.ndarray, scale: float) -> np.ndarray:
        """Alongamento horizontal usando interpolação simples."""
        new_width = int(img.shape[1] * scale)
        stretched = np.zeros((img.shape[0], new_width))
        x_indices = np.linspace(0, img.shape[1] - 1, new_width).astype(int)
        stretched[:, :] = img[:, x_indices]
        return stretched
    
    def mirror_horizontal(img: np.ndarray) -> np.ndarray:
        """Espelha a imagem horizontalmente."""
        return np.fliplr(img)
    
    def barrel_distort(img: np.ndarray) -> np.ndarray:
        """Aplica uma distorção radial (barrel distortion) simples."""
        h, w = img.shape
        y, x = np.indices((h, w))
        cx, cy = w / 2, h / 2  # Centro da imagem
        r = np.sqrt((x - cx)**2 + (y - cy)**2) / max(cx, cy)
        factor = 1 + 0.2 * r**2  # Fator de distorção radial
        
        new_x = np.clip(((x - cx) * factor + cx).astype(int), 0, w - 1)
        new_y = np.clip(((y - cy) * factor + cy).astype(int), 0, h - 1)
        distorted = img[new_y, new_x]
        return distorted
    
    return {
        "translated": translate(img, 10, 10),
        "rotated": rotate_90_clockwise(img),
        "stretched": stretch_horizontal(img, 1.5),
        "mirrored": mirror_horizontal(img),
        "distorted": barrel_distort(img)
    }
    