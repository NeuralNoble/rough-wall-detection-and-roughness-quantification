import cv2
import os
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from scipy.stats import skew


# def fractal_dimension(image, threshold=128, box_sizes=None):
#     """Calculate fractal dimension using box-counting method"""
#     if box_sizes is None:
#         box_sizes = [2 ** i for i in range(1, 7)]  # 2 to 64 pixel box sizes
#
#     # Convert to binary image
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
#
#     counts = []
#     for size in box_sizes:
#         # Grid dimensions
#         h, w = binary.shape
#         grid_h = h // size
#         grid_w = w // size
#
#         # Count non-empty boxes
#         count = 0
#         for i in range(grid_h):
#             for j in range(grid_w):
#                 box = binary[i * size:(i + 1) * size, j * size:(j + 1) * size]
#                 if np.any(box < 255):  # Contains any dark pixels
#                     count += 1
#         counts.append(count)
#
#     # Linear fit in log-log space
#     coeffs = np.polyfit(np.log(box_sizes), np.log(counts), 1)
#     return -coeffs[0]  # Fractal dimension is negative slope
def fractal_dimension(image, threshold=128, box_sizes=None):
    """Calculate fractal dimension using box-counting method"""
    if box_sizes is None:
        box_sizes = [2 ** i for i in range(1, 7)]  # 2 to 64 pixel box sizes

    # Convert to binary image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    counts = []
    valid_box_sizes = []
    for size in box_sizes:
        # Grid dimensions
        h, w = binary.shape
        grid_h = h // size
        grid_w = w // size

        # Skip if grid would be empty
        if grid_h == 0 or grid_w == 0:
            continue

        # Count non-empty boxes
        count = 0
        for i in range(grid_h):
            for j in range(grid_w):
                box = binary[i * size:(i + 1) * size, j * size:(j + 1) * size]
                if np.any(box < 255):  # Contains any dark pixels
                    count += 1

        # Only add non-zero counts
        if count > 0:
            counts.append(count)
            valid_box_sizes.append(size)

    # Ensure we have enough points for fitting
    if len(counts) < 2:
        return 0  # Default value if not enough data

    # Linear fit in log-log space
    coeffs = np.polyfit(np.log(valid_box_sizes), np.log(counts), 1)
    return -coeffs[0]  # Fractal dimension is negative slope


def extract_features(image_path):
    """Extract comprehensive set of roughness features"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            return None

        # Convert to grayscale for texture analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 1. Edge Features
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        # 2. Contour Features
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_count = len(contours)
        contour_areas = [cv2.contourArea(c) for c in contours]
        avg_contour_area = np.mean(contour_areas) if contour_count > 0 else 0

        # 3. Texture Features
        glcm = graycomatrix(gray,
                            distances=[1, 3],
                            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                            levels=256,
                            symmetric=True,
                            normed=True)

        contrast = graycoprops(glcm, 'contrast').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()

        # 4. LBP Features
        lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
        lbp_hist, _ = np.histogram(lbp, bins=59, range=(0, 58))
        lbp_uniformity = lbp_hist.max() / lbp_hist.sum()

        # 5. Gradient Features
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        gradient_skew = skew(gradient_magnitude.flatten())

        # 6. Fractal Dimension
        fractal_dim = fractal_dimension(image)

        return [
            edge_density,
            fractal_dim,
            contour_count,
            avg_contour_area,
            contrast,
            homogeneity,
            lbp_uniformity,
            gradient_skew
        ]

    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None


def prepare_data(rough_folder, smooth_folder):
    """Create dataset from image folders"""
    features = []
    labels = []

    # Process rough images
    print("Processing rough images...")
    for img_name in os.listdir(rough_folder):
        img_path = os.path.join(rough_folder, img_name)
        feat = extract_features(img_path)
        if feat is not None:
            features.append(feat)
            labels.append(1)  # 1 for rough

    # Process smooth images
    print("Processing smooth images...")
    for img_name in os.listdir(smooth_folder):
        img_path = os.path.join(smooth_folder, img_name)
        feat = extract_features(img_path)
        if feat is not None:
            features.append(feat)
            labels.append(0)  # 0 for smooth

    # Create DataFrame
    columns = [
        'Edge Density',
        'Fractal Dimension',
        'Contour Count',
        'Avg Contour Area',
        'GLCM Contrast',
        'GLCM Homogeneity',
        'LBP Uniformity',
        'Gradient Skewness'
    ]

    df = pd.DataFrame(features, columns=columns)
    df['Label'] = labels

    # Remove rows with missing values
    df.dropna(inplace=True)

    return df


# Example usage
if __name__ == "__main__":
    dataset = prepare_data('data_svr/rough', 'data_svr/smooth')
    dataset.to_csv('wall_roughness_dataset.csv', index=False)
    print("\nDataset created with shape:", dataset.shape)