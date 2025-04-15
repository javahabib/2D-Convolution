import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import cv2
from skimage import data

def manual_convolution_2d(input_image, kernel=None, kernel_size=3, stride=1, padding='same', mode='convolution'):


    image_path = "/content/greyscale.jpg"
    input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    input_image = np.array(input_image, dtype=np.float32)
    if kernel is None:
        np.random.seed(42)  # For reproducibility
        kernel = np.random.randn(kernel_size, kernel_size)
        kernel = kernel / np.sum(np.abs(kernel))
    elif isinstance(kernel, list):
        kernel = np.array(kernel, dtype=np.float32)

    # Get dimensions
    h, w = input_image.shape
    kh, kw = kernel.shape

    # Flip kernel for convolution mode
    if mode == 'convolution':
        kernel = np.flip(np.flip(kernel, axis=0), axis=1)

    # Calculate padding
    if padding == 'same':
        pad_h = (kh - 1) // 2
        pad_w = (kw - 1) // 2
        padded_image = np.pad(input_image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    else:  # 'valid' padding
        padded_image = input_image

    # Calculate output dimensions
    out_h = (padded_image.shape[0] - kh) // stride + 1
    out_w = (padded_image.shape[1] - kw) // stride + 1

    # Initialize output
    output = np.zeros((out_h, out_w), dtype=np.float32)

    # Perform convolution
    for i in range(0, out_h):
        for j in range(0, out_w):
            # Get the current patch
            i_start, j_start = i * stride, j * stride
            patch = padded_image[i_start:i_start + kh, j_start:j_start + kw]

            # Calculate the convolution for this position
            output[i, j] = np.sum(patch * kernel)

    return output

def apply_kernels_and_visualize(image):

    # Define kernels
    # Edge detection kernels
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    sobel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])

    laplacian = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ])

    # Blurring kernels
    box_blur = np.ones((3, 3)) / 9

    gaussian_blur = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ]) / 16

    # Sharpening kernel
    sharpen = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    # Symmetric and non-symmetric kernels for comparison
    symmetric = np.array([
        [0.25, 0.25],
        [0.25, 0.25]
    ])

    non_symmetric = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])

    # Create a dictionary of kernels and their descriptions
    kernels = {
        'Sobel X (Edge Detection)': sobel_x,
        'Sobel Y (Edge Detection)': sobel_y,
        'Laplacian (Edge Detection)': laplacian,
        'Box Blur': box_blur,
        'Gaussian Blur': gaussian_blur,
        'Sharpen': sharpen,
        'Symmetric 2x2': symmetric,
        'Non-symmetric (Sobel Y)': non_symmetric
    }

    # Create figures
    plt.figure(figsize=(12, 10))
    plt.subplot(3, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Apply each kernel and visualize
    idx = 2
    for name, kernel in kernels.items():
        # Manual convolution
        result = manual_convolution_2d(image, kernel=kernel, padding='same')

        # Normalize for display if needed
        if np.min(result) < 0:
            result = (result - np.min(result)) / (np.max(result) - np.min(result))

        plt.subplot(3, 3, idx)
        plt.imshow(result, cmap='gray')
        plt.title(name)
        plt.axis('off')
        idx += 1

        if idx > 9:
            break

    plt.tight_layout()
    plt.savefig('kernel_effects.png', dpi=300)
    plt.show()

    return kernels

def compare_convolution_vs_correlation(image, symmetric_kernel, non_symmetric_kernel):
    # Results with symmetric kernel
    symmetric_conv = manual_convolution_2d(image, symmetric_kernel, mode='convolution')
    symmetric_corr = manual_convolution_2d(image, symmetric_kernel, mode='correlation')

    # Results with non-symmetric kernel
    non_symmetric_conv = manual_convolution_2d(image, non_symmetric_kernel, mode='convolution')
    non_symmetric_corr = manual_convolution_2d(image, non_symmetric_kernel, mode='correlation')

    # Normalize for display
    def normalize(img):
        if np.min(img) < 0:
            return (img - np.min(img)) / (np.max(img) - np.min(img))
        return img

    symmetric_conv = normalize(symmetric_conv)
    symmetric_corr = normalize(symmetric_corr)
    non_symmetric_conv = normalize(non_symmetric_conv)
    non_symmetric_corr = normalize(non_symmetric_corr)

    # Display results
    plt.figure(figsize=(16, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(symmetric_conv, cmap='gray')
    plt.title('Symmetric Kernel - Convolution')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(symmetric_corr, cmap='gray')
    plt.title('Symmetric Kernel - Correlation')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(symmetric_kernel, cmap='gray')
    plt.title('Symmetric Kernel')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(non_symmetric_conv, cmap='gray')
    plt.title('Non-symmetric Kernel - Convolution')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.imshow(non_symmetric_corr, cmap='gray')
    plt.title('Non-symmetric Kernel - Correlation')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('convolution_vs_correlation.png', dpi=300)
    plt.show()

    # Calculate and display difference
    symmetric_diff = np.abs(symmetric_conv - symmetric_corr)
    non_symmetric_diff = np.abs(non_symmetric_conv - non_symmetric_corr)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(symmetric_diff, cmap='hot')
    plt.title('Difference: Symmetric Kernel')
    plt.axis('off')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(non_symmetric_diff, cmap='hot')
    plt.title('Difference: Non-symmetric Kernel')
    plt.axis('off')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig('convolution_correlation_difference.png', dpi=300)
    plt.show()

    # Print statistics about the differences
    print("Symmetric Kernel - Max Difference:", np.max(symmetric_diff))
    print("Symmetric Kernel - Mean Difference:", np.mean(symmetric_diff))
    print("Non-symmetric Kernel - Max Difference:", np.max(non_symmetric_diff))
    print("Non-symmetric Kernel - Mean Difference:", np.mean(non_symmetric_diff))

def explore_kernel_parameters(image):

    # Define a basic edge detection kernel
    edge_kernel = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ])

    # Apply with different kernel sizes
    kernel_sizes = [3, 5, 7]
    results_size = []
    for size in kernel_sizes:
        # Create a scaled version of the kernel
        if size == 3:
            kernel = edge_kernel
        else:
            # Create a larger kernel that approximates the edge detection effect
            kernel = np.ones((size, size)) * -1
            center = size // 2
            kernel[center, center] = size * size - 1

        result = manual_convolution_2d(image, kernel=kernel, padding='same')
        # Normalize for visualization
        if np.min(result) < 0:
            result = (result - np.min(result)) / (np.max(result) - np.min(result))
        results_size.append(result)

    # Apply with different strides
    strides = [1, 2, 3]
    results_stride = []
    for stride in strides:
        result = manual_convolution_2d(image, kernel=edge_kernel, stride=stride, padding='same')
        # Normalize for visualization
        if np.min(result) < 0:
            result = (result - np.min(result)) / (np.max(result) - np.min(result))
        results_stride.append(result)

    # Apply with different padding options
    paddings = ['same', 'valid']
    results_padding = []
    for padding in paddings:
        result = manual_convolution_2d(image, kernel=edge_kernel, padding=padding)
        # Normalize for visualization
        if np.min(result) < 0:
            result = (result - np.min(result)) / (np.max(result) - np.min(result))
        results_padding.append(result)

    # Visualize results
    plt.figure(figsize=(15, 12))

    # Original image
    plt.subplot(3, 4, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Kernel size comparison
    for i, (size, result) in enumerate(zip(kernel_sizes, results_size)):
        plt.subplot(3, 4, i + 2)
        plt.imshow(result, cmap='gray')
        plt.title(f'Kernel Size: {size}x{size}')
        plt.axis('off')

    # Stride comparison
    for i, (stride, result) in enumerate(zip(strides, results_stride)):
        plt.subplot(3, 4, i + 5)
        plt.imshow(result, cmap='gray')
        plt.title(f'Stride: {stride}')
        plt.axis('off')

    # Padding comparison
    for i, (padding, result) in enumerate(zip(paddings, results_padding)):
        plt.subplot(3, 4, i + 8)
        plt.imshow(result, cmap='gray')
        plt.title(f'Padding: {padding}')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('parameter_effects.png', dpi=300)
    plt.show()

def compare_manual_vs_numpy(image, kernel):


    # Manual convolution
    manual_result = manual_convolution_2d(image, kernel=kernel, padding='same')

    # NumPy-based convolution using scipy.signal.convolve2d
    numpy_result = signal.convolve2d(image, kernel, mode='same', boundary='fill', fillvalue=0)

    # Calculate difference
    difference = np.abs(manual_result - numpy_result)

    # Normalize for display
    def normalize(img):
        if np.min(img) < 0:
            return (img - np.min(img)) / (np.max(img) - np.min(img))
        return img

    manual_result = normalize(manual_result)
    numpy_result = normalize(numpy_result)

    # Display results
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(manual_result, cmap='gray')
    plt.title('Manual Convolution')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(numpy_result, cmap='gray')
    plt.title('NumPy Convolution')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(difference, cmap='hot')
    plt.title('Absolute Difference')
    plt.axis('off')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig('manual_vs_numpy.png', dpi=300)
    plt.show()

    # Print statistics
    print("Max Difference:", np.max(difference))
    print("Mean Difference:", np.mean(difference))
    print("Standard Deviation of Difference:", np.std(difference))

def demonstrate_multiple_kernels(image):

    # Define kernels
    edge_detection = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ])

    gaussian_blur = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ]) / 16

    sharpen = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    # Apply individual kernels
    edge_result = manual_convolution_2d(image, edge_detection, padding='same')
    blur_result = manual_convolution_2d(image, gaussian_blur, padding='same')
    sharpen_result = manual_convolution_2d(image, sharpen, padding='same')

    # Apply sequential kernels
    # First blur, then edge detection
    blur_then_edge = manual_convolution_2d(blur_result, edge_detection, padding='same')

    # First sharpen, then edge detection
    sharpen_then_edge = manual_convolution_2d(sharpen_result, edge_detection, padding='same')

    # Normalize for display
    def normalize(img):
        if np.min(img) < 0:
            return (img - np.min(img)) / (np.max(img) - np.min(img))
        return img

    edge_result = normalize(edge_result)
    blur_result = normalize(blur_result)
    sharpen_result = normalize(sharpen_result)
    blur_then_edge = normalize(blur_then_edge)
    sharpen_then_edge = normalize(sharpen_then_edge)

    # Display results
    plt.figure(figsize=(18, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(edge_result, cmap='gray')
    plt.title('Edge Detection')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(blur_result, cmap='gray')
    plt.title('Gaussian Blur')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(sharpen_result, cmap='gray')
    plt.title('Sharpening')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(blur_then_edge, cmap='gray')
    plt.title('Blur → Edge Detection')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.imshow(sharpen_then_edge, cmap='gray')
    plt.title('Sharpen → Edge Detection')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('multiple_kernels.png', dpi=300)
    plt.show()

def main():
    # Load a sample grayscale image
   # image = data.camera()  # Standard test image
    image_path = "/content/greyscale.jpg"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 1. Apply various kernels and visualize
    kernels = apply_kernels_and_visualize(image)

    # 2. Compare convolution vs correlation
    symmetric_kernel = np.array([
        [0.25, 0.25],
        [0.25, 0.25]
    ])

    non_symmetric_kernel = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])

    compare_convolution_vs_correlation(image, symmetric_kernel, non_symmetric_kernel)

    # 3. Explore effects of kernel parameters
    explore_kernel_parameters(image)

    # 4. Compare manual vs NumPy implementation
    edge_kernel = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ])

    compare_manual_vs_numpy(image, edge_kernel)

    # 5. Demonstrate advantages of multiple kernels
    demonstrate_multiple_kernels(image)

    # Print analysis summary
    print("\n==== Analysis Summary ====")
    print("1. Edge Detection Kernels:")
    print("   - Sobel kernels (X and Y) detect vertical and horizontal edges")
    print("   - Laplacian kernel detects edges in all directions")
    print("2. Blurring Kernels:")
    print("   - Box blur: simple averaging filter")
    print("   - Gaussian blur: weighted averaging for more natural blur")
    print("3. Sharpening Kernels:")
    print("   - Enhances edges while maintaining original image features")
    print("4. Convolution vs Correlation:")
    print("   - For symmetric kernels: virtually no difference")
    print("   - For non-symmetric kernels: significant differences in edge direction and intensity")
    print("5. Parameter Effects:")
    print("   - Kernel size: larger kernels consider wider context")
    print("   - Stride: higher stride reduces output size but loses information")
    print("   - Padding: 'same' preserves image dimensions while 'valid' reduces them")
    print("6. Multiple Kernel Application:")
    print("   - Sequential application allows for combined effects")
    print("   - Pre-processing with blur reduces noise before edge detection")
    print("   - Pre-processing with sharpening enhances edges before detection")

if __name__ == "__main__":
    main()
