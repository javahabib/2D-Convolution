# 2D-Convolution
# 🧠 Image Kernel Visualizer

A Python-based tool to manually implement 2D convolution and apply various image processing filters. This project also compares **convolution vs. correlation**, explores the effect of **kernel size, stride, and padding**, and visualizes all results using `matplotlib`.

## 📸 Features

- Manual 2D convolution (from scratch using NumPy)
- Apply standard filters:
  - Sobel (X/Y), Laplacian, Sharpen, Gaussian Blur
- Visualize and save filtered image outputs
- Compare **convolution vs. correlation** on symmetric and non-symmetric kernels
- Explore the effect of:
  - Kernel size
  - Stride
  - Padding

## 🧰 Technologies Used

- Python 3.x
- NumPy
- OpenCV (`cv2`)
- SciPy
- scikit-image
- Matplotlib


## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/image-kernel-visualizer.git
cd image-kernel-visualizer
```
### 2. Install dependencies
```bash
pip install numpy opencv-python matplotlib scipy scikit-image
```

### 3. Add your grayscale image
Place your image as greyscale.jpg in the root directory or modify the path in main.py.

### 4. Run the script
```bash
python manual_2D_convolution.py
```
