import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

def spatial_frequency_analysis(image1_path, image2_path): # spatial frequency analysis for two images

    try:
        img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
        img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            raise ValueError("Images could not be loaded. Check file paths.")

        # Ensure images are the same size (or resize them) for meaningful comparison
        if img1.shape != img2.shape:
            min_height = min(img1.shape[0], img2.shape[0])
            min_width = min(img1.shape[1], img2.shape[1])
            img1 = cv2.resize(img1, (min_width, min_height))
            img2 = cv2.resize(img2, (min_width, min_height))
            print("Images resized to have the same dimensions for comparison.")


        # Fourier Transform
        f1 = np.fft.fft2(img1)
        f2 = np.fft.fft2(img2)

        # Magnitude Spectrum (for spatial frequency representation)
        magnitude_spectrum1 = np.abs(f1)
        magnitude_spectrum2 = np.abs(f2)

        # Log transform for better visualization of the magnitude spectrum
        log_magnitude_spectrum1 = 20 * np.log(np.abs(f1))
        log_magnitude_spectrum2 = 20 * np.log(np.abs(f2))


        # Image similarity metrics
        ssim_score = ssim(img1, img2)
        rmse = np.sqrt(np.mean((img1 - img2)**2))
        correlation = np.corrcoef(img1.flatten(), img2.flatten())[0, 1]  # Flatten for correlation


        results = {
            "fft_magnitude1": magnitude_spectrum1,
            "fft_magnitude2": magnitude_spectrum2,
            "fft_log_magnitude1": log_magnitude_spectrum1,
            "fft_log_magnitude2": log_magnitude_spectrum2,
            "ssim": ssim_score,
            "rmse": rmse,
            "correlation": correlation,
        }
        return results

    except ValueError as e:
        print(f"Error: {e}")
        return None
    except Exception as e: 
        print(f"An unexpected error occurred: {e}")
        return None



# Replace with your image paths
image1_path = r"image1.png"  
image2_path = r"image2.png"

analysis_results = spatial_frequency_analysis(image1_path, image2_path)

if analysis_results:
    # Display or analyze the results:
    plt.figure(figsize=(12, 6))

    plt.subplot(221)
    plt.imshow(analysis_results["fft_log_magnitude1"], cmap='gray')
    plt.title("Log Magnitude Spectrum - Image 1")

    plt.subplot(222)
    plt.imshow(analysis_results["fft_log_magnitude2"], cmap='gray')
    plt.title("Log Magnitude Spectrum - Image 2")

    plt.subplot(223)
    plt.imshow(cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE), cmap='gray') # Display original image 1
    plt.title("Image 1")

    plt.subplot(224)
    plt.imshow(cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE), cmap='gray') # Display original image 2
    plt.title("Image 2")


    plt.show()

    print(f"SSIM: {analysis_results['ssim']}")
    print(f"RMSE: {analysis_results['rmse']}")
    print(f"Correlation: {analysis_results['correlation']}")
