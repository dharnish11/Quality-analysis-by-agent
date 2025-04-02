import torch
import cv2
import numpy as np
import kornia.feature as KF
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.preprocessing.image import img_to_array
import hashlib
import matplotlib.pyplot as plt

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load LoFTR model
loftr_matcher = KF.LoFTR(pretrained='outdoor').to(device).eval()
sift = cv2.SIFT_create()

def compute_md5(image_path):
    with open(image_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def load_image(image):
    image = cv2.resize(image, (640, 480))
    image = torch.tensor(image, dtype=torch.float32) / 255.0
    image = image.unsqueeze(0).unsqueeze(0).to(device)
    return image

def find_unmatched_keypoints(all_kpts, matched_kpts, tolerance=2):
    return np.array([kp for kp in all_kpts if not np.any(np.all(np.isclose(matched_kpts, kp, atol=tolerance), axis=1))])

def detect_defects_loftr(good_image_path, test_image_path):
    if compute_md5(good_image_path) == compute_md5(test_image_path):
        return "Good"
    
    img1 = cv2.imread(good_image_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    matcher = KF.LoFTR(pretrained="outdoor").to(device).eval()
    img0_t, img1_t = load_image(img1), load_image(img2)
    
    with torch.no_grad():
        output_dict = matcher({"image0": img0_t, "image1": img1_t})
    
    mkpts0 = output_dict["keypoints0"].cpu().numpy()
    mkpts1 = output_dict["keypoints1"].cpu().numpy()
    
    if len(mkpts1) > 0.8 * len(cv2.SIFT_create().detect(img2, None)):
        return "Good"
    return "Defective"
# ================== TASK 2: SIFT Matching ==================
import cv2
import numpy as np
import torch
import kornia.feature as KF
import matplotlib.pyplot as plt

# Device setup for Kornia
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sift_descriptor = KF.SIFTDescriptor(41).to(device).eval()


def detect_defects_sift(good_img_path, test_img_path):
    """Detect defects and highlight them in red on a good image."""

    # Load images in grayscale
    good_image = cv2.imread(good_img_path, cv2.IMREAD_GRAYSCALE)
    test_image = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)

    if good_image is None or test_image is None:
        print("Error: Could not load one or both images.")
        return

    # Resize images for consistency
    good_image = cv2.resize(good_image, (640, 480))
    test_image = cv2.resize(test_image, (640, 480))

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect and compute SIFT features
    keypoints0, descriptors0 = sift.detectAndCompute(good_image, None)
    keypoints1, descriptors1 = sift.detectAndCompute(test_image, None)

    if descriptors0 is None or descriptors1 is None:
        print("Error: Could not compute descriptors.")
        return

    # BFMatcher for feature matching
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors0, descriptors1)
    matches = sorted(matches, key=lambda x: x.distance)  # Sort matches by distance (best first)

    # Compute unmatched keypoints
    matched_keypoints = set([m.queryIdx for m in matches])
    unmatched_keypoints = len(keypoints0) - len(matched_keypoints)  # Keypoints without a match

    # Dynamic threshold: keypoints0 - 50
    match_threshold = max(len(keypoints0) - 50, 0)

    # Draw matches between good and test images
    match_img = cv2.drawMatches(
        cv2.cvtColor(good_image, cv2.COLOR_GRAY2BGR), keypoints0,
        cv2.cvtColor(test_image, cv2.COLOR_GRAY2BGR), keypoints1,
        matches, None
    )

    # Compute absolute difference between images
    diff = cv2.absdiff(good_image, test_image)

    # Apply threshold to extract defect regions
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Find contours of defects
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Highlight defects on the original good image
    defect_highlight = cv2.cvtColor(good_image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(defect_highlight, contours, -1, (0, 0, 255), 2)  # Draw in red

    # Display results
   

    return "GOOD" if len(matches) >= match_threshold else "DEFECTIVE"

# ================== TASK 3: Histogram Comparison ==================
def compare_histograms(good_img_path, test_img_path, threshold=0.9):
    """Compare images using histogram correlation."""
    good_image = cv2.imread(good_img_path)
    test_image = cv2.imread(test_img_path)

    if good_image is None or test_image is None:
        return "Error: Could not load images"

    test_image = cv2.resize(test_image, (good_image.shape[1], good_image.shape[0]))

    hist_good = [cv2.calcHist([good_image], [i], None, [256], [0, 256]) for i in range(3)]
    hist_test = [cv2.calcHist([test_image], [i], None, [256], [0, 256]) for i in range(3)]

    hist_good = [cv2.normalize(h, h).flatten() for h in hist_good]
    hist_test = [cv2.normalize(h, h).flatten() for h in hist_test]

    similarity_scores = [cv2.compareHist(hist_good[i], hist_test[i], cv2.HISTCMP_CORREL) for i in range(3)]
    avg_similarity = sum(similarity_scores) / 3

    return "GOOD" if avg_similarity > threshold else "DEFECTIVE"

# ================== TASK 4: Edge & SSIM Analysis ==================
def compare_images_strict(good_img_path, test_img_path, edge_threshold=0.7, ssim_threshold=0.85):
    """Compare images using edge detection and SSIM for stricter defect detection."""
    good_image = cv2.imread(good_img_path, cv2.IMREAD_GRAYSCALE)
    test_image = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)

    if good_image is None or test_image is None:
        return "Error: Could not load images"

    test_image = cv2.resize(test_image, (good_image.shape[1], good_image.shape[0]))

    edges_good = cv2.Canny(good_image, 50, 150)
    edges_test = cv2.Canny(test_image, 50, 150)
    edge_similarity = np.sum(edges_good == edges_test) / edges_good.size
    ssim_score = ssim(good_image, test_image)

    return "GOOD" if edge_similarity >= edge_threshold and ssim_score >= ssim_threshold else "DEFECTIVE"

# ================== TASK 5: Autoencoder Reconstruction ==================
def build_autoencoder():
    input_img = Input(shape=(224, 224, 3))  # RGB images (224x224x3)

    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    return Model(input_img, decoded)

# ==============================
# 2️⃣ Load and Preprocess a Single Image
# ==============================
def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None

    image = cv2.resize(image, (224, 224))  # Resize to match model input
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = img_to_array(image) / 255.0  # Normalize pixel values (0 to 1)
    return np.expand_dims(image, axis=0)  # Add batch dimension

# ==============================
# 3️⃣ Train Autoencoder on a Single Good Image
# ==============================
def train_autoencoder(autoencoder, good_img_path, epochs=10):
    good_image = load_and_preprocess_image(good_img_path)
    if good_image is None:
        return None

    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(good_image, good_image, epochs=epochs, verbose=0)  # Silent training

    return autoencoder

# ==============================
# 4️⃣ Defect Detection Function (Silent)
# ==============================
def detect_defect_autoencoder(autoencoder, test_img_path, threshold=0.01):
    """ Detect defects based on reconstruction error without printing """
    test_image = load_and_preprocess_image(test_img_path)
    if test_image is None:
        return "ERROR"

    reconstructed = autoencoder.predict(test_image, verbose=0)  # Silent prediction
    error = np.mean(np.square(test_image - reconstructed))

    return "GOOD" if error < threshold else "DEFECTIVE"


def compare_images_orb(good_img_path, test_img_path, threshold=0.5, scale_percent=100):
    """Compare images using ORB-based feature matching and show only the defect-highlighted image."""
    
    # Load images in grayscale
    good_image = cv2.imread(good_img_path, cv2.IMREAD_GRAYSCALE)
    test_image = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)

    if good_image is None or test_image is None:
        print("Error: Could not load one or both images.")
        return

    # Resize test image to match good image
    test_image = cv2.resize(test_image, (good_image.shape[1], good_image.shape[0]))

    # Compute absolute difference
    diff = cv2.absdiff(good_image, test_image)
    _, thresholded = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
    
    # Remove small noise
    kernel = np.ones((3, 3), np.uint8)
    thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)

    # Convert grayscale test image to BGR
    defect_image = cv2.cvtColor(test_image, cv2.COLOR_GRAY2BGR)

    # Create a red color mask (where defect is found)
    red_mask = np.zeros_like(defect_image)
    red_mask[:, :, 2] = thresholded  # Apply red channel only

    # Overlay red mask on the original image
    highlighted_defects = cv2.addWeighted(defect_image, 1, red_mask, 0.7, 0)

    # Resize image
    def resize_image(image, scale_percent):
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

    highlighted_resized = resize_image(highlighted_defects, scale_percent)

    # Show only the highlighted defect image
    cv2.imshow("Highlighted Defects", highlighted_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return "Defective" if np.sum(thresholded) > 0 else "Good"
