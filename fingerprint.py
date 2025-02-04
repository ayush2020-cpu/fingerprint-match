import cv2
import numpy as np

def extract_features(image_path):
    """Extract keypoints and descriptors from a fingerprint image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return img, keypoints, descriptors

def match_fingerprints(des1, des2, kp1, kp2):
    """Match two fingerprint descriptors and return match percentage and matches."""
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    match_percentage = (len(good_matches) / max(len(kp1), len(kp2))) * 100
    return good_matches, match_percentage

def draw_matches(img1, kp1, img2, kp2, good_matches):
    """Draw matched keypoints between two fingerprint images."""
    return cv2.drawMatches(img1, kp1, img2, kp2, good_matches[:30], None, flags=2)
