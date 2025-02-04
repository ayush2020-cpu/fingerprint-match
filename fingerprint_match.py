from fingerprint import extract_features, match_fingerprints, draw_matches
import cv2
import matplotlib.pyplot as plt

def fingerprint_matching(img1_path, img2_path):
    img1, kp1, des1 = extract_features(img1_path)
    img2, kp2, des2 = extract_features(img2_path)
    
    good_matches, match_percentage = match_fingerprints(des1, des2, kp1, kp2)
    result_img = draw_matches(img1, kp1, img2, kp2, good_matches)
    
    # Display the result with matplotlib
    plt.figure(figsize=(10, 6))
    plt.imshow(result_img, cmap='gray')
    plt.title(f"Fingerprint Match Percentage: {match_percentage:.2f}%")
    plt.axis('off')
    plt.show()

    # Print fingerprint matching results
    print("\nFINGERPRINT MATCHING RESULTS")
    print("--------------------------------")
    print(f"Total Keypoints in Image 1: {len(kp1)}")
    print(f"Total Keypoints in Image 2: {len(kp2)}")
    print(f"Good Matches Found: {len(good_matches)}")
    print("Match Confidence: {:.2f}%".format(match_percentage))
    
    # Match Confidence Threshold
    if match_percentage > 40:
        print("✅ Fingerprints MATCH!")
    else:
        print("❌ Fingerprints DO NOT MATCH!")

if __name__ == "__main__":
    img1_path = "fingerprint1.jpg"
    # If we compare with fingerprint2 then it wont match 
    img2_path = "fingerprint3.jpg"
    fingerprint_matching(img1_path, img2_path)
