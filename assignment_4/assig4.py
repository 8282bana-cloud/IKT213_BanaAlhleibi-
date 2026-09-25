import cv2
import numpy as np
import os


class DocumentAligner:

    def __init__(self, reference_path, target_path):
        self.reference_path = reference_path
        self.target_path = target_path

        if not os.path.exists(self.reference_path) or not os.path.exists(self.target_path):
            raise FileNotFoundError("Kunne ikke finne alle inndata-bildene.")

    def run_harris_detection(self, output_path="harris.png"):
        img = cv2.imread(self.reference_path)
        if img is None:
            raise FileNotFoundError(f"Kunne ikke laste {self.reference_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_float = np.float32(gray)

        corners = cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=0.04)
        corners = cv2.dilate(corners, None)

        img[corners > 0.01 * corners.max()] = [0, 0, 255]

        if not cv2.imwrite(output_path, img):
            raise RuntimeError(f"Kunne ikke lagre {output_path}")

        print(f"Created: {output_path}")

    def align(self, image_to_align, reference_image, max_features, good_match_precent):
        img_to_align = cv2.imread(image_to_align)
        ref_img = cv2.imread(reference_image)

        if img_to_align is None or ref_img is None:
            raise FileNotFoundError("Et av bildene kunne ikke lastes")

        gray_target = cv2.cvtColor(img_to_align, cv2.COLOR_BGR2GRAY)
        gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()
        kp_target, des_target = sift.detectAndCompute(gray_target, None)
        kp_ref, des_ref = sift.detectAndCompute(gray_ref, None)

        if des_target is None or des_ref is None:
            raise RuntimeError("SIFT klarte ikke å finne trekk i bildene.")

        print(f"SIFT features in image_to_align: {len(kp_target)}")
        print(f"SIFT features in reference_image: {len(kp_ref)}")

        flann = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5),
            dict(checks=50)
        )
        matches = flann.knnMatch(des_target, des_ref, k=2)

        good_matches = [
            m for m, n in matches
            if m.distance < (good_match_precent * n.distance)
        ]

        good_matches = sorted(good_matches, key=lambda x: x.distance)
        if max_features > 0 and len(good_matches) > max_features:
            good_matches = good_matches[:max_features]

        print(f"Good matches used: {len(good_matches)}")

        if len(good_matches) < 4:
            raise RuntimeError("For få matcher for å kunne beregne homografi.")

        src_pts = np.float32([kp_target[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)


        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            raise RuntimeError("Klarte ikke å beregne homografi.")

        h, w = ref_img.shape[:2]
        aligned = cv2.warpPerspective(img_to_align, H, (w, h))

        inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i][0] == 1]
        matches_img = cv2.drawMatches(
            img_to_align, kp_target,
            ref_img, kp_ref,
            inlier_matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        cv2.imwrite("aligned.png", aligned)
        cv2.imwrite("matches.png", matches_img)

        print("Created: aligned.png")
        print("Created: matches.png")


def main():
    ref_file = "reference_img.png"
    target_file = "align_this.jpg"

    print("\nPart 1: Harris Corner Detection")

    aligner = DocumentAligner(ref_file, target_file)
    aligner.run_harris_detection()

    print("\nPart 2: Feature-Based Image Alignment")
    print("max_features = 10")
    print("good_match_precent = 0.7\n")

    aligner.align(
        image_to_align=target_file, reference_image=ref_file, max_features=10, good_match_precent=0.7
    )


if __name__ == "__main__":
    main()