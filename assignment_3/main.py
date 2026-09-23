import cv2
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))


def sobel_edge_detection(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    blurred = cv2.GaussianBlur(gray, (3, 3), sigmaX=0)

    sobel_img = cv2.Sobel(
        blurred, cv2.CV_16S,dx=1,dy=1,ksize=1
    )

    abs_sobel = cv2.convertScaleAbs(sobel_img)
    output_path = os.path.join(script_dir, "sobel_edges.png")
    cv2.imwrite(output_path, abs_sobel)

    print(f"Sobel-kanter lagret i {output_path}")
    return abs_sobel


def canny_edge_detection(image, threshold_1, threshold_2):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    blurred = cv2.GaussianBlur(gray, (3, 3), sigmaX=0)

    edges = cv2.Canny( blurred,threshold_1,threshold_2)

    output_path = os.path.join(script_dir, "canny_edges.png")
    cv2.imwrite(output_path, edges)

    print(f"Canny-kanter lagret i {output_path}")

    return edges


def template_match(image, template):

    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    temp_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template.copy()

    result = cv2.matchTemplate(
        img_gray,temp_gray,cv2.TM_CCOEFF_NORMED
    )

    threshold = 0.9
    locations = np.where(result >= threshold)

    output_img = image.copy()

    h, w = temp_gray.shape

    for pt in zip(*locations[::-1]):
        cv2.rectangle(output_img,pt,(pt[0] + w, pt[1] + h),(0, 0, 255),2)

    output_path = os.path.join(script_dir, "template_matched.png")
    cv2.imwrite(output_path, output_img)

    print(f"Resultat av mal-sammenligning lagret i {output_path}")

    return output_img


def resize(image, scale_factor: int, up_or_down: str):

    resized_img = image.copy()

    if up_or_down.lower() == "up":

        for _ in range(scale_factor):
            resized_img = cv2.pyrUp(resized_img)

    elif up_or_down.lower() == "down":

        for _ in range(scale_factor):
            resized_img = cv2.pyrDown(resized_img)

    else:
        raise ValueError("up_or_down må være enten 'up' eller 'down'")

    output_path = os.path.join(
        script_dir, f"resized_{up_or_down}_{scale_factor}.png"
    )

    cv2.imwrite(output_path, resized_img)
    print(f"Bilde med ny størrelse lagret i {output_path}")

    return resized_img


if __name__ == "__main__":

    image_paths = {
        "lambo": os.path.join(script_dir, "lambo.png"),
        "shapes": os.path.join(script_dir, "shapes.png"),
        "template": os.path.join(script_dir, "shapes_template.jpg")
    }

    images = {}

    for name, path in image_paths.items():
        images[name] = cv2.imread(path)

    missing_images = [name
        for name, img in images.items()
        if img is None
    ]

    if missing_images:

        print(f"Feil: Kunne ikke laste inn følgende bilder: "
            f"{', '.join(missing_images)}"
        )

        print(f"Skriptet kjører fra: {script_dir}")
        print(f"Filer i mappen: {os.listdir(script_dir)}")

    else:

        print("Kjører Sobel kantdeteksjon...")
        sobel_edge_detection(images["lambo"])

        print("Kjører Canny kantdeteksjon...")
        canny_edge_detection(images["lambo"], 50, 50)

        print("Kjører mal-sammenligning...")
        template_match(images["shapes"],images["template"]
        )

        print("Gjør bildet større")
        resize(images["lambo"],scale_factor=2,up_or_down="up"
        )

        print("Gjør bildet mindre ")
        resize(images["lambo"],scale_factor=2,up_or_down="down"
        )

