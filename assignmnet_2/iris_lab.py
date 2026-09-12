import cv2
import numpy as np


'''legger en reflekterende kant til bilden'''
def padding(image, border_width):
    padded_image = cv2.copyMakeBorder(
        image,
        top=border_width,
        bottom=border_width,
        left=border_width,
        right=border_width,
        borderType=cv2.BORDER_REFLECT
    )
    cv2.imwrite('padded_iris.png', padded_image)
    print("Lagret som: padded_iris.png")
    return padded_image

''' basert på angitte koordinater, beskjærer bildet'''
def crop(image, x_0, x_1, y_0, y_1):
    cropped_image = image[y_0:y_1, x_0:x_1]
    cv2.imwrite('cropped_iris.png', cropped_image)
    print("Lagret som: cropped_iris.png")
    return cropped_image

'''bildestørrelsen endres baser på angitt bredde og høyde'''
def resize(image, width, height):
    resized_image = cv2.resize(image, (width, height))
    cv2.imwrite('resized_iris.png', resized_image)
    print("Lagret som: resized_iris.png")
    return resized_image

'''ved hjelp av løkker blir piksler kopiert manulet'''
def copy(image, emptyPictureArray):
    h, w, c = image.shape
    for y in range(h):
        for x in range(w):
            emptyPictureArray[y, x] = image[y, x]

    cv2.imwrite('copied_iris.png', emptyPictureArray)
    print("Lagret som: copied_iris.png")
    return emptyPictureArray

'''gråtoner'''
def grayscale(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('grayscale_iris.png', gray_image)
    print("Lagret som: grayscale_iris.png")
    return gray_image

'''bildet BLIR konvertert fra BGR TIL HSV fargesom'''
def hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imwrite('hsv_iris.png', hsv_image)
    print("Lagret som: hsv_iris.png")
    return hsv_image

'''Skifter på fargeverdiene for alle kanaler med angitt hue'''
def hue_shifted(image, emptyPictureArray, hue):
    h, w, c = image.shape
    for y in range(h):
        for x in range(w):
            for chan in range(c):
                original_val = int(image[y, x, chan])
                new_val = original_val + hue
                final_val = new_val % 256
                emptyPictureArray[y, x, chan] = final_val

    cv2.imwrite('hue_shifted_iris.png', emptyPictureArray)
    print("Lagret som: hue_shifted_iris.png")
    return emptyPictureArray

'''utjevning ved hjelp av Gaussian'''
def smoothing(image):
    smoothed_image = cv2.GaussianBlur(image, (15, 15), 0)
    cv2.imwrite('smoothed_iris.png', smoothed_image)
    print("Lagret som: smoothed_iris.png")
    return smoothed_image

'''roterer bildet 90 eller 180'''
def rotation(image, rotation_angle):
    if rotation_angle == 90:
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite('rotated_90_iris.png', rotated_image)
        print("Lagret som: rotated_90_iris.png")
    elif rotation_angle == 180:
        rotated_image = cv2.rotate(image, cv2.ROTATE_180)
        cv2.imwrite('rotated_180_iris.png', rotated_image)
        print("Lagret som: rotated_180_iris.png")
    else:
        print(f"Vinkel {rotation_angle} støttes ikke.")
        return image

    return rotated_image

'''laster inn bildet iris.png'''
def main():
    image = cv2.imread('iris.png')

    if image is None:
        print("ERROR: Kunne ikke finne 'iris.png'. Kan du Sjekke filnavnet og sti.")
        return

    height, width = image.shape[:2]
    print(f"Bilde lastet: Bredde={width}, Høyde={height}")

    print("\n1. Padding")
    padding(image, 100)

    print("\n2. Cropping")
    x_0 = 200
    x_1 = width - 130
    y_0 = 200
    y_1 = height - 130

    if x_1 > x_0 and y_1 > y_0:
        crop(image, x_0, x_1, y_0, y_1)
    else:
        print("Advarsel: Beskjæringskoordinatene er ikke gyldige for dette bildet.")

    print("\n3. Resizing")
    resize(image, 200, 200)

    print("\n4. Manual Copy")
    empty_copy_array = np.zeros((height, width, 3), dtype=np.uint8)
    copy(image, empty_copy_array)

    print("\n5. Grayscale")
    grayscale(image)

    print("\n6. HSV conversion")
    hsv(image)

    print("\n7. Hue Shift")
    empty_shift_array = np.zeros((height, width, 3), dtype=np.uint8)
    hue_shifted(image, empty_shift_array, 50)

    print("\n8. Smoothing")
    smoothing(image)

    print("\n9. Rotation 180 degrees")
    rotation(image, 180)


if __name__ == "__main__":
    main()