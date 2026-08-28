import cv2  # Bibliotek
import os  # Filer


def print_image_information(image):  # Funksjon
    height, width = image.shape[:2]  # Dimensjoner
    channels = image.shape[2] if len(image.shape) == 3 else 1  # Kanaler
    size = image.size  # Antall
    data_type = image.dtype  # Type
    # Utskrift
    print(f"height: {height}")
    print(f"width: {width}")
    print(f"channels: {channels}")
    print(f"size: {size}")
    print(f"data type: {data_type}")

 # Funksjon
def save_camera_info():
    camera = cv2.VideoCapture(0)  # Start
    # Henter
    fps = camera.get(cv2.CAP_PROP_FPS)
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))

    camera.release()  # Lukk

    os.makedirs("solutions", exist_ok=True)  # Mappe
    fil_sti = os.path.join("solutions", "camera_outputs.txt")  # Sti

    with open(fil_sti, "w") as fil:
        fil.write(f"fps: {fps}\n")
        fil.write(f"height: {height}\n")
        fil.write(f"width: {width}\n")

    print(f"Fil lagret til: {fil_sti}")  # Bekreft


def main():  # Hoved
    print("Bildeinformasjon")  # Tittel

    bilde = cv2.imread("iris-1.jpg")  # Les

    if bilde is None:  # Sjekk
        print("FeiL: finner ikke iris-1.jpg")  # Feil
    else:
        print_image_information(bilde)  # Kall

    print("\n" + "-" * 50 + "\n")  # Skill

    print("info-Webkamera")  # Tittel
    save_camera_info()  # Kall


if __name__ == "__main__":  # Sjekk
    main()  # Kjør