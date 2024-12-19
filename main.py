import cv2

def ex1():
    cap = cv2.VideoCapture(0)
    object_detector = cv2.createBackgroundSubtractorMOG2()  # Correction ici

    while True:
        ret, frame = cap.read()

        mask = object_detector.apply(frame)
        cv2.imshow('video', mask)  # Affiche le masque généré par le détecteur d'objets

        if cv2.waitKey(30) & 0xFF == 27:  # Quitter avec la touche Échap (27)
            break

    cap.release()
    cv2.destroyAllWindows()

ex1()
