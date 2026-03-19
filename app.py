import streamlit as st
import cv2

st.title("Inspection intelligente - Huile d'olive 🫒")

# Bouton détecter
start = st.button("🔍 Détecter")

if start:
    st.text("Webcam ouverte... appuyez sur Q pour fermer")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Caméra non détectée")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Erreur capture")
                break

            cv2.imshow("Camera - Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
