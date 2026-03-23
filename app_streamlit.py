import streamlit as st
import cv2
import numpy as np
import requests
from PIL import Image
import io
import time

# ======================
# CONFIG
# ======================
BACKEND_URL = "http://127.0.0.1:5000/predict-image"
st.set_page_config(layout="wide")

# ======================
# STYLE
# ======================
st.markdown("""
<style>

body { background:#f7f7f4; }

.header{
background:#6f8249;
padding:30px;
border-radius:15px;
color:white;
text-align:center;
}

.card{
background:white;
padding:25px;
border-radius:12px;
box-shadow:0px 0px 10px rgba(0,0,0,0.1);
}

.title{
font-size:24px;
color:#6f8249;
font-weight:bold;
border-bottom:2px solid #6f8249;
margin-bottom:10px;
}

.section{
border:2px solid #6f8249;
padding:20px;
border-radius:12px;
}

</style>
""", unsafe_allow_html=True)

# ======================
# HEADER
# ======================
st.markdown("""
<div class="header">
<h1>🫒 Olivia Hack 2026</h1>
<p>Détection intelligente des corps étrangers dans l’huile d’olive</p>
</div>
""", unsafe_allow_html=True)

st.write("")

# ======================
# INFO CARDS
# ======================
c1, c2 = st.columns(2)

with c1:
    st.markdown("""
    <div class="card">

    <div class="title">Objectif du Projet</div>
    <p>
    Système automatisé de détection des corps étrangers utilisant
    l’intelligence artificielle et la vision par ordinateur.
    </p>

    <div class="title">Contaminants détectés</div>
    <ul>
        <li>Insectes</li>
        <li>Plastique</li>
        <li>Verre</li>
        <li>Fibres textiles</li>
    </ul>

    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="card">
    <div class="title">Performance</div>
    <p>✔ Précision modèle : 98.5%</p>
    <p>✔ Temps moyen : 2 secondes</p>
    </div>
    """, unsafe_allow_html=True)

# ======================
# BACKEND
# ======================
def send_to_backend(img):
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")

    response = requests.post(
        BACKEND_URL,
        files={"image":("image.jpg",buffer.getvalue(),"image/jpeg")},
        timeout=5
    )

    pred = response.json()["prediction"][0][0]
    return pred

# ======================
# ZONE ANALYSE
# ======================
col_cam, col_up = st.columns(2)

# ---------------- CAMERA ----------------
with col_cam:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.subheader("📷 Caméra")

    start = st.button("▶ Démarrer Caméra")
    stop = st.button("⏹ Arrêter Caméra")

    cam_view = st.empty()
    result_cam = st.empty()
    percent_text = st.empty()
    bar_cam = st.empty()

    if "run_cam" not in st.session_state:
        st.session_state.run_cam = False

    if start:
        st.session_state.run_cam = True

    if stop:
        st.session_state.run_cam = False

    if st.session_state.run_cam:

        cap = cv2.VideoCapture(0)
        last_send = 0

        while st.session_state.run_cam:

            ret, frame = cap.read()
            if not ret:
                break

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv,(5,50,40),(30,255,255))
            mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))

            contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

            bottle = None

            for cnt in contours:
                if cv2.contourArea(cnt) > 8000:
                    x,y,w,h = cv2.boundingRect(cnt)
                    if h/w > 2:
                        bottle = frame[y:y+h, x:x+w]
                        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                        break

            if bottle is not None and time.time()-last_send>1:

                last_send=time.time()
                pil = Image.fromarray(cv2.cvtColor(bottle,cv2.COLOR_BGR2RGB))

                try:
                    pred = send_to_backend(pil)
                    percent = int(pred*100)

                    if pred < 0.5:
                        result_cam.success("✅ Bouteille conforme")
                    else:
                        result_cam.error("❌ Bouteille défectueuse")

                    percent_text.markdown(f"### 🔢 Confiance modèle : {percent}%")
                    bar_cam.progress(percent)

                except:
                    result_cam.error("Backend indisponible")

            cam_view.image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))

        cap.release()

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- UPLOAD ----------------
with col_up:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.subheader("📂 Image")

    uploaded = st.file_uploader("Glisser ou sélectionner une image",type=["jpg","png","jpeg"])

    result_up = st.empty()
    percent_up = st.empty()
    bar_up = st.empty()

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img,width=350)

        if st.button("🔍 Analyser image"):

            try:
                pred = send_to_backend(img)
                percent = int(pred*100)

                if pred < 0.5:
                    result_up.success("✅ Bouteille conforme")
                else:
                    result_up.error("❌ Bouteille défectueuse")

                percent_up.markdown(f"### 🔢 Confiance modèle : {percent}%")
                bar_up.progress(percent)

            except:
                result_up.error("Backend indisponible")

    st.markdown("</div>", unsafe_allow_html=True)
