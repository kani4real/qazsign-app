"""
QazSign — Streamlit веб-қосымшасы
====================================
Үш бөлімнен тұрады (жоғарыдағы бетбелгілер/tabs арқылы ауысады):
  1) Деректер жинау   — камерадан қол қимылын түсіріп, CSV-ге жинау
  2) Модельді оқыту    — жиналған деректер бойынша MLP классификаторын үйрету
  3) Тікелей тану      — камера алдында қимылды нақты уақытта тану,
                          экранға қазақша мәтін шығару және дауыстау

Іске қосу (компьютерде сынау үшін):
    streamlit run app.py

Интернетке жариялау (бір сілтеме арқылы ашылатындай) үшін
осы қалтаны GitHub-қа жүктеп, Streamlit Community Cloud-қа
қосу керек — толық нұсқаулық README.md файлында.
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import av
import cv2
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from gtts import gTTS
import io

from qazsign_features import calculate_features, DEFAULT_GESTURES, KAZ_TEXT, FEATURE_NAMES

# ---------------------------------------------------------------
# Жалпы баптаулар
# ---------------------------------------------------------------
st.set_page_config(page_title="QazSign", page_icon="🤟", layout="wide")

DATA_PATH = "qazsign_dataset.csv"
MODEL_PATH = "qazsign_model.joblib"
ENCODER_PATH = "qazsign_encoder.joblib"

RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

if "gestures" not in st.session_state:
    st.session_state.gestures = list(DEFAULT_GESTURES)
if "dataset_rows" not in st.session_state:
    # егер бұрын сақталған csv болса — соны жүктейміз
    if os.path.isfile(DATA_PATH):
        st.session_state.dataset_rows = pd.read_csv(DATA_PATH).values.tolist()
    else:
        st.session_state.dataset_rows = []


# ---------------------------------------------------------------
# Камера процессоры — деректер жинау режимі (тек landmark сурет салу)
# ---------------------------------------------------------------
class CollectProcessor:
    def __init__(self):
        self.hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.landmarks = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        self.landmarks = None
        if result.multi_hand_landmarks:
            hl = result.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(img, hl, mp_hands.HAND_CONNECTIONS)
            self.landmarks = [[lm.x, lm.y, lm.z] for lm in hl.landmark]

        cv2.putText(img, "Qol korinip tur ma? -> joğarыdaғы belgi",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if self.landmarks else (0, 0, 255), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------------------------------------------------------------
# Камера процессоры — тану режимі (модель арқылы болжам жасау)
# ---------------------------------------------------------------
class RecognizeProcessor:
    def __init__(self):
        self.hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.model = None
        self.encoder = None
        self.last_word = None
        self.last_conf = 0.0
        
        # Добавляем счетчик кадров и кэш текста
        self.frame_count = 0
        self.current_text = "..."
        
        if os.path.isfile(MODEL_PATH) and os.path.isfile(ENCODER_PATH):
            self.model = joblib.load(MODEL_PATH)
            self.encoder = joblib.load(ENCODER_PATH)

    def recv(self, frame):
        self.frame_count += 1
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        # ОПТИМИЗАЦИЯ: прогоняем через модель только каждый 3-й кадр
        if self.frame_count % 3 == 0:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = self.hands.process(rgb)

            if result.multi_hand_landmarks and self.model is not None:
                hl = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(img, hl, mp_hands.HAND_CONNECTIONS)
                feats = np.array([calculate_features(hl.landmark)])
                probs = self.model.predict_proba(feats)[0]
                idx = int(np.argmax(probs))
                conf = float(probs[idx])
                
                if conf >= 0.6:
                    word = self.encoder.inverse_transform([idx])[0]
                    self.current_text = f"{KAZ_TEXT.get(word, word)}  ({conf*100:.0f}%)"
                    self.last_word = word
                    self.last_conf = conf
                else:
                    self.current_text = "..."
            elif self.model is None:
                self.current_text = "Alдымен modeldi oqytyңыз!"
            else:
                self.current_text = "..."

        # Рисуем черную плашку и текст абсолютно на каждом кадре (чтобы не мерцало)
        cv2.rectangle(img, (0, 0), (img.shape[1], 55), (0, 0, 0), -1)
        cv2.putText(img, self.current_text, (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def speak_kazakh_word(word):
    """gTTS арқылы дауыс файлын жасап, браузерде ойнатады."""
    text_to_say = KAZ_TEXT.get(word, word)
    try:
        tts = gTTS(text=text_to_say, lang="ru")  # қазақша толық қолдау болмаса, ru жақын дыбысталады
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        st.audio(buf, format="audio/mp3", autoplay=True)
    except Exception as e:
        st.warning(f"Дыбыстау мүмкін болмады (интернет керек): {e}")


# =================================================================
# БЕТБЕЛГІЛЕР (TABS)
# =================================================================
st.title("🤟 QazSign — қазақ ым-ишара тілін тану жүйесі")

tab1, tab2, tab3 = st.tabs(["1️⃣ Деректер жинау", "2️⃣ Модельді оқыту", "3️⃣ Тікелей тану"])

# -----------------------------------------------------------------
# TAB 1 — ДЕРЕКТЕР ЖИНАУ
# -----------------------------------------------------------------
with tab1:
    st.header("Деректер жинау")
    st.write("Камераны қосып, әр ым-ишара сөз үшін қолыңызды әртүрлі бұрышпен көрсетіп, "
             "**«Кадрды жазу»** батырмасын 40–60 рет басыңыз.")

    colA, colB = st.columns([1, 1])
    with colA:
        new_word = st.text_input("Жаңа сөз қосу (ағылшын әріптерімен, мыс. 'ana')")
        if st.button("➕ Сөздікке қосу") and new_word.strip():
            w = new_word.strip().lower().replace(" ", "_")
            if w not in st.session_state.gestures:
                st.session_state.gestures.append(w)
                st.success(f"'{w}' сөздікке қосылды.")

        current_word = st.selectbox("Қазір жинайтын сөз:", st.session_state.gestures)

    with colB:
        df_counts = pd.DataFrame(st.session_state.dataset_rows,
                                  columns=FEATURE_NAMES + ["label"]) if st.session_state.dataset_rows else pd.DataFrame(columns=["label"])
        if not df_counts.empty:
            st.write("Жиналған үлгі саны (сөз бойынша):")
            st.dataframe(df_counts["label"].value_counts().rename("саны"))
        else:
            st.info("Әзірге деректер жиналған жоқ.")

    ctx = webrtc_streamer(
        key="collect",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_processor_factory=CollectProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📸 Кадрды жазу", type="primary", use_container_width=True):
            if ctx.video_processor and ctx.video_processor.landmarks:
                feats = calculate_features(ctx.video_processor.landmarks)
                st.session_state.dataset_rows.append(feats + [current_word])
                st.toast(f"'{current_word}' үшін жазылды! (барлығы: "
                         f"{sum(1 for r in st.session_state.dataset_rows if r[-1]==current_word)})")
            else:
                st.warning("Қол көрінбей тұр — камераға қолыңызды көрсетіңіз.")
    with col2:
        if st.button("💾 CSV-ге сақтау", use_container_width=True):
            if st.session_state.dataset_rows:
                df = pd.DataFrame(st.session_state.dataset_rows, columns=FEATURE_NAMES + ["label"])
                df.to_csv(DATA_PATH, index=False)
                st.success(f"Сақталды: {DATA_PATH} ({len(df)} жол)")
            else:
                st.warning("Алдымен деректер жинаңыз.")
    with col3:
        if st.button("🗑️ Барлығын тазалау", use_container_width=True):
            st.session_state.dataset_rows = []
            st.rerun()

    if st.session_state.dataset_rows:
        df_export = pd.DataFrame(st.session_state.dataset_rows, columns=FEATURE_NAMES + ["label"])
        st.download_button("⬇️ Deректерді CSV ретінде жүктеп алу (сақтық көшірме)",
                            df_export.to_csv(index=False).encode("utf-8"),
                            file_name="qazsign_dataset.csv", mime="text/csv")

    uploaded = st.file_uploader("Немесе бұрын сақталған CSV файлды жүктеу", type="csv")
    if uploaded is not None:
        df_up = pd.read_csv(uploaded)
        st.session_state.dataset_rows = df_up.values.tolist()
        st.success(f"{len(df_up)} жол жүктелді.")
        st.rerun()

# -----------------------------------------------------------------
# TAB 2 — МОДЕЛЬДІ ОҚЫТУ
# -----------------------------------------------------------------
with tab2:
    st.header("Модельді оқыту")
    n_rows = len(st.session_state.dataset_rows)
    st.write(f"Қазір жадыда **{n_rows}** үлгі бар.")

    hidden1 = st.slider("1-жасырын қабат нейрон саны", 8, 256, 64, step=8)
    hidden2 = st.slider("2-жасырын қабат нейрон саны", 8, 128, 32, step=8)
    max_iter = st.slider("Оқыту итерациясы (epochs)", 50, 1000, 300, step=50)

    if st.button("🚀 Модельді оқыту", type="primary"):
        if n_rows < 20:
            st.error("Деректер тым аз (кемінде ~20 үлгі керек). Алдымен 1-бетбелгіде деректер жинаңыз.")
        else:
            df = pd.DataFrame(st.session_state.dataset_rows, columns=FEATURE_NAMES + ["label"])
            X = df[FEATURE_NAMES].values
            y_raw = df["label"].values

            encoder = LabelEncoder()
            y = encoder.fit_transform(y_raw)

            if len(np.unique(y)) < 2:
                st.error("Кемінде 2 түрлі сөз болу керек.")
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42,
                    stratify=y if min(np.bincount(y)) >= 2 else None,
                )

                model = MLPClassifier(hidden_layer_sizes=(hidden1, hidden2),
                                       activation="relu", solver="adam",
                                       max_iter=max_iter, random_state=42)
                with st.spinner("Оқытылуда..."):
                    model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                joblib.dump(model, MODEL_PATH)
                joblib.dump(encoder, ENCODER_PATH)

                st.session_state.model = model
                st.session_state.encoder = encoder

                st.success(f"Дайын! Тест дәлдігі (accuracy): **{acc*100:.1f}%**")

                st.text("Толық есеп (classification report):")
                st.text(classification_report(y_test, y_pred, target_names=encoder.classes_))

                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=encoder.classes_, yticklabels=encoder.classes_, ax=ax)
                ax.set_xlabel("Болжанған")
                ax.set_ylabel("Нақты")
                st.pyplot(fig)

                st.info("Модель сақталды. 3-бетбелгідегі камераны **тоқтатып, қайта іске қосыңыз** "
                        "(Stop → Start), сонда жаңа модель жүктеледі.")

    if os.path.isfile(MODEL_PATH):
        st.caption(f"✅ Дискіде сақталған модель бар: `{MODEL_PATH}`")

# -----------------------------------------------------------------
# TAB 3 — ТІКЕЛЕЙ ТАНУ
# -----------------------------------------------------------------
with tab3:
    st.header("Тікелей тану және аудару")
    if not os.path.isfile(MODEL_PATH):
        st.warning("Әзірге оқытылған модель жоқ. Алдымен 2-бетбелгіде модельді оқытыңыз.")

    ctx2 = webrtc_streamer(
        key="recognize",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_processor_factory=RecognizeProcessor,
        media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False
        }

    if st.button("🔊 Танылған сөзді дауыстау"):
        if ctx2.video_processor and ctx2.video_processor.last_word:
            speak_kazakh_word(ctx2.video_processor.last_word)
        else:
            st.warning("Әзірге ешбір сөз танылған жоқ.")

    st.caption("Кеңес: жарық жеткілікті бөлмеде, қолды камераға жақынырақ ұстап көріңіз.")
