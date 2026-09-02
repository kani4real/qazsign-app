"""
QazSign — ортақ модуль: MediaPipe landmark-тарынан
инвариантты сандық белгілер (features) есептеу.

Бұл функция data_collection, train_model, realtime_recognition
скрипттерінің үшеуінде де бірдей қолданылады — сондықтан оны
бөлек файлға шығарып алдық (кодтың қайталанбауы үшін).

Теориялық негіз: қолдың "конфигурациясын" (C) сипаттайтын
G = f(C, L, M, O) моделінің C бөлігі — саусақтардың бір-біріне
қатысты орналасуы. Тікелей (x, y, z) координаттарын пайдаланбай,
біз екі инвариантты белгі түрін есептейміз:

  1) Әр саусақ ұшының білекке (0-нүкте) дейінгі қашықтығы
     (Евклид формуласы: d = sqrt((x1-x2)^2+(y1-y2)^2+(z1-z2)^2))
  2) Әр саусақтың ортаңғы буынындағы бүгілу бұрышы
     (вектор алгебрасы: cos(theta) = (a·b)/(|a|*|b|))

Бұл екеуі де камераға қашық/жақын тұрудан (масштабтан) және
қолдың кадр ішіндегі орнынан (позициядан) тәуелсіз — сондықтан
модель әртүрлі жағдайда да тұрақты жұмыс істейді.
"""

import numpy as np

# Негізгі сөздік — қажет болса осында жаңа сөз қосуға болады
DEFAULT_GESTURES = ["salem", "raqmet", "iya", "joq", "komek"]

KAZ_TEXT = {
    "salem": "Сәлем!",
    "raqmet": "Рақмет!",
    "iya": "Иә",
    "joq": "Жоқ",
    "komek": "Көмек керек",
}

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_JOINTS = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12),
                  (13, 14, 15, 16), (17, 18, 19, 20)]

FEATURE_NAMES = (
    [f"dist_{i}" for i in range(len(FINGER_TIPS))]
    + [f"angle_{i}" for i in range(len(FINGER_JOINTS))]
)


def calculate_features(landmark_xyz):
    """
    landmark_xyz: MediaPipe hand_landmarks.landmark нысаны НЕМЕСЕ
                  21 x 3 өлшемді list/array (әр нүкте үшін x,y,z).
    Қайтарады: ұзындығы 10 болатын сандар тізімі (5 қашықтық + 5 бұрыш).
    """
    if hasattr(landmark_xyz, "__iter__") and hasattr(next(iter(landmark_xyz)), "x"):
        pts = np.array([[lm.x, lm.y, lm.z] for lm in landmark_xyz])
    else:
        pts = np.array(landmark_xyz)

    base = pts[0]
    pts = pts - base
    scale = np.linalg.norm(pts[9])
    if scale > 0:
        pts = pts / scale

    features = []
    for tip in FINGER_TIPS:
        dist = np.linalg.norm(pts[tip] - pts[0])
        features.append(float(dist))

    for a, b, c, d in FINGER_JOINTS:
        v1 = pts[a] - pts[b]
        v2 = pts[c] - pts[b]
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        features.append(angle)

    return features
