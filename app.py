import streamlit as st
import cv2
import numpy as np
from PIL import Image
import re

# ==========================================
# KONFIGURASI HALAMAN STREAMLIT
# ==========================================
st.set_page_config(page_title="Scanner Gizi Cerdas", page_icon="🍬", layout="wide")
st.title("🍬 Scanner Informasi Nilai Gizi")
st.write("Unggah foto kemasan jajananmu, dan biarkan AI mendeteksi kandungan Gula & Garam secara otomatis!")

# ==========================================
# CACHE MODEL (Agar RAM tidak meledak di-load berulang kali)
# ==========================================
@st.cache_resource
def load_models():
    import tensorflow as tf
    from paddleocr import PaddleOCR
    # Load Model V3 (Deteksi Kotak)
    model_deteksi = tf.keras.models.load_model('bestv3.1.keras')
    # Load PaddleOCR (Ekstraksi Teks)
    ocr_engine = PaddleOCR(use_textline_orientation=True, lang='id')
    return model_deteksi, ocr_engine

model_deteksi, ocr_engine = load_models()

# ==========================================
# FUNGSI PENDUKUNG (SMART BBOX)
# ==========================================
def get_center_y(bbox):
    try:
        if isinstance(bbox[0], (list, tuple)): return (bbox[0][1] + bbox[2][1]) / 2
        else: return (bbox[1] + bbox[3]) / 2
    except Exception:
        return 0 

def extract_nutrition_value_smart(lines, keywords):
    num_pattern = re.compile(r'(<\s*)?([\d]+(?:[.,]\d+)?)\s*(mg|g|kkal|kcal|%)', re.IGNORECASE)
    target_item = None
    for item in lines:
        if any(kw.lower() in item['text'].lower() for kw in keywords):
            target_item = item
            break
            
    if not target_item: return None

    # Cek di baris yang sama
    match = num_pattern.search(target_item['text'])
    if match:
        return f"{match.group(2).replace(',', '.')} {match.group(3)}"

    # Cek baris sejajar
    target_y = get_center_y(target_item['bbox'])
    kandidat_sejajar = []
    for item in lines:
        if item == target_item: continue
        match = num_pattern.search(item['text'])
        if match:
            y_diff = abs(target_y - get_center_y(item['bbox']))
            if y_diff < 25:
                kandidat_sejajar.append({'y_diff': y_diff, 'match': match})

    if kandidat_sejajar:
        kandidat_sejajar.sort(key=lambda x: x['y_diff'])
        terbaik = kandidat_sejajar[0]['match']
        return f"{terbaik.group(2).replace(',', '.')} {terbaik.group(3)}"

    return "Angka tidak sejajar"

# ==========================================
# UI: UPLOAD & PROSES GAMBAR
# ==========================================
uploaded_file = st.file_uploader("Pilih foto kemasan (JPG/PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file)
    img_asli = np.array(image_pil)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Foto Asli")
        st.image(image_pil, use_container_width=True)

    with st.spinner('Menganalisis gambar dan mencari tabel gizi...'):
        img_rgb = img_asli if len(img_asli.shape) == 3 and img_asli.shape[2] == 3 else cv2.cvtColor(img_asli, cv2.COLOR_BGR2RGB)
        h_asli, w_asli, _ = img_rgb.shape

        img_input = cv2.resize(img_rgb, (640, 640)) / 255.0
        img_input = np.expand_dims(img_input, axis=0)

        prediksi = model_deteksi.predict(img_input, verbose=0)[0]
        x1 = max(0, int(prediksi[0] * w_asli))
        y1 = max(0, int(prediksi[1] * h_asli))
        x2 = min(w_asli, int(prediksi[2] * w_asli))
        y2 = min(h_asli, int(prediksi[3] * h_asli))

        potongan_ocr = img_rgb[y1:y2, x1:x2]
        
        with col2:
            st.subheader("✂️ Hasil Potong Tabel Gizi")
            st.image(potongan_ocr, use_container_width=True)

    with st.spinner('Membaca teks menggunakan PaddleOCR...'):
        ocr_result = ocr_engine.ocr(potongan_ocr)
        
        all_lines = []
        if ocr_result and isinstance(ocr_result[0], dict):
            for i in range(len(ocr_result[0].get('rec_texts', []))):
                all_lines.append({
                    'text': ocr_result[0]['rec_texts'][i], 
                    'bbox': ocr_result[0]['rec_boxes'][i].tolist()
                })
        elif ocr_result and isinstance(ocr_result[0], list):
            for line_info in ocr_result[0]:
                if len(line_info) == 2:
                    all_lines.append({'text': line_info[1][0], 'bbox': line_info[0]})

        # 4. Ekstraksi Nilai Gizi
        hasil_gula = extract_nutrition_value_smart(all_lines, ['gula'])
        hasil_garam = extract_nutrition_value_smart(all_lines, ['garam', 'natrium'])

    # Tampilkan Hasil Akhir
    st.divider()
    st.subheader("💡 Hasil Ekstraksi Gizi")
    
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric(label="🍭 Gula", value=hasil_gula if hasil_gula else "Tidak Ditemukan")
    with metric_col2:
        st.metric(label="🧂 Garam / Natrium", value=hasil_garam if hasil_garam else "Tidak Ditemukan")