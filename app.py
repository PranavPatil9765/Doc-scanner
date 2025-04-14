import streamlit as st
import numpy as np
import cv2
from scipy.fftpack import dct, idct
from PIL import Image
import tempfile

# --- Page Configuration ---
st.set_page_config(page_title="Document Scanner & Compressor", layout="wide", page_icon="📄")

# --- Custom Light Styling ---
st.markdown("""
    <style>
        .stApp {
            background-color: #f9fafb;
            font-family: 'Segoe UI', sans-serif;
            color: #111827;
        }
        h1, h2, h3, h4, h5, h6, label, p, span, div {
            color: #111827 !important;
        }
        .block-container {
            padding: 2rem 3rem;
        }
        .card {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            margin-bottom: 2rem;
            border: 1px solid #e5e7eb;
            color: #111827 !important;
        }
        .step-label {
            font-weight: bold;
            color: #111827;
            font-size: 1.1rem;
        }
        .info {
            color: #374151;
            font-size: 0.95rem;
            margin-top: 0.5rem;
        }
        .data-block {
            background-color: #f3f4f6;
            padding: 0.6rem 1rem;
            border-radius: 8px;
            margin: 0.4rem 0;
            border: 1px solid #e5e7eb;
            color: #111827;
        }
        button {
            background-color: #3b82f6 !important;
            color: white !important;
            font-weight: bold;
            border-radius: 8px;
        }
        button:hover {
            background-color: #2563eb !important;
        }
    </style>
""", unsafe_allow_html=True)


st.markdown("<h1 style='text-align:center'>📄 DocuScan Pro</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color: #4b5563;'>A clean document scanner & compressor using DCT + RLE</h4>", unsafe_allow_html=True)
st.markdown("---")

# --- Utility Functions ---
def to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def to_binary(gray):
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return binary

def rle_encode(img):
    flat = img.flatten()
    result = []
    count = 1
    for i in range(1, len(flat)):
        if flat[i] == flat[i-1]:
            count += 1
        else:
            result.append((flat[i-1], count))
            count = 1
    result.append((flat[-1], count))
    return result

def rle_ratio(original_img, rle):
    orig_size = original_img.size
    compressed_size = len(rle) * 2
    ratio = round(orig_size / compressed_size, 2) if compressed_size != 0 else 0
    return orig_size, compressed_size, ratio

def apply_color_dct(image):
    image = np.float32(image) / 255.0
    dct_channels = []
    for i in range(3):
        channel = image[:, :, i]
        dct_ch = dct(dct(channel.T, norm='ortho').T, norm='ortho')
        dct_channels.append(dct_ch)
    return dct_channels

def apply_color_idct(dct_channels):
    reconstructed_channels = []
    for ch in dct_channels:
        idct_ch = idct(idct(ch.T, norm='ortho').T, norm='ortho')
        idct_ch = np.clip(idct_ch, 0, 1)
        reconstructed_channels.append(idct_ch)
    reconstructed = np.stack(reconstructed_channels, axis=2)
    return np.uint8(reconstructed * 255)

def detect_and_crop_document(image):
    gray = to_grayscale(image)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for c in contours:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4:
            doc_cnt = approx
            break
    else:
        return image

    pts = doc_cnt.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

    dst = np.array([[0, 0], [width - 1, 0],
                    [width - 1, height - 1], [0, height - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (width, height))
    return warped

# --- App Layout ---
left, right = st.columns([1, 2])

with left:
    st.markdown("### 🔹 Step 1: Upload File")
    option = st.radio("Choose Input Type:", ("Upload Image", "Upload Video"))

    image = None
    file = None

    if option == "Upload Image":
        file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
        if file:
            image = np.array(Image.open(file).convert("RGB"))

    elif option == "Upload Video":
        file = st.file_uploader("Upload a Video", type=["mp4", "avi"])
        if file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(file.read())
            cap = cv2.VideoCapture(tfile.name)
            ret, frame = cap.read()
            cap.release()
            if ret:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

with right:
    if image is not None:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📷 Original Preview")
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### ✂️ Detected & Cropped Document")
        doc_image = detect_and_crop_document(image)
        st.image(doc_image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # --- Processing ---
        gray = to_grayscale(doc_image)
        binary = to_binary(gray)

        # RLE Compression
        rle = rle_encode(binary)
        orig_size, rle_size, rle_ratio_val = rle_ratio(binary, rle)

        # DCT Compression
        dct_channels = apply_color_dct(doc_image)
        reconstructed = apply_color_idct(dct_channels)
        dct_size = sum([np.count_nonzero(np.round(ch, 2)) for ch in dct_channels])
        dct_orig_size = np.prod(doc_image.shape)
        dct_ratio_val = round(dct_orig_size / dct_size, 2) if dct_size != 0 else 0

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📦 Compressed Output")
        st.image(reconstructed, use_container_width=True)

        st.markdown(f"<div class='step-label'>Compression Stats</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='data-block'>🧾 Original Size (pixels): {orig_size}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='data-block'>🧮 DCT Compressed Size (non-zero coeffs): {dct_size}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='data-block'>🎯 DCT Compression Ratio: {dct_ratio_val}x</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='data-block'>🗜️ RLE Compressed Size (approx): {rle_size}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='data-block'>📉 RLE Compression Ratio: {rle_ratio_val}x</div>", unsafe_allow_html=True)

        def rle_decode(rle, shape):
            decoded = []
            for value, count in rle:
                decoded.extend([value] * count)
            return np.array(decoded, dtype=np.uint8).reshape(shape)

        rle_decoded = rle_decode(rle, binary.shape)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🧵 RLE Decoded (Binary Image)")
        st.image(rle_decoded, use_container_width=True, clamp=True)
        st.markdown('</div>', unsafe_allow_html=True)

        result_img = Image.fromarray(reconstructed)
        img_bytes = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        result_img.save(img_bytes.name)
        with open(img_bytes.name, "rb") as f:
            st.download_button("⬇️ Download Compressed Document", f, file_name="compressed_doc.png", mime="image/png")

        st.markdown('</div>', unsafe_allow_html=True)
