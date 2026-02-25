import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt

# ---------- Page configuration ----------
st.set_page_config(page_title="Color Space Analyzer", layout="wide")

# ---------- Helper function for histograms ----------
def plot_histogram(img_rgb, space):
    """
    Plots the three channels of the given RGB image in the specified color space.
    Returns a matplotlib figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    
    if space == "RGB":
        # Direct RGB channels
        channels = ('Red', 'Green', 'Blue')
        colors = ('red', 'green', 'blue')
        data = img_rgb  # shape (h,w,3) RGB
        for i, (ax, ch, col) in enumerate(zip(axes, channels, colors)):
            ax.hist(data[:, :, i].ravel(), bins=256, range=(0, 255), color=col, alpha=0.7)
            ax.set_title(ch)
    
    elif space == "HSV":
        # Convert to HSV
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        channels = ('Hue', 'Saturation', 'Value')
        colors = ('purple', 'orange', 'gray')
        for i, (ax, ch, col) in enumerate(zip(axes, channels, colors)):
            ax.hist(img_hsv[:, :, i].ravel(), bins=256, range=(0, 255), color=col, alpha=0.7)
            ax.set_title(ch)
    
    else:  # YCbCr
        # Convert to YCrCb (OpenCV order: Y, Cr, Cb)
        img_ycc = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
        channels = ('Y (Luma)', 'Cr (R-chrominance)', 'Cb (B-chrominance)')
        colors = ('black', 'red', 'blue')
        for i, (ax, ch, col) in enumerate(zip(axes, channels, colors)):
            ax.hist(img_ycc[:, :, i].ravel(), bins=256, range=(0, 255), color=col, alpha=0.7)
            ax.set_title(ch)
    
    for ax in axes:
        ax.set_xlim(0, 255)
        ax.set_xlabel('Pixel value')
        ax.set_ylabel('Frequency')
    plt.tight_layout()
    return fig

# ---------- Custom CSS for modern look and dark/light mode ----------
def set_theme_style(is_dark):
    if is_dark:
        bg_color = "#1e1e2f"
        card_bg = "#2d2d44"
        text_color = "#ffffff"
        border_color = "#444"
    else:
        bg_color = "#f0f2f6"
        card_bg = "#ffffff"
        text_color = "#000000"
        border_color = "#ddd"
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {bg_color};
            color: {text_color};
            transition: background-color 0.3s, color 0.3s;
        }}
        .card {{
            background-color: {card_bg};
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            border: 1px solid {border_color};
            transition: transform 0.2s;
        }}
        .card:hover {{
            transform: scale(1.01);
        }}
        .stButton>button {{
            border-radius: 10px;
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 8px 16px;
            transition: background-color 0.3s;
        }}
        .stButton>button:hover {{
            background-color: #45a049;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Initialize session state for dark mode
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# Dark mode toggle in sidebar
with st.sidebar:
    st.markdown("## 🎨 Theme")
    dark_toggle = st.toggle("Dark Mode", value=st.session_state.dark_mode)
    if dark_toggle != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_toggle
        st.rerun()

# Apply the selected theme
set_theme_style(st.session_state.dark_mode)

# ---------- Title and description ----------
st.title("🎓 Advanced Color Space Analyzer")
st.markdown("Interactive tool to explain RGB, HSV and YCbCr channels to students.")

# ---------- Initialize session state for image ----------
if "img_bgr" not in st.session_state:
    st.session_state.img_bgr = None
if "img_rgb" not in st.session_state:
    st.session_state.img_rgb = None
if "file_name" not in st.session_state:
    st.session_state.file_name = None

# ---------- Sidebar controls ----------
with st.sidebar:
    st.markdown("## 📂 Image")
    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"], key="uploader"
    )

    if uploaded_file is not None:
        # Read image only if new file or if not already stored
        if st.session_state.file_name != uploaded_file.name:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            st.session_state.img_bgr = cv2.imdecode(file_bytes, 1)
            st.session_state.img_rgb = cv2.cvtColor(st.session_state.img_bgr, cv2.COLOR_BGR2RGB)
            st.session_state.file_name = uploaded_file.name
            st.success("Image loaded successfully!")
            # Reset sliders will be handled later by session state default values

    st.markdown("---")
    st.markdown("## 🎛️ Controls")

    # Color space selection
    color_space = st.selectbox("Color Space", ("RGB", "HSV", "YCbCr"), index=0)

    # Initialize default slider values if not set
    if "slider_defaults" not in st.session_state:
        st.session_state.slider_defaults = {
            "RGB": [1.0, 1.0, 1.0],
            "HSV": [0, 1.0, 1.0],
            "YCbCr": [1.0, 1.0, 1.0],
        }

    # Show sliders according to selected space
    if color_space == "RGB":
        st.markdown("### 🔴 Red, Green, Blue multipliers")
        r = st.slider("Red multiplier", 0.0, 2.0, st.session_state.slider_defaults["RGB"][0], 0.01)
        g = st.slider("Green multiplier", 0.0, 2.0, st.session_state.slider_defaults["RGB"][1], 0.01)
        b = st.slider("Blue multiplier", 0.0, 2.0, st.session_state.slider_defaults["RGB"][2], 0.01)
        params = [r, g, b]

    elif color_space == "HSV":
        st.markdown("### 🌈 Hue shift, Saturation & Value")
        h = st.slider("Hue shift (°)", -180, 180, st.session_state.slider_defaults["HSV"][0], 1)
        s = st.slider("Saturation multiplier", 0.0, 2.0, st.session_state.slider_defaults["HSV"][1], 0.01)
        v = st.slider("Value multiplier", 0.0, 2.0, st.session_state.slider_defaults["HSV"][2], 0.01)
        params = [h, s, v]

    else:  # YCbCr
        st.markdown("### 🎚️ Luma (Y) and Chrominance (Cb, Cr)")
        y = st.slider("Y multiplier", 0.0, 2.0, st.session_state.slider_defaults["YCbCr"][0], 0.01)
        cb = st.slider("Cb multiplier", 0.0, 2.0, st.session_state.slider_defaults["YCbCr"][1], 0.01)
        cr = st.slider("Cr multiplier", 0.0, 2.0, st.session_state.slider_defaults["YCbCr"][2], 0.01)
        params = [y, cb, cr]

    # Reset button
    if st.button("↺ Reset to defaults"):
        st.session_state.slider_defaults[color_space] = (
            [1.0, 1.0, 1.0] if color_space == "RGB" else
            [0, 1.0, 1.0] if color_space == "HSV" else
            [1.0, 1.0, 1.0]
        )
        st.rerun()

    st.markdown("---")
    st.markdown("### 📥 Download")
    download_btn = st.empty()  # placeholder for download button (shown only if image exists)

# ---------- Main content (only if image loaded) ----------
if st.session_state.img_bgr is not None:
    img_bgr = st.session_state.img_bgr
    img_rgb = st.session_state.img_rgb

    # Apply modifications based on selected space
    if color_space == "RGB":
        # RGB: apply multipliers directly
        modified = img_rgb.astype(np.float32)
        modified[:, :, 0] *= params[0]  # R
        modified[:, :, 1] *= params[1]  # G
        modified[:, :, 2] *= params[2]  # B
        modified = np.clip(modified, 0, 255).astype(np.uint8)
        modified_rgb = modified

    elif color_space == "HSV":
        # Convert BGR to HSV, apply adjustments
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + params[0]) % 180  # Hue shift
        hsv[:, :, 1] *= params[1]                        # Saturation
        hsv[:, :, 2] *= params[2]                        # Value
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        modified_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        modified_rgb = cv2.cvtColor(modified_bgr, cv2.COLOR_BGR2RGB)

    else:  # YCbCr
        # Convert BGR to YCrCb (OpenCV uses YCrCb order)
        ycc = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        ycc[:, :, 0] *= params[0]  # Y
        ycc[:, :, 1] *= params[2]  # Cr (index 1 in OpenCV)
        ycc[:, :, 2] *= params[1]  # Cb (index 2)
        ycc = np.clip(ycc, 0, 255).astype(np.uint8)
        modified_bgr = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
        modified_rgb = cv2.cvtColor(modified_bgr, cv2.COLOR_BGR2RGB)

    # ---------- Display images side by side ----------
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(img_rgb, caption="Original", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(modified_rgb, caption=f"Modified ({color_space})", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------- Histograms ----------
    with st.expander("📊 Show histograms (original vs modified)"):
        col_h1, col_h2 = st.columns(2)
        # Original histogram
        with col_h1:
            st.markdown("**Original channels**")
            fig_orig = plot_histogram(img_rgb, color_space)
            st.pyplot(fig_orig)
        # Modified histogram
        with col_h2:
            st.markdown("**Modified channels**")
            fig_mod = plot_histogram(modified_rgb, color_space)
            st.pyplot(fig_mod)

    # ---------- Download button ----------
    # Convert modified image to bytes for download
    pil_img = Image.fromarray(modified_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    with st.sidebar:
        download_btn.download_button(
            label="📸 Download modified image",
            data=byte_im,
            file_name="modified_image.png",
            mime="image/png"
        )

else:
    # No image loaded: show instructions
    st.info("👆 Please upload an image from the sidebar to begin.")