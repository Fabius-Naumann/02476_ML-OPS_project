import os

import requests
import streamlit as st
from google.cloud import run_v2
from PIL import Image


@st.cache_resource
def get_backend_url():
    """Get the URL of the backend service."""
    try:
        parent = "projects/<project>/locations/<region>"
        client = run_v2.ServicesClient()
        services = client.list_services(parent=parent)
        for service in services:
            if service.name.split("/")[-1] == "production-model":
                return service.uri
    except Exception as e:
        st.warning(f"Could not fetch backend URL from Cloud Run: {e}")

    # Fall back to environment variable
    backend_url = os.environ.get("BACKEND", None)
    if not backend_url:
        st.error("Backend URL not found. Please set the BACKEND environment variable.")
    return backend_url


def check_backend_health(backend_url):
    """Check if the backend is healthy and ready."""
    try:
        response = requests.get(f"{backend_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data
        return None
    except Exception:
        return None


def get_class_name(class_id):
    """Map class ID to traffic sign name based on your dataset."""
    class_names = {
        0: "Speed limit (5km/h)",
        1: "Speed limit (15km/h)",
        2: "Speed limit (30km/h)",
        3: "Speed limit (40km/h)",
        4: "Speed limit (50km/h)",
        5: "Speed limit (60km/h)",
        6: "Speed limit (70km/h)",
        7: "Speed limit (80km/h)",
        8: "Don't Go straight or left",
        9: "Give way",
        10: "Don't Go straight",
        11: "Don't Go Left",
        12: "Don't Go Left or Right",
        13: "Don't Go Right",
        14: "Don't overtake from Left",
        15: "No U-turn",
        16: "No Car",
        17: "No horn",
        18: "No entry",
        19: "No stopping",
        20: "Go straight or right",
        21: "Go straight",
        22: "Go Left",
        23: "Go Left or right",
        24: "Go Right",
        25: "Keep Left",
        26: "Keep Right",
        27: "Roundabout mandatory",
        28: "Watch out for cars",
        29: "Horn",
        30: "Bicycles crossing",
        31: "U-turn",
        32: "Road Divider",
        33: "Unknown6",
        34: "Danger Ahead",
        35: "Zebra Crossing",
        36: "Bicycles crossing",
        37: "Children crossing",
        38: "Dangerous curve to the left",
        39: "Dangerous curve to the right",
        40: "Steep descent / Steep downhill road ahead",
        41: "Falling rocks / Landslide area",
        42: "Roadwork / Men at work",
        43: "Go right or straight",
        44: "Go left or straight",
        45: "Merging traffic from the right",
        46: "ZigZag Curve",
        47: "Train Crossing",
        48: "Under Construction",
        49: "Uneven road ahead",
        50: "Fences",
        51: "Heavy Vehicle Accidents",
    }
    return class_names.get(class_id, f"Unknown Sign (Class {class_id})")


def main():
    st.set_page_config(page_title="Traffic Sign Classification", page_icon="🚦", layout="wide")

    st.title("🚦 Traffic Sign Classification")
    st.markdown("Upload a traffic sign image to get real-time predictions from our ML model")

    # Get backend URL
    backend_url = get_backend_url()

    # Create two columns for layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Upload Image")

        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a traffic sign image...",
            type=["jpg", "jpeg", "png"],
            help="Upload an image file (JPG, JPEG, or PNG)",
        )

        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Image info
            st.caption(f"📏 Size: {image.size[0]}x{image.size[1]} pixels")
            st.caption(f"📝 Format: {image.format}")

    with col2:
        st.subheader("🔮 Prediction Results")

        if uploaded_file is not None:
            # Predict button
            if st.button("🚀 Classify Traffic Sign", use_container_width=True, type="primary"):
                if not backend_url:
                    st.error("❌ Backend URL is not configured. Cannot make predictions.")
                else:
                    with st.spinner("🔄 Analyzing image..."):
                        try:
                            # Reset file pointer
                            uploaded_file.seek(0)

                            # Prepare the file for upload
                            files = {"image": (uploaded_file.name, uploaded_file, uploaded_file.type)}

                            # Send request to backend
                            response = requests.post(f"{backend_url}/predict", files=files, timeout=30)

                            if response.status_code == 200:
                                result = response.json()

                                # Extract data from response
                                predicted_class = result["predicted_class"]
                                probabilities = result["probabilities"]
                                num_classes = result["num_classes"]

                                # Display prediction
                                st.success("✅ Classification Complete!")

                                # Main prediction
                                class_name = get_class_name(predicted_class)
                                confidence = probabilities[predicted_class] * 100

                                st.metric(
                                    label="Predicted Traffic Sign",
                                    value=class_name,
                                    delta=f"{confidence:.1f}% confidence",
                                )

                                # Show top 3 predictions
                                st.markdown("### Top 3 Predictions")

                                # Get top 3 classes
                                top_3_indices = sorted(
                                    range(len(probabilities)), key=lambda i: probabilities[i], reverse=True
                                )[:3]

                                for idx in top_3_indices:
                                    prob = probabilities[idx] * 100
                                    name = get_class_name(idx)
                                    st.progress(prob / 100)
                                    st.caption(f"{name}: {prob:.2f}%")

                                # Full response in expandable section
                                with st.expander("🔍 View Full Response"):
                                    st.json(result)

                            elif response.status_code == 503:
                                st.error("❌ Model not loaded on backend. Please check backend status.")
                            elif response.status_code == 415:
                                st.error("❌ Unsupported file type. Please upload a valid image.")
                            elif response.status_code == 413:
                                st.error("❌ File too large. Please upload a smaller image.")
                            else:
                                st.error(f"❌ Error: Received status code {response.status_code}")
                                with st.expander("View error details"):
                                    st.code(response.text)

                        except requests.exceptions.ConnectionError:
                            st.error("❌ Could not connect to backend. Please check if the backend is running.")
                        except requests.exceptions.Timeout:
                            st.error("❌ Request timed out. The backend might be slow or unavailable.")
                        except Exception as e:
                            st.error(f"❌ An error occurred: {e!s}")
        else:
            st.info("👆 Upload an image to get started")

    # Sidebar with backend info and features
    with st.sidebar:
        st.header("ℹ️ System Information")

        # Backend status
        if backend_url:
            health_data = check_backend_health(backend_url)

            if health_data:
                status = health_data.get("status", "unknown")
                is_loaded = health_data.get("is_loaded", False)
                num_classes = health_data.get("num_classes")
                weights_file = health_data.get("weights_file", "N/A")

                if status == "ok" and is_loaded:
                    st.success("✅ Backend Online")
                    st.caption(f"**Model Classes:** {num_classes}")
                    st.caption(f"**Weights:** {weights_file.split('/')[-1]}")
                else:
                    st.warning("⚠️ Backend Not Ready")
                    if health_data.get("detail"):
                        st.caption(f"Error: {health_data['detail']}")

                st.caption(f"**URL:** {backend_url}")
            else:
                st.error("❌ Backend Unavailable")
                st.caption(f"**URL:** {backend_url}")
        else:
            st.warning("⚠️ Backend Not Configured")

        st.divider()

        st.header("📖 How to Use")
        st.markdown("""
        1. **Upload** a traffic sign image
        2. Click **Classify Traffic Sign**
        3. View the **prediction results**

        The model will identify the traffic sign and show confidence scores for all possible classes.
        """)

        st.divider()

        st.header("🎯 Model Info")
        st.markdown("""
        This application uses a PyTorch-based CNN model trained on traffic sign images.

        **Supported formats:**
        - JPG/JPEG
        - PNG

        **Max file size:** 5 MB
        """)

        # Link to documentation or admin features
        with st.expander("🔧 Admin Features"):
            st.markdown("""
            Backend admin endpoints available:
            - `/health` - Health check
            - `/admin/status` - Job status
            - `/admin/train_sync` - Train model
            - `/admin/evaluate_sync` - Evaluate model
            """)


if __name__ == "__main__":
    main()
