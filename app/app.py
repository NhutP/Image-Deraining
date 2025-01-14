# import streamlit as st
# import requests
# from PIL import Image
# from io import BytesIO

# # Streamlit app title
# st.title("Image Upload and Processing App")

# # FastAPI endpoint URL
# API_URL = "http://localhost:8000/process-image"  # Replace with your FastAPI endpoint

# # File uploader for the user to upload an image
# uploaded_file = st.file_uploader("Upload an image")#, type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Display the uploaded image
#     st.subheader("Uploaded Image")
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     # Convert the image to bytes
#     img_bytes = BytesIO()
#     image.save(img_bytes, format=image.format)
#     img_bytes = img_bytes.getvalue()

#     # Button to send the image to the API for processing
#     if st.button("Process Image"):
#         st.subheader("Processing...")

#         # Send the image to the API
#         try:
#             response = requests.post(API_URL, files={"file": img_bytes})
#             if response.status_code == 200:
#                 # Load and display the processed image
#                 processed_img = Image.open(BytesIO(response.content))
#                 st.subheader("Processed Image")
#                 st.image(processed_img, caption="Processed Image", use_column_width=True)
#             else:
#                 st.error(f"Error: {response.status_code} - {response.text}")
#         except Exception as e:
#             st.error(f"An error occurred: {e}")

# import streamlit as st
# import requests
# from PIL import Image
# from io import BytesIO

# # Streamlit app title
# st.title("Image Upload and Processing App")

# # FastAPI endpoint URLs for two different processing methods
# API_URL_METHOD1 = "http://localhost:8000/process-image"  # Replace with your FastAPI endpoint for method 1
# API_URL_METHOD2 = "http://localhost:8001/process-image"  # Replace with your FastAPI endpoint for method 2

# # File uploader for the user to upload an image
# uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Display the uploaded image
#     st.subheader("Uploaded Image")
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     # Convert the image to bytes
#     img_bytes = BytesIO()
#     image.save(img_bytes, format=image.format)
#     img_bytes = img_bytes.getvalue()

#     # Checkboxes for selecting the processing methods
#     use_method1 = st.checkbox("Use DGUNet")
#     use_method2 = st.checkbox("Use Restormer")

#     # Submit button to process the image
#     if st.button("Submit"):
#         if use_method1 or use_method2:
#             # Create columns for side-by-side display of results
#             col1, col2 = st.columns(2)

#             if use_method1:
#                 with col1:
#                     st.subheader("Processing with DGUNet...")
#                     try:
#                         response = requests.post(API_URL_METHOD1, files={"file": img_bytes})
#                         if response.status_code == 200:
#                             # Load and display the processed image from method 1
#                             processed_img1 = Image.open(BytesIO(response.content))
                            
#                             # Resize processed image to match original image size
#                             processed_img1 = processed_img1.resize(image.size)
                            
#                             st.subheader("Processed Image - DGUNet")
#                             st.image(processed_img1, caption="Processed Image - DGUNet", use_column_width=True)
#                         else:
#                             st.error(f"Error: {response.status_code} - {response.text}")
#                     except Exception as e:
#                         st.error(f"An error occurred: {e}")

#             if use_method2:
#                 with col2:
#                     st.subheader("Processing with Restormer...")
#                     try:
#                         response = requests.post(API_URL_METHOD2, files={"file": img_bytes})
#                         if response.status_code == 200:
#                             # Load and display the processed image from method 2
#                             processed_img2 = Image.open(BytesIO(response.content))
                            
#                             # Resize processed image to match original image size
#                             processed_img2 = processed_img2.resize(image.size)
                            
#                             st.subheader("Processed Image - Restormer")
#                             st.image(processed_img2, caption="Processed Image - Restormer", use_column_width=True)
#                         else:
#                             st.error(f"Error: {response.status_code} - {response.text}")
#                     except Exception as e:
#                         st.error(f"An error occurred: {e}")
#         else:
#             st.warning("Please select at least one method to process the image.")


import streamlit as st
import requests
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

# Streamlit app title
st.title("Image Deraining App")

# FastAPI endpoint URLs for two different processing methods
API_URL_METHOD1 = "http://localhost:8000/process-image"  # Replace with your FastAPI endpoint for method 1
API_URL_METHOD2 = "http://localhost:8001/process-image"  # Replace with your FastAPI endpoint for method 2

# Helper function to send API requests
def process_image(api_url, img_bytes):
    try:
        response = requests.post(api_url, files={"file": img_bytes})
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"An error occurred: {e}"

# File uploader for the user to upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.subheader("Uploaded Image")
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert the image to bytes
    img_bytes = BytesIO()
    image.save(img_bytes, format=image.format)
    img_bytes = img_bytes.getvalue()

    # Checkboxes for selecting the processing methods
    use_method1 = st.checkbox("Use DGUNet")
    use_method2 = st.checkbox("Use Restormer")

    # Submit button to process the image
    if st.button("Submit"):
        if use_method1 or use_method2:
            with st.spinner("Processing the image..."):
                # Create columns for side-by-side display of results
                col1, col2 = st.columns(2)

                # Create a list of tasks for parallel execution
                tasks = []
                if use_method1:
                    tasks.append((API_URL_METHOD1, col1, "DGUNet"))
                if use_method2:
                    tasks.append((API_URL_METHOD2, col2, "Restormer"))

                # Process tasks in parallel
                with ThreadPoolExecutor() as executor:
                    futures = {executor.submit(process_image, api_url, img_bytes): (col, caption) for api_url, col, caption in tasks}

                    for future in futures:
                        col, caption = futures[future]
                        result = future.result()
                        if isinstance(result, Image.Image):
                            # Resize processed image to match original image size
                            result = result.resize(image.size)
                            col.subheader(caption)
                            col.image(result, caption=caption, use_column_width=True)
                        else:
                            col.error(result)
        else:
            st.warning("Please select at least one method to process the image.")

