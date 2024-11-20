# -*- coding: utf-8 -*-
import streamlit as st
from img2table.ocr import AzureOCR
from img2table.document import Image
import cv2
from PIL import Image as PILImage
import numpy as np
import tempfile
import time

# Azure OCR credentials (replace with your actual key and endpoint)
subscription_key = "gMYpHRCnHqA8r2MxdtL203rBZ3WLTg4qFlH9wkxU40441ZLI302qJQQJ99AKACGhslBXJ3w3AAAFACOG50Ds"
endpoint = "https://image-extration.cognitiveservices.azure.com/"

# Initialize AzureOCR
azure_ocr = AzureOCR(subscription_key=subscription_key, endpoint=endpoint)

# Retry logic for Azure OCR calls
def extract_table_with_retry(image, retries=3, delay=2):
    for attempt in range(retries):
        try:
            return image.extract_tables(
                ocr=azure_ocr,
                implicit_rows=True,
                borderless_tables=False,
                min_confidence=30
            )
        except Exception as e:
            if 'Too Many Requests' in str(e):
                st.warning(f"Rate limit exceeded. Retrying in {delay} seconds... (Attempt {attempt + 1}/{retries})")
                time.sleep(delay)
            else:
                raise e
    raise Exception("Max retries exceeded for Azure OCR.")

# Streamlit app layout
st.title("Azure OCR Table Extraction with Retry Logic")
st.markdown("Upload an image to extract tables using Azure OCR. Handles rate limits gracefully.")

# Upload image
uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    # Display uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    cv_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(cv_img, caption="Uploaded Image", use_column_width=True)

    # Process the uploaded image
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        image_path = tmp_file.name

    img = Image(src=image_path)

    # Extract tables using Azure OCR with retry logic
    try:
        extracted_tables = extract_table_with_retry(img, retries=3, delay=5)
    except Exception as e:
        st.error(f"Failed to extract tables: {e}")
        extracted_tables = None

    # Display results
    if not extracted_tables:
        st.warning("No tables were detected in the image.")
        
        # Optional: Debug OCR output
        try:
            ocr_output = img.extract_text(ocr=azure_ocr)
            st.text("Raw OCR Output:")
            st.write(ocr_output)
        except Exception as e:
            st.error(f"Failed to extract raw text: {e}")
    else:
        # Display and process extracted tables
        for i, table in enumerate(extracted_tables):
            st.markdown(f"### Table {i + 1}")
            st.markdown(table.html_repr(title=f"Extracted Table {i + 1}"), unsafe_allow_html=True)

            # Highlight table cells on the image
            for row in table.content.values():
                for cell in row:
                    cv2.rectangle(cv_img, (cell.bbox.x1, cell.bbox.y1), (cell.bbox.x2, cell.bbox.y2), (255, 0, 0), 2)

        # Display the image with highlighted table cells
        st.image(cv_img, caption="Detected Table Cells", use_column_width=True)

        # Allow users to download the tables as an Excel file
        excel_path = "extracted_tables.xlsx"
        img.to_xlsx(
            excel_path,
            ocr=azure_ocr,
            implicit_rows=True,
            borderless_tables=False,
            min_confidence=50
        )
        with open(excel_path, "rb") as excel_file:
            st.download_button(
                label="Download Extracted Tables as Excel",
                data=excel_file,
                file_name="extracted_tables.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
