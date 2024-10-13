import os 
import firebase_admin
from firebase_admin import credentials, storage
import streamlit as st
import time

class FirebaseConnection:
    def __init__(self):
        firebase_secrets = st.secrets["firebase"]
        cred = credentials.Certificate({
            "type": firebase_secrets["type"],
            "project_id": firebase_secrets["project_id"],
            "private_key_id": firebase_secrets["private_key_id"],
            "private_key": firebase_secrets["private_key"].replace("\\n", "\n"),
            "client_email": firebase_secrets["client_email"],
            "client_id": firebase_secrets["client_id"],
            "auth_uri": firebase_secrets["auth_uri"],
            "token_uri": firebase_secrets["token_uri"],
            "auth_provider_x509_cert_url": firebase_secrets["auth_provider_x509_cert_url"],
            "client_x509_cert_url": firebase_secrets["client_x509_cert_url"]
        })
        try:
            firebase_admin.get_app()
        except ValueError:
            firebase_admin.initialize_app(cred, {
                'storageBucket': firebase_secrets["storage_bucket"]
            })

    def get_bucket(self):
        return storage.bucket()
    
    def check_local_dir(self, local_image_directory, file_count_check=10):
        if os.path.exists(local_image_directory):
            local_files = [f for f in os.listdir(local_image_directory) if os.path.isfile(os.path.join(local_image_directory, f))]
            if len(local_files) >= file_count_check:
                result = st.success(f"Local image directory found with more than {file_count_check} files. Skipping download.", icon="👌")
                return result, True
            else:
                result = st.warning(f"Local image directory found with fewer than {file_count_check} files. Proceeding with download.", icon="🚨")
                return result, False
        else:
            result = st.warning("No local image directory found. Proceeding with download.", icon="🚨")
        return result, False

    def download_images(self, blobs, local_image_directory, max_download=5):
        progress_bar = st.progress(0)
        total_files = min(len(blobs), max_download)
        blobs = [blob for blob in blobs if not blob.name.endswith('/')]
        for index, blob in enumerate(blobs[:max_download]):
            local_path = os.path.join(local_image_directory, os.path.basename(blob.name))
            blob.download_to_filename(local_path)
            progress_text = "Downloading Images: {}/{}".format(index + 1, total_files)
            progress_bar.progress((index + 1) / total_files, text=progress_text)
        time.sleep(1)
        progress_bar.empty()
        st.success("Download complete!", icon="🎉")