import streamlit as st
import os 
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage

# Replace with the path to your downloaded JSON key file
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

firebase_admin.initialize_app(cred, {
    'storageBucket': firebase_secrets["storageBucket"]
})

bucket = storage.bucket()

DATASET_FOLDER = "MSRC_ObjCategImageDatabase_v2"
DESCRIPTOR_FOLDER = "descriptors"
# Example: Download an image file
blob = bucket.blob(os.path.join(DATASET_FOLDER, 'Images', '1_1_s.bmp'))
blob.download_to_filename(os.path.join(DESCRIPTOR_FOLDER, 'fire_base_image.bmp'))
