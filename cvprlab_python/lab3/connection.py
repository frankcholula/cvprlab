import streamlit as st
import os 
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage

import streamlit as st
import firebase_admin
from firebase_admin import credentials, storage

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
