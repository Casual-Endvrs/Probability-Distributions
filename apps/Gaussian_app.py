import streamlit as st
from distributions import c_Gaussian


def app():
    if "dist" not in st.session_state:
        st.session_state["dist"] = c_Gaussian.Gaussian_distribution(
            key_root="dist", session_state=st.session_state
        )
