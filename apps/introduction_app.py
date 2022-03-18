import streamlit as st
from pyprojroot import here as get_proj_root
import os

from helpers.helper_fns import load_dict_txt, st_expandable_box


def app():
    if "intro_txt_dict" not in st.session_state:
        proj_root = get_proj_root()
        fil = os.path.join(proj_root, "text_files", "introduction_page.txt")
        (st.session_state["intro_txt_dict"]) = load_dict_txt(fil)

    st_expandable_box(
        st.session_state["intro_txt_dict"], "dashboard_title", "dashboard_usage"
    )

    st_expandable_box(
        st.session_state["intro_txt_dict"], "level_outline_title", "level_outline_text"
    )

