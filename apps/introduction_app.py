import streamlit as st
from pyprojroot import here as get_proj_root
import os

from helpers.helper_fns import load_dict_txt, st_expandable_box


def app():
    st.markdown(
        "### Known issue:\n"
        "There is a known bug where the app may stop responding. "
        "Sliders and options will still change but the plot will not update. "
        "If the app stops responding for more than 5 seconds and you do not see "
        "the running icon in the top right corner, please refresh the page."
        "\n#"
    )

    if "intro_txt_dict" not in st.session_state:
        proj_root = get_proj_root()
        fil = os.path.join(proj_root, "text_files", "introduction_page.txt")
        (st.session_state["intro_txt_dict"]) = load_dict_txt(fil)

    st_expandable_box(
        st.session_state["intro_txt_dict"],
        "dashboard_title",
        "dashboard_usage",
        expanded=True,
    )

    st_expandable_box(
        st.session_state["intro_txt_dict"],
        "level_outline_title",
        "level_outline_text",
        expanded=True,
    )

    st.markdown(
        "### Source Code:\n"
        "If you are interested in the source code for this project, it can be found at:  \n"
        "https://github.com/Casual-Endvrs/Probability-Distributions"
    )
