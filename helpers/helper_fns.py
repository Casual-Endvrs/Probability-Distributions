import streamlit as st
from typing import Optional
import numpy as np
import os


def load_dict_txt(fil: str) -> dict:
    """Reads in text from a file and parses it into text_key and text_body. A
        dictionary is returned using text_key as the key values and text_body 
        as the dictionary entries. The text file is parsed into text_key and 
        text_body using the following separators:

        $---$   --> Separates each text_key and text_body.
        #---#   --> Separates each text_key/text_body pair for the next pair.

    :param str fil: String providing the absolute path of the file.
    :dict: A dictionary of the text_key:text_body entries as created from the 
        text file.
    """
    if not os.path.isfile(fil):
        return {}

    with open(fil, "r") as f:
        text = f.readlines()

    text = "\n".join(text)
    text = text.split("#---#")

    if text[0].strip() == "":
        del text[0]

    text_dic = {}
    for entry in text:
        entry = entry.split("$---$")
        key = entry[0].strip()
        text_dic[key] = entry[1].strip()

    return text_dic


def text_keys_in_dict(dictionary: dict, *keys: list[str]) -> bool:
    results = []
    for key in keys:
        results.append(key in dictionary)

    return np.all(results)


def st_expandable_box(
    txt_dict: dict,
    title_key: str,
    text_markdown: Optional[str] = None,
    text_latex: Optional[str] = None,
):
    keys = [key for key in [text_markdown, text_latex] if key is not None]

    if text_keys_in_dict(txt_dict, *keys):
        with st.expander(txt_dict[title_key], expanded=True):
            if text_markdown is not None:
                st.markdown(txt_dict[text_markdown])
            if text_latex is not None:
                st.latex(txt_dict[text_latex])
