import streamlit as st
from typing import Optional, List
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


def text_keys_in_dict(dictionary: dict, *keys: List[str]) -> bool:
    results = []
    for key in keys:
        results.append(key in dictionary)

    return np.all(results)


def st_expandable_box(
    txt_dict: dict, title_key: str, text_key: Optional[str] = None, expanded=True
):
    if not text_keys_in_dict(txt_dict, title_key, text_key):
        return

    text = txt_dict[text_key]

    with st.expander(txt_dict[title_key], expanded=expanded):
        while True:  # while printing to text to screen
            # if there is an indicated latex equation in the text body
            if "latex_eq{" in text:
                # split the starting normal text from the start of the latex equation
                text_split = text.split("latex_eq{", 1)
                # print the normal text as markdown
                st.markdown(text_split[0])

                # find and seperate the end of the latex equation from the rest of the text
                text_split = text_split[1].split("}end_eq", 1)
                # print the latex equation
                st.latex(text_split[0])

                # keep only the remaining, undisplayed text to continue working with
                text = text_split[1]

            # if there is no latex equation to be displayed
            else:
                # display all available text as markdown
                st.markdown(text)
                # end the display loop as all text has been shown
                break
