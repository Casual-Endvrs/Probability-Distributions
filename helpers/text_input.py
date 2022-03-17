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
    with open(fil, "r") as f:
        text = f.readlines()

    text = "".join(text)
    text = text.split("#---#")

    if text[0].strip() == "":
        del text[0]

    text_dic = {}
    for entry in text:
        entry = entry.split("$---$")
        text_dic[entry[0].strip()] = entry[1].strip()

    return text_dic

