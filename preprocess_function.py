import re
import pandas as pd

def remove_accent(string):
    """
    args: string
    return: string

    """
    string = string.replace('á', 'a')
    string = string.replace('â', 'a')

    string = string.replace('é', 'e')
    string = string.replace('è', 'e')
    string = string.replace('ê', 'e')
    string = string.replace('ë', 'e')

    string = string.replace('î', 'i')
    string = string.replace('ï', 'i')

    string = string.replace('ö', 'o')
    string = string.replace('ô', 'o')
    string = string.replace('ò', 'o')
    string = string.replace('ó', 'o')

    string = string.replace('ù', 'u')
    string = string.replace('û', 'u')
    string = string.replace('ü', 'u')

    string = string.replace('ç', 'c')

    
    return string

def lower_case(text): # convert all text to lower case
    """
    args: string
    return: string
    """
    text = text.lower().strip()
    return text

def remove_htmltags(text): # remove all html tags
    """
    args: string
    return: string
    """
    text = re.sub('<[^<]+?>', '',text)
    return text

def keeping_essentiel(text): # remove all special characters
    """
    args: string
    return: string
    """
    text = re.sub(r"[^a-zA-Z]+", " ", text)
    return text