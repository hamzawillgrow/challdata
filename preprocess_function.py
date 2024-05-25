import re

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

def lower_case(text):
    text = text.lower().strip()
    return text

def create_text(text1, text2):
    if pd.isna(text2):
        text = text1
    else:
        text = text1 + text2
    return text

def removltags(text):
    text = re.sub('<[^<]+?>', '',text)
    return text

def keeping_essentiel(text):
    text = re.sub(r"[^a-zA-Z]+", " ", text)
    return text