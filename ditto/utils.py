import unicodedata


stopwords = {'at', 'based', 'in', 'of', 'for', 'on', 'and', 'to', 'an', 'using', 'with', 'the', 'by', 'we', 'be',
             'is', 'are', 'can'}


def remove_marks(txt):
    """This method removes all diacritic marks from the given string"""
    norm_txt = unicodedata.normalize('NFD', txt)
    shaved = ''.join(c for c in norm_txt if not unicodedata.combining(c))
    return unicodedata.normalize('NFC', shaved)



def normalize_text(text):
    text = text.lower()
    text = text.replace("-", " ")
    text = remove_marks(text)
    text = " ".join([x for x in text.split() if x not in stopwords])
    return text
