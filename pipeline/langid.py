import langid

def detect_language(text):
    lang, confidence = langid.classify(text)
    return lang, confidence