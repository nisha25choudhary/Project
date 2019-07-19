
# Natural Language Processing

import re
import numpy as np
import pickle
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#pridiction process

def check_sms(user_input):
    with open("static/models/trained_models.pkl", "rb") as f:
        # Load the pickled model
        models = pickle.load(f)
    #user_input = "Ringtone Club: Gr8 new polys direct to your mobile every week !"
    user_input = re.sub('[^a-zA-Z]', ' ', user_input)
    user_input = user_input.lower()
    user_input = user_input.split()

    user_input = [word for word in user_input if not word in set(stopwords.words('english'))]

    ps = PorterStemmer()
    user_input = [ps.stem(word) for word in user_input]
    user_input = ' '.join(user_input)

    user_input = np.ravel(user_input)
    user_input = models["CV"].transform(user_input).toarray()

    # Use the loaded pickled model to make predictions
    result = models["Logistic_Reg"].predict(user_input)

    if result[0] == 0:
        return "Ham"
    else:
        return "spam"
