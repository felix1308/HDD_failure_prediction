import pickle

def HDD_prediction(x_test):
    filename = "./HDD_classification_model.pkl"
    model = pickle.load(open(filename, 'rb'))

    res = model.predict(x_test)
    return res