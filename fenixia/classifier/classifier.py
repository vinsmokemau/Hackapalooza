import numpy as np
from pyaudioclassification import feature_extraction, train, predict, print_leaderboard

parent_dir = '.'

# step 1: preprocessing
if np.DataSource().exists("./feat.npy") and np.DataSource().exists("./label.npy"):
    features, labels = np.load('./feat.npy'), np.load('./label.npy')
else:
    features, labels = feature_extraction('./real/')
    np.save('./feat.npy', features)
    np.save('./label.npy', labels)

# step 2: training
if np.DataSource().exists("./model.h5"):
    from keras.models import load_model
    model = load_model('./model.h5')
else:
    model = train(features, labels, epochs=1050)
    model.save('./model.h5')

# step 3: prediction
pred = predict(model, './alerta_test.ogg')
print_leaderboard(pred, './real/')
