import scipy.signal
import scipy.stats
import spafe.features.spfeats
import numpy as np
import pandas as pd
import pickle
from spafe.frequencies.fundamental_frequencies import compute_yin
from spafe.frequencies.dominant_frequencies import get_dominant_frequencies
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def spectral_prop(d,r):
    nfft = 512
    v1 = np.fft.rfft(d,nfft)
    v2 = (1/nfft)*(np.abs(v1))
    v3 = (1/nfft)**2 * v2**2
    v4 = np.fft.fftfreq(len(v3),1/r)
    v4 = v4[np.where(v4 >= 0)] // 2 + 1
    v4 = v4/1000.0
    v2 = v2[:len(v4)]
    v3 = v3[:len(v4)]
    v5 = v3
    v6 = compute_yin(d,r)
    v7 = np.cumsum(v5)
    v8 = v6[0]/1000.0
    v9 = get_dominant_frequencies(d,r)/1000.0
    v10 = np.abs(v9[:-1] - v9[1:])
    v11 = v9.max()-v9.min()
    v12 = 0
    if v11 != 0:
        v12 = v10.mean()/v11
    v13 = np.fft.rfft(d)
    v14 = np.abs(np.fft.fftfreq(len(d), 1.0 / r)[:len(d) // 2 + 1])
    v15 = np.sum(v13 * v14) / np.sum(v13)
    v15 = np.abs(v15)/1000.0
    res = {
        "meanfreq": [v4.mean()],
        "sd": [v4.std()],
        "median": [np.median(v4)],
        "Q25": [v4[len(v7[v7<=0.25])-1]],
        "Q75": [v4[len(v7[v7<=0.75])-1]],
        "IQR": [0.0],
        "skew": [scipy.stats.skew(v4)],
        "kurt": [scipy.stats.kurtosis(v4)],
        "sp.ent": [scipy.stats.entropy(v4)],
        "sfm": [spafe.features.spfeats.spectral_flatness(d)],
        "mode": [v4[v5.argmax()]],
        "centroid": [v15],
        "meanfun": [v8.mean()],
        "minfun": [v8.min()],
        "maxfun": [v8.max()],
        "meandom": [v9.mean()],
        "mindom": [v9.min()],
        "maxdom": [v9.max()],
        "dfrange": [0.0],
        "modindx": [v12],
        "label": ["male"]
    }
    res["IQR"][0] = res["Q75"][0]-res["Q25"][0]
    res["dfrange"][0] = res["maxdom"][0]-res["mindom"][0]
    return res

def gender(rate,data):
    spec_prop = spectral_prop(data,rate)
    filename = "C:/Users/sones/Desktop/Program/Python/ML-1/Files/voice_model.pickle"
    loaded_model = pickle.load(open(filename,'rb'))
    specprop = pd.DataFrame.from_dict(spec_prop)
    #print(specprop)
    spec_prop_new = specprop.drop(['dfrange','kurt','sfm','meandom','meanfreq'],axis = 1)
    y = specprop['label']
    x = specprop.drop(['label'], axis = 1)
    final = loaded_model.score(x,y)
    if (final == 0.0):
        return "female"
    return "male"

# r1,d1 = scipy.io.wavfile.read("C:/Users/sones/Downloads/standard_recording.wav")
rate,data = scipy.io.wavfile.read("C:/Users/sones/Desktop/Program/Python/vr/new.wav")
if (type(data[0]) == np.ndarray):
    data = data[:,0]
res = gender(rate,data)
print(res)