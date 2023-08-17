import audio_test
import csv

rate,data = audio_test.scipy.io.wavfile.read("C:/Users/sones/Desktop/Program/Python/vr/dadu.wav")
if (type(data[0]) == audio_test.np.ndarray):
    data = data[:,0]

l = audio_test.spectral_prop(data,rate)
l["IQR"][0] = l["Q75"][0]-l["Q25"][0]
l["dfrange"][0] = l["maxdom"][0]-l["mindom"][0]
a = [l['meanfreq'][0],l['sd'][0],l['median'][0],l['Q25'][0],l['Q75'][0],l['IQR'][0],l['skew'][0],l['kurt'][0],l['sp.ent'][0],l['sfm'][0],l['mode'][0],l['centroid'][0],l['meanfun'][0],l['minfun'][0],l['maxfun'][0],l['meandom'][0],l['mindom'][0],l['maxdom'][0],l['dfrange'][0],l['modindx'][0],"male"]
with open('C:/Users/sones/Desktop/Program/Python/ML-1/Files/gender_voice_dataset.csv','a') as f:
    w = csv.writer(f)
    w.writerow(a)
    f.close()