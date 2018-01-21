# Import libraries
import numpy as np
import soundfile as sf
import librosa
import json
import datetime
import os

def feature_extract(pathToJson, npy_function_list, label_family_map, label_cap, count_cap=float('inf'), postfix='',):
    # Load dataset description
    with open(pathToJson+'examples.json') as data_file:
        data = json.load(data_file)

    datamap={} #function -> and a list
    count=0

    for func in npy_function_list:
        datamap[func]=[]
    dataorder=[]
    familylabel=[]
    label_cap_copy=label_cap
    # Feature computations for data
    for datum in data:
        #only limiting to librosa stuff, not perfect, but much cleaner
        if label_family_map[data[datum]['instrument_family']] != -1 and label_cap_copy[label_family_map[data[datum]['instrument_family']]]>0:
            familylabel.append(label_family_map[data[datum]['instrument_family']])
            label_cap_copy[label_family_map[data[datum]['instrument_family']]]-=1
            temp_audio, sample_rate = sf.read(pathToJson+'audio/'+data[datum]['note_str']+'.wav')
            for func in npy_function_list:
                datamap[func].append(func[0](temp_audio, sample_rate)) # if your function take less or more parameters, you need to create a wrapper or lambda function

            dataorder.append(data[datum])
            print(count)
            count += 1
            if count >= count_cap:
                break

    path='data'+postfix+'\\'
    os.makedirs(path)
    vectorfiles=['dataorder'+postfix+'.npy\n','familylabel'+postfix+'.npy\n']
    np.save(path+'dataorder'+postfix+'.npy',dataorder)
    np.save(path+'familylabel'+postfix+'.npy', familylabel)

    for func in npy_function_list:
        vectorfiles.append(func[1]+postfix+'.npy\n')
        np.save(path+func[1]+postfix+'.npy',datamap[func])

    with open(path+'filelist'+postfix+'.txt','w') as fp:
        fp.writelines(vectorfiles)

def mfccanddeltas(audio, samplerate):
    mfcc=librosa.feature.mfcc(audio,samplerate)
    result = mfcc
    result = np.append(result, librosa.feature.delta(mfcc, order=1))
    result = np.append(result, librosa.feature.delta(mfcc, order=2))
    return result

if __name__ == '__main__':
    postfix=''.join(list(filter(str.isalnum, datetime.datetime.isoformat(datetime.datetime.now()))))
    npyfunctionlist=[]
    npyfunctionlist.append((librosa.feature.mfcc, 'mfcc'))
    #npyfunctionlist.append((mfccanddeltas, 'mfccdlanddldl'))
    #npyfunctionlist.append(((lambda audio, samplerate: librosa.feature.melspectrogram(y=audio, sr=samplerate, n_fft=2048, hop_length=512, power=2.0)),'melspectrogram'))
    #npyfunctionlist.append(((
    #                        lambda audio, samplerate: librosa.feature.chroma_cqt(y=audio, sr=samplerate, hop_length=512)),
    #                        'chromacqt'))
    #npyfunctionlist.append((librosa.feature.spectral_rolloff, 'spectral_rolloff'))
    #npyfunctionlist.append((librosa.feature.spectral_centroid, 'spectral_centroid'))
    #npyfunctionlist.append((lambda audio,samplerate:librosa.feature.zero_crossing_rate(audio), 'zcr')) #example of lambda function
    labelfamilymap = {8: 0, 2: 1, 6: 1, 7: 1, 1: 2, 4: 3, 3: 0, 5: 3, 0: -1, 9: -1, 10: -1}
    labelcap={0:8000,1:8000,2:8000,3:8000}
    feature_extract('Nsynth_Dataset/nsynth-train/',npyfunctionlist,labelfamilymap,labelcap,float('inf'), postfix)