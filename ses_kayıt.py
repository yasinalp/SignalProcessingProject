import scipy.io.wavfile
import sounddevice as sd
import time


numOfSounds = 1
soundName = 'testsound'
samplerate = 16000
duration = 1 #seconds


def record(duration,samplerate,record=True,soundName='testsound', soundNum=0,):
    print("RECORD STARTED...")
    print("+"*10)
    myrecord = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    print("RECORD FINISHED!")
    print("-"*10)
    if record:
        save_it(soundName,soundNum,samplerate,myrecord)
    return myrecord


def save_it(soundName, soundNum,samplerate,record,):
    scipy.io.wavfile.write('sesler/{}{}.wav'.format(soundName, soundNum),samplerate,record)


for i in range(numOfSounds):
    record(duration, samplerate, True, soundName, i+1)
    print('-'*30)
    time.sleep(0.5)
