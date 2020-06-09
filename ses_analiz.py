import numpy
import scipy.io.wavfile
from scipy.fftpack import dct
import plotly
import plotly.graph_objs as go
import matplotlib.pyplot as plt
numpy.set_printoptions(threshold=numpy.nan)

numOfSounds = 10  # her sesten kaç adet örnek işleyeceğimiz.
soundName = 'testsound'  # karşılaştırılacak ses dosyasının adı
samplerate = 16000
duration = 1  # saniye
threshHoldValue = 30
# soundNames = ["yasin","ömer","elif","göktuğ"]
soundNames = ["aaa","eee","iii","ooo"]  # örnek ses havuzumuz.
# soundNames = ["one","two","three"]
soundPlotColor = ['r','g','b','y','c','m']  # sırasıyla plot renklerinin kodları.
signals = []  # ses dosyalarını tutmak için.
mfccs = []  # her sesin ortalama mfcc'lerini tutmak için.
legends = []  # grafiklerin etiket isimlerini tutmak için.


# ses dosyasından yada doğrudan bir sinyalden mfcc özniteliklerinin çıkarımı.
def get_mfcc_of_sound(soundName=None, soundNum=1, _signal=None, _samplerate=16000):
    # ses dosyası adı verilmişse ses dosyasından sinyali çekiyoruz.
    if soundName:
        # sesin bilgisini dizi olarak alıyoruz.
        sample_rate,signal = scipy.io.wavfile.read('sesler/{}{}.wav'.format(soundName,soundNum))
        # print('signal: ', signal)
    # ses dosyası değil de sinyal verilmişse sinyali doğrudan kullanıyoruz.
    elif _signal.any():
        sample_rate,signal = _samplerate, _signal
    # hem ses dosyası hem de sinyal verilmemişse.
    else:
        print("There is neither soundName nor signal")
        return None

    # önvurgu uygulanımı.
    pre_emphasis = 0.95  # ön vurgu parametresi.
    emphasized_signal = numpy.append(signal[0],signal[1:] - pre_emphasis * signal[:-1])
    #

    # çerçeveleme
    frame_size = 0.025  # 25ms'lik ses parçaları alıyoruz.
    frame_stride = 0.01  # aldığımız ses örnekleri birbiriyle 15ms iç içe.
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # saniye-örnekleme dönüşümü.
    signal_length = len(emphasized_signal)  # önvurgu uygulanmış ses dosyasının dizi uzunluğu.
    frame_length = int(round(frame_length))  # tamsayı dönüşümü.
    frame_step = int(round(frame_step))  # tamsayı dönüşümü.
    # elde ettiğimiz frame sayısı. ceil fonksiyonu sayesinde en az bir.
    num_frames = int(numpy.ceil(float(numpy.abs(signal_length - frame_length)) / frame_step))
    # burada yaptığımız şey elimizdeki ses dosyası dizisi, bir frame'in dizi uzunluğuna tam bölünmediği zaman
    # son frame'in uzunluğunu da eşitlemek adına sonuna sıfırlar eklemek.
    pad_signal_length = num_frames * frame_step + frame_length  # 16080
    z = numpy.zeros((pad_signal_length - signal_length))  # 80
    # Tüm frame'lerin eşit sayıda sample'a sahip olmasını sağlıyoruz.
    pad_signal = numpy.append(emphasized_signal,z)
    # print("pad signal: ", pad_signal)
    # numpy.tile(a,b) = a dizisini b kere tekrar et.
    # her bir frame'in içerisindeki elemanların asıl ses dizisinin kaçıncı elemanı olduğunun belirlenmesi.
    # ilk frame: (0,399), ikinci frame: (160,399+160) ...
    indices = numpy.tile(numpy.arange(0,frame_length),(num_frames,1)) + \
        numpy.tile(numpy.arange(0,num_frames * frame_step,frame_step),(frame_length,1)).T
    # indisler aracılığıyla her bir frame'in ses sinyalinden doldurulması.
    # örn: ilk frame: ses sinyali dizisinin 0,399 arasındaki elemanlarının değerleri,
    # ikinci frame: ses sinyali dizisinin 160,559 arasındaki elemanlarının değerleri...
    frames = pad_signal[indices.astype(numpy.int32,copy=False)]
    #

    # pencereleme
    frames *= numpy.hamming(frame_length)
    #

    # fourier ve güç işlemleri.
    NFFT = 512
    mag_frames = numpy.absolute(numpy.fft.rfft(frames,NFFT))  # FFT'nin büyüklüğü.
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Güç spektrumu.
    #

    #filterbanks
    nfilt = 40
    low_freq_mel = 0
    high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))  # Frekanstan Mel'e geçiş.
    mel_points = numpy.linspace(low_freq_mel,high_freq_mel,nfilt + 2)  # Mel ölçeğinde filtre sayısı kadar eşit aralık.
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Mel'den frekansa geçiş.
    # bin:  [  0.   1.   2.   4.   6.   8.  10.  12.  14.  16.  19.  21.  24.  27.
    # 30.  33.  37.  41.  45.  49.  54.  59.  64.  69.  75.  81.  88.  95.
    # 103. 110. 119. 128. 137. 148. 158. 170. 182. 195. 209. 224. 239. 256.]
    # bin: mel ölçeğinde eşit aralıklarla ayrılmış frekans noktaları.
    bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)

    # fbank i,j: 40,257
    fbank = numpy.zeros((nfilt,int(numpy.floor(NFFT / 2 + 1))))
    # mel filtre bankaları için gereken üçgen filtreleri oluşturuyoruz.
    for m in range(1,nfilt + 1):
        f_m_minus = int(bin[m - 1])   # sol
        f_m = int(bin[m])             # orta
        f_m_plus = int(bin[m + 1])    # sağ

        for k in range(f_m_minus,f_m):
            fbank[m - 1,k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m,f_m_plus):
            fbank[m - 1,k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        # plt.plot(fbank[m-1])  # -> üçgen filtrelerin çizimi.
    # plt.show()
    #
    filter_banks = numpy.dot(pow_frames,fbank.T)  # -> üçgen filtre altında güç spektrumu.
    filter_banks = numpy.where(filter_banks == 0,numpy.finfo(float).eps,filter_banks)  # 0'lardan kaçınmak için.
    filter_banks = 20 * numpy.log10(filter_banks)  # dB'e dönüşüm.
    #

    #mfcc
    num_ceps = 15  # denemelerimiz sonucunda liftering için seçtiğimiz değer
    mfcc = dct(filter_banks,type=2,axis=1,norm='ortho')[:, 3: (num_ceps + 1)]  # liftering -> indis kısıtları.
    return signal,mfcc  # burada hem sinyali hem mfcc'yi döndürüyoruz.


# Kelimelerin ortalama değerlerini alıp grafiklerini çizdiriyoruz.
def plotAveragesOfSounds():
    # Kelime havuzumuzdaki her bir kelime için döngüye giriyoruz.
    for soundname, i in zip(soundNames, range(len(soundNames))):
        signals_of_sound = []
        mfccs_of_sound = []  # kelimenin her bir örneğinin mfcc vektörlerini tutacak olan değişken.
        # Kelimenin her bir örneği için döngüye giriyoruz.
        for sample in range(numOfSounds):
            zeros, processed_mfcc = getAveragesOfFrames(sample, soundname)  # fonksiyonun tanımı aşağıda.
            mfccs_of_sound.append(zeros)  # kelimenin mfcc konteynerine bu örneğin mfcc'sini ekliyoruz.
            # plt.plot(zeros,soundPlotColor[i]) # kelimenin her bir örneğinin mfcc'lerini plot ediyoruz.
            # legends.append(soundname)
        # kelimenin her bir örneğinin değerlerinin ortalamasını alıyoruz.
        mfccs_of_sound = numpy.average(mfccs_of_sound, axis=0)
        # kelimenin ortalama değerlerinin grafiğini çizdiriyoruz.
        plt.plot(mfccs_of_sound, soundPlotColor[i])
        signals.append(signals_of_sound)
        mfccs.append(mfccs_of_sound)  # tüm kelimelerimizin mfcc'lerinin bilgisini tutuyoruz.
        legends.append(soundname)  # etiketleme işlemi.


# Sesin frame'lerinin mfcc vektörlerini alıp eşik değerinden küçük olan bileşenleri sıfırlıyoruz.
# Bunu yapmamızın yararı hem başlangıçtaki ve sondaki konuşmasız kısımdan kurtulmuş oluyoruz
# hem de görece küçük kısımları da sıfırlayarak konuşmanın anlamlı bir vektörünü oluşturmaya çalışıyoruz.
# filtrebankasının her bir filtresindeki değerleri frameler boyunca toplayarak da zamandaki kaymaları
# önemsizleştirmiş oluyoruz.
# Normalde mfcc özniteliklerini HMM gibi yöntemlerle kullanmak daha iyi sonuçlar verse de bunu yapamadığımız için
# kendimizce bir yöntem oluşturmaya çalıştık.
# Kesinliği yüksek olmasa da yine de tatmin edici seviyede sonuçlar alabiliyoruz.
def getAveragesOfFrames(sample=0, soundname='testsound'):
    signal, mfcc = get_mfcc_of_sound(soundname, sample + 1, _samplerate=samplerate)
    # (len(mfcc[0])) = bir frame'in içerisindeki katsayı sayısı. (filtrebankasındaki filtre sayısı)
    # zeros, sesin işlenmiş frame'lerinin değerlerinin toplamını tutacak.
    zeros = numpy.zeros(len(mfcc[0]))  # katsayı sayısı uzunluğunda, sıfırlardan oluşan bir dizi oluşturuyoruz.
    numOfFrames = len(mfcc)  # ses kaydından aldığımız frame sayısı.
    # her bir frame için döngüye giriyoruz.
    for frame in range(numOfFrames):
        numOfBanks = len(mfcc[frame])  # bu frame'in içerisindeki katsayı sayısı.
        # bu frame'in katsayıları arasında dönüyoruz.
        for bank in range(numOfBanks):
            # eğer bir katsayı eşik değerinden küçükse;
            if abs(mfcc[frame][bank]) < threshHoldValue:
                # sıfırlıyoruz.
                mfcc[frame][bank] = 0
        # bu frame'in işlenmiş değerlerini toplama ekliyoruz.
        zeros += abs(mfcc[frame])
        # print('zeros: ', zeros)
    return zeros, mfcc


def compareAndPlotDiff():
    plotAveragesOfSounds()  # Elimizdeki seslerin örneklerini alıp filtre değerlerinin ortalamasını alıyoruz.
    testSoundVector, processed_mfcc = getAveragesOfFrames()  # test kaydımızın işlemden geçirilmiş mfcc'sini alıyoruz.
    plot_it_3d(abs(processed_mfcc),"Mfcc of Testsound, if value<TH: value=0")
    plt.plot(testSoundVector, 'y--')  # test kaydımızı çizgili şekilde sarı renkte çizdiriyoruz.
    legends.append('TestSound')  # etiketliyoruz.
    plt.legend(legends)
    # Eksenlerin isimlerini yazıyoruz.
    plt.xlabel("Filterbanks")
    plt.ylabel("Coefficients")
    # Grafiği ekrana veriyoruz.
    plt.show()
    diff_scalars = []  # test kaydının diğer seslerle olan farklarını skaler olarak tutuyoruz.
    diff_vectors = []  # vektörel olarak.
    # önceki işlemlerden elde ettiğimiz her bir kelimenin mfcc'leri içerisinde dönüyoruz.
    for soundfeatures, i in zip(mfccs, range(len(mfccs))):
        # kelimenin fark bilgilerini tutacak değişkenleri tanımlıyoruz.
        diff_scalar = 0
        diff_vector = []
        # o sesin vektörü içerisinde dönüyoruz.
        for soundfeature, j in zip(soundfeatures, range(len(soundfeatures))):
            # vektörel ve skaler fark bilgileri:
            diff_vector.append(abs(soundfeature - testSoundVector[j]))
            diff_scalar += abs(soundfeature - testSoundVector[j])
        plt.plot(diff_vector)  # fark vektörünün grafiğini çizdiriyoruz.
        legends.append(soundNames[i])  # etiketleme.
        plt.legend(legends)
        # tüm kelimelerin farklarına ekliyoruz.
        diff_scalars.append(int(diff_scalar))
        diff_vectors.append(diff_vector)
    # her bir kelime ile arasındaki farklar
    print("diff scalars: ", diff_scalars)
    # aralarından en küçüğü seçiyoruz.
    indexOfMinimum = diff_scalars.index(min(diff_scalars))
    print("My choice is: ", soundNames[indexOfMinimum])
    # Fark grafiğinin eksen isimlendirmeleri.
    plt.xlabel("Filterbanks")
    plt.ylabel("Differences with sample")
    plt.show()
    # Skaler farklarının grafiği ve etiketlemesi.
    plt.plot(soundNames, diff_scalars,'o')
    plt.xlabel("Words")
    plt.ylabel("Differences with sample")
    plt.ylim(bottom=0)
    plt.show()


# ses sinyalinin kendisini çizdirmek için olan fonksiyonumuz.
def showTestSound(soundname="testsound",soundnum=1):
    signal, mfcc = get_mfcc_of_sound(soundname,soundnum)
    plt.plot(signal)
    plt.show()
    plot_it_3d(mfcc,"Mfcc of Testsound")


def plot_it_3d(mfcc,title):
    data = []
    data.append(go.Surface(z=mfcc))
    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(title="Filterbanks"),
            yaxis=dict(title="Frames"),
            zaxis=dict(title="Coefficients")))
    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig)


showTestSound()
compareAndPlotDiff()
