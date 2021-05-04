import speech_recognition as sr
r = sr.Recognizer()


#pop = sr.AudioFile('data/genres_original/pop/pop.00000.wav')
#with pop as source:
#    audio = r.record(source)

#print(type(audio))
#print(r.recognize_google(audio, language = 'pt', show_all=True))



for i in range(10, 100):
    pop = sr.AudioFile('data/genres_original/blues/blues.000' + str(i) + '.wav')
    with pop as source:
        audio = r.record(source)
    print(str(i) + ':' + str(r.recognize_google(audio, show_all=True)))