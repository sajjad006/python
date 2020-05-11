import speech_recognition as sr
import pyttsx3

r = sr.Recognizer()
mic = sr.Microphone()
engine = pyttsx3.init()
engine.setProperty('rate', 120)

while True:
    with mic as source:
        print('say something')
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)

    print("recognizing it...")

    try:
        # engine.say("You said " + r.recognize_google(audio))
        # engine.runAndWait()
        say = r.recognize_google(audio)
        print(say)
        if 'google' in say.lower():
            engine.say('What can I do for you ?')
            engine.runAndWait()

            with mic as source:
                print('What can I do for you?')
                r.adjust_for_ambient_noise(source)
                audio = r.listen(source)

            print("recognizing it...")

            try:
                cmd = r.recognize_google(audio)

                engine.say('You said ' + cmd)
                engine.runAndWait()
            except Exception as e:
                engine.say('error')
                engine.runAndWait()
        else:
            print('You said ' + say)
    except sr.RequestError:
        engine.say("api unavailable")
        engine.runAndWait()
    except sr.UnknownValueError:
        # engine.say("unable to recognize speech")
        # engine.runAndWait()
        pass
