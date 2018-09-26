#Python 2.x program to transcribe an Audio file
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence

sample_rate = 48000
chunk_size = 2048

## AudioSegment.converter = r"C:\\projects\\ffmpeg\\bin\\ffmpeg.exe"

sound_file = AudioSegment.from_wav("example.wav")
audio_chunks = split_on_silence(sound_file, 
    # must be silent for at least half a second
    min_silence_len=500,

    # consider it silent if quieter than -16 dBFS
    silence_thresh=-16
)


for i, chunk in enumerate(audio_chunks):
    out_file = "chunk{0}.wav".format(i)
    print ("exporting "+ out_file)
    chunk.export(out_file, format="wav")
##    r = sr.Recognizer()
##    with sr.AudioFile(out_file) as source:
##        audio = r.record(source)
##    try:
##        print(r.recognize_google(audio))
##    except sr.UnknownValueError:
##            print("Google Speech Recognition could not understand audio")
##    except sr.RequestError as e:
##            print("Could not request results from Google Speech recognition service; {0}".format(e))
