import os
from google.cloud import speech
import moviepy.editor as mp
import speech_recognition as sr
import pandas as pd
import numpy as np
import math
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from flask import Flask, render_template, request, send_file, send_from_directory,session
import mysql.connector
from pytube import YouTube
import time
from pydub import AudioSegment
from pydub.silence import split_on_silence 
from tabulate import tabulate


# client_file = "tugas-akhir-399907-4dd3b7ba20a3.json"
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = client_file
# speech_client = speech.SpeechClient()

app = Flask(__name__, static_folder="css") 



def Download(link):
    youtubeObject = YouTube(link)
    youtubeObject = youtubeObject.streams.get_highest_resolution()
    try:
        youtubeObject.download(output_path="css/static/",filename="videoYt.mp4")
    except:
        print("An error has occurred")
    print("Download is completed successfully")

def get_duration(file_path):
   audio_file = AudioSegment.from_file(file_path)
   duration = audio_file.duration_seconds
   return duration

def long_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    text = ""
    chunks = split_on_silence(audio,
        min_silence_len=700, silence_thresh=-20, keep_silence=700
    )
    try: 
        os.mkdir('audio_chunks') 
    except(FileExistsError): 
        pass
  
    # move into the directory to 
    # store the audio files. 
    os.chdir('audio_chunks') 
  
    i = 0
    # process each chunk 
    for chunk in chunks: 
              
        chunk_silent = AudioSegment.silent(duration = 10) 
        audio_chunk = chunk_silent + chunk + chunk_silent 

        print("saving chunk{0}.wav".format(i)) 
        # specify the bitrate to be 192 k 
        audio_chunk.export("./chunk{0}.wav".format(i), format ="wav") 
  
        # the name of the newly created chunk 
        filename = 'chunk'+str(i)+'.wav'
  
        print("Processing chunk "+str(i)) 
  
        # get the name of the newly created chunk 
        # in the AUDIO_FILE variable for later use. 
        file = filename 
  
        # create a speech recognition object 
        r = sr.Recognizer() 

        with sr.AudioFile(file) as source:
                r.adjust_for_ambient_noise(source)
                data = r.record(source)
        try:
            text_data = r.recognize_google(data, language="id-ID")
            text += text_data+" "
        except sr.UnknownValueError:
                print('Audio unintelligible')
        except sr.RequestError as e:
                print("Cannot obtain result: {0}".format(e))
  
        i += 1
  
    os.chdir('..')
    return text

@app.route("/")
def main():
   return render_template("index.html", linkVideo="")

@app.route('/', methods = ['POST'])   
def success():
    if request.method == 'POST':
        linkVideo = ""
        input = ""
        if "file" in request.files:
            f = request.files['file']
            f.save(os.path.join('css/static', f.filename))
            linkVideo = f.filename
            video = mp.VideoFileClip("css/static/"+f.filename)
        else:
            input = "yt"
            ytLink = request.form['link']
            Download(ytLink)
            linkVideo = "videoYt.mp4"
            video = mp.VideoFileClip("css/static/videoYt.mp4")
        audio = video.audio
        audio_file = audio.write_audiofile("data.wav")
        #LIBRARY
        start_time = time.time()
        if((get_duration("data.wav")<=200) or (input == "yt")):
            r = sr.Recognizer()
            with sr.AudioFile("data.wav") as source:
                r.adjust_for_ambient_noise(source)
                data = r.record(source)
            try:
                text_data = r.recognize_google(data, language="id-ID")
                # text2 = r.recognize_google_cloud(data, credentials_json=client_file, language="id-ID")
                print(text_data)
                print(" ")
                # print(text2)
            except sr.UnknownValueError:
                print('Audio unintelligible')
            except sr.RequestError as e:
                print("Cannot obtain result: {0}".format(e))
        else:
            text_data = long_audio("data.wav")
            print(text_data)
        end_time = time.time()
        ex_time = end_time-start_time
        print("speech to text = ",ex_time)
        start_time = time.time()
        mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="tugas_akhir"
            )
        text = []
        konteks = []
        hate_speech = []

        mycursor = mydb.cursor()
        mycursor.execute("SELECT Speech FROM datasets")
        myresult = mycursor.fetchall()
        for x in myresult:
            text.append(x)

        mycursor.execute("SELECT Hatespeech FROM datasets")
        myresult = mycursor.fetchall()
        for y in myresult:
            hate_speech.append(y)

        mycursor.execute("SELECT Konteks FROM datasets")
        myresult = mycursor.fetchall()
        for z in myresult:
            konteks.append(z)
        data_train=[]
        # test_text = []
        # print(test_text)
        #preprocess
        factory = StemmerFactory()
        stopword_factory = StopWordRemoverFactory()
        stemmer = factory.create_stemmer()
        stopword = stopword_factory.create_stop_word_remover()
        
        for i in text:
            stem = stemmer.stem(i[0])
            stop = stopword.remove(stem)
            data_train.append(stop)
        stemTest = stemmer.stem(text_data)
        print(stemTest)
        stopTest = stopword.remove(stemTest)
        print(stopTest)
        # print(stopTest)
        data_train.append(stopTest)
        # print(data_train)
        # stem_test = stemmer.stem(test_text[0])
        # stop_test = stopword.remove(stem_test)
        # print(stop_test)
        test_data = []
        # test_data.append(stop_test)
        # tf = CountVectorizer(max_features=1000)
        # TF_vector = tf.fit_transform(temp)
        # print(TF_vector)

        tfidf = TfidfVectorizer()
        tfidf.fit(data_train)
        uji_tfidf = tfidf.transform(data_train)
        test_data.append(data_train[len(data_train)-1])
        # print(test_data)
        data_train.pop(len(data_train)-1)
        # temp.pop(len(temp)-1)
        tfidf_train = tfidf.transform(data_train)
        tfidf_test = tfidf.transform(test_data)
        terms = tfidf.get_feature_names_out()
        tampil_tfidf = dict(zip(terms, uji_tfidf.toarray().sum(axis=0)))
        tampil_tfidf = dict(sorted(tampil_tfidf.items(), key=lambda x: x[1], reverse=True))
        tfidf_df = pd.DataFrame.from_dict(tampil_tfidf, orient='index', columns=['tfidf'])
        tfidf_df.index = tfidf_df.index.rename('token')
        pd.set_option('display.max_rows', 20)
        print(tfidf_df)
        # print(tabulate(tfidf_df, headers='keys', tablefmt='psql'))
        # print(terms)
        # print(tfidf_test)

        #Prediction secara bersamaan
        naive_bayes = MultinomialNB()
        naive_bayes.fit(tfidf_train,hate_speech)
        prediction = naive_bayes.predict(tfidf_test)
        print(prediction)

        k = round(math.sqrt(len(data_train)))
        knn = KNeighborsClassifier(n_neighbors=k ,p=2)#euclidean
        knn.fit(tfidf_train, konteks)
        pred = knn.predict(tfidf_test)
        print(pred)
        #Prediction Konteks hanya pada data hate speech
        if(prediction[0] == "hate speech"):
            text = []
            konteks = []
            data_train=[]
            mycursor = mydb.cursor()
            mycursor.execute("SELECT Speech FROM datasets WHERE Hatespeech='hate speech'")
            myresult = mycursor.fetchall()
            for x in myresult:
                text.append(x)
            mycursor.execute("SELECT Konteks FROM datasets WHERE Hatespeech='hate speech'")
            myresult = mycursor.fetchall()
            for y in myresult:
                konteks.append(y)
            for i in text:
                stem = stemmer.stem(i[0])
                stop = stopword.remove(stem)
                data_train.append(stop)
            print(data_train)
            data_train.append(stopTest)
            test_data = []
            tfidf = TfidfVectorizer()
            tfidf.fit(data_train)
            test_data.append(data_train[len(data_train)-1])
            print(test_data)
            data_train.pop(len(data_train)-1)
            # temp.pop(len(temp)-1)
            tfidf_train = tfidf.transform(data_train)
            tfidf_test = tfidf.transform(test_data)
            print(tfidf_test)
            k = round(math.sqrt(len(data_train)))
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(tfidf_train, konteks)
            pred2 = knn.predict(tfidf_test)
        else:
            pred2 = ['']
        end_time = time.time()
        ex_time = end_time-start_time
        print("Klasifikasi = ",ex_time)
        return render_template("index.html", text=stopTest, linkVideo=linkVideo,label=prediction[0], konteks=pred[0], label2=prediction[0], konteks2=pred2[0])

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/regulation')
def regulation():
    return render_template("regulations.html")



# SHORT AUDIO
# with open("data.wav", 'rb') as f1:
#     byte_data = f1.read()
# video_mp4 = speech.RecognitionAudio(content=byte_data)

# config_mp4 = speech.RecognitionConfig(
#     encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
#     sample_rate_hertz= 44100,
#     language_code= "id-ID",
#     audio_channel_count = 2,
#     enable_automatic_punctuation=True,
# )

# response_standard_mp4 = speech_client.recognize(
#     config = config_mp4,
#     audio=video_mp4,
# )

# print(response_standard_mp4)


# LONG AUDIO
# media_uri = "gs://tugas_akhir-1/data.wav"
# long_audio_wav = speech.RecognitionAudio(uri=media_uri)

# config_wav_enabled = speech.RecognitionConfig(
#     sample_rate_hertz= 48000,
#     language_code= "id-ID",
#     model='video',
#     use_enhanced=True,
#     enable_automatic_punctuation=True,
# )


# operation = speech_client.long_running_recognize(
#     config= config_mp4,
#     audio=long_audio_wav,
# )

# response = operation.result(timeout=90)
# print(response)





if __name__ == '__main__':
    app.run(debug=True)