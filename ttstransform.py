import azure.cognitiveservices.speech as speechsdk
import pandas as pd

import os
import natsort
from playsound import playsound

class azuretts():
    def __init__(self,key,language='en-US'):
        self.key = key
        self.language = language

    @staticmethod
    def match_voice(n_crt, t_crt):
        n_crt['TTS']=None
        TTS_db=t_crt.sort_values('emo_control',ascending=False)
        # TTS_db_y,TTS_db_n = TTS_db[TTS_db['emo_control']=='Y'],TTS_db[TTS_db['emo_control']=='N']
        n_crt = n_crt.sort_values(by='script_num' ,ascending=False)

        TTS_db_m,TTS_db_f = TTS_db[TTS_db['gender']=='Male'],TTS_db[TTS_db['gender']=='Female']

        m, fix_m, max_m = 0, len(TTS_db_m[TTS_db_m['emo_control']=='Y']),len(TTS_db_m)
        f, fix_f, max_f = 0, len(TTS_db_f[TTS_db_f['emo_control']=='Y']),len(TTS_db_f)

        for i in range(len(n_crt)):
            if m == max_m:
                m = fix_m+1
            elif f == max_f:
                f = fix_f+1

            if n_crt.iloc[i]['gender'] == 'Male':
                crt_name=n_crt.iloc[i]['crt_name']
                n_crt.loc[n_crt['crt_name']==crt_name,'TTS'] = TTS_db_m.iloc[m]['tts_crt_name']
                m+=1
            elif n_crt.iloc[i]['gender'] == 'Female':
                crt_name=n_crt.iloc[i]['crt_name']
                n_crt.loc[n_crt['crt_name']==crt_name,'TTS'] = TTS_db_f.iloc[f]['tts_crt_name']
                f+=1
            else:
                print('not match TTS')
        
        return n_crt
    
    def text_to_speach(self, output, content):
        fail=None
        # This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
        speech_config = speechsdk.SpeechConfig(subscription=self.key[0], region=self.key[1])#region
        audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True,filename=f'{output}.wav')
        speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
        speech_synthesis_result = speech_synthesizer.speak_ssml_async(f'''
        <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="{self.language}">
            <voice name="{content[0]}">
                <mstts:express-as style="{content[1]}" styledegree="2">
                    <prosody pitch="{content[2]}%" rate="{content[3]}%">
                    "{content[4]}"
                    </prosody>
                </mstts:express-as>
            </voice>
        </speak>''').get()
        if speech_synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print(f"{output} save wav file")
        elif speech_synthesis_result.reason == speechsdk.ResultReason.Canceled:
            fail=output
            cancellation_details = speech_synthesis_result.cancellation_details
            print("Speech synthesis canceled: {}".format(cancellation_details.reason))
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                if cancellation_details.error_details:
                    print("Error details: {}".format(cancellation_details.error_details))
                    print("Did you set the speech resource key and region values?")
        return fail

class playnovel():
    def __init__(self,bk_num,chapter, line=None):
        self.bk_num=bk_num
        self.chapter=chapter
        self.line=line
    def play_file(self,path,fileform):
        path = path
        file_list = natsort.natsorted(os.listdir(path))
        file_list_py = [file for file in file_list if file.endswith(fileform)]
        category_chapter = []
        for books in file_list_py:
            if f"c{self.chapter}" in books:
                category_chapter.append(books)
        if self.line == None:     
            for fname in category_chapter[:]:
                playsound(path+fname)
        else:
            for fname in category_chapter[self.line:]:
                playsound(path+fname)

if __name__ == '__main__':
    pn = playnovel(1,1)
    pn.play_file('C:/Users/norebo/Desktop/Norobo/b1/','.wav')