import time
import pandas as pd

from transformers import AutoTokenizer, pipeline, BertTokenizer, BertForSequenceClassification
from nltk.corpus import stopwords

from concurrent.futures import ThreadPoolExecutor

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import re

from sklearn.preprocessing import StandardScaler
import torch

import multiprocessing as mp
print(mp.cpu_count(), mp.current_process().name)

class EmotionModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
        self.stop_words = set(stopwords.words('english')).union({'“', '”', '.', ','})
        self.model = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=2, device=0)

    def preprocess_data(self,db_data):
        # start_time = time.time()
        novel_data = db_data.copy()
        novel_data['stopwords_contents'] = novel_data['content'].apply(self.preprocess_text)
        # end_time = time.time()
        # print(f"Data loading time: {end_time - start_time} seconds")

        return novel_data

    def preprocess_text(self, text):
        if pd.isnull(text):
            return ''
        tokens = [token for token in text.split() if token.lower() not in self.stop_words]
        return ' '.join(tokens)

    def analyze_emotion(self, context):
        emotions = self.model(context)
        return emotions

    def emotion_analysis_loop(self,pre_data):
        # start_time = time.time()
        context_list=[]
        emotions_dict = {'crt_name': [], 'label1': [], 'score1': [], 'label2': [], 'score2': []}
        num_sentences = len(pre_data)
        for i in range(num_sentences):
            current_sentence = pre_data.iloc[i]['stopwords_contents']
            # 이후 문장 가져오기 (문맥 윈도우 설정에 따라 달라질 수 있음)
            if i < len(pre_data) - 1:
                if pre_data.iloc[i]['crt_name'] == 'narrator': # narrator문장은 한문장만 분석
                    next_sentence = ""
                else : # narrator가 아닌경우
                    if pre_data.iloc[i+1]['crt_name'] != 'narrator':
                        next_sentence = ""
                    else:
                        next_sentence = pre_data.iloc[i+1]['stopwords_contents']
            else: #마지막문장
                next_sentence = ""
            context = f"{current_sentence} {next_sentence}"
            context_list.append(context)
        with ThreadPoolExecutor() as executor:
            emotions_results = list(executor.map(self.analyze_emotion, 
                                                 context_list))
        for i, emotions in enumerate(emotions_results):
            if pre_data.iloc[i]['crt_name'] == 'narrator':
                emotions_dict['label1'].append('neutral')
                emotions_dict['score1'].append(1.0)
                emotions_dict['label2'].append(None)
                emotions_dict['score2'].append(None)
            else:
                emotions_dict['label1'].append(emotions[0][0]['label'])
                emotions_dict['score1'].append(emotions[0][0]['score'])

                if len(emotions[0]) > 1:
                    emotions_dict['label2'].append(emotions[0][1]['label'])
                    emotions_dict['score2'].append(emotions[0][1]['score'])
                else:
                    emotions_dict['label2'].append(None)
                    emotions_dict['score2'].append(None)

        emotions_dict['crt_name'] = list(pre_data['crt_name'])

        emotion_df = pd.DataFrame(emotions_dict)

        # end_time = time.time()
        # print(f"emotion_analysis time: {end_time - start_time} seconds")
        return emotion_df

    @staticmethod
    def get_group_name(emotion):
        
        groups = {
            "sad": ['disappointment', 'grief', 'sadness'],
            "cheerful": ['amusement', 'joy'],
            "angry": ['anger', 'annoyance'],
            "terrified": ['confusion', 'embarrassment', 'fear', 'nervousness'],
            "unfriendly": ['disapproval', 'pride', 'remorse', 'disgust'],
            "friendly": ['approval', 'caring', 'love', 'gratitude'],
            "hopeful": ['optimism', 'relief'],
            "excited": ['admiration', 'curiosity', 'desire', 'excitement', 'realization', 'surprise'],
            "neutral": ['neutral']
        }
        for group, emotions in groups.items():
            if emotion in emotions:
                return group

        return 'None'

    def match_emotion_voice(self, emotion_df, threshold_value=0.5):
        # start_time = time.time()

        temp_group_df = pd.DataFrame()
        temp_group_df['label1_group'] = emotion_df['label1'].apply(self.get_group_name)
        temp_group_df['label2_group'] = emotion_df['label2'].apply(self.get_group_name)
        temp_group_df['same_group'] = temp_group_df['label1_group'] == temp_group_df['label2_group']

        result_emotions = []

        for index, row in emotion_df.iterrows():
            if not temp_group_df.at[index, 'same_group']:
                if row['score1'] < threshold_value:
                    result_emotions.append('neutral')
                else:
                    result_emotions.append(row['label1'])
            else:
                score_sum = row['score1'] + row['score2']
                if score_sum < threshold_value:
                    result_emotions.append('neutral')
                else:
                    result_emotions.append(row['label1'])

        emotion_df['emotion'] = result_emotions
        emotion_df.loc[emotion_df['crt_name'] == 'narrator', 'emotion'] = 'neutral'
        emotion_df['emotion_voice'] = emotion_df['emotion'].apply(self.get_group_name)

        # end_time = time.time()
        # print(f"changing & setting threshold time: {end_time - start_time} seconds")
        return emotion_df

class PersonalityModel:
    def __init__(self):
        # GPU 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 모델 및 토그나이져 설정
        self.tokenizer = BertTokenizer.from_pretrained("Minej/bert-base-personality")
        self.bert_model = BertForSequenceClassification.from_pretrained("Minej/bert-base-personality").to(self.device)
        self.scaler = StandardScaler()

        # 주요/비주요 crt df
        self.character_contents = {} 
        self.character_contents_n = {}
        self.n_character = []
        self.script_num = {}

    def pre_character_content(self,db_data):
        db_data.rename(columns = {'character' : 'crt_name'}, inplace = True)
        db_data.rename(columns = {'contents' : 'content'}, inplace = True)

        characters = db_data['crt_name'].unique()
        
        # 특수문자가 포함된 등장인물 이름에 대한 이스케이프 처리
        self.characters = [character.replace("'", "\'") if isinstance(character, str) else character for character in characters]
        self.novel = db_data[['crt_name', 'content']]

        # 전처리는 수행
        self.novel['content'] = self.novel['content'].apply(self.preprocess_text)

    @staticmethod
    def preprocess_text(text):
        if not isinstance(text, str):
            text = ''
        else:
            text = text.lower()
            text = re.sub(r'\W', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
        return text

    def extract_character_contents(self, character):
        # start_time = time.time()

        crt_contents = self.novel[self.novel['crt_name'] == character]
        if character != 'narrator':
            character_dialogues = crt_contents['content'].tolist()
            character_dialogues = [self.preprocess_text(dialogue) for dialogue in character_dialogues]
            self.character_contents[character] = character_dialogues
            self.script_num[character]=len(character_dialogues)# 대사 수를 추가

        else :
            self.character_contents[character] = None
            self.n_character.append(character)
            self.script_num[character]=0  #narrator: 대사 수 0
    
        # end_time = time.time()
        # print(f"Extracting dialogues for {character}: {end_time - start_time} seconds")

    def personality_detection(self, texts):
        # start_time = time.time()

        # 배치 처리를 위해 토큰화 및 패딩
        inputs = self.tokenizer(texts, truncation=True, padding=True, return_tensors="pt", max_length=512).to(self.device)
        
        with torch.no_grad():
            # 모델로부터 예측값 계산
            outputs = self.bert_model(**inputs)
        predictions = outputs.logits.detach().cpu().numpy()  # 각 텍스트에 대한 예측값 가져오기

        # end_time = time.time()
        # print(f"Personality detection time: {end_time - start_time} seconds")
        return predictions

    def loop_dialogue(self, dialogues):
        # start_time = time.time()

        # 배치 사이즈 설정
        batch_size = 8
        num_batches = (len(dialogues) + batch_size - 1) // batch_size
        predictions = []

        for i in range(num_batches):
            # 배치 생성
            batch_texts = dialogues[i * batch_size: (i + 1) * batch_size]
            batch_predictions = self.personality_detection(batch_texts)
            predictions.append(batch_predictions)

        # 예측 결과를 하나로 병합
        predictions = np.concatenate(predictions, axis=0)
        label_name = ['Var_E', 'Var_N', 'Var_A', 'Var_C', 'Var_O']
        df = pd.DataFrame(predictions, columns=label_name)

        end_time = time.time()
        # print(f"Loop dialogue time: {end_time - start_time} seconds")
        return df

    def content_by_character(self):
        characters = [character for character in self.characters if character != 'narrator']
        with ThreadPoolExecutor() as executor:
            list(executor.map(self.extract_character_contents, characters))
    
    def feature_scaled_mode(self):
        # start_time = time.time()

        # 캐릭터 대사 추출
        with ThreadPoolExecutor() as executor:
            executor.map(self.extract_character_contents, self.characters)

        character_ocean = {}  # script num 계산하기
        for character, dialogues in self.character_contents.items():
            if dialogues:
                ocean_df = self.loop_dialogue(dialogues)
                std_df = np.std(ocean_df, axis=0).round(2)

                character_ocean[character] = std_df.tolist()

        character_ocean_df = pd.DataFrame.from_dict(character_ocean, orient='index', columns=['Var_E', 'Var_N', 'Var_A', 'Var_C', 'Var_O'])
        ocean_scaled_array = self.scaler.fit_transform(character_ocean_df)
        ocean_scaled = np.round(ocean_scaled_array, 2)
        character_ocean_scaled = pd.DataFrame(ocean_scaled, index=character_ocean_df.index, columns=character_ocean_df.columns)

        for n_character in self.n_character:
            character_ocean_scaled.loc[n_character] = [-1,-1,-1,-1,-1]

        # script_num을 DataFrame으로 변환
        script_num_df = pd.DataFrame(self.script_num,index=['script_num']).T
        character_ocean_scaled = pd.concat([character_ocean_scaled, script_num_df],axis=1)
        # end_time = time.time()
        # print(f"Feature scaled mode time: {end_time - start_time} seconds")             

        return character_ocean_scaled
    
    @staticmethod
    def filtering(row, bk_num):
        ocean_values = ['Var_E', 'Var_N', 'Var_A', 'Var_C', 'Var_O']
        speed = [] 
        pitch = [] 

        for ocean_value in ocean_values:
            if row.get(ocean_value, -1) != -1:  
                value = row[ocean_value]
                if ocean_value in ['Var_E', 'Var_O']:  
                    if value >= 0.75:
                        speed.append(3) 
                        pitch.append(3)
                    elif value >= 0.50:
                        speed.append(2)
                        pitch.append(2) 
                    elif value >= 0.25:
                        speed.append(1) 
                        pitch.append(1)
                    else:
                        speed.append(0)
                        pitch.append(0)
                elif ocean_value == 'Var_N':  
                    if value >= 0.70:
                        speed.append(2) 
                        pitch.append(-2)
                    elif value >= 0.40:
                        speed.append(1)
                        pitch.append(-1)
                    else:
                        speed.append(0) 
                        pitch.append(0)
                elif ocean_value == 'Var_A':  
                    if value >= 0.70:
                        speed.append(2) 
                        pitch.append(-2)
                    elif value >= 0.40:
                        speed.append(1)
                        pitch.append(-1)
                    else:
                        speed.append(0) 
                        pitch.append(0)
                elif ocean_value == 'Var_C':  
                    if value >= 0.50:
                        speed.append(-2) 
                        pitch.append(0)
                    elif value >= 0.25:
                        speed.append(-1) 
                        pitch.append(0)
                    else:
                        speed.append(0)
                        pitch.append(0)
            else:
                speed.append(0)
                pitch.append(0)

        result = pd.Series({'bk_num': str(bk_num), 'speed': np.mean(speed), 'pitch': np.mean(pitch),'script_num':row['script_num']})
        return result

    @staticmethod
    def append_gender(ocean_df, p_model_db, characters):
        for character in characters:
            if character in ocean_df.index and character in p_model_db.index:
                if ocean_df.loc[character, 'crt_name'] == p_model_db.loc[character, 'crt_name']:
                    ocean_df.loc[character, 'gender'] = p_model_db.loc[character, 'gender']
                else:
                    ocean_df.loc[character, 'gender'] = None  # 캐릭터 이름이 다를 때도 추가하지 않음
            else:
                ocean_df.loc[character, 'gender'] = None  # 캐릭터가 없을 때도 추가하지 않음
        return ocean_df


# if __name__ == '__main__':
    # start = time.time()
    # data = pd.read_excel('novel.xlsx')
    # p_model = PersonalityModel()   
    # p_model.pre_character_content(data)

    # # 등장인물 변수 생성  # 'narrator'인 경우에 제외
    # characters = [character for character in p_model.characters if character != 'narrator']

    # # `extract_character_contents` 메서드 호출을 통해 등장인물별 대화 내용 추출
    # with ThreadPoolExecutor() as executor:
    #     result = list(executor.map(p_model.extract_character_contents, characters))
    # # 등장인물별 성격 특성 분석 수행
    # character_ocean = pd.DataFrame(p_model.feature_scaled_mode())

    # end = time.time()
    # print(character_ocean)
    # print(f'** Personality model total time ',round((end-start)/60, 2),' minutes **')
    
    # character_ocean.to_excel('character_personality_final.xlsx')



# if __name__ == '__main__':
#     data = pd.read_excel('novel.xlsx')
#     data.columns=['chapter', 'crt_name', 'content']

#     e_model = EmotionModel()
#     pre_data = e_model.preprocess_data(data)
#     match_emotion_voice = e_model.match_emotion_voice(e_model.emotion_analysis_loop(pre_data))

#     data['emotion_voice']=match_emotion_voice['emotion_voice']
#     print(data)