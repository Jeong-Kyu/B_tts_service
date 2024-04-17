import pandas as pd
import time
import os
import gptcontrol
import database
import bertmodel
import openai
import ttstransform
from config import config
import sys

# EPUB 파일 경로 설정
epub_file = sys.argv[1].strip() # Gatsby
# epub_file = 'epubfile/pg16-images-3.epub'  # Peter pan
# epub_file = 'pg11-images-3.epub' # Alice
# epub_file = 'pg768-images-3.epub' # Wuthering Heights

# GPT 모델 및 API 키 설정
model = config['gpt_model']
openai.api_key = config['gpt_api_key']
tts_subscription = config['azure']
db_uri = config['mongodb_uri']

# 책 처리 객체 생성
processor = gptcontrol.BookProcessor(epub_file, openai.api_key, model)
book = processor.book_info
# MongoDB Class
Mongo_consol = database.MongoClass(uri=db_uri)
nl_data, bk_title = database.MongoClass.check_duplicate_title(Mongo_consol, db='Novel', collection='novellist', bk_title=book['bk_title'])

# MongoDB에서 중복된 값 확인
if bk_title:
    print("중복된 값입니다.")
    bk_num = int(nl_data[nl_data['bk_title']==book['bk_title']]['bk_num'].values)
    novel_df =Mongo_consol.load_mongo(db='Novel', collection='noveldetail_line',category={'bk_num':bk_num})
    
    # 저장된 최종 결과 불러오는 곳#######
else:
    # 비어 있는 경우, 데이터프레임을 삽입할 때 예외 처리
    if nl_data.empty:
        last_item = 0
    else:
        last_index = len(nl_data) - 1
        last_item = nl_data['bk_num'][last_index]
    book['bk_num'] = last_item + 1
    bk_num = int(book['bk_num'])
    info_df = pd.DataFrame.from_dict(book, orient='index').T
    Mongo_consol.insert_mongo(db='Novel', collection='novellist', data=info_df)

    # 책 처리 및 결과 저장
    processor.process_book() # gpt 사용
    processor.save_results() # gpt 사용
    processor.load_results() # 저장된 파일 사용
    novel_df = gptcontrol.DataTransformer.compare_data(processor.book_df, processor.gpt_df, book["bk_title"])
    novel_df = gptcontrol.DataTransformer.add_data(novel_df, book['bk_num'])
    gptcontrol.DataTransformer.add_crt(novel_df) # ? 채우기
    processor.crt_gpt(novel_df) # 캐릭터 합치기 gpt
    processor.merge_crt(novel_df) # 캐릭터 합치기 df 적용
    novel_df.to_excel(f'result/novel_{book['bk_num']}_{book["bk_title"]}.xlsx', index=False)
    Mongo_consol.insert_mongo(db='Novel', collection='noveldetail_line', data=novel_df)

print('successfully saved to MONGODB')

#############################################################################################################
# emotion model 실행
start = time.time()
emotion_df = novel_df.copy()
e_model = bertmodel.EmotionModel()
pre_data = e_model.preprocess_data(emotion_df)
match_emotion_voice = e_model.match_emotion_voice(e_model.emotion_analysis_loop(pre_data))
emotion_df['emotion_voice']=match_emotion_voice['emotion_voice']
end = time.time()
print(f'** Emotion model total time ',round((end-start)/60, 2),' minutes **')
emotion_df.to_excel(f'result/emotion_{book['bk_num']}_{book['bk_title']}.xlsx')
print('successfully running the emotion model')
#############################################################################################################
# personality model 실행
start = time.time()
p_model = bertmodel.PersonalityModel()   
p_model.pre_character_content(novel_df)
characters = [character for character in p_model.characters if character != 'narrator']
p_model.content_by_character()
character_ocean = pd.DataFrame(p_model.feature_scaled_mode())
character_filtering = character_ocean.apply(p_model.filtering, axis=1, bk_num=bk_num)
end = time.time()
print(f'** Personality model total time ',round((end-start)/60, 2),' minutes **')
character_ocean.to_excel(f'result/personality_{book['bk_num']}_{book['bk_title']}.xlsx')
print('successfully running the personality model')
#############################################################################################################
start = time.time()
novel_character_detail = character_filtering.copy()
novel_character_detail.insert(1,'crt_name',novel_character_detail.index)
novel_character_detail['gender']=None
novel_character_detail=novel_character_detail.reset_index(drop=True)
gender_list = processor.gender_gpt(novel_character_detail)
for i in range(len(novel_character_detail)):
    if novel_character_detail.iloc[i]['crt_name'] in gender_list.keys():
        gender = gender_list[novel_character_detail.iloc[i]['crt_name']]
    else:
        gender = 'Male'
    novel_character_detail.at[i,'gender'] = gender
novel_character_detail['bk_num']=novel_character_detail['bk_num'].astype(int)
end = time.time()
print(f'** gender gpt total time ',round((end-start)/60, 2),' minutes **')
character_ocean.to_excel(f'result/chracter_ttsinfo_{book['bk_num']}_{book['bk_title']}.xlsx')
Mongo_consol.insert_mongo(db='Novel', collection='novelcharacter', data=novel_character_detail)
# novel_character_detail = Mongo_consol.load_mongo(db='Novel', collection='novelcharacter', category={'bk_num':bk_num})
############################################################################################################
tts_control = ttstransform.azuretts(tts_subscription)
NC_db = novel_character_detail
TTS_db = Mongo_consol.load_mongo(db='TTS', collection='TTS_character')
matching_voice = tts_control.match_voice(NC_db,TTS_db)

start = time.time()
os.makedirs(f"static/audiofile/book{bk_num}")
for i in range(1,10):
    os.mkdir(f"static/audiofile/book{bk_num}/chapter{i}")
fails=[]
for emo in emotion_df.values[:]:
    match_crt = matching_voice[matching_voice['crt_name']==emo[3]]
    fail=tts_control.text_to_speach(f'audio/book{bk_num}/chapter{emo[1]}/line{emo[2]}',[match_crt['TTS'].values[0],emo[5],match_crt['pitch'].values[0],match_crt['speed'].values[0],emo[4]])
    if fail != None:
        fails.append(fail)
end = time.time()
print(f'** TTS-transform total time ',round((end-start)/60, 2),' minutes **')
print('successfully running the TTS')

# ** TTS-transform total time  16.13-20  minutes **
# novel_list = Mongo_consol.load_mongo(db='Novel', collection='novellist',category={'bk_num':bk_num})
# novel_detail =Mongo_consol.load_mongo(db='Novel', collection='noveldetail_line',category={'bk_num':bk_num})
# novel_detail['bk_title'] = novel_list['bk_title']
# novel_detail['author'] = novel_list['author']
# Mongo_consol.insert_mongo(db='Web', collection='result', data=novel_detail)

# pn = ttstransform.playnovel(1,1,1)
# pn.play_file(f'b{bk_num}/','.wav')#mp3알아보기