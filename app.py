from flask import Flask, render_template, send_from_directory, url_for, request, redirect
from pymongo import MongoClient
import certifi
import os
import subprocess
import pyautogui
import pandas as pd
from config import config
db_uri = config['mongodb_uri']

app = Flask(__name__, template_folder='templates')
UPLOAD_FOLDER = 'epubfile/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# MongoDB 연결
client = MongoClient(db_uri, tlsCAFile=certifi.where())
db = client['Novel']
collection_1 = db['noveldetail_line']
collection_2 = db['novellist']

novellist=[]
for noveltitle in list(collection_2.find({},{'_id':0,'bk_title':1})):
    novellist.append(noveltitle['bk_title'])

# 오디오 파일이 있는 디렉토리 경로
audio_dir = 'static/audiofile/'   # 수정해야될 부분
# 전역으로 사용할 오디오 플레이어 객체 생성
audio_player = None

@app.route('/')
def main():
    return render_template('main_page.html', novellist=novellist)

@app.route('/<novelname>')
def index(novelname):
    global audio_player  # 전역으로 사용할 오디오 플레이어 객체를 가져옴
    novel_info = pd.DataFrame(collection_2.find({'bk_title':novelname},{"_id":0, 'bk_num':1}))
    bk_num = int(novel_info['bk_num'].values[0])
    novel_data = pd.DataFrame(collection_1.find({"bk_num": bk_num}, {"_id": 0,"chapter": 1,"content": 1,"line": 1}))
    chapter_list = list(set(novel_data.chapter.to_list()))
    # 각 챕터별로 데이터를 가져오기
    chapters_data = []
    for chapter in chapter_list:
        df_each_chapter = novel_data[novel_data['chapter']==chapter]
        content_list = [{"content":df_each_chapter.iloc[i]['content'],
                         "audio": f"book{bk_num}/chapter{chapter}/line{df_each_chapter.iloc[i]['line']}.wav"} for i in range(len(df_each_chapter))]
        chapters_data.append({"chapter_num": chapter, "content_list": content_list})

    # 책 제목과 저자 정보 가져오기
    book_info = collection_2.find_one({"bk_num": bk_num}, {"_id": 0, "bk_title": 1, "author": 1})
    bk_title = book_info["bk_title"]
    author = book_info["author"]

    return render_template('template.html', bk_title=bk_title, author=author, chapters=chapters_data, audio_player=audio_player, novelname=novelname)

@app.route('/audio/<path:filename>')
def download_file(filename):
    return send_from_directory(audio_dir, filename)

@app.route('/add_name', methods=['POST'])
def add_name():
    # 파일 받기
    file = request.files['file']
    if file:
        if file.filename in os.listdir(app.config['UPLOAD_FOLDER']):
            pyautogui.alert('해당 소설은 목록에서 찾으실 수 있습니다.')
        else:
            # 파일을 업로드 폴더에 저장
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            
            # main.py 호출하여 파일 경로 전달하고 결과 받아오기
            process = subprocess.Popen(['python', 'titlecheck.py', file_path], stdout=subprocess.PIPE)
            output, _ = process.communicate()
            new_name = output.decode().strip()
            # 새로운 이름을 names 리스트에 추가
            novellist.append(new_name)
    
            subprocess.Popen(['python', 'main.py', file_path])
            # os.remove(file_path)  # 파일 삭제
    return redirect(url_for('main'))

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)