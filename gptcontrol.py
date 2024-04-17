import time
import re
import json
import pandas as pd
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import openai
from openai.error import APIError
import math
import tiktoken

class BookProcessor:
    def __init__(self, epub_file, api_key, model):
        self.epub_file = epub_file
        self.api_key = api_key
        self.model = model
        self.book_info = {'bk_num':'', 'bk_title':'', 'author':'', 'language':''}
        self.book_dict = {}
        self.who_dict = {}
        self.crt_dict = {}
        self.book_df = pd.DataFrame(columns=['chapter','content'])
        self.gpt_df = pd.DataFrame(columns=['content'])
        self.book = self.read_epub()

    def read_epub(self):
        book = epub.read_epub(self.epub_file)

        title = book.get_metadata("DC", "title")
        if title:
            title = title[0][0]
        else:
            title = "Unknown"
        author = book.get_metadata("DC", "creator")
        if author:
            author = author[0][0]
        else:
            author = "Unknown"
        language = book.get_metadata("DC", "language")
        if language:
            language = language[0][0]
        else:
            language = "Unknown"

        self.book_info['bk_title'] = title
        self.book_info['author'] = author
        self.book_info['language'] = language

        return book

    def extract_book_info(self, book):
        source = BeautifulSoup("", "html.parser")
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                source.append(BeautifulSoup(item.get_body_content(), "html.parser"))

        chapters = source.find_all('div', class_='chapter')

        for index, chapter in enumerate(chapters, start=1):
            chapter_title = chapter.find('h2').text.strip()
            chapter_content = ""
            for paragraph in chapter.find_all('p'):
                chapter_content += paragraph.get_text().strip()
            cleaned_text = chapter_content.replace('\n', ' ').replace('\r', '')
            self.book_dict[f"chapter{index}"] = cleaned_text
            
        for i in range(0, len(self.book_dict)):
            self.book_dict[f"chapter{i+1}"] = self.attach_who(self.separate_sentences(self.book_dict[f"chapter{i+1}"]))

    def separate_sentences(self, text):
        sentences = []
        current_sentence = ''
        abbreviations = ['Mr.', 'Mrs.', 'Dr.', 'Ms.', 'Prof.']
        alphabet_list = [char + '.' for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']
        abbreviations += alphabet_list

        within_dict = {'“':0, '(':0, '”':0, ')':0}

        for char in text:
            if char in ['“', '(']:
                within_dict[char] = 1
                check_within_start = within_dict['“'] + within_dict['(']
                if check_within_start == 1:
                    if current_sentence.strip() != '':
                        sentences.append(current_sentence.strip())
                        current_sentence = ''
                current_sentence += char  
            elif char in ['”', ')']:
                within_dict[char] = -1
                current_sentence += char
                check_within_end = sum(within_dict.values())
                if check_within_end == 0:
                    sentences.append(current_sentence.strip())
                    current_sentence = ''
                    within_dict = dict.fromkeys(within_dict, 0)
            elif char == '.' :
                current_sentence += char 
                word = current_sentence.split()
                check_within_end = sum(within_dict.values())
                if word[-1] in abbreviations:
                    continue
                if check_within_end != 0:
                    continue
                else:
                    sentences.append(current_sentence.strip())
                    current_sentence = ''
            else:
                current_sentence += char

        if current_sentence:
            sentences.append(current_sentence.strip())

        merged_sentences = []
        for idx, sentence in enumerate(sentences):
            if len(sentence) == 1 and idx > 0:
                merged_sentences[-1] += sentence
            else:
                merged_sentences.append(sentence)

        return merged_sentences

    def attach_who(self, sentences):
        who_sentence = []
        for sentence in sentences:
            if '“' in sentence:
                sentence = '<who=?>' + sentence + '</who>'
            else:
                sentence = '<who=narrator>' + sentence + '</who>'
            who_sentence.append(sentence)
            who_sentence_text = ''.join(who_sentence)
        return who_sentence_text

    def divide_text(self, text, num_parts):
        total_length = len(text)
        avg_length = total_length // num_parts

        parts = []
        start_idx = 0
        for i in range(num_parts):
            if i == num_parts - 1:
                parts.append(text[start_idx:])
            else:
                end_idx = start_idx + avg_length
                while text[end_idx] != "<" or not text.startswith("</who>", end_idx):
                    end_idx -= 1
                parts.append(text[start_idx:end_idx + len("</who>")])
                start_idx = end_idx + len("</who>")
        return parts
    
    def generate_token_list(self):
        encoder = tiktoken.encoding_for_model(self.model)
        tokens_list = []
        for i in range(0,len(self.book_dict)):
            encoded_text = encoder.encode(self.book_dict["chapter"+str(i+1)])
            total_tokens = len(encoded_text)
            tokens_list.append(math.ceil(total_tokens/3000))
        return tokens_list

    def gpt(self, text):
        for i in range(0, len(self.book_dict[text])):
            prompt = f"{self.book_dict[text][i]}\
            The above novel is part of <{self.book_info['bk_title']}>.\
            I would like to analyze the novel to identify the characters in the dialogue.\
            In <who=?>, please fill in the ? with a character.Don't miss the original text.\
            You can write the <who=narrator> part as it is in the original text.\
            Please do not say anything except the original text filled with ?."
            # fewshot 추가(오류 방지)
            messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
            ]

            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages
            )
            answer = response['choices'][0]['message']['content']
            self.who_dict[text].append(answer)
            time.sleep(3)

    def crt_gpt(self, df) :
        anwser_list = []
        df['crt_name'] = df['crt_name'].str.lower()
        characters = df['crt_name'].astype(str).unique()
        prompt = f"""
        {characters}
        Above is a list of characters from the novel <{self.book_info['bk_title']}>.
        I would like you to analyze the novel {self.book_info['bk_title']} and replace any overlapping characters, excluding the narrator, with one representative character.
        At this time, the judgment of overlapping characters is that they are the same person but have different names.
        Use dict as output method.
        Example) {{'Peter Pan':['peter','Peter','Peter Pan']}}
        Don't say anything other than example answers.
        """

        messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
        ]

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages
        )
        answer = response['choices'][0]['message']['content']
        anwser_list.append(answer)

        pattern = r'{[^{}]+}'
        matches = re.findall(pattern, anwser_list[0])
        character_text = matches[0]
        self.crt_dict = eval(character_text)
        
    def merge_crt(self,df):
        for index, row in df.iterrows():
            crt_name = row['crt_name']
            for key, value in self.crt_dict.items():
                if crt_name in value:
                    df.at[index, 'crt_name'] = key
                    break
        df.loc[df['crt_name'].isna(), 'crt_name'] = 'narrator'

    def process_book(self):
        self.extract_book_info(self.book)
        self.tokens_list = self.generate_token_list()
        for i in range(0, len(self.book_dict)):
            self.who_dict[f"chapter{i+1}"] = []
            self.book_dict[f"chapter{i+1}"] = self.divide_text(self.book_dict[f"chapter{i+1}"], self.tokens_list[i])
            result_text = "\n".join(self.book_dict[f'chapter{i + 1}'])
            result_text = result_text.replace("</who>", "</who>\n")
            lines = result_text.strip().split('\n')
            chapter_df = pd.DataFrame(lines, columns=['content'])
            chapter_df['chapter'] = i + 1    
            self.book_df = pd.concat([self.book_df, chapter_df], ignore_index=True)
        self.book_df = self.book_df[self.book_df['content'].str.len() >= 1]

        i = 0  # 초기화
        while i < len(self.book_dict):
            try:
                self.gpt(f"chapter{i+1}")
                result_text = "\n".join(self.who_dict[f'chapter{i + 1}'])
                result_text = result_text.replace("</who>", "</who>\n")
                lines = result_text.strip().split('\n')
                chapter_df = pd.DataFrame(lines, columns=['content'])
                self.gpt_df = pd.concat([self.gpt_df, chapter_df], ignore_index=True)
                i += 1 
            except APIError as e:
                print(f"An API error occurred: {e}, chapter{i + 1}")
                print("3분동안 실행을 중지합니다 ..")
                if self.who_dict[f'chapter{i + 1}'] != []:
                    i -= 1
                time.sleep(180)
                print("실행을 다시 재개합니다 ..")
                continue
        self.gpt_df = self.gpt_df[self.gpt_df['content'].str.len() >= 1]               
                
    def save_results(self):
        self.book_df.to_excel(f'{self.book_info["bk_title"]}.xlsx', index=False)
        self.gpt_df.to_excel(f'{self.book_info["bk_title"]}_gpt.xlsx', index=False)
        file_name = f'who_dict_{self.book_info["bk_title"]}.json'
        with open(file_name, "w") as json_file:
            json.dump(self.who_dict, json_file, indent=4)
            
    def load_results(self):
        self.book_df = pd.read_excel(f'{self.book_info["bk_title"]}.xlsx')
        self.gpt_df = pd.read_excel(f'{self.book_info["bk_title"]}_gpt.xlsx')
        file_name = f'who_dict_{self.book_info["bk_title"]}.json'
        with open(file_name, "r") as json_file:
            self.who_dict = json.load(json_file)
        
    def gender_gpt(self, df) :
        anwser_list = []
        characters = df['crt_name'].astype(str).unique().tolist()
        num_crt = len(characters)
        prompt = f"""
        {characters}
        Above is a list of characters from the novel <{self.book_info['bk_title']}>.
        I would like you to analyze the novel {self.book_info['bk_title']} and identify the genders of the {num_crt} characters.
        There are only two genders: Male and Female.
        If you don't know the gender, enter Male.
        If a word contains an apostrophe ('), enclose the word with double quotes (") instead of single quotes (').
        When you return the result, please enter it in the order of the characters entered.
        There is a 1:1 correspondence between the characters and gender.
        Please make sure to include all {num_crt} characters.
        Just tell me the results I requested without any explanation.
        Below is an example when you have completed the task.
        {{'narrator':'Male',
        "Nick's father":'Male',
        'Daisy':'Female'}}
        """
        messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
        ]
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages
        )
        answer = response['choices'][0]['message']['content']
        anwser_list.append(answer)
        pattern = r'{[^{}]+}'
        matches = re.findall(pattern, anwser_list[0])
        character_text = matches[0]
        gender_dict = eval(character_text)
        gender_dict = {key: 'Male' if value != 'Male' and value != 'Female' else value for key, value in gender_dict.items()}
        return gender_dict
    
class DataTransformer:
    def __init__(self) -> None:
        pass

    def transform_data(book_df,gpt_df):
        who = []
        content = []
        chapter = []
        for line in book_df.values:
            search_who = re.search('<who=(.+?)>', line[1]).group(1)
            search_content=(line[1].split('>'))[1].split('<')[0]
            who.append(search_who)
            content.append(search_content)
            chapter.append(line[0])
        transfrom_book = pd.DataFrame({'crt_name':who,'content':content,'chapter':chapter})
        # transfrom_html.to_excel('book_trf.xlsx', index=False)

        who = []
        content = []
        gpt_df = gpt_df[gpt_df['content'].str.startswith('<')]
        for line in gpt_df['content'].values:
            search_content = line.split('>')[1].split('<')
            if len(search_content) == 1:
                who.append('narrator')
                content.append(line)   
            else:
                search_who = re.search('<who=(.+?)>', line).group(1)
                who.append(search_who)
                content.append(search_content[0])
        transfrom_gpt = pd.DataFrame({'crt_name':who,'content':content})
        # transfrom_gpt.to_excel('book_gpt_trf.xlsx', index=False)

        return transfrom_book, transfrom_gpt

    def compare_data(book_df,gpt_df, title):
        transfrom_book, transfrom_gpt = DataTransformer.transform_data(book_df,gpt_df)
        original_content=transfrom_book['content']
        gpt_content=transfrom_gpt['content']

        gpt_max = len(gpt_content)-1
        gpt_index = 0
        gpt_first_word = [gpt_content[gpt_index].split(" ")[0],gpt_content[gpt_index+1].split(" ")[0]]
        # pp = []
        
        for origin_index, origin_ctt in enumerate(original_content):
            origin_spl = origin_ctt.split(" ")
            if len(origin_spl) == 1:
                origin_first_word = origin_spl[0]
                # pp.append([origin_first_word,'None', gpt_first_word])
            else:
                origin_first_word = [origin_spl[0],origin_spl[0]+origin_spl[1]]
                # pp.append([origin_first_word[0],origin_ctt.split(" ")[0]+origin_ctt.split(" ")[1], gpt_first_word])
            
            if  gpt_first_word[0] in origin_first_word:
                transfrom_book['crt_name'][origin_index]=transfrom_gpt['crt_name'][gpt_index]
                if gpt_index < gpt_max-1:
                    gpt_index += 1

            elif gpt_first_word[1] in origin_first_word:
                transfrom_book['crt_name'][origin_index]=transfrom_gpt['crt_name'][gpt_index+1]
                if gpt_index < gpt_max-2:
                    gpt_index += 2

            else:
                if gpt_first_word == 'nan':
                    if gpt_index < gpt_max-1:
                        gpt_index += 1
            
            gpt_first_word = [str(gpt_content[gpt_index]).split(" ")[0],str(gpt_content[gpt_index+1]).split(" ")[0]]

        unknown = transfrom_book[transfrom_book['crt_name']=='?']
        for ui,uv in zip(unknown.index,unknown.values):
            if len(transfrom_gpt[transfrom_gpt['content']==uv[1]]) == 1:
                transfrom_book['crt_name'][ui]=transfrom_gpt['crt_name'][transfrom_gpt[transfrom_gpt['content']==uv[1]].index].values
                
        transfrom_book.to_excel(f'novel_{title}.xlsx', index=False)
        return transfrom_book
    
    def add_crt(df):
        df.loc[df['crt_name'] == '?', 'crt_name'] = 'narrator'  

    def add_data(df, bk_num):
        df['bk_num'] = bk_num
        # 'line' 열 추가하고 행 번호로 채우기
        df['line'] = range(1, len(df) + 1)
        # 새로운 열 순서 정의
        new_column_order = ['bk_num', 'chapter', 'line', 'crt_name', 'content']
        # 열 순서 변경
        df = df.reindex(columns=new_column_order)
        return df
    
if __name__ == '__main__':
    # EPUB 파일 경로 설정
#     epub_file = 'pg64317-images-3.epub' # Gatsby
    epub_file = 'pg16-images-3.epub'  # Peter pan
#     epub_file = 'pg11-images-3.epub' # Alice
    # epub_file = 'pg768-images-3.epub' # Wuthering Heights

    # GPT 모델 및 API 키 설정
    model = "gpt-4-0125-preview"

    # openai.api_key = 

    # 책 처리 객체 생성
    processor = BookProcessor(epub_file, openai.api_key, model)
    book = processor.book_info
    
    # MongoDB에서 중복된 값 확인
    def check_duplicate_title(Mongo_consol, db, collection, bk_title):
        nl_data = Mongo_consol.load_mongo(db=db, collection=collection)
    #     return nl_data, bk_title in nl_data['bk_title'].values if not nl_data.empty else False
    # # url
    # db_uri = 'mongodb+srv://norebodbmaster:no1re2bo3@norebodb.w2gsoyl.mongodb.net/?retryWrites=true&w=majority&appName=NoreboDB'
    # # MongoDB Class
    # Mongo_consol = ongoClass(uri=db_uri)
    # nl_data, bk_title = check_duplicate_title(Mongo_consol, db='test', collection='pjwtestlist', bk_title=book['bk_title'])
    # # MongoDB에서 중복된 값 확인
    # if bk_title:
    #     print("중복된 값입니다.")
    # else:
    #     # 비어 있는 경우, 데이터프레임을 삽입할 때 예외 처리
    #     if nl_data.empty:
    #         last_item = 0
    #     else:
    #         last_index = len(nl_data) - 1
    #         last_item = nl_data['bk_num'][last_index]
    #     book['bk_num'] = last_item + 1
    #     info_df = pd.DataFrame.from_dict(book, orient='index').T
    #     Mongo_consol.insert_mongo(db='test', collection='pjwtestlist', data=info_df)
    
    # 책 처리 및 결과 저장
#     processor.process_book() # gpt 사용
#     processor.save_results() # gpt 사용
    processor.load_results() # 저장된 파일 사용
