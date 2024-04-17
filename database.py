from pymongo import MongoClient
import pymongo
import certifi
import pandas as pd

class MongoInfo:
    def __init__(self,uri):
        self.uri = uri
        ca = certifi.where()
        self.client = MongoClient(uri, tlsCAFile=ca)
    def database_list(self):
        '''DB 목록 불러오기 함수'''
        db_list = self.client.list_database_names()
        return db_list
    def collection_list(self, db):
        '''해당 DB의 collection 목록 불러오기 함수'''
        # db : Database명
        col_list = self.client[db].list_collection_names()
        return col_list
    
class MongoClass:
    def __init__(self, uri):
        self.uri = uri
        ca = certifi.where()
        self.client = MongoClient(uri, tlsCAFile=ca)
    def insert_mongo(self, db, collection, data, primarykey=None):
        '''DB 저장 함수'''
        # db : Database명(기존에 없으면 해당 이름으로 새롭게 생성)
        # collection : Collection명(기존에 없으면 해당 이름으로 새롭게 생성)
        # data : insert할 데이터(Dataframe형태로)
        db_conn = self.client[db]
        col_conn = db_conn[collection]
        if primarykey != None:
            db.collection.create_index([(primarykey, pymongo.ASCENDING)], unique=True)
        datalist = []
        data_dict = data.to_dict('records')
        for i in data_dict:
            datalist.append(i)
        col_conn.insert_many(datalist)
        print(f"[{db}-{collection}] {len(data_dict)} data insert")
    def load_mongo(self, db, collection, category=False,id=False):
        '''DB 불러오기 함수'''
        # db : Database명
        # collection : Collection명
        # category : 특정 조건 검색시 입력
        # id : mongoDB 저장 id 필요시 입력
        db_conn = self.client[db]
        col_conn = db_conn[collection]
        if category==False:
            find_category = {}
        else:
            find_category = category
        result = col_conn.find(find_category,{'_id':id})
        db_list=[]
        for i in result:
            db_list.append(i)
        print(f"[{db}-{collection}] data load")
        return pd.DataFrame(db_list)
    def del_data_mongo(self, db, collection,category=False):
        '''DB 데이터 삭제 함수'''
        # db : Database명
        # collection : Collection명
        # category : 특정 조건 삭제시 입력
        db_conn = self.client[db]
        col_conn = db_conn[collection]
        if category==False:
            delete_category = {}
        else:
            delete_category = category
        col_conn.delete_many(delete_category)
        print(f"[{db}-{collection}] data delete")
    def del_collection_mongo(self, db, collection):
        '''DB collection 삭제 함수'''
        # db : Database명
        # collection : Collection명
        db_conn = self.client[db]
        db_conn.drop_collection(collection)
        print(f"[{db}-{collection}] collection delete")
    # MongoDB에서 중복된 값 확인
    def check_duplicate_title(self, db, collection, bk_title):
        nl_data = self.load_mongo(db=db, collection=collection)
        return nl_data, bk_title in nl_data['bk_title'].values if not nl_data.empty else False
    
if __name__ == '__main__':
    # db_uri = 'mongodb+srv://norebodbmaster:no1re2bo3@norebodb.w2gsoyl.mongodb.net/?retryWrites=true&w=majority&appName=NoreboDB'
    # # MongoDB에 연결합니다.
    # client = MongoClient(db_uri)
    # db = client['Novel']  # 데이터베이스 선택
    data = pd.read_excel('TTS_character.xlsx')












