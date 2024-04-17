# B_tts_service
## BERT 성격, 감정분석 기반 TTS 독서 도우미 서비스

* 개요 : 단순 TTS를 이용한 독서를 넘어 각 인물의 성격과 대사의 감정을 적용한 목소리 적용
  
* 핵심기술내용 : GPT, Bert Pretrain model, Azure TTS service
  
* 코드구성
  app.py : flask를 이용한 페이지 생성 및 작동 << 실행파일
  bertmodel.py : 성격, 감정 분석에 활용하는 bert 사전학습모델
  config.py : api 및 db key 입력 << 사전입력필요
  database.py : Mongodb 연결 및 함수 정리
  gptcontrol.py : chatgpt연결 및 프롬프트 입력 작업
  main.py : 내부 프로세스 연결 작동
  titlecheck.py : e-book 형태에서 책정보 추출
  ttstransform.py : Azure TTS service 연결 및 음원 추출
  epubfile : e-book 업로드 파일 저장소
  result : 각 모델 결과 저장소
  static : 음원, 이미지 등 고객 제공 내용 저장소
  templates : 웹 구성 저장소

* 프로세스 요약
  1. e-book 데이터 업로드
  2. GPT를 이용하여 소설을 스크립트화
  3. bert 사전 모델을 이용하여 각 대사의 감정 추출
  4. bert 사전 모델을 이용하여 각 인물의 성격 추출
  5. 소설 등장인물마다 특정 TTS 목소리 부여
  6. 부여된 목소리, 인물에 성격에 따라 목소리 크기와 속도 조절, 대사마다 감정 추가의 정보로 TTS 음원 추출 
