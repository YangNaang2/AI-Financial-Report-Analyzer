# 📈 AI 기반 증권사 리포트 '숨은 매도 시그널' 탐지하기

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/Transformers-FFD21E?logo=huggingface&logoColor=black)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)

## 👨‍💻 프로젝트 동기

<div align="center">
  <strong>"목표 주가는 하향하지만, 투자의견은 매수를 유지합니다."</strong>
</div>
<br>

투자를 좋아하기에 직접 공부도 하고 투자를 하는 과정에 있어서 여러 증권사의 리포트를 필연적으로 읽게 된다.  
이때 많이 접하게 되는 문장이다.  
최근 금융 시장에 대한 대중의 관심이 매우 높아졌다.   
하지만 과거와는 다르게 매도 리포트, 혹은 애널리스트의 소신을 담은 보고서가 갈수록 점점 보기 힘들어진다는 생각이 들었다.

<p align="center">
  <a href="https://youtu.be/E5Ue34SvIcw?si=iub3zFq9MFhaagnV">
    <img src="https://img.youtube.com/vi/E5Ue34SvIcw/maxresdefault.jpg" width="60%" alt="슈카월드 비판의 종말">
  </a>
  <br>
  <em>▲ 슈카월드 "비판의 종말" (Ctrl+이미지 클릭 시 영상 재생)</em>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/d736a030-ab0d-49d9-98c8-e6d61c6a7107" width="49%" />
  <img src="https://github.com/user-attachments/assets/f14b42de-e728-44af-ac98-eb82f4070764" width="49%" />
</p>
그렇다면 보고서 내에 숨겨진 뜻, 애널리스트가 하고싶은 말을 찾는 것은 독자의 몫이다.
겉으로는 긍정적인 뉘앙스를 띠고 있더라도, 실제 리포트 발간 후 주가가 하락했던 과거의 패턴들을 AI에게 학습시켜 실질적인 매도 시그널을 탐지하고자 하였다.

## ✨ 프로젝트 소개
증권사 리포트의 텍스트(비정형 데이터)를 분석하여, 명시적인 투자의견(Buy/Hold) 이면에 숨겨진 **'실질적 매도(Negative) 시그널'을 탐지하는 자연어 처리(NLP) 파이프라인**이다. 
웹 크롤링을 통한 원시 데이터 수집부터 PDF 텍스트 전처리, 주가 데이터를 활용한 자동 라벨링, 그리고 금융 특화 언어모델(KoFinBERT) 파인튜닝까지 전 과정을 엔드투엔드(End-to-End)로 자동화했다.

## ⚙️ 파이프라인 (Data Pipeline)

1. **데이터 수집 (`crawler.py`)**
   * 네이버 금융에서 과거(N개월 전) 증권사 리포트 PDF 자동 다운로드
2. **비정형 데이터 전처리 (`data_processor.py` / `main.py`)**
   * `pdfplumber`를 활용한 텍스트 추출
   * 정규표현식(Regex)을 활용한 노이즈(이메일, 전화번호, 주가 테이블 등) 제거
   * 증권사별 다양한 포맷에서 종목코드(6자리) 및 영문/국문 발간일 자동 파싱
3. **오토 라벨링 (`labeler.py`)**
   * `FinanceDataReader`를 연동하여 리포트 발간일 기준 28일(약 1개월) 후의 실제 주가 수익률 계산
   * 수익률 -5% 이하 하락 시 '숨은 매도(1)', 그 외 '유지/매수(0)'로 퀀트 기반 자동 라벨링
4. **AI 모델 파인튜닝 (`train.py`)**
   * 구축된 정답셋을 바탕으로 `kwoncho/KoFinBERT` 모델 파인튜닝
   * 버전 관리 기법(Timestamp)을 적용하여 모델 가중치(`safetensors`) 자동 저장
5. **실전 추론 (`inference.py`)**
   * 저장된 최신 모델을 로드하여 새로운 문장에 대한 숨은 매도 시그널 확률 추론

## 📂 디렉토리 구조 (Directory Structure)
```text
📦 AI-Financial-Report-Analyzer
 ┣ 📜 crawler.py           # 리포트 다운로드 크롤러
 ┣ 📜 data_processor.py    # PDF 텍스트 추출 및 정제 모듈
 ┣ 📜 labeler.py           # 주가 연동 및 수익률 기반 라벨링 모듈
 ┣ 📜 train.py             # KoFinBERT 파인튜닝 모듈
 ┣ 📜 main.py              # 전체 파이프라인 통합 및 실행
 ┣ 📜 inference.py         # 최신 모델 자동 로드 및 텍스트 추론(터미널용)
 ┣ 📜 app.py               # 최신 모델 자동 로드 및 텍스트 추론(GUI용)
 ┣ 📜 .gitignore           # 대용량 모델 및 캐시 파일 업로드 방지
 ┗ 📂 pdf_data/            # (Local) 크롤링된 원본 PDF 파일
 ┗ 📂 models/              # (Local) 학습 완료된 AI 모델 가중치
```

## 🌟 실행 방법 (Getting Started)

**1. 패키지 설치**
```bash
pip install requests beautifulsoup4 pdfplumber finance-datareader pandas torch transformers
```
**2. 파이프라인 가동**
```bash
# 1. 학습용 과거 리포트 PDF 수집
python crawler.py

# 2. 데이터 정제, 라벨링 및 모델 학습 (Model 생성)
python main.py

# 3. 최신 모델을 활용한 문장 추론 테스트(터미널용) (파일 내에 원하는 문장 입력 후)
python inference.py

# 4. 🌐 Streamlit 웹 대시보드 실행 (GUI)
streamlit run app.py

```

## 📊 실행 결과 (Results)

<p align="center">
  <img src="https://github.com/user-attachments/assets/3c9b1d15-3355-4605-bb67-d4ac1d68c87f" width="49%" />
  <br>
  <em>▲ 1. 백엔드(Terminal) 환경에서의 파이프라인 추론 결과</em>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/b4097d2b-55ec-4faa-b93a-a1e08152ab06" width="49%" />
  <br>
  <em>▲ 2. Streamlit을 활용한 사용자 친화적 웹 대시보드(GUI) 시연</em>
</p>

## 🎯 최종 결론 및 성능 향상 가이드

현재 깃허브에 업로드된 파이프라인은 시스템 검증을 위해 소량의 데이터로 학습된 베이스라인 모델이다. 한국 증권사 리포트 특성상 '부정적(매도)' 데이터가 극히 적은 **클래스 불균형(Class Imbalance)** 문제가 존재한다.

실제 퀀트 트레이딩 수준의 **정확도**를 얻기 위해서는...

1. **데이터 볼륨 확대 (수집량 증가)**
   * `crawler.py`의 다운로드 페이지 범위를 대폭 늘려(예: `start_page=1, end_page=200`) 수천 건 이상의 리포트를 수집.
2. **모델 학습 반복 횟수(Epoch) 증가**
   * 데이터가 많아진 만큼, `train.py` 내부의 학습 파라미터(예: `num_train_epochs`)를 3~5 이상으로 늘려 AI가 금융 텍스트의 미세한 뉘앙스 차이를 더 깊게 학습하도록 유도.
3. **학습 데이터 밸런싱**
   * 수집된 데이터 중 긍정(0)과 부정(1) 데이터의 비율을 인위적으로 1:1에 가깝게 맞춰주면(Under-sampling 등), AI가 '모르면 일단 유지/매수라고 찍는' 현상을 방지할 수 있다.
