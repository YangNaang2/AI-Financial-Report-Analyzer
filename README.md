# 📈 AI 기반 증권사 리포트 '숨은 매도 시그널' 탐지 파이프라인

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/Transformers-FFD21E?logo=huggingface&logoColor=black)

## 📌 프로젝트 소개
증권사 리포트의 텍스트(비정형 데이터)를 분석하여, 명시적인 투자의견(Buy/Hold) 이면에 숨겨진 **'실질적 매도(Negative) 시그널'을 탐지하는 자연어 처리(NLP) 파이프라인**입니다. 
웹 크롤링을 통한 원시 데이터 수집부터 PDF 텍스트 전처리, 주가 데이터를 활용한 자동 라벨링, 그리고 금융 특화 언어모델(KoFinBERT) 파인튜닝까지 전 과정을 엔드투엔드(End-to-End)로 자동화했습니다.

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

## 🛠️ 기술 스택 (Tech Stack)
* **언어:** Python
* **데이터 수집 및 가공:** `requests`, `BeautifulSoup4`, `pdfplumber`, `re` (Regex)
* **금융 데이터 연동:** `FinanceDataReader`, `pandas`
* **AI / Deep Learning:** `torch`, `transformers` (Hugging Face)

## 🚀 실행 방법 (Getting Started)

**1. 패키지 설치**
```bash
pip install requests beautifulsoup4 pdfplumber finance-datareader pandas torch transformers

**2. 파이프라인 가동**
```bash
# 1. 학습용 과거 리포트 PDF 수집
python crawler.py

# 2. 데이터 정제, 라벨링 및 모델 학습 (Model 생성)
python main.py

# 3. 최신 모델을 활용한 문장 추론 테스트
python inference.py
```

## 📂 디렉토리 구조 (Directory Structure)
```text
📦 AI-Financial-Report-Analyzer
 ┣ 📜 crawler.py           # 리포트 다운로드 크롤러
 ┣ 📜 data_processor.py    # PDF 텍스트 추출 및 정제 모듈
 ┣ 📜 labeler.py           # 주가 연동 및 수익률 기반 라벨링 모듈
 ┣ 📜 train.py             # KoFinBERT 파인튜닝 모듈
 ┣ 📜 main.py              # 전체 파이프라인 통합 및 실행
 ┣ 📜 inference.py         # 최신 모델 자동 로드 및 텍스트 추론
 ┣ 📜 .gitignore           # 대용량 모델 및 캐시 파일 업로드 방지
 ┗ 📂 pdf_data/            # (Local) 크롤링된 원본 PDF 파일
 ┗ 📂 models/              # (Local) 학습 완료된 AI 모델 가중치
```
