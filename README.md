# 📈 AI 기반 증권사 리포트 '숨은 매도 시그널' 탐지하기

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/Transformers-FFD21E?logo=huggingface&logoColor=black)

## 👨‍💻 프로젝트 동기
증권사의 리포트를 읽는 것이 취미이다. 
하지만 근래 금융/증권에 대한 높아진 관심을 바탕으로 매도 리포트, 혹은 애널리스트의 소신을 밝히는 보고서가 많이 줄어든 추세이다.

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
 ┣ 📜 inference.py         # 최신 모델 자동 로드 및 텍스트 추론
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

<img width="588" height="90" alt="Image" src="https://github.com/user-attachments/assets/3c9b1d15-3355-4605-bb67-d4ac1d68c87f" />


<img width="959" height="692" alt="Image" src="https://github.com/user-attachments/assets/b4097d2b-55ec-4faa-b93a-a1e08152ab06" />

# 4. 🌐 Streamlit 웹 대시보드 실행 (GUI)
streamlit run app.py

```
