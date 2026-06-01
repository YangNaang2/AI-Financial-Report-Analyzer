import os
import re
import time
from data_processor import extract_text_from_pdf, clean_financial_report
from labeler import calculate_post_report_return
from train import train_model

def extract_metadata_from_raw_text(raw_text):
    search_text = re.sub(r'(?<=\d)\s+(?=\d)', '', raw_text) 
    
    # 1. 종목코드 탐지 (완벽 작동 중)
    ticker_match = re.search(r'(?<!\d)(?:A)?([0-9]{6})(?!\d)', search_text)
    
    # 2. 날짜 탐지 A: 기존 숫자 포맷 (2026.05.18)
    date_match = re.search(r'(202[0-6]|2[0-6])[\.\-\/]\s*(0[1-9]|1[0-2])[\.\-\/]\s*(0[1-9]|[12]\d|3[01])', search_text)
    
    # 3. 💡 날짜 탐지 B: 영문 포맷 추가 (May 18, 2026)
    eng_date_match = re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{1,2}),\s*(202[0-6])', search_text, re.IGNORECASE)
    
    ticker = ticker_match.group(1) if ticker_match else None
    date_str = None
    
    if eng_date_match:
        # 영문 달(Month)을 숫자로 변환
        month_str = eng_date_match.group(1).capitalize()
        month_dict = {'Jan':'01', 'Feb':'02', 'Mar':'03', 'Apr':'04', 'May':'05', 'Jun':'06', 'Jul':'07', 'Aug':'08', 'Sep':'09', 'Oct':'10', 'Nov':'11', 'Dec':'12'}
        year = eng_date_match.group(3)
        month = month_dict[month_str]
        day = eng_date_match.group(2).zfill(2) # 1일이면 '01'로 변환
        date_str = f"{year}.{month}.{day}"
        
    elif date_match:
        year = date_match.group(1)
        if len(year) == 2:  
            year = "20" + year
        month = date_match.group(2)
        day = date_match.group(3)
        date_str = f"{year}.{month}.{day}"
        
    return ticker, date_str

def run_pipeline():
    pdf_dir = './pdf_data'
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    print(f"📦 총 {len(pdf_files)}개의 PDF 리포트를 발견했습니다. 데이터 가공을 시작합니다...\n")
    
    valid_texts = []
    valid_labels = []
    
    for idx, pdf_file in enumerate(pdf_files):
        file_path = os.path.join(pdf_dir, pdf_file)
        
        try:
            # 1. 텍스트 추출
            raw_text = extract_text_from_pdf(file_path)
            
            # 2. 메타데이터(종목코드, 날짜) 추출
            ticker, date_str = extract_metadata_from_raw_text(raw_text)
            
            if not ticker or not date_str:
                print(f"⚠️ [{idx+1}/{len(pdf_files)}] {pdf_file} - 종목코드나 날짜를 찾지 못해 건너뜁니다.")
                continue
                
            # 3. 주가 데이터 연동 및 라벨링
            time.sleep(0.5)
            label_info = calculate_post_report_return(ticker, date_str)
            
            if not label_info:
                print(f"⚠️ [{idx+1}/{len(pdf_files)}] {pdf_file} - 주가 데이터를 불러올 수 없어 건너뜁니다.")
                continue
                
            # 4. 텍스트 정제 (학습용)
            clean_text = clean_financial_report(raw_text)
            
            # 성공한 데이터만 리스트에 추가
            valid_texts.append(clean_text)
            valid_labels.append(label_info['label_id'])
            
            print(f"✅ [{idx+1}/{len(pdf_files)}] {pdf_file} - 처리 완료! (라벨: {label_info['label_id']}, 종목: {ticker})")
            
        except Exception as e:
            print(f"❌ [{idx+1}/{len(pdf_files)}] {pdf_file} - 에러 발생: {e}")
            continue

    # --- 데이터 가공 완료 및 AI 학습 시작 ---
    print("\n" + "="*50)
    print(f"🎉 데이터 파이프라인 완료! 총 {len(valid_texts)}개의 유효한 학습 데이터를 확보했습니다.")
    print("="*50 + "\n")
    
    if len(valid_texts) > 0:
        print("🧠 AI 모델(KoFinBERT) 파인튜닝을 시작합니다...")
        # train.py 에 있는 함수를 호출하여 학습 시작!
        train_model(valid_texts, valid_labels)
    else:
        print("학습할 유효한 데이터가 부족합니다.")

if __name__ == "__main__":
    run_pipeline()