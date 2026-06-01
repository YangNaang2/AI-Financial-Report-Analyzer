import pdfplumber
import re

def extract_text_from_pdf(file_path):
    text_content = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_content.append(page_text)
    return "\n".join(text_content)

def clean_financial_report(text):
    # 줄바꿈 및 이메일 제거
    text = re.sub(r'(?<!\.)\n', ' ', text) 
    text = re.sub(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', '', text)
    
    # 띄어쓰기가 망가진 영문 노이즈 제거
    text = re.sub(r'[A-Za-z]\s[A-Za-z]\s[A-Za-z].*?(?=[가-힣])', '', text)
    
    # 전화번호, 날짜, 종목코드 제거
    text = re.sub(r'\d{2,3}-\d{3,4}-\d{4}', '', text) 
    text = re.sub(r'\d{4}\.\d{2}\.\d{2}', '', text)
    text = re.sub(r'\b\d{6}\b', '', text) 
    
    # 주가 및 재무 데이터 노이즈 제거
    noise_patterns = [
        r'목표주가.*?\d+원', r'현재주가.*?\d+원', r'상승여력.*?%', 
        r'KOSPI.*?[pP][tT]', r'시가총액.*?[억백십만]+원', r'발행주식수.*?[주]', 
        r'액면가.*?원', r'자본금.*?원', r'60일 평균거래.*',
        r'매수\s*\([가-힣]+\)', r'Stock Data'
    ]
    for pattern in noise_patterns:
        text = re.sub(pattern, '', text)
        
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# 테스트용 코드 (직접 실행할 때만 작동함)
if __name__ == "__main__":
    pdf_path = r"C:\Users\양진우\Desktop\finance\한미반도체.pdf"
    raw_text = extract_text_from_pdf(pdf_path)
    clean_text = clean_financial_report(raw_text)
    print("✅ 데이터 정제 모듈 테스트 완료!")