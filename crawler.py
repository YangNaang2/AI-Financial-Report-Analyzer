import os
import time
import requests
from bs4 import BeautifulSoup

def download_naver_reports(save_dir='./pdf_data', start_page=1, end_page=1):
    """
    네이버 증권 기업분석 리포트를 지정한 페이지 구간에서 자동으로 다운로드합니다.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    base_url = "https://finance.naver.com/research/company_list.naver"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

    print(f"🚀 네이버 증권 리포트 수집 시작... ({start_page}페이지 ~ {end_page}페이지)")
    download_count = 0

    # 💡 수정된 부분: 1부터가 아니라 start_page부터 end_page까지 반복합니다.
    for page in range(start_page, end_page + 1):
        print(f"\n[ {page} 페이지 탐색 중... ]")
        url = f"{base_url}?&page={page}"
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')

        for row in soup.select('table.type_1 tr'):
            file_td = row.select_one('td.file a')
            title_td = row.select_one('td:nth-child(2) a')

            if file_td and title_td:
                pdf_link = file_td.get('href')
                report_title = "".join(c for c in title_td.text.strip() if c not in r'\/:*?"<>|') 

                download_url = pdf_link if pdf_link.startswith('http') else f"https://finance.naver.com{pdf_link}"

                try:
                    pdf_res = requests.get(download_url, headers=headers)
                    file_path = os.path.join(save_dir, f"{report_title}.pdf")
                    
                    with open(file_path, 'wb') as f:
                        f.write(pdf_res.content)

                    print(f"✅ 저장 완료: {report_title}.pdf")
                    download_count += 1
                    
                    time.sleep(1) 

                except Exception as e:
                    print(f"❌ 다운로드 실패 ({report_title}): {e}")

    print(f"\n🎉 총 {download_count}개의 리포트 다운로드가 완료되었습니다! ({save_dir} 폴더 확인)")

if __name__ == "__main__":
    # 원하는 페이지 타겟팅
    download_naver_reports(save_dir='./pdf_data', start_page=60, end_page=61)