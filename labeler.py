import FinanceDataReader as fdr
from datetime import datetime, timedelta

def calculate_post_report_return(ticker, report_date_str, window_days=30):
    start_date = datetime.strptime(report_date_str, '%Y.%m.%d')
    end_date = start_date + timedelta(days=window_days + 20) 
    
    df = fdr.DataReader(ticker, start_date, end_date)
    if df.empty:
        return None
        
    base_price = df.iloc[0]['Close']
    target_date = start_date + timedelta(days=window_days)
    future_df = df[df.index >= target_date]
    
    if future_df.empty:
        return None
        
    future_price = future_df.iloc[0]['Close']
    return_rate = ((future_price - base_price) / base_price) * 100
    
    # -5% 이하면 1(숨은 매도), 아니면 0(유지/매수) 숫자로 라벨링
    label_id = 1 if return_rate <= -5.0 else 0
    
    return {
        '수익률': round(return_rate, 2),
        'label_id': label_id
    }

if __name__ == "__main__":
    result = calculate_post_report_return('042700', '2024.05.29')
    print("✅ 주가 데이터 라벨링 모듈 테스트 완료!", result)