"""Clearly fictional reports. No fabricated classifier scores."""
DEMO_REPORTS = [
    ('한빛반도체_2025-01-15.txt', '''한빛반도체 (123456)
새봄증권 | 2025.01.15
투자의견: 매수 | 목표주가: 82,000원

2024년 매출액은 2조 3,500억원, 영업이익은 3,200억원으로 집계됐다. 데이터센터 수요 회복에 따라 고부가 제품 비중이 확대되었다.

다만 재고 부담과 환율 변동은 수익성의 불확실성을 높인다. 주요 고객사의 투자 지연이 이어질 경우 실적 추정치 하향 가능성이 있다.

장기 수요 성장 전망은 유지하나 단기 마진 둔화에 유의할 필요가 있다. 본 문서는 앱 체험을 위해 작성된 가상의 보고서이다.'''),
    ('한빛반도체_2025-04-16.txt', '''한빛반도체 (123456)
새봄증권 | 2025.04.16
투자의견: 중립 | 목표주가: 71,000원

2025년 1분기 매출액은 5,600억원, 영업이익은 540억원을 기록했다. 전분기 대비 출하량은 증가했으나 판매단가 하락으로 이익 개선은 제한적이었다.

수요 둔화와 비용 증가를 반영해 실적 전망을 하향한다. 해외 경쟁 심화와 투자 지연에 따른 불확실성이 남아 있다.

신제품 양산은 하반기 회복의 확인 지표다. 본 문서는 앱 체험을 위해 작성된 가상의 보고서이다.'''),
    ('한빛반도체_2025-04-18.txt', '''한빛반도체 (123456)
다온증권 | 2025.04.18
투자의견: 매수 | 목표주가: 76,000원

2025년 매출액 전망은 2조 6,000억원이다. 차세대 메모리 공급 확대와 고객 다변화가 매출 성장을 뒷받침할 것으로 예상한다.

상반기 비용 부담은 지속되지만 하반기 가동률 회복을 기대한다. 환율 변동과 재고 조정 속도는 확인이 필요하다.

신규 수주와 이익률 추이를 함께 점검해야 한다. 본 문서는 앱 체험을 위해 작성된 가상의 보고서이다.'''),
]

def seed_demo(library, owner):
    from data_processor import document_from_text
    from inference import analyze_document
    for filename, text in DEMO_REPORTS:
        doc = document_from_text(text, filename=filename)
        doc['metadata'].update(company='한빛반도체', ticker='123456', broker='다온증권' if '04-18' in filename else '새봄증권', report_date=filename[-14:-4])
        analysis = analyze_document(doc, engine='rules')
        analysis['engine'] = 'demo'
        analysis['warnings'].append('가상 기업·수치로 구성한 데모입니다. 실제 투자 자료가 아닙니다.')
        ident = library.save_document(owner, doc)
        library.save_analysis(owner, ident, analysis)
