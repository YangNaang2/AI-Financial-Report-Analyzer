from pathlib import Path
from streamlit.testing.v1 import AppTest

APP=Path(__file__).resolve().parents[1]/'app.py'


def boot(tmp_path,monkeypatch):
    monkeypatch.setenv('REPORT_LENS_DB',str(tmp_path/'library.sqlite3'))
    monkeypatch.setenv('REPORT_LENS_DATA_DIR',str(tmp_path))
    monkeypatch.setenv('REPORT_LENS_MODELS_DIR',str(tmp_path/'models'))
    app=AppTest.from_file(str(APP),default_timeout=15).run()
    assert not app.exception
    return app


def test_model_free_empty_workspace_and_every_page(tmp_path,monkeypatch):
    app=boot(tmp_path,monkeypatch)
    assert any(m.label=='준비된 모델' and m.value=='0' for m in app.metric)
    for page in ['리포트 분석','분석 상세','리포트 비교','분석 보관함','데이터 · 모델']:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, (page,list(app.exception))
    assert any(b.label=='CPU 기준선 학습 · 평가 시작' and b.disabled for b in app.button)


def test_demo_navigation_and_isolation(tmp_path,monkeypatch):
    app=boot(tmp_path,monkeypatch)
    app.sidebar.toggle[0].set_value(True).run()
    assert not app.exception
    assert any(m.label=='보관한 리포트' and m.value=='3' for m in app.metric)
    for page in ['분석 상세','리포트 비교','분석 보관함','데이터 · 모델']:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, (page,list(app.exception))
    app.sidebar.toggle[0].set_value(False).run()
    app.sidebar.radio[0].set_value('대시보드').run()
    assert any(m.label=='보관한 리포트' and m.value=='0' for m in app.metric)


def test_text_preview_preserves_financial_data(tmp_path,monkeypatch):
    app=boot(tmp_path,monkeypatch)
    app.sidebar.radio[0].set_value('리포트 분석').run()
    app.text_area[0].set_value('한빛반도체 (123456)\n발간일: 2031.01.15\n목표주가: 82,000원\n\n영업이익 300억원. 수요 둔화와 재고 부담을 확인한다.').run()
    next(b for b in app.button if b.label=='텍스트 확인').click().run()
    assert not app.exception
    assert any(t.value=='2031-01-15' for t in app.text_input)
    assert any('82,000원' in t.value for t in app.text_area)
    assert not any(r.label=='분석 방식' and '학습된 분류 모델' in r.options for r in app.radio)


def test_invalid_model_is_visible_without_crashing(tmp_path,monkeypatch):
    broken=tmp_path/'models'/'broken-model';broken.mkdir(parents=True)
    (broken/'config.json').write_text('{}')
    app=boot(tmp_path,monkeypatch)
    app.sidebar.radio[0].set_value('데이터 · 모델').run()
    assert not app.exception
    assert any('broken-model' in e.label for e in app.expander)
    assert app.warning


def test_text_analysis_persists_once_across_reruns(tmp_path,monkeypatch):
    import time
    app=boot(tmp_path,monkeypatch)
    app.sidebar.radio[0].set_value('리포트 분석').run()
    app.text_area[0].set_value('검증기업 (123456)\n발간일: 2025.01.15\n\n실적 부진 우려와 수요 둔화로 전망을 하향한다.').run()
    next(b for b in app.button if b.label=='텍스트 확인').click().run()
    next(b for b in app.button if b.label=='분석하고 보관함에 저장').click().run()
    for _ in range(30):
        app.run()
        assert not app.exception
        if any(b.label=='마지막 분석 상세 보기 →' for b in app.button): break
        time.sleep(.05)
    assert any(b.label=='마지막 분석 상세 보기 →' for b in app.button)
    app.sidebar.radio[0].set_value('대시보드').run()
    for _ in range(2): app.run()
    assert any(m.label=='보관한 리포트' and m.value=='1' for m in app.metric)
    assert any(m.label=='분석 기록' and m.value=='1' for m in app.metric)


def test_workspace_recovery_callback_and_nullable_label_info(tmp_path,monkeypatch):
    app=boot(tmp_path,monkeypatch)
    key='test-workspace-recovery-key-123456789012345'
    next(t for t in app.text_input if t.label=='기존 작업 공간 키').set_value(key).run()
    next(b for b in app.button if b.label=='작업 공간 열기').click().run()
    assert not app.exception
    assert app.session_state['workspace_key']==key
    app.session_state['dataset']={'records':[{'text':'가상 데이터','label_info':None}]}
    app.sidebar.radio[0].set_value('데이터 · 모델').run()
    assert not app.exception
