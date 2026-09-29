"""Report Lens: local Korean financial report reading workspace."""
from __future__ import annotations
import hashlib
import html
import json
import os
from pathlib import Path
import re
import secrets
from copy import deepcopy
import pandas as pd
import streamlit as st
from data_processor import document_from_text, extract_document, DocumentError
from inference import analyze_document, list_models
from storage import Library, LibraryError
from exports import analyses_to_csv, analyses_to_json, analysis_to_html
from jobs import JobManager, JobError
from demo import seed_demo
from presentation import unprocessed_ranges, expression_changes

st.set_page_config(page_title='Report Lens · 증권 리포트 분석', page_icon='◈', layout='wide')
DATA = Path(os.getenv('REPORT_LENS_DATA_DIR', 'data'))
DB = os.getenv('REPORT_LENS_DB', str(DATA / 'library.sqlite3'))
MODEL_DIR = Path(os.getenv('REPORT_LENS_MODELS_DIR', 'models'))
PAGES = ['대시보드', '리포트 분석', '분석 상세', '리포트 비교', '분석 보관함', '데이터 · 모델']
ENGINE_NAMES = {'rules':'규칙 기반', 'model':'분류 모델', 'demo':'가상 데모'}

@st.cache_resource
def manager():
    return JobManager()

@st.cache_resource
def library(path):
    return Library(path)

@st.cache_resource
def demo_library():
    return Library(':memory:')


def go(page, run_id=None):
    st.session_state['nav'] = page
    if run_id:
        st.session_state['selected_run'] = run_id


def reset_workspace(key):
    for name in list(st.session_state):
        if name not in ('nav','owner_jobs'):
            del st.session_state[name]
    st.session_state['workspace_key'] = key
    st.session_state['nav'] = '대시보드'


def restore_workspace():
    incoming=st.session_state.get('workspace_recovery_input','')
    if re.fullmatch(r'[A-Za-z0-9_-]{32,128}',incoming):
        reset_workspace(incoming)
    else:
        st.session_state['workspace_error']='저장한 32자 이상의 작업 공간 키를 입력하세요.'


def notice(text, label='안내'):
    st.markdown(f'<div class="notice"><b>{html.escape(label)}</b><br>{html.escape(str(text))}</div>', unsafe_allow_html=True)


def title(kicker, headline, description):
    st.markdown(f'<div class="eyebrow">{html.escape(kicker)}</div><h1>{html.escape(headline)}</h1><p class="lede">{html.escape(description)}</p>', unsafe_allow_html=True)


def score_label(analysis):
    score = analysis.get('metrics', {}).get('negative_score')
    return f'{score:.3f}' if isinstance(score, (float, int)) else '—'


def records_table(rows):
    result = []
    for row in rows:
        a = row['analysis']; m = a.get('metadata', {})
        result.append({'기업':m.get('company') or '미확인', '발간일':m.get('report_date') or '미확인', '증권사':m.get('broker') or '미확인', '방식':ENGINE_NAMES.get(a.get('engine'),a.get('engine')), '음성 클래스 점수':score_label(a), '상태':a.get('status',''), '분석일':row['created_at'][:16].replace('T',' ')})
    return pd.DataFrame(result)


def available_models(owner):
    return list_models(MODEL_DIR) + list_models(DATA / 'workspaces' / owner / 'models')


def clear_staged():
    st.session_state.pop('staged_docs',None)
    st.session_state.pop('stage_errors',None)


def clear_context():
    for key in ('staged_docs','stage_errors','selected_run','dataset','active_job','reported_job','last_job'):
        st.session_state.pop(key,None)


def get_workspace():
    st.session_state.setdefault('workspace_key', secrets.token_urlsafe(32))
    st.session_state.setdefault('nav', '대시보드')
    with st.sidebar:
        st.markdown('<div class="brand">◈ REPORT LENS</div><p class="muted">금융 리포트 분석 워크스페이스</p>', unsafe_allow_html=True)
        st.radio('메뉴', PAGES, key='nav', label_visibility='collapsed')
        st.divider()
        demo = st.toggle('데모 둘러보기', key='demo_mode', on_change=clear_context, help='가상의 문서 3개로 체험합니다. 실제 보관함과 분리됩니다.')
        with st.expander('내 작업 공간 · 복구 키'):
            st.caption('세션마다 별도 공간을 만듭니다. 아래 키를 안전하게 보관하면 다른 세션에서 같은 기록을 열 수 있습니다. 키를 아는 사람은 이 공간에 접근할 수 있습니다.')
            st.download_button('작업 공간 키 저장', st.session_state['workspace_key'], file_name='report-lens-workspace-key.txt', mime='text/plain')
            st.text_input('기존 작업 공간 키', type='password',key='workspace_recovery_input')
            st.button('작업 공간 열기', disabled=demo,on_click=restore_workspace)
            if st.session_state.get('workspace_error'):
                st.error(st.session_state['workspace_error'])
        st.caption('원문에 근거한 검토 도구입니다. 후행 수익률 라벨과 모델 점수는 애널리스트의 의도나 실제 하락 확률을 뜻하지 않습니다.')
    owner = hashlib.sha256(st.session_state['workspace_key'].encode()).hexdigest()
    if demo:
        owner = 'demo-' + owner
        lib = demo_library()
        if not lib.list_documents(owner):
            seed_demo(lib, owner)
        st.info('DEMO · 가상의 기업·보고서·수치입니다. 실제 보관함과 별도로 저장됩니다.')
    else:
        lib = library(DB)
    return owner, lib, demo


@st.fragment(run_every='2s')
def render_job(owner):
    ident = st.session_state.get('owner_jobs',{}).get(owner) or manager().active(owner)
    if not ident:
        return
    job = manager().get(owner, ident)
    if not job:
        return
    if job['kind']=='리포트 분석':
        st.progress(job['progress'], text=f"{job['kind']} · {job['message']}")
    else:
        st.info(f"{job['kind']} · {job['message']}")
    if job['state'] in ('queued','running'):
        if st.button('작업 중지 요청', key='cancel_job'):
            manager().cancel(owner, ident)
            st.caption('현재 문서 또는 학습 단계가 끝나면 중지합니다. 이미 저장한 결과는 유지됩니다.')
    elif st.session_state.get('reported_job') != ident:
        st.session_state['reported_job'] = ident
        st.session_state['last_job'] = job
        if job['state'] == 'completed' and isinstance(job['result'], dict):
            result = job['result']
            if result.get('run_ids'):
                st.session_state['selected_run'] = result['run_ids'][-1]
            if 'records' in result:
                st.session_state['dataset'] = result
        st.rerun()
    elif job['state'] == 'completed':
        st.success('작업이 완료되었습니다. 결과를 아래에서 확인할 수 있습니다.')
        with st.expander('작업 결과'):
            result = deepcopy(job['result'])
            if isinstance(result, dict) and 'records' in result:
                result['records'] = f"문서 {len(result['records'])}개 (데이터 관리에서 확인)"
            st.json(result)
    elif job['state'] == 'failed':
        st.error(job['error'])
    else:
        st.warning(job['error'] or '작업을 중지했습니다.')
    if job['state'] not in ('queued','running') and st.button('작업 알림 닫기',key='dismiss_job'):
        st.session_state.pop('active_job',None)
        st.session_state.setdefault('owner_jobs',{}).pop(owner,None)
        st.rerun()


def start_job(owner, kind, fn):
    try:
        ident = manager().submit(owner, kind, fn)
        st.session_state['active_job'] = ident
        st.session_state.setdefault('owner_jobs',{})[owner]=ident
        st.session_state.pop('reported_job', None)
        st.rerun()
    except JobError as exc:
        st.warning(str(exc))


def dashboard(owner, lib, models):
    title('RESEARCH WORKSPACE', '리포트의 숫자와 문맥을 함께 읽다', '문서를 모으고, 변화와 근거를 확인하고, 나만의 분석 기록을 남기세요.')
    docs = lib.list_documents(owner); rows = lib.list_analyses(owner)
    ready = [m for m in models if m['status'] == 'ready']
    cols = st.columns(4)
    for col,label,value in zip(cols, ['보관한 리포트','분석 기록','기업','준비된 모델'], [len(docs),len(rows),len({d.get('metadata',{}).get('company') for d in docs if d.get('metadata',{}).get('company')}),len(ready)]):
        col.metric(label, value)
    left,right = st.columns([1.8,1],gap='large')
    with left:
        st.subheader('최근 분석')
        if rows:
            st.dataframe(records_table(rows[:8]), hide_index=True, width='stretch')
            st.button('분석 보관함 열기 →', on_click=go, args=('분석 보관함',))
        else:
            notice('아직 저장된 분석이 없습니다. 텍스트를 붙여 넣거나 PDF를 올려 첫 문서를 살펴보세요.', '첫 리포트를 시작하세요')
            st.button('새 리포트 분석 →', type='primary', on_click=go, args=('리포트 분석',))
        st.subheader('기업별 기록')
        counts = {}
        for d in docs:
            company = d.get('metadata',{}).get('company') or '미확인'
            counts[company] = counts.get(company,0)+1
        if counts:
            st.bar_chart(pd.Series(counts, name='리포트 수'), color='#147D72')
        else:
            st.caption('기업 정보를 확인한 문서를 저장하면 기록이 나타납니다.')
    with right:
        st.subheader('모델 준비 상태')
        if ready:
            st.success(f'{len(ready)}개 모델을 사용할 수 있습니다.')
            for m in ready:
                st.write(m['id']); st.caption(f"{m.get('manifest',{}).get('model_type','model')} · 실제 평가 지표는 모델 관리에서 확인")
        else:
            notice('현재 검증 가능한 모델 파일이 없습니다. PDF 추출, 규칙 탐지, 원문 확인, 비교와 저장은 바로 사용할 수 있습니다.', '모델 없음')
        st.button('데이터 · 모델 관리 →', on_click=go, args=('데이터 · 모델',))
        st.subheader('세 가지 읽기 기준')
        st.markdown('**01 원문을 보존합니다**  \n숫자와 단위, 페이지·문단을 함께 확인합니다.\n\n**02 방식의 차이를 드러냅니다**  \n규칙 탐지와 학습한 모델의 점수를 구분합니다.\n\n**03 빈틈도 기록합니다**  \n누락 페이지와 분석 범위, 모델 버전을 남깁니다.')


def analysis_page(owner, lib, models, demo):
    title('READ & ANALYZE', '새 리포트 분석', '텍스트 또는 PDF에서 시작하세요. 추출 결과와 메타데이터를 확인한 뒤 분석합니다.')
    source = st.radio('입력 방식', ['텍스트 입력','PDF 업로드'], horizontal=True,key='input_source',on_change=clear_staged)
    if source == '텍스트 입력':
        raw = st.text_area('리포트 원문', height=220, placeholder='기업명, 발간일, 목표주가와 본문을 그대로 붙여 넣으세요.')
        if st.button('텍스트 확인', type='primary', disabled=not raw.strip()):
            try:
                st.session_state['staged_docs'] = [document_from_text(raw)]
                st.session_state['stage_errors'] = []
                st.session_state['stage_version'] = secrets.token_hex(4)
            except ValueError as exc:
                st.error(str(exc))
    else:
        uploads = st.file_uploader('PDF 선택 · 최대 5개 / 각 20MB', type=['pdf'], accept_multiple_files=True)
        st.caption('100페이지 이하 문서를 지원합니다. 저장 가능한 추출문은 최대 200만 자입니다. 스캔 PDF는 OCR이 필요하며, 암호화·손상·빈 문서는 오류로 구분합니다.')
        if st.button('PDF 추출 및 미리보기', type='primary', disabled=not uploads):
            if len(uploads)>5:
                st.error('한 번에 최대 5개 파일을 선택하세요.')
            else:
                docs,errors = [],[]
                with st.status('PDF에서 원문을 읽는 중', expanded=True) as status:
                    for upload in uploads:
                        st.write(f'추출 중: {upload.name}')
                        try:
                            doc = extract_document(upload.getvalue(), filename=upload.name)
                            if not doc.get('text','').strip():
                                errors.append(f'{upload.name}: OCR 필요 — 텍스트가 없습니다.')
                            else:
                                docs.append(doc)
                        except (DocumentError, ValueError) as exc:
                            errors.append(f'{upload.name}: {exc}')
                    status.update(label=f'{len(docs)}개 추출 · {len(errors)}개 확인 필요', state='complete' if docs else 'error', expanded=False)
                st.session_state.update(staged_docs=docs, stage_errors=errors, stage_version=secrets.token_hex(4))
    for err in st.session_state.get('stage_errors',[]):
        st.warning(err)
    staged = st.session_state.get('staged_docs',[])
    if not staged:
        return
    st.subheader('추출 미리보기 · 정보 확인')
    version = st.session_state.get('stage_version','0')
    with st.form('metadata_form'):
        updated = []
        for index,doc in enumerate(staged):
            m = deepcopy(doc.get('metadata',{})); key=f'meta_{version}_{index}'
            with st.expander(f"{index+1}. {doc['filename']} · {len(doc['pages'])}페이지", expanded=index==0):
                for warning in doc.get('warnings',[]): st.warning(warning)
                c1,c2,c3 = st.columns(3)
                m['company'] = c1.text_input('기업',value=m.get('company') or '',key=key+'company') or None
                m['ticker'] = c2.text_input('종목코드',value=m.get('ticker') or '',key=key+'ticker') or None
                m['broker'] = c3.text_input('증권사',value=m.get('broker') or '',key=key+'broker') or None
                c1,c2,c3 = st.columns(3)
                m['report_date'] = c1.text_input('발간일 (YYYY-MM-DD)',value=m.get('report_date') or '',key=key+'date') or None
                m['opinion'] = c2.text_input('투자의견',value=str(m.get('opinion') or ''),key=key+'opinion') or None
                m['target_price'] = c3.text_input('목표주가 · 단위 포함',value=str(m.get('target_price') or ''),key=key+'target') or None
                st.text_area('추출 원문 미리보기',value=doc['text'][:15000],height=180,disabled=True,key=key+'preview')
                if len(doc['text'])>15000: st.caption('미리보기만 15,000자로 제한됩니다. 분석 상세에서 페이지별 전체 추출문을 확인할 수 있습니다.')
                st.caption('자동 추출은 오류가 있을 수 있습니다. 수정한 값은 사용자 확인 정보로 기록됩니다.')
                revised = deepcopy(doc)
                if revised.get('metadata') != m:
                    revised['metadata_review'] = {'source':'user','original':deepcopy(revised.get('metadata',{}))}
                revised['metadata'] = m
                updated.append(revised)
        ready = [m for m in models if m['status']=='ready']
        mode = st.radio('분석 방식', ['규칙 기반 원문 검토'] + (['학습된 분류 모델'] if ready and not demo else []), horizontal=True)
        chosen = st.selectbox('모델 선택',ready,format_func=lambda m:m['id']) if ready and not demo else None
        if not ready: st.caption('모델 없음 · 분류 모델 분석은 비활성화되어 있습니다. 규칙 기반 검토는 점수를 만들지 않습니다.')
        max_segments = st.slider('최대 분석 구간',50,1000,500,50,help='제한을 초과한 원문은 분석하지 않은 범위로 표시됩니다.')
        submit = st.form_submit_button('분석하고 보관함에 저장',type='primary')
    if submit:
        try:
            for doc in updated:
                m=doc['metadata']
                if m.get('ticker') and not re.fullmatch(r'\d{6}',m['ticker']): raise ValueError('종목코드는 6자리 숫자로 입력하세요.')
                if m.get('report_date'):
                    from datetime import date
                    date.fromisoformat(m['report_date'])
        except ValueError as exc:
            st.error(f'메타데이터를 확인하세요: {exc}'); return
        engine = 'model' if mode == '학습된 분류 모델' else 'rules'
        model_path = chosen['path'] if engine=='model' else None
        settings = {'max_segments':max_segments}
        def work(ctx):
            run_ids,errors = [],[]
            for i,doc in enumerate(updated):
                ctx.progress(i/len(updated),f"{i+1}/{len(updated)} · {doc['filename']}")
                try:
                    a = analyze_document(doc,engine=engine,model_path=model_path,settings=settings)
                    if demo:
                        a['engine']='demo'; a['warnings'].append('데모 공간의 분석입니다.')
                    a['document_snapshot']=deepcopy(doc)
                    ident = lib.save_document(owner,doc)
                    lib.update_document(owner,ident,metadata=doc['metadata'])
                    run_ids.append(lib.save_analysis(owner,ident,a))
                except (ValueError,RuntimeError,OSError) as exc:
                    errors.append({'filename':doc['filename'],'error':str(exc)})
            if not run_ids: raise ValueError('; '.join(e['error'] for e in errors))
            return {'run_ids':run_ids,'errors':errors,'status':'partial' if errors else 'complete'}
        start_job(owner,'리포트 분석',work)
    if st.session_state.get('selected_run'):
        st.button('마지막 분석 상세 보기 →',on_click=go,args=('분석 상세',))


def safe_mark(text, terms):
    if not terms: return html.escape(text)
    regex = re.compile('('+'|'.join(re.escape(t) for t in sorted(set(terms),key=len,reverse=True) if t)+')')
    return ''.join(f'<mark>{html.escape(part)}</mark>' if i%2 else html.escape(part) for i,part in enumerate(regex.split(text)))


def detail_page(owner,lib):
    title('EVIDENCE VIEW', '분석 상세', '점수와 함께 원문 위치, 분석 범위, 처리 설정을 확인하세요.')
    rows = lib.list_analyses(owner)
    if not rows:
        notice('리포트를 분석하면 문단과 페이지 단위로 결과를 살펴볼 수 있습니다.'); return
    ids = [r['id'] for r in rows]; mapping = {r['id']:r for r in rows}
    selected = st.session_state.get('selected_run')
    rid = st.selectbox('분석 기록',ids,index=ids.index(selected) if selected in ids else 0,format_func=lambda i:f"{mapping[i]['analysis'].get('metadata',{}).get('company') or '미확인'} · {mapping[i]['analysis'].get('metadata',{}).get('report_date') or '날짜 미확인'} · {mapping[i]['created_at'][:16]} · {i[:8]}")
    st.session_state['selected_run']=rid
    row=mapping[rid]; a=row['analysis']; doc=a.get('document_snapshot') or lib.get_document(owner,row['document_id'])
    if not doc: st.error('원문을 찾을 수 없습니다.'); return
    m=a.get('metadata',{}); metrics=a.get('metrics',{}); segments=a.get('segments',[])
    c1,c2,c3,c4=st.columns(4)
    c1.metric('기업',m.get('company') or '미확인');c2.metric('음성 클래스 점수',score_label(a));c3.metric('분석 구간',len(segments))
    total=metrics.get('total_chars',0); covered=metrics.get('analyzed_chars',0)
    c4.metric('추출문 분석 범위', f'{min(100,100*covered/total):.1f}%' if total else '—')
    st.caption(f"{ENGINE_NAMES.get(a.get('engine'),a.get('engine'))} · 모델 {a.get('model_id','없음')} · {a.get('created_at','')} · {covered:,}/{total:,}자 (추출된 텍스트 기준)")
    if metrics.get('negative_score') is not None:
        st.info('음성 클래스는 관측 기간의 주가 하락으로 만든 대리 라벨입니다. 점수는 보정된 하락 확률·모델 정확도·애널리스트 의도가 아닙니다. 높은 점수의 구간은 인과적 설명이 아닙니다.')
    else:
        st.caption('규칙 기반 표현 탐지입니다. 모델 분류 점수는 제공하지 않습니다.')
    with st.expander('분석 방식과 처리 안내', expanded=a.get('status')!='ready'):
        for warning in a.get('warnings',[]): st.write('• '+warning)
    if metrics.get('unprocessed_pages'):
        st.warning(f"일부 또는 전체 미처리 페이지: {metrics['unprocessed_pages']}")
        with st.expander('분석하지 않은 원문 범위'):
            for missing in unprocessed_ranges(doc,a):
                st.markdown(f"**p.{missing['page']} · 문자 {missing['start']}–{missing['end']}**")
                st.text(missing['text'][:3000])
                if len(missing['text'])>3000: st.caption('이 미처리 구간의 미리보기는 3,000자입니다. 전체는 페이지 원문에서 확인하세요.')
    tab1,tab2,tab3=st.tabs(['원문과 분석','추출 정보 · 비교 항목','실행 정보 · 내보내기'])
    with tab1:
        left,right=st.columns([1,1.35],gap='large')
        with left:
            st.subheader('출처를 연결한 발췌')
            for s in a.get('summary',[])[:8]:
                st.markdown(f"**p.{s.get('page','?')} · {s.get('paragraph_id','')}**")
                st.write(s['text'])
            st.subheader('구간별 분석')
            if segments:
                options=list(range(len(segments)))
                si=st.selectbox('원문으로 이동할 구간',options,format_func=lambda i:f"p.{segments[i]['page']} · {segments[i].get('paragraph_id','')} · {segments[i]['text'][:42]}",key=f'segment_{rid}')
                segment=segments[si]
                st.caption(f"문자 위치 {segment.get('start',0)}–{segment.get('end',0)} · 모델 점수 {segment.get('negative_score') if segment.get('negative_score') is not None else '없음'}")
                st.write(segment['text'])
                st.json(segment.get('rule_hits',[]),expanded=False)
                if any(s.get('negative_score') is not None for s in segments):
                    chart=pd.DataFrame([{'페이지':s['page'],'점수':s['negative_score']} for s in segments]).groupby('페이지')['점수'].mean()
                    st.bar_chart(chart,color='#147D72'); st.caption('차트는 페이지 내 구간 점수의 산술 평균입니다. 문서 점수 집계는 실행 정보를 확인하세요.')
            else:
                segment=None; st.caption('분석 가능한 구간이 없습니다.')
        with right:
            st.subheader('추출 원문')
            pages=doc.get('pages',[]);nums=[p['number'] for p in pages]
            page_num=segment['page'] if segments else (nums[0] if nums else 1)
            manual=st.toggle('다른 페이지 직접 보기',key=f'manual_{rid}')
            if manual and nums: page_num=st.selectbox('페이지',nums,index=nums.index(page_num) if page_num in nums else 0,key=f'page_{rid}')
            page=next((p for p in pages if p['number']==page_num),None)
            if page:
                text=page['text']; start=segment.get('start',0) if segments else 0;end=segment.get('end',0) if segments else 0
                if segments and page_num==segment['page'] and 0<=start<end<=len(text):
                    rendered=html.escape(text[:start])+'<mark>'+html.escape(text[start:end])+'</mark>'+html.escape(text[end:])
                else: rendered=html.escape(text)
                st.markdown(f'<div class="source-label">PAGE {page_num} · 원문 위치</div><div class="source">{rendered}</div>',unsafe_allow_html=True)
                if not text.strip(): st.warning('이 페이지는 텍스트가 추출되지 않았습니다. OCR이 필요할 수 있습니다.')
    with tab2:
        st.dataframe(pd.DataFrame([{'항목':k,'값':str(v) if v is not None else '미확인'} for k,v in m.items()]),hide_index=True,width='stretch')
        st.subheader('자동 추출 근거')
        st.json(doc.get('evidence',{}))
        facts=doc.get('financial_facts',[])
        if facts:
            st.subheader('실적 수치 · 단위와 원문 위치')
            st.dataframe(pd.DataFrame(facts),hide_index=True,width='stretch')
        if doc.get('metadata_review'): st.json(doc['metadata_review'])
        st.caption('메타데이터를 직접 수정했다면 원문 근거와 다를 수 있습니다. 분석 당시 정보는 각 실행에 별도로 보관합니다.')
    with tab3:
        st.json({k:v for k,v in a.items() if k not in ('segments','summary','document_snapshot')})
        st.caption('집계 규칙과 분할 설정은 기록된 metrics/settings에 보존됩니다. PDF 표의 행·열 구조는 텍스트 추출 과정에서 달라질 수 있습니다.')
        c1,c2,c3=st.columns(3)
        c1.download_button('HTML 보고서',analysis_to_html(doc,a),file_name=f'report-{rid[:8]}.html',mime='text/html')
        c2.download_button('분석 JSON',analyses_to_json([row]),file_name=f'analysis-{rid[:8]}.json',mime='application/json')
        c3.download_button('요약 CSV',analyses_to_csv([row]),file_name=f'analysis-{rid[:8]}.csv',mime='text/csv')


def compare_page(owner,lib):
    title('COMPARE REPORTS','시간이 지나며 무엇이 달라졌을까','같은 기업의 시점별·증권사별 보고서에서 의견과 목표주가, 표현의 변화를 확인하세요.')
    rows=lib.list_analyses(owner)
    company_keys={}
    for r in rows:
        m=r['analysis'].get('metadata',{}); key=m.get('ticker') or m.get('company')
        if key: company_keys.setdefault(key,[]).append(r)
    eligible={k:v for k,v in company_keys.items() if len({r['document_id'] for r in v})>=2}
    if not eligible:
        notice('동일 기업의 서로 다른 리포트 2개 이상을 저장하세요. 데모에는 비교할 수 있는 보고서 3개가 있습니다.'); return
    company=st.selectbox('기업',list(eligible),format_func=lambda k:f"{eligible[k][0]['analysis'].get('metadata',{}).get('company') or k} ({k})")
    choices=eligible[company]; mapping={r['id']:r for r in choices};ids=list(mapping)
    fmt=lambda i:f"{mapping[i]['analysis'].get('metadata',{}).get('report_date') or '날짜 미확인'} · {mapping[i]['analysis'].get('metadata',{}).get('broker') or '증권사 미확인'} · {i[:6]}"
    l,r=st.columns(2);aid=l.selectbox('기준 리포트',ids,index=len(ids)-1,format_func=fmt);bid=r.selectbox('비교 리포트',ids,index=0,format_func=fmt)
    ar,br=mapping[aid],mapping[bid]; a,b=ar['analysis'],br['analysis']
    if ar['document_id']==br['document_id']: st.warning('같은 문서의 서로 다른 실행입니다. 보고서 간 변화는 별도 문서를 선택하세요.')
    am,bm=a.get('metadata',{}),b.get('metadata',{})
    fields={'report_date':'발간일','broker':'증권사','opinion':'투자의견','target_price':'목표주가','current_price':'현재주가'}
    st.dataframe(pd.DataFrame([{'항목':label,'기준':str(am.get(k) or '미확인'),'비교':str(bm.get(k) or '미확인'),'변화':'변경' if am.get(k)!=bm.get(k) else '동일'} for k,label in fields.items()]),hide_index=True,width='stretch')
    full_coverage=all(x.get('status')=='ready' and not x.get('metrics',{}).get('unprocessed_pages') for x in (a,b))
    if not full_coverage:
        st.warning('일부 미처리 원문이 있습니다. 표현 변화는 분석된 구간에서만 비교하며 문서 점수 차이는 계산하지 않습니다.')
    compatible=(full_coverage and a.get('engine')=='model' and b.get('engine')=='model' and a.get('model_fingerprint')==b.get('model_fingerprint') and a.get('settings')==b.get('settings') and a.get('preprocessing_version')==b.get('preprocessing_version'))
    if compatible and all(x.get('metrics',{}).get('negative_score') is not None for x in (a,b)):
        st.metric('음성 클래스 점수 변화', f"{b['metrics']['negative_score']-a['metrics']['negative_score']:+.3f}")
    else: st.caption('같은 모델·전처리·설정의 분류 결과에서만 점수 차이를 계산합니다.')
    st.subheader('표현과 원문 근거')
    changes=expression_changes(a,b)
    st.write('비교 보고서의 분석된 구간에서 새로 탐지: '+(', '.join(changes['added']) or '없음'))
    st.write('비교 보고서의 분석된 구간에서 미탐지: '+(', '.join(changes['removed']) or '없음'))
    st.caption('표현의 출현 여부 차이입니다. 의견 방향의 정답이나 작성자의 의도를 의미하지 않습니다.')
    for column,row in zip(st.columns(2),[ar,br]):
        with column:
            analysis=row['analysis'];doc=analysis.get('document_snapshot') or lib.get_document(owner,row['document_id'])
            st.markdown(f"**{analysis.get('metadata',{}).get('broker') or '미확인'}**")
            hits=[]
            for s in analysis.get('segments',[]):
                if s.get('rule_hits'): hits.append(s)
            st.caption(f'규칙 표현이 포함된 구간 {len(hits)}개 · 길이가 다른 보고서의 빈도는 직접 비교하기 어렵습니다.')
            for s in hits[:10]:
                st.markdown(f"**p.{s['page']} · {s.get('paragraph_id','')}**")
                st.write(s['text'])
            with st.expander('목표주가·의견 등의 자동 추출 근거'):
                st.json(doc.get('evidence',{}) if doc else {})
            st.button('분석 상세에서 원문 열기',key='open_'+row['id']+('left' if row is ar else 'right'),on_click=go,args=('분석 상세',row['id']))


def library_page(owner,lib):
    title('YOUR LIBRARY','분석 보관함','보고서를 검색하고 검토 메모와 즐겨찾기를 관리하세요.')
    c1,c2,c3=st.columns([2,1,1]);q=c1.text_input('기업·파일·본문 검색');favorites=c2.checkbox('즐겨찾기만');engine=c3.selectbox('분석 방식',['전체','rules','model','demo'],format_func=lambda x:ENGINE_NAMES.get(x,x))
    docs=lib.list_documents(owner,query=q,favorite_only=favorites)
    broker_options=sorted({d.get('metadata',{}).get('broker') for d in docs if d.get('metadata',{}).get('broker')})
    broker=st.selectbox('증권사 필터',['전체']+broker_options)
    if broker!='전체': docs=[d for d in docs if d.get('metadata',{}).get('broker')==broker]
    doc_ids={d['id'] for d in docs};rows=[r for r in lib.list_analyses(owner) if r['document_id'] in doc_ids and (engine=='전체' or r['analysis'].get('engine')==engine)]
    st.caption(f'문서 {len(docs)}개 · 분석 {len(rows)}개')
    if rows:
        st.dataframe(records_table(rows),hide_index=True,width='stretch')
        c1,c2=st.columns(2);c1.download_button('검색 결과 CSV',analyses_to_csv(rows),file_name='report-lens.csv',mime='text/csv');c2.download_button('검색 결과 JSON',analyses_to_json(rows),file_name='report-lens.json',mime='application/json')
    if docs:
        mapping={d['id']:d for d in docs}
        did=st.selectbox('관리할 문서',list(mapping),format_func=lambda i:f"{'★ ' if mapping[i].get('favorite') else ''}{mapping[i]['filename']} · {i[:6]}")
        doc=mapping[did]
        with st.form('document_notes'):
            favorite=st.checkbox('즐겨찾기',value=bool(doc.get('favorite')))
            note=st.text_area('나의 검토 메모',value=doc.get('note') or '')
            tags=st.text_input('태그 (쉼표로 구분)',value=', '.join(doc.get('tags') or []))
            if st.form_submit_button('메모 저장'):
                lib.update_document(owner,did,note=note,tags=[t.strip() for t in tags.split(',') if t.strip()],favorite=favorite);st.rerun()
        ownrows=[r for r in rows if r['document_id']==did]
        for row in ownrows:
            st.button(f"분석 열기 · {row['created_at'][:16]} · {row['id'][:6]}",key='library_'+row['id'],on_click=go,args=('분석 상세',row['id']))
        with st.expander('문서 삭제'):
            confirm=st.checkbox('이 문서와 모든 분석 기록을 삭제합니다.',key='delete_'+did)
            if st.button('삭제',disabled=not confirm,key='delete_button_'+did):
                lib.delete_document(owner,did);st.rerun()
    else: notice('검색 조건에 맞는 문서가 없습니다.')
    st.divider(); st.subheader('백업과 복구')
    st.caption('현재 작업 공간의 추출 원문·메타데이터·분석·메모를 JSON으로 백업합니다. 원본 PDF와 모델 파일은 포함하지 않습니다.')
    st.download_button('작업 공간 백업',lib.export_backup(owner),file_name='report-lens-backup.json',mime='application/json')
    backup=st.file_uploader('백업 JSON 가져오기',type=['json'],key='backup_upload')
    if st.button('백업 병합 복구',disabled=backup is None):
        try:
            result=lib.import_backup(owner,backup.getvalue().decode('utf-8'));st.success(f'복구 완료: {result}')
        except (ValueError,UnicodeError) as exc: st.error(f'복구 실패: {exc}')


def model_page(owner,lib,models,demo):
    title('DATA & MODELS','데이터와 모델을 투명하게','수집부터 라벨링, 시간 분할과 평가까지 실제 처리 결과를 확인합니다.')
    tabs=st.tabs(['모델 현황','데이터 준비','학습 · 평가'])
    with tabs[0]:
        if not models: notice('등록된 모델이 없습니다. 데이터 준비 후 CPU 기준선 모델부터 학습할 수 있습니다. 기존 가중치는 삭제하거나 자동 변환하지 않습니다.')
        for m in models:
            with st.expander(f"{m['id']} · {m['status']}",expanded=True):
                if m['status']!='ready': st.warning(m.get('reason','모델 준비가 완료되지 않았습니다.'))
                manifest=m.get('manifest') or {}
                st.write(f"유형: {manifest.get('model_type','미확인')} · 생성: {manifest.get('created_at','미확인')}")
                if manifest.get('evaluation_status')=='insufficient': st.warning('평가 불충분 · 표본 수와 평가 제한을 확인하세요.')
                st.json(manifest)
                st.caption('저장된 평가 지표는 해당 데이터 분할에서만 유효합니다. 실전 수익·의도 탐지 능력을 보증하지 않습니다.')
        st.button('모델 목록 새로고침',on_click=lambda:None)
    workdir=DATA/'workspaces'/owner
    with tabs[1]:
        st.subheader('공개 리포트 수집')
        st.caption('네이버 리서치의 공개 링크를 수집합니다. 요청 제한과 출처 이용 조건을 지키세요. 다운로드 기록을 남겨 재시도 시 완료 파일을 건너뜁니다.')
        c1,c2=st.columns(2);first=c1.number_input('시작 페이지',1,100,1);last=c2.number_input('마지막 페이지',1,100,1)
        if st.button('리포트 수집 시작',disabled=demo):
            if last<first or last-first>9: st.error('시작부터 최대 10페이지를 선택하세요.')
            else:
                def collect(ctx):
                    from crawler import download_naver_reports
                    ctx.progress(0.05,'목록과 PDF 다운로드 중')
                    result=download_naver_reports(str(workdir/'reports'),int(first),int(last),progress=lambda msg:ctx.progress(.3,msg))
                    ctx.check_cancelled();return result
                start_job(owner,'리포트 수집',collect)
        reports=list((workdir/'reports').glob('*.pdf')) if (workdir/'reports').exists() else []
        st.caption(f'이 작업 공간에 수집한 PDF: {len(reports)}개')
        st.subheader('원문·라벨 데이터 만들기')
        st.caption('기본값: 발간일 다음 거래일 종가 진입 → 발간일로부터 30달력일 이후 첫 거래일 종가. 수익률 -5% 이하가 음성 클래스(1)입니다. 평가일 전이거나 가격이 없으면 미라벨로 남깁니다.')
        c1,c2=st.columns(2);window=c1.number_input('관측 기간 (달력일)',1,365,30);threshold=c2.number_input('하락 기준 (%)',-50.,0.,-5.,step=1.)
        dataset_file=st.file_uploader('기존 문서 데이터 JSON 불러오기',type=['json'],key='dataset_upload')
        if st.button('데이터 JSON 확인',disabled=dataset_file is None or demo):
            try:
                parsed=json.loads(dataset_file.getvalue())
                records=parsed.get('records') if isinstance(parsed,dict) else parsed
                if not isinstance(records,list) or len(records)>10000: raise ValueError('최대 10,000개 문서 배열이 필요합니다.')
                if any(not isinstance(r,dict) or not isinstance(r.get('text'),str) for r in records): raise ValueError('각 문서에 text 문자열이 필요합니다.')
                if any(r.get('label_info') is not None and not isinstance(r.get('label_info'),dict) for r in records): raise ValueError('label_info는 객체 또는 null이어야 합니다.')
                st.session_state['dataset']=dict(parsed,records=records) if isinstance(parsed,dict) else {'records':records,'status':'imported'};st.success(f'{len(records)}개 문서 불러옴')
            except (ValueError,TypeError) as exc: st.error(str(exc))
        if st.button('수집 PDF 전처리 · 라벨링',disabled=demo or not reports):
            def prepare(ctx):
                from main import prepare_dataset
                ctx.progress(.05,'PDF 추출 및 가격 조회 중')
                result=prepare_dataset(str(workdir/'reports'),progress=lambda msg:ctx.progress(.3,msg),window_days=int(window),threshold=float(threshold),output_path=str(workdir/'dataset.json'))
                ctx.check_cancelled();return result
            start_job(owner,'전처리 · 라벨링',prepare)
        dataset=st.session_state.get('dataset')
        saved=workdir/'dataset.json'
        if not dataset and saved.exists():
            try:
                dataset=json.loads(saved.read_text());st.session_state['dataset']=dataset
            except (ValueError,OSError): st.warning('저장된 데이터 파일을 읽을 수 없습니다.')
        if dataset:
            records=dataset.get('records',[]);st.json({k:v for k,v in dataset.items() if k!='records'})
            st.dataframe(pd.DataFrame([{'발간일':r.get('report_date'),'종목':r.get('ticker'),'라벨':r.get('label_id'),'관측 종료':r.get('label_end_date'),'상태':(r.get('label_info') if isinstance(r.get('label_info'),dict) else {}).get('status',r.get('status','확인 필요'))} for r in records]),hide_index=True,width='stretch')
            st.download_button('문서 데이터 JSON 저장',json.dumps(dataset,ensure_ascii=False,indent=2),file_name='dataset.json',mime='application/json')
    with tabs[2]:
        st.subheader('재현 가능한 CPU 기준선')
        st.write('TF-IDF + 로지스틱 회귀를 학습하고 다수 클래스 기준선과 비교합니다. 문서·날짜로 먼저 분할하고, 다음 평가 구간과 수익률 관측 기간이 겹치는 학습 문서는 제외합니다.')
        st.caption('임계값은 검증셋에서만 조정합니다. 테스트셋은 최종 평가에만 사용합니다. 학습에 필요한 문서·날짜·클래스가 부족하면 학습을 중단합니다. 평가 표본이 부족하면 평가 불충분 상태를 기록합니다.')
        c1,c2,c3=st.columns(3);max_tokens=c1.select_slider('구간 최대 토큰',options=[128,256,384,512],value=256);overlap=c2.number_input('겹침 토큰',0,64,32);seed=c3.number_input('랜덤 시드',0,999999,42)
        dataset=st.session_state.get('dataset',{});records=dataset.get('records',[])
        valid=sum(r.get('label_id') in (0,1) for r in records)
        st.metric('현재 데이터의 라벨 문서',valid)
        if st.button('CPU 기준선 학습 · 평가 시작',type='primary',disabled=demo or not valid):
            snapshot=deepcopy(records)
            def train(ctx):
                from train import train_model
                ctx.progress(.05,'날짜 분할·누수 제거·기준선 학습 중')
                result=train_model(progress=lambda msg:ctx.progress(.3,msg),records=snapshot,output_dir=str(workdir/'models'),backend='baseline',max_tokens=max_tokens,overlap=int(overlap),seed=int(seed))
                ctx.check_cancelled();return result
            start_job(owner,'모델 학습 · 평가',train)
        st.caption('Transformer 학습은 선택 의존성 설치 후 CLI에서 명시적으로 실행합니다. 앱이 대형 모델을 자동 다운로드하지 않습니다.')
        with st.expander('평가 지표 읽는 법'):
            st.write('Precision은 음성으로 분류한 문서의 정밀도, recall은 음성 문서의 재현율, F1은 두 지표의 조화평균입니다. PR-AUC와 혼동행렬, 클래스별 표본 수를 함께 확인하세요. 부족한 분할의 점수를 임의로 채우지 않습니다.')


def main():
    st.markdown('''<style>
    html,body,[class*="css"],.stApp {font-family:'Noto Sans KR','Apple SD Gothic Neo','Malgun Gothic',sans-serif;}
    .block-container {max-width:1440px;padding-top:2.5rem;padding-bottom:4rem;}
    h1 {font-size:2.25rem!important;letter-spacing:-.065rem;line-height:1.35!important;margin-bottom:.3rem!important;}
    h2,h3 {letter-spacing:-.035rem;} .eyebrow{font-size:.72rem;letter-spacing:.16rem;font-weight:700;color:#147D72;margin-bottom:.7rem;}
    .lede{font-size:1.03rem;color:#617185;line-height:1.8;margin-bottom:2rem;}.brand{font-size:1.22rem;font-weight:800;letter-spacing:.05rem;margin-top:.2rem;}.muted{font-size:.78rem;color:#778396;margin:.5rem 0 2rem;}
    [data-testid="stMetric"]{background:white;border:1px solid #E2E8F0;border-radius:14px;padding:18px 20px;min-height:112px;margin-bottom:20px;}
    [data-testid="stMetricValue"]{font-size:1.8rem;} .notice{background:#EDF4F3;border:1px solid #D6E8E3;border-radius:12px;padding:22px;line-height:1.8;margin:12px 0 18px;color:#345A55;}
    .source{white-space:pre-wrap;overflow-wrap:anywhere;background:#fff;border:1px solid #DDE4EC;border-radius:0 0 12px 12px;padding:26px;line-height:1.95;max-height:800px;overflow:auto;font-size:.92rem;}
    .source-label{background:#182B44;color:white;padding:11px 20px;border-radius:12px 12px 0 0;font-size:.76rem;letter-spacing:.05rem;}
    mark{background:#DDF3BC;padding:2px 0;border-radius:2px;}[data-testid="stSidebar"]{border-right:1px solid #E0E6ED;}
    [data-testid="stDataFrame"]{border-radius:10px;} .stButton>button{border-radius:9px;} [data-baseweb="tab-list"]{gap:24px;}
    @media(max-width:760px){.block-container{padding-top:1.5rem;}h1{font-size:1.7rem!important;}}
    </style>''',unsafe_allow_html=True)
    try:
        owner,lib,demo=get_workspace()
        models=available_models(owner) if not demo else []
        render_job(owner)
        page=st.session_state['nav']
        if page=='대시보드': dashboard(owner,lib,models)
        elif page=='리포트 분석': analysis_page(owner,lib,models,demo)
        elif page=='분석 상세': detail_page(owner,lib)
        elif page=='리포트 비교': compare_page(owner,lib)
        elif page=='분석 보관함': library_page(owner,lib)
        elif page=='데이터 · 모델': model_page(owner,lib,models,demo)
    except (LibraryError,ValueError,OSError) as exc:
        st.error(f'작업을 완료하지 못했습니다: {exc}')
        st.caption('입력과 설정을 확인한 뒤 다시 시도하세요. 저장된 데이터는 자동으로 삭제하지 않습니다.')

if __name__=='__main__':
    main()
