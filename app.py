import streamlit as st
import time

# 진우님이 만든 inference.py에서 핵심 함수들을 가져옵니다.
from inference import predict_hidden_sell_signal, get_latest_model_path

# 1. 웹페이지 기본 설정
st.set_page_config(page_title="AI 리포트 해독기", page_icon="🕵️‍♂️", layout="centered")

# 2. 메인 타이틀 및 소개
st.title("📉 AI 증권사 리포트 해독기")
st.markdown("리포트의 핵심 문장을 입력하면 AI가 숨겨진 진짜 의도(Negative Signal)를 파악합니다.")

# 3. 사이드바 (현재 로드된 모델 정보 표시)
st.sidebar.header("⚙️ 시스템 정보")
try:
    model_path = get_latest_model_path()
    st.sidebar.success(f"현재 구동 중인 AI 뇌:\n{model_path.split('/')[-1]}")
except Exception as e:
    st.sidebar.error("학습된 모델을 찾을 수 없습니다. main.py를 먼저 실행하세요.")

# 4. 사용자 텍스트 입력창
st.subheader("🔍 텍스트 분석")
user_input = st.text_area(
    "리포트에서 의심스러운 뉘앙스의 문장을 복사해서 붙여넣어 보세요:", 
    height=150, 
    placeholder="예: 최근 메모리 반도체 수요 부진으로 인해 실적 추정치를 하향 조정합니다."
)

# 5. 분석 실행 버튼
if st.button("🚀 AI 분석 시작", use_container_width=True):
    if user_input.strip() == "":
        st.warning("문장을 먼저 입력해 주세요!")
    else:
        # 로딩 스피너 (열심히 분석하는 척 애니메이션)
        with st.spinner("AI가 금융 데이터를 바탕으로 문장의 행간을 분석 중입니다..."):
            time.sleep(1.5) 
            
            try:
                # 💡 진우님의 백엔드 AI 모델이 실제로 작동하는 부분!
                label, conf = predict_hidden_sell_signal(user_input, model_path=model_path)
                
                st.divider()
                st.subheader("💡 분석 결과")
                
                # 결과에 따라 시각적 효과(색상, 프로그레스 바)를 다르게 줍니다.
                if "유지/매수" in label:
                    st.success(f"**안심하세요!** 평범한 {label} 의견입니다. (확률: {conf:.1f}%)")
                    st.progress(int(conf) / 100)
                else:
                    st.error(f"**🚨 주의!** 이면에 감춰진 {label} 입니다! (확률: {conf:.1f}%)")
                    st.progress(int(conf) / 100)
                    
            except Exception as e:
                st.error(f"분석 중 에러가 발생했습니다: {e}")