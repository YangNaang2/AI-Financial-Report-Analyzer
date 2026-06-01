import os
os.environ["USE_TF"] = "NO"
os.environ["USE_KERAS"] = "NO"

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def predict_hidden_sell_signal(text, model_path='./my_finbert_model'):
    # 허깅페이스 서버가 아닌, 방금 학습시켜 저장한 폴더에서 모델을 불러옴
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = F.softmax(outputs.logits, dim=-1)[0]
    predicted_class = torch.argmax(probabilities).item()
    
    labels = {0: "🟢 유지/매수 (Positive/Neutral)", 1: "🔴 숨은 매도 (Negative Signal)"}
    pred_label = labels[predicted_class]
    confidence = probabilities[predicted_class].item() * 100
    
    return pred_label, confidence

def get_latest_model_path(base_dir='./models'):
    """models 폴더 내에서 가장 최근에 수정된 폴더의 경로를 찾아옵니다."""
    # base_dir 안의 모든 폴더 목록을 가져옵니다.
    folders = [f.path for f in os.scandir(base_dir) if f.is_dir()]
    
    if not folders:
        raise FileNotFoundError(f"🚨 {base_dir} 경로에 학습된 모델이 없습니다!")
        
    # 생성(수정) 시간을 기준으로 가장 최신 폴더를 찾습니다.
    latest_folder = max(folders, key=os.path.getmtime)
    return latest_folder

if __name__ == "__main__":
    # 💡 이제 수동으로 이름을 바꿀 필요 없이 알아서 최신 모델을 물고 옵니다!
    my_model_path = get_latest_model_path()
    print(f"📂 로드된 최신 모델 경로: {my_model_path}\n")
    
    test_sentence = "최근 메모리 반도체 수요 폭등으로 인해 실적 추정치를 초상향합니다."
    label, conf = predict_hidden_sell_signal(test_sentence, model_path=my_model_path)
    
    print(f"▶ AI 판독 결과: {label} (확률: {conf:.1f}%)")