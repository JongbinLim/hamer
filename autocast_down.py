import torch

# 🔥 PyTorch 기본 데이터 타입을 float32로 강제 설정
torch.set_default_dtype(torch.float32)

# 🔥 TF32(자동 연산 최적화)를 비활성화하여 예기치 않은 precision 변경 방지
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# 🔥 GPU 캐시 완전 초기화
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# 🔥 Autocast를 무조건 False로 설정하는 새로운 컨텍스트 적용
with torch.cuda.amp.autocast(enabled=False):
    pass  # 실행하여 Autocast를 False로 강제 변경

# ✅ Autocast 상태 확인
print(f"Autocast status after reset: {torch.is_autocast_enabled()}")  # 🔥 반드시 False여야 함
