# Streamlit Cloud 배포 실패 시 해결 방법

## 🔴 현재 에러 원인

**Python 3.14** 사용 중 → Pillow, NumPy, Pandas 등이 소스 빌드를 시도하다 실패/타임아웃

## ✅ 해결 방법 (Python 3.12로 재배포)

Streamlit Cloud에서는 **배포 후에는 Python 버전을 변경할 수 없습니다.**  
반드시 **앱을 삭제하고 다시 배포**하면서 Python 3.12를 선택해야 합니다.

### 단계별 안내

1. **[share.streamlit.io](https://share.streamlit.io)** 접속 후 로그인

2. **기존 앱 삭제**
   - 앱 옆 **⋮ (더보기)** 클릭
   - **Settings** → **Delete app** 선택
   - (나중에 같은 서브도메인 쓰려면 기억해두기)

3. **새 앱 배포**
   - **Create app** 클릭
   - Repository, Branch, Main file path 입력

4. **⚠️ 중요: Advanced settings**
   - **Advanced settings** 버튼 클릭
   - **Python version** 드롭다운에서 **3.12** 선택 (3.14 아님!)
   - Secrets 있으면 여기서 다시 입력
   - **Save** 클릭

5. **Deploy** 실행

---

## 프로젝트에 포함된 설정

- `runtime.txt` (python-3.12) – 일부 환경에서 참조
- `packages.txt` (zlib1g-dev) – Pillow 빌드용 zlib 헤더 (Python 3.14 시도 시)

**가장 확실한 방법은 위처럼 Advanced settings에서 Python 3.12를 직접 선택하는 것입니다.**
