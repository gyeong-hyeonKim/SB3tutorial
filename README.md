# Stable Baselines3 커스텀 환경 프로젝트

이 프로젝트는 Gymnasium API를 사용하여 직접 환경을 만들고, Stable Baselines3의 PPO 알고리즘으로 에이전트를 학습시킨 결과입니다.<br>
액션 : 왼쪽, 오른쪽<br>
상태 : 카트위치, 카트속도, 막대 각도, 막대 각속도<br>
<br>
# 실행방법
1. 파이썬 가상환경 생성<br>
py -<버전> -m venv <가상환경 이름>
필자는 py -3.10 -m venv .venv 로 실행함

2. 가상환경 실행 : .\.venv\Scripts\activate

3. 모듈 및 라이브러리 설치
pip install -r requirements.txt

4. 메인파일 실행
 python .\main.py

## 학습 결과

학습된 에이전트가 환경에서 임무를 수행하는 모습입니다.

![학습결과](./SB3/videos/go-left-step-0-to-step-500.gif)

## 프로젝트 구조
