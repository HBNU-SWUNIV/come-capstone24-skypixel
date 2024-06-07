# 한밭대학교 컴퓨터공학과 Sky Pixel팀

**팀 구성**
- 20191785 이지상
- 20191735 서형원

## <u>Teammate</u> Project Background
- ### 필요성
  - SAR(Synthetic Aperture Radar) 이미지를 활용한 지식 증류(Knowledge Distillation, KD) 기술의 필요성
  - 복잡한 도시 환경 내에서 건물 탐지의 중요성
- ### 기존 해결책의 문제점
  - 기존 EO(Electro-Optical) 이미지 분석의 한계
  - 레이블링 된 SAR 데이터의 부족 문제와 SAR 이미지의 낮은 해상도

## System Design
  - ### System Requirements
    - SAR 이미지를 활용한 건물 탐지 알고리즘 개발
    - EO 이미지와 SAR 이미지 간의 지식 증류를 위한 모델 구축
    
## Case Study
  - ### Description
    본 프로젝트는 EO 이미지를 teacher 모델로, SAR 이미지를 student 모델로 사용하여 지식 증류를 수행함. EO 이미지에서 학습된 지식을 SAR 이미지 분석에 적용하여 SAR 데이터의 분석 성능을 향상시키는 것을 목표로 함. 또한, 생성 모델(GAN 및 디퓨전 모델)을 활용하여 SAR 데이터의 부족 문제를 해결하고자 함. 추가적으로 SAR 이미지의 특징에 기반하여 배경을 고려한 데이터 증대 기법인 Crop-Paste 기법을 제안하여 object-detection 성능을 올리는 연구를 수행함.

## Conclusion
  - ### 연구 결과
    - SAR 세그멘테이션의 성능을 높이기 위한 생성 모델 및 데이터 증대 기법의 효과 확인
  - ### 향후 연구 방향
    - Optical 이미지를 teacher 모델로 사용하고, student 모델의 학습에 가이드를 주는 방향으로 학습하여 모델 성능 향상
    - 캡스톤 1에서 연구한 전처리 및 증대 기법을 적용하여 지식 증류(Knowledge Distillation, KD) 기법에서의 SOTA(State-Of-The-Art) 성능을 달성하고, 이를 기반으로 SCI 논문 작성

## Project Outcome
- ### KCC 2024(한국정보과학회 2024) 2편 submit/accept
  - 합성개구레이다 영상에서 광학 영상으로의 이미지 변환을 위한 데이터 전처리 및 증대 기법
  - 합성 개구 레이다 이미지의 객체 탐지 성능 향상을 위한 절삭-붙여넣기 기법

- ### 제6회 2024 연구개발특구 AI SPARK 챌린지 글로벌 산불 감지 챌린지🌋 :  AI를 활용한 향상된 위성 이미지 분석
  - 대상: 연구개발특구진흥재단 이사장상 수상
