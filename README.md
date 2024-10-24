# 한밭대학교 컴퓨터공학과 Sky Pixel팀

## 팀 구성
- 20191785 이지상
- 20191735 서형원

## 프로젝트 배경

### 필요성
- 🌐 SAR(Synthetic Aperture Radar) 이미지를 활용한 지식 증류(Knowledge Distillation, KD) 기술의 필요성
- 🏙️ 복잡한 도시 환경 내에서 건물 탐지의 중요성

### 기존 해결책의 문제점
- 📉 기존 EO(Electro-Optical) 이미지 분석의 한계
- 🖼️ 레이블링 된 SAR 데이터의 부족 문제와 SAR 이미지의 낮은 해상도

## 시스템 설계

### 시스템 요구사항
- 🏗️ SAR 이미지를 활용한 건물 탐지 알고리즘 개발
- 🧠 EO 이미지와 SAR 이미지 간의 지식 증류를 위한 모델 구축

## 사례 연구

### 설명
본 프로젝트는 다음과 같은 목표와 방법을 가지고 진행됨:

- EO 이미지를 teacher 모델로, SAR 이미지를 student 모델로 사용하여 지식 증류를 수행함
  - EO 이미지에서 학습된 지식을 SAR 이미지 분석에 적용하여 SAR 데이터의 분석 성능을 향상시키는 것을 목표로 함.

- 생성 모델(GAN 및 디퓨전 모델)을 활용하여 SAR 데이터의 부족 문제를 해결하고자 함
  - 생성 모델을 사용하여 SAR 데이터의 부족을 보완함.

- SAR 이미지의 특징에 기반하여 배경을 고려한 데이터 증대 기법인 Crop-Paste 기법을 제안하여 object-detection 성능을 올리는 연구를 수행함
  - SAR 이미지의 특성을 반영한 데이터 증대 기법을 통해 탐지 성능을 개선함.

- **Overview**
  ![지식증류](https://github.com/HBNU-SWUNIV/come-capstone24-skypixel/assets/98447471/69dbab16-bb39-45d1-9507-d14ec636df40)
  ![생성모델](https://github.com/HBNU-SWUNIV/come-capstone24-skypixel/blob/main/assets/sar2eo_pipeline.png)
  ![Crop-Paste](https://github.com/HBNU-SWUNIV/come-capstone24-skypixel/blob/main/assets/crop-paste_pipeline.png)
  
## 결론

### 연구 결과
- 📈 SAR 세그멘테이션의 성능을 높이기 위한 생성 모델 및 데이터 증대 기법의 효과 확인
  ![생성모델](https://github.com/HBNU-SWUNIV/come-capstone24-skypixel/blob/main/assets/sar2eo_inference.png)
  ![Crop-Paste](https://github.com/HBNU-SWUNIV/come-capstone24-skypixel/blob/main/assets/crop-paste_inference.png)

### 향후 연구 방향
- 📚 Optical 이미지를 teacher 모델로 사용하고, student 모델의 학습에 가이드를 주는 방향으로 학습하여 모델 성능 향상
- 📝 캡스톤 1에서 연구한 전처리 및 증대 기법을 적용하여 지식 증류(Knowledge Distillation, KD) 기법에서의 SOTA(State-Of-The-Art) 성능을 달성하고, 이를 기반으로 SCI 논문 작성

## 프로젝트 성과

### 제6회 2024 연구개발특구 AI SPARK 챌린지 글로벌 산불 감지 챌린지🌋
- AI를 활용한 향상된 위성 이미지 분석
- 🏆 대상: 연구개발특구진흥재단 이사장상 수상 ([링크](https://aifactory.space/task/2723/overview))

### KJRS(scopus, 한국원격탐사 논문지) 1편 제출/게재
- 📝 Parcel-Based Crop Type Classification in UAV Imagery with SAM for Smallholder Farms

### KCC 2024(한국정보과학회 2024) 2편 제출/발표
- 📝 합성개구레이다 영상에서 광학 영상으로의 이미지 변환을 위한 데이터 전처리 및 증대 기법
- 📝 합성 개구 레이다 이미지의 객체 탐지 성능 향상을 위한 절삭-붙여넣기 기법

### KTCP(정보과학회 컴퓨팅의 실제 논문지) 2편 제출
- 📝 확산 모델을 활용한 SAR-광학 영상 변환을 위한 데이터 전처리와 증대 기법
- 📝 합성 개구 레이다 이미지에서 객체 탐지 성능 향상을 위한 Crop-Paste 데이터 증대 기법

