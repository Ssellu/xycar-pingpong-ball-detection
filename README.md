# Xycar Pingpong Ball Detection

## 목표

- Xycar의 RGB 카메라로부터 입력된 이미지에 존재하는 **탁구공을 검출**한다.
- 탁구공의 **실제 위치를 추정**한다.
- 2D 지도에 탁구공의 위치를 표시한다. (Optional)

## 요구사항

- Xycar의 RGB 카메라로부터 입력된 이미지에 존재하는 **탁구공을 검출**한다.
  - 데이터 수집
  - 데이터 처리 - 라벨링, 증강(Augmentation)
  - 데이터 학습 
  - Inference
- 탁구공의 **실제 위치를 추정**한다.
  - Calibration 
  - (+) Sensor Fustion with LiDAR
  - Location Inference of Bbox FOV
- 2D 지도에 탁구공의 위치를 표시한다. (Optional)

## Project Pipeline

![image-20230208141515128](https://user-images.githubusercontent.com/33347724/217444724-ec3a9bb0-8d37-4be0-9258-d70c09e45dc6.png)


1. 카메라로부터 이미지를 입력 받는다.
   - OpenCV
2. 카메라 Intrinsic Calibration
3. 데이터 준비 및 처리
   - `.jpg`
   - Labeling
     - CVAT
   - Augmentation
   - Calibration
     - 전체 화면?
     - 라벨링한 Bbox/Segmentation만? - Display 화면(전체 화면)에는 왜곡은 있음(무시가능, 디버깅 어려움)
4. 탁구공 인식 OB Model
   - Yolo V3 Tiny
   - ONNX 
   - TensorRT
5. 탁구공 위치(카메라와 탁구공의 거리) 추정
   1. 기하학적 사전 조건
   2. Extrinsic Calibration
   3. (?) 자이카와 탁구공의 거리인지? 아니면 카메라와 탁구공의 거리인지?
   4. (?) Sensor Fusion 써야할까?
6. 지도 상에서 탁구공의 위치를 표시한다. (Optional)

## 상세기능

### Data Collection

1. 자이카에 장착된 카메라를 사용하여 탁구공 이미지(또는 비디오)를 촬영한다.
2. (비디오의 경우) 연속된 이미지를 적절한 시간 간격으로 이미지를 추출한다.
3. 데이터 라벨링을 위해 별도 디렉토리에 저장한다.

### Data Labeling

1.  `CVAT` 를 사용하여 라벨링한다.

   - Data Collection 단계에서 수집한 데이터에 대한 라벨링을 수행한다.

     (?) bbox? Segmentation?

2. 레이블링 형식을 확인한다. (?) Yolo 포맷이 아무래도...

### Model Training

1. 적절한 OB 모델 선택 
2. 선택한 모델의 학습 데이터 형식 확인
3. 레이블링 데이터를 모델의 학습 데이터 형식에 맞게 변환
4. 모델 학습 수행
5. 모델 학습의 튜닝 포인트 조절하며 학습 실험
   - Traning Hyperparameter 
   - More dataset & Additional Dataset Augementation
6. 학습 결과 확인

### Model Inference 

1. 학습 그래프 추이를 확인하며 적절한 모델 파일을 선택

2. 비디오 데이터를 입력으로 하는 Model Inference 코드 작성 및 확인((?) AWS)

3. 실제 자이카 환경에서 동작 가능한 Model Inference 코드를 작성
   `Model 축소`, `Pruning`, `GPU Acceleration`

   다음 두가지 사항을 고려해야함.

   1. Model Inference FPS - 한 장의 이미지를 처리하는데 걸리는 시간(15FPS, 8FPS)
   2. Model Prediction Accuracy - 실제 환경에서의 정확도

### Object Position Estimation

1. 탁구공의 실제 지름을 알고있어야 한다.
2. Model의 예측 결과(Bbox의 크기 및 위치, 클래스) 중에서 활용 가능한 정보를 선택한다.
3. 자신만의 객체 위치 추정 알고리즘을 작성한다. 
4. 실제 자이카에서 실험 결과를 비교, 분석한다.
   - Object의 거리 정확도 (종/횡 방향 정확도)

## Project Workflow

| 수                      | 목                                                           | 금   | 토   | 일   | 월   | 화   | 수   |
| ----------------------- | ------------------------------------------------------------ | ---- | ---- | ---- | ---- | ---- | ---- |
| 라벨링<br />Augemtation | 추가 라벨링<br />자이카 캘리브레이션<br />자이카 거리 추정 알고리즘 작성 |      |      |      |      |      | 발표 |

