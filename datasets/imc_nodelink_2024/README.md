# 인천시 표준노드링크

2023-02-23 기준 전국표준노드링크 데이터에서 인천시 데이터만 뽑은 데이터. 원본 전국표준노드링크로 작업하면 로딩시간이 엄청 걸리므로 인천시에 해당하는 데이터만 추출. NODE_ID, LINK_ID 앞 세자리가 지역구 코드이므로 인천시에 해당하는 구만 선택.

## 파일 설명

기본적으로 모든 파일은 Shapefile 형식을 따른다

### imc_link

인천시 도로 Link 데이터

### imc_node

인천시 교차로 Node 데이터

### imc_turninfo

인천시 도로 규칙 데이터, 유턴 가능 좌/우회전 가능/불가능 등 정보 포함

### sensor_node

송도 내 루프 검지기 등 센서 설치 위치 표시. 센서가 생각보다 많지 않다. 실제로 송도 교통량 데이터는 제대로 확보 불가능.


## 데이터 생성 방법 (Data Generation)

터미널을 열고 프로젝트 폴더에서 다음을 입력한다. 해당 명령어는 songdo_traffic_core.dataset.nodelink 모듈을 실행하는 코드이다.

``` bash
python3 -m songdo_traffic_core.dataset.nodelink --nodelink_dir {표준노드링크 폴더 경로} --songdo_traffic_file {송도교통량데이터 파일 경로} --outpu_dir {파일 출력 경로}
```

## References

1. 전국표준노드링크
    - https://www.its.go.kr/nodelink/nodelinkRef