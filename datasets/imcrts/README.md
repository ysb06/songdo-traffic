# 인천시 교통량 데이터

인천시 교통량 데이터는 WebAPI를 통해 JSON 형태로 제공하므로 데이터를 받아서 이를 Pandas 및 Excel로 변환 작업을 수행. API는 Call 제한도 있고 전체 데이터 크기가 크지도 않으므로 디스크에 저장하는 것이 효율적이다.

# 데이터 불러오기

다음 명령어를 실행한다.

``` bash
python3 -m songdo_traffic_core.dataset.imcrts --key {API 호출 Key 파일 경로} --date_range {(Start Date)-(End Date)} --outpu_dir {데이터 출력 폴더 경로}
```

여기서 key는 반드시 있어야 함. 텍스트 파일 형태로 제공하면 된다. Key는 Data.go.kr에서 회원 가입 및 해당 데이터 이용 신청을 해야 획득 및 사용이 가능하다.

# 기타

Pickle 저장은 성공했지만 openpyxl과 같은 엑셀 모듈이 없어서 엑셀 저장에 실패했다면 collector.py의 IMCRTSExcelConverter 클래스를 사용하면 된다.


# References

1. 인천시 교통량 데이터
    - https://www.data.go.kr/data/15113145/openapi.do


