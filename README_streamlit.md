# ERA5 바람장 항로 분석 웹 앱

## GitHub에 올릴 파일
- `app.py`
- `asia_wind_route_interactive_hf.py`
- `requirements.txt`

ERA5 NetCDF 파일은 Hugging Face Dataset에서 자동으로 다운로드합니다.

## 로컬 실행
```bash
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

## Streamlit Community Cloud 배포
1. 세 파일을 GitHub 저장소의 최상위 폴더에 업로드합니다.
2. https://share.streamlit.io/ 에 접속해 GitHub 저장소를 연결합니다.
3. Main file path에 `app.py`를 지정하고 Deploy합니다.

## 참고
- 최초 실행 시 ERA5 파일 다운로드에 시간이 걸릴 수 있습니다.
- 앱은 한 시각의 바람을 비행 내내 고정하는 교육용 단순 모델입니다.
- 실제 항공 운항·항법 용도로 사용할 수 없습니다.
