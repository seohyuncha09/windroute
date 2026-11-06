from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xarray as xr

import asia_wind_route_interactive_hf as core


st.set_page_config(
    page_title="ERA5 바람장 항로 분석",
    page_icon="✈️",
    layout="wide",
)

st.title("ERA5 바람장 기반 항로 분석")
st.caption(
    "ERA5의 특정 시각·기압면 바람 자료를 이용해 기준 대권항로와 "
    "바람을 고려한 교육용 최적항로를 비교합니다."
)


@st.cache_resource(show_spinner="ERA5 데이터를 불러오는 중입니다...")
def load_dataset():
    # core 모듈이 Hugging Face에서 파일을 내려받고 캐시 경로를 제공합니다.
    return xr.open_dataset(core.DATA_FILE)


def normalize_longitude(lon, lon_min_data, lon_max_data):
    if lon_min_data >= 0 and lon_max_data > 180:
        return lon % 360
    return ((lon + 180) % 360) - 180


def build_region(ds, start, goal):
    lon_min_data = float(ds.longitude.min())
    lon_max_data = float(ds.longitude.max())

    start = (
        float(start[0]),
        normalize_longitude(float(start[1]), lon_min_data, lon_max_data),
    )
    goal = (
        float(goal[0]),
        normalize_longitude(float(goal[1]), lon_min_data, lon_max_data),
    )

    for label, point in (("출발지", start), ("도착지", goal)):
        if not (
            float(ds.latitude.min()) <= point[0] <= float(ds.latitude.max())
            and lon_min_data <= point[1] <= lon_max_data
        ):
            raise ValueError(
                f"{label}가 ERA5 데이터 범위 밖입니다: "
                f"위도 {point[0]:.3f}, 경도 {point[1]:.3f}"
            )

    lat_min = max(
        min(start[0], goal[0]) - core.CORRIDOR_MARGIN_DEG,
        float(ds.latitude.min()),
    )
    lat_max = min(
        max(start[0], goal[0]) + core.CORRIDOR_MARGIN_DEG,
        float(ds.latitude.max()),
    )
    lon_min = max(
        min(start[1], goal[1]) - core.CORRIDOR_MARGIN_DEG,
        lon_min_data,
    )
    lon_max = min(
        max(start[1], goal[1]) + core.CORRIDOR_MARGIN_DEG,
        lon_max_data,
    )

    # ERA5 위도 좌표는 보통 북쪽에서 남쪽 순서로 저장됩니다.
    lat_values = ds.latitude.values
    if lat_values[0] > lat_values[-1]:
        region = ds.sel(
            latitude=slice(lat_max, lat_min),
            longitude=slice(lon_min, lon_max),
        )
    else:
        region = ds.sel(
            latitude=slice(lat_min, lat_max),
            longitude=slice(lon_min, lon_max),
        )

    lats = region.latitude.values
    lons = region.longitude.values
    if len(lats) < 2 or len(lons) < 2:
        raise ValueError(
            "선택한 좌표 주변의 ERA5 격자가 부족합니다. "
            "출발지와 도착지를 확인해 주세요."
        )
    return region, start, goal, lats, lons, (lat_min, lat_max, lon_min, lon_max)


def make_figure(
    lats, lons, u, v, start, goal, route, level,
    show_speed, show_arrows, bounds
):
    lat_min, lat_max, lon_min, lon_max = bounds
    speed = np.sqrt(u**2 + v**2)
    fig = go.Figure()

    stride = core.WIND_STRIDE
    lat_sample = lats[::stride]
    lon_sample = lons[::stride]
    u_sample = u[::stride, ::stride]
    v_sample = v[::stride, ::stride]
    speed_sample = speed[::stride, ::stride]

    if show_speed:
        fig.add_trace(
            go.Scattergeo(
                lon=np.tile(lon_sample, len(lat_sample)),
                lat=np.repeat(lat_sample, len(lon_sample)),
                mode="markers",
                marker=dict(
                    size=7,
                    color=speed_sample.ravel(),
                    colorscale="Turbo",
                    cmin=0,
                    cmax=max(40, float(np.nanpercentile(speed, 98))),
                    opacity=0.32,
                    colorbar=dict(title="풍속<br>m/s"),
                    showscale=True,
                ),
                name="풍속",
                customdata=np.column_stack(
                    [u_sample.ravel(), v_sample.ravel()]
                ),
                hovertemplate=(
                    "위도 %{lat:.2f}°<br>"
                    "경도 %{lon:.2f}°<br>"
                    "풍속 %{marker.color:.1f} m/s<br>"
                    "U %{customdata[0]:.1f} m/s<br>"
                    "V %{customdata[1]:.1f} m/s"
                    "<extra></extra>"
                ),
            )
        )

    if show_arrows:
        arrow_lon, arrow_lat = [], []
        for i in range(0, len(lats), stride):
            for j in range(0, len(lons), stride):
                uu, vv = float(u[i, j]), float(v[i, j])
                lat, lon = float(lats[i]), float(lons[j])
                scale = 0.10
                dlat = vv * scale / 20.0
                dlon = uu * scale / (
                    20.0 * max(0.25, np.cos(np.radians(lat)))
                )
                arrow_lon.extend([lon - dlon / 2, lon + dlon / 2, None])
                arrow_lat.extend([lat - dlat / 2, lat + dlat / 2, None])

        fig.add_trace(
            go.Scattergeo(
                lon=arrow_lon,
                lat=arrow_lat,
                mode="lines",
                line=dict(color="#263238", width=1),
                name="바람 방향",
                hoverinfo="skip",
            )
        )

    direct = core.great_circle_points(start, goal)
    fig.add_trace(
        go.Scattergeo(
            lon=[p[1] for p in direct],
            lat=[p[0] for p in direct],
            mode="lines",
            line=dict(color="#555555", width=3, dash="dash"),
            name="기준 대권항로",
        )
    )
    fig.add_trace(
        go.Scattergeo(
            lon=[p[1] for p in route],
            lat=[p[0] for p in route],
            mode="lines",
            line=dict(color="#d62728", width=4),
            name=f"{level} hPa 최적항로",
        )
    )
    fig.add_trace(
        go.Scattergeo(
            lon=[start[1], goal[1]],
            lat=[start[0], goal[0]],
            mode="markers+text",
            marker=dict(
                size=11,
                color=["#00a58a", "#e45756"],
                line=dict(color="white", width=1),
            ),
            text=["출발", "도착"],
            textposition="top right",
            name="출발·도착",
        )
    )

    fig.update_layout(
        title=f"ERA5 바람장 및 항로 비교 — {level} hPa",
        geo=dict(
            scope="asia",
            projection_type="mercator",
            showland=True,
            landcolor="#f1eee6",
            showocean=True,
            oceancolor="#dcecf5",
            showcoastlines=True,
            showcountries=True,
            center=dict(
                lat=(start[0] + goal[0]) / 2,
                lon=(start[1] + goal[1]) / 2,
            ),
            lataxis=dict(range=[lat_min, lat_max]),
            lonaxis=dict(range=[lon_min, lon_max]),
        ),
        height=720,
        margin=dict(l=10, r=10, t=60, b=20),
        legend=dict(orientation="h", y=-0.08),
    )
    return fig


with st.sidebar:
    st.header("항로 설정")
    st.subheader("출발지")
    start_lat = st.number_input(
        "출발지 위도", min_value=-90.0, max_value=90.0,
        value=37.4602, step=0.01, format="%.4f"
    )
    start_lon = st.number_input(
        "출발지 경도", min_value=-180.0, max_value=180.0,
        value=126.4407, step=0.01, format="%.4f"
    )

    st.subheader("도착지")
    goal_lat = st.number_input(
        "도착지 위도", min_value=-90.0, max_value=90.0,
        value=13.6900, step=0.01, format="%.4f"
    )
    goal_lon = st.number_input(
        "도착지 경도", min_value=-180.0, max_value=180.0,
        value=100.7501, step=0.01, format="%.4f"
    )

    level = st.selectbox("기압면", options=core.PRESSURE_LEVELS, index=0)
    show_speed = st.toggle("풍속 표시", value=True)
    show_arrows = st.toggle("풍향 표시", value=True)
    calculate = st.button("경로 계산", type="primary", use_container_width=True)

st.info(
    "기본 좌표는 인천공항 → 방콕 수완나품공항입니다. "
    "다른 지역을 분석하려면 위도·경도를 바꿔 주세요."
)

if calculate:
    try:
        ds = load_dataset()

        if "valid_time" not in ds.sizes:
            st.error("데이터에 valid_time 차원이 없습니다.")
            st.stop()

        if core.TIME_INDEX >= ds.sizes["valid_time"]:
            st.error("설정된 분석 시각이 데이터 시간 범위를 벗어났습니다.")
            st.stop()

        selected_time = pd.to_datetime(
            ds.valid_time.values[core.TIME_INDEX], utc=True
        )

        region, start, goal, lats, lons, bounds = build_region(
            ds, (start_lat, start_lon), (goal_lat, goal_lon)
        )

        if level not in region.pressure_level.values:
            st.error(f"{level} hPa 자료가 파일에 없습니다.")
            st.stop()

        u = region.u.isel(valid_time=core.TIME_INDEX).sel(
            pressure_level=level
        ).values
        v = region.v.isel(valid_time=core.TIME_INDEX).sel(
            pressure_level=level
        ).values

        if np.isnan(u).any() or np.isnan(v).any():
            st.error("선택한 기압면에 결측치가 있습니다.")
            st.stop()

        with st.spinner("바람을 고려한 항로를 계산하고 있습니다..."):
            route = core.dijkstra_time_route(
                lats, lons, u, v, start, goal
            )
            distance_km, time_hours = core.path_metrics(
                route, u, v, lats, lons
            )
            direct_points = core.great_circle_points(start, goal)
            direct_distance_km = sum(
                core.distance_bearing(a, b)[0]
                for a, b in zip(direct_points[:-1], direct_points[1:])
            ) / 1000.0

        fig = make_figure(
            lats, lons, u, v, start, goal, route, level,
            show_speed, show_arrows, bounds
        )

        st.subheader("분석 결과")
        col1, col2, col3 = st.columns(3)
        col1.metric("최적항로 거리", f"{distance_km:,.1f} km")
        col2.metric("예상 비행 시간", f"{time_hours:.2f} 시간")
        col3.metric("기준 대권항로 거리", f"{direct_distance_km:,.1f} km")

        st.caption(
            f"분석 시각: {selected_time:%Y-%m-%d %H:%M UTC} · "
            f"기압면: {level} hPa · 경로점: {len(route)}개"
        )
        st.plotly_chart(fig, use_container_width=True)

        result_df = pd.DataFrame(
            [{
                "기압면(hPa)": level,
                "최적항로 거리(km)": round(distance_km, 2),
                "예상 시간(h)": round(time_hours, 3),
                "기준 대권항로 거리(km)": round(direct_distance_km, 2),
                "분석 시각(UTC)": str(selected_time),
            }]
        )
        st.download_button(
            "분석 결과 CSV 다운로드",
            data=result_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="wind_route_summary.csv",
            mime="text/csv",
        )

    except Exception as exc:
        st.error(f"계산 중 문제가 발생했습니다: {exc}")
else:
    st.markdown(
        "왼쪽에서 출발지와 도착지를 설정한 뒤 **경로 계산**을 눌러 주세요."
    )

st.divider()
st.caption(
    "주의: 이 앱은 한 시각의 바람을 비행 내내 고정하는 교육용 단순 모델입니다. "
    "실제 항공 운항·항법에 사용할 수 없습니다."
)
