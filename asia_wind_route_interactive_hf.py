"""ERA5 바람장 + 사용자 지정 항로 시각화 (교육용)

설치:
    py -m pip install xarray netCDF4 numpy pandas scipy plotly huggingface_hub

실행:
    이 Python 파일만 있으면 됩니다. ERA5 파일은 Hugging Face에서 자동으로 내려받습니다.
    출발지/도착지 위도·경도를 입력하면 항로를 다시 계산합니다.
"""
from pathlib import Path
from huggingface_hub import hf_hub_download
import heapq
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

DATASET_REPO = "seohyuncha09/era5-wind-data"
DATASET_FILENAME = "26a1fb3fbce4fd258e932986267c23fb.nc"

def download_era5_file():
    """Hugging Face Dataset에서 ERA5 파일을 가져옵니다.
    최초 실행 시 다운로드하고, 이후에는 로컬 캐시를 재사용합니다.
    """
    return Path(hf_hub_download(
        repo_id=DATASET_REPO,
        filename=DATASET_FILENAME,
        repo_type="dataset"
    ))

DATA_FILE = download_era5_file()
TIME_INDEX = 12
PRESSURE_LEVELS = [300, 250, 200]
TAS_MPS = 230.0
CORRIDOR_MARGIN_DEG = 8.0
EARTH_RADIUS_M = 6_371_000.0
MIN_GROUND_SPEED_MPS = 30.0
WIND_STRIDE = 2  # 바람 화살표/점 간격. 숫자를 키우면 더 성긴 표시

DEFAULT_START = (37.4602, 126.4407)  # 인천공항
DEFAULT_GOAL = (13.6900, 100.7501)   # 방콕 수완나품


def ask_point(label, default):
    print(f"\n{label} 좌표를 입력하세요. 기본값: 위도 {default[0]}, 경도 {default[1]}")
    raw = input("위도,경도 (그대로 기본값을 쓰려면 Enter): ").strip()
    if not raw:
        return default
    try:
        lat_s, lon_s = raw.replace(" ", "").split(",")
        point = (float(lat_s), float(lon_s))
        if not (-90 <= point[0] <= 90 and -180 <= point[1] <= 180):
            raise ValueError
        return point
    except Exception:
        print("좌표 형식이 올바르지 않아 기본값을 사용합니다. 예: 37.4602,126.4407")
        return default


def great_circle_points(start, goal, n=201):
    lat1, lon1 = np.radians(start); lat2, lon2 = np.radians(goal)
    p1 = np.array([np.cos(lat1)*np.cos(lon1), np.cos(lat1)*np.sin(lon1), np.sin(lat1)])
    p2 = np.array([np.cos(lat2)*np.cos(lon2), np.cos(lat2)*np.sin(lon2), np.sin(lat2)])
    omega = np.arccos(np.clip(np.dot(p1, p2), -1, 1))
    f = np.linspace(0, 1, n)
    if np.isclose(omega, 0): xyz = np.repeat(p1[None, :], n, axis=0)
    else: xyz = (np.sin((1-f)*omega)[:, None]*p1 + np.sin(f*omega)[:, None]*p2) / np.sin(omega)
    xyz /= np.linalg.norm(xyz, axis=1)[:, None]
    return list(zip(np.degrees(np.arcsin(xyz[:, 2])), np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))))


def distance_bearing(p1, p2):
    lat1, lon1 = np.radians(p1); lat2, lon2 = np.radians(p2)
    dlat, dlon = lat2-lat1, lon2-lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    dist = 2*EARTH_RADIUS_M*np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    bearing = np.arctan2(np.sin(dlon)*np.cos(lat2), np.cos(lat1)*np.sin(lat2)-np.sin(lat1)*np.cos(lat2)*np.cos(dlon))
    return dist, bearing


def nearest_grid_index(lats, lons, point):
    return int(np.abs(lats-point[0]).argmin()), int(np.abs(lons-point[1]).argmin())


def dijkstra_time_route(lats, lons, u_grid, v_grid, start, goal):
    nlat, nlon = u_grid.shape
    si, sj = nearest_grid_index(lats, lons, start); gi, gj = nearest_grid_index(lats, lons, goal)
    best = np.full((nlat, nlon), np.inf); parent = {}
    best[si, sj] = 0.0; queue = [(0.0, si, sj)]
    directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    while queue:
        elapsed, i, j = heapq.heappop(queue)
        if elapsed > best[i,j]: continue
        if (i,j) == (gi,gj): break
        p1 = (lats[i], lons[j])
        for di,dj in directions:
            ni,nj=i+di,j+dj
            if not (0<=ni<nlat and 0<=nj<nlon): continue
            p2=(lats[ni],lons[nj]); dist,bearing=distance_bearing(p1,p2)
            u=(u_grid[i,j]+u_grid[ni,nj])/2; v=(v_grid[i,j]+v_grid[ni,nj])/2
            along=u*np.sin(bearing)+v*np.cos(bearing); cross=u*np.cos(bearing)-v*np.sin(bearing)
            if abs(cross)>=TAS_MPS: continue
            gs=np.sqrt(TAS_MPS**2-cross**2)+along
            if gs<=MIN_GROUND_SPEED_MPS: continue
            cand=elapsed+dist/gs
            if cand<best[ni,nj]:
                best[ni,nj]=cand; parent[(ni,nj)]=(i,j); heapq.heappush(queue,(cand,ni,nj))
    if not np.isfinite(best[gi,gj]): raise RuntimeError("경로를 찾지 못했습니다. 출발지·도착지가 데이터 범위 안인지 확인하세요.")
    path=[]; cur=(gi,gj)
    while cur!=(si,sj):
        path.append((float(lats[cur[0]]),float(lons[cur[1]])))
        cur=parent[cur]
    path.append((float(lats[si]),float(lons[sj])))
    return path[::-1]


def path_metrics(path, u, v, lats, lons):
    # 각 구간 중간 지점의 가장 가까운 ERA5 격자 바람을 이용한 간이 계산
    total_d=total_t=0.0
    for a,b in zip(path[:-1],path[1:]):
        d,brg=distance_bearing(a,b)
        mid=((a[0]+b[0])/2,(a[1]+b[1])/2); i,j=nearest_grid_index(lats,lons,mid)
        uu,vv=u[i,j],v[i,j]
        along=uu*np.sin(brg)+vv*np.cos(brg); cross=uu*np.cos(brg)-vv*np.sin(brg)
        if abs(cross)>=TAS_MPS: return d/1000, np.inf
        gs=np.sqrt(TAS_MPS**2-cross**2)+along
        if gs<=MIN_GROUND_SPEED_MPS: return d/1000,np.inf
        total_d+=d; total_t+=d/gs
    return total_d/1000,total_t/3600


def main():
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        raise ImportError("CMD에서 py -m pip install plotly 를 실행하세요.") from e

    print("=== 사용자 지정 항로 ===")
    start=ask_point("출발지",DEFAULT_START); goal=ask_point("도착지",DEFAULT_GOAL)
    ds=xr.open_dataset(DATA_FILE)
    if TIME_INDEX>=ds.sizes['valid_time']: raise IndexError("TIME_INDEX가 데이터 시간 범위를 벗어났습니다.")
    selected_time=pd.to_datetime(ds.valid_time.values[TIME_INDEX],utc=True)
    # 경도 좌표 체계(0~360 또는 -180~180)에 맞춤
    lon_min_data=float(ds.longitude.min()); lon_max_data=float(ds.longitude.max())
    def fix_lon(lon): return lon % 360 if lon_min_data>=0 and lon_max_data>180 else ((lon+180)%360)-180
    start=(start[0],fix_lon(start[1])); goal=(goal[0],fix_lon(goal[1]))
    lat_min=min(start[0],goal[0])-CORRIDOR_MARGIN_DEG; lat_max=max(start[0],goal[0])+CORRIDOR_MARGIN_DEG
    # 경도 범위는 데이터 좌표 체계에서 단순 구간으로 선택
    lon_min=min(start[1],goal[1])-CORRIDOR_MARGIN_DEG; lon_max=max(start[1],goal[1])+CORRIDOR_MARGIN_DEG
    lat_min=max(lat_min,float(ds.latitude.min())); lat_max=min(lat_max,float(ds.latitude.max()))
    lon_min=max(lon_min,lon_min_data); lon_max=min(lon_max,lon_max_data)
    region=ds.sel(latitude=slice(lat_max,lat_min),longitude=slice(lon_min,lon_max))
    lats=region.latitude.values; lons=region.longitude.values
    if len(lats)<2 or len(lons)<2: raise ValueError("선택한 좌표 주변의 ERA5 격자가 부족합니다. 데이터 범위를 확인하세요.")
    for label,pt in [("출발지",start),("도착지",goal)]:
        if not (float(ds.latitude.min())<=pt[0]<=float(ds.latitude.max()) and lon_min_data<=pt[1]<=lon_max_data):
            raise ValueError(f"{label}가 ERA5 데이터 범위 밖입니다: {pt}")
    direct=great_circle_points(start,goal)
    fig=go.Figure(); visibility_groups=[]; summary=[]
    palette={300:"#e45756",250:"#2878b5",200:"#2a9d70"}
    for level in PRESSURE_LEVELS:
        if level not in ds.pressure_level.values: continue
        u=region.u.isel(valid_time=TIME_INDEX).sel(pressure_level=level).values
        v=region.v.isel(valid_time=TIME_INDEX).sel(pressure_level=level).values
        if np.isnan(u).any() or np.isnan(v).any(): raise ValueError(f"{level} hPa에 결측치가 있습니다.")
        speed=np.sqrt(u**2+v**2)
        # 속도 색상 점
        sl=(slice(None,None,WIND_STRIDE),slice(None,None,WIND_STRIDE))
        fig.add_trace(go.Scattergeo(lon=np.tile(lons[::WIND_STRIDE],len(lats[::WIND_STRIDE])),
            lat=np.repeat(lats[::WIND_STRIDE],len(lons[::WIND_STRIDE])),mode='markers',
            marker=dict(size=6,color=speed[sl].ravel(),colorscale='Turbo',cmin=0,cmax=max(40,float(np.nanpercentile(speed,98))),
                        colorbar=dict(title='바람 속도<br>m/s') if level==PRESSURE_LEVELS[0] else None,
                        showscale=True,opacity=0.85),
            name=f'{level} hPa 바람 속도',visible=(level==PRESSURE_LEVELS[0]),
            customdata=np.column_stack([u[sl].ravel(),v[sl].ravel()]),
            hovertemplate='위도 %{lat:.2f}°<br>경도 %{lon:.2f}°<br>풍속 %{marker.color:.1f} m/s<br>U %{customdata[0]:.1f}, V %{customdata[1]:.1f} m/s<extra></extra>'))
        # 방향 화살표: 각 격자에서 동서/남북 성분을 짧은 선분으로 표시
        arrow_lon=[]; arrow_lat=[]
        for ii in range(0,len(lats),WIND_STRIDE):
            for jj in range(0,len(lons),WIND_STRIDE):
                uu,vv=u[ii,jj],v[ii,jj]; lat=float(lats[ii]); lon=float(lons[jj])
                scale=0.10  # 시각화용 길이 배율(도 단위)
                dlat=vv*scale/20.0
                dlon=uu*scale/(20.0*max(0.25,np.cos(np.radians(lat))))
                arrow_lon.extend([lon-dlon/2,lon+dlon/2,None]); arrow_lat.extend([lat-dlat/2,lat+dlat/2,None])
        fig.add_trace(go.Scattergeo(lon=arrow_lon,lat=arrow_lat,mode='lines',line=dict(color='#263238',width=1),
            name=f'{level} hPa 바람 방향',visible=(level==PRESSURE_LEVELS[0]),hoverinfo='skip'))
        route=dijkstra_time_route(lats,lons,u,v,start,goal)
        dist,time=path_metrics(route,u,v,lats,lons); summary.append((level,dist,time))
        fig.add_trace(go.Scattergeo(lon=[p[1] for p in direct],lat=[p[0] for p in direct],mode='lines',
            line=dict(color='#444',width=3,dash='dash'),name='기준 대권항로',visible=(level==PRESSURE_LEVELS[0])))
        fig.add_trace(go.Scattergeo(lon=[p[1] for p in route],lat=[p[0] for p in route],mode='lines',
            line=dict(color=palette.get(level,'#9467bd'),width=4),name=f'{level} hPa 최적항로',visible=(level==PRESSURE_LEVELS[0]),
            hovertemplate=f'{level} hPa 최적항로<extra></extra>'))
        visibility_groups.append((level,len(fig.data)-4,len(fig.data)))
        print(f"{level} hPa 완료: 거리 {dist:.1f} km / 예상 시간 {time:.2f} h / 경로점 {len(route)}개")

    # 출발·도착 표식은 모든 고도에서 공통 표시
    fig.add_trace(go.Scattergeo(lon=[start[1],goal[1]],lat=[start[0],goal[0]],mode='markers+text',
        marker=dict(size=11,color=['#00a58a','#e45756'],line=dict(color='white',width=1)),
        text=['출발','도착'],textposition='top right',name='출발·도착',visible=True,
        hovertemplate='%{text}<br>%{lat:.3f}, %{lon:.3f}<extra></extra>'))
    buttons=[]
    for level,first,last in visibility_groups:
        vis=[False]*len(fig.data)
        for k in range(first,last): vis[k]=True
        vis[-1]=True
        buttons.append(dict(label=f'{level} hPa',method='update',args=[{'visible':vis},{'title':f'ERA5 바람장과 항로 비교 — {level} hPa'}]))
    fig.update_layout(title=f'ERA5 바람장과 항로 비교 — {PRESSURE_LEVELS[0]} hPa',
        geo=dict(scope='asia',projection_type='mercator',showland=True,landcolor='#f1eee6',showocean=True,
                 oceancolor='#dcecf5',showcoastlines=True,showcountries=True,center=dict(lat=(start[0]+goal[0])/2,lon=(start[1]+goal[1])/2),
                 lataxis=dict(range=[lat_min,lat_max]),lonaxis=dict(range=[lon_min,lon_max])),
        updatemenus=[dict(type='buttons',direction='right',x=0.02,y=1.12,buttons=buttons)],
        height=780,margin=dict(l=20,r=20,t=110,b=60),legend=dict(orientation='h',y=-0.08),
        annotations=[dict(text=f'분석 시각: {selected_time:%Y-%m-%d %H:%M UTC} | 바람 점 색: 풍속(m/s) | 화살표: 바람 방향 | 교육용 단순 모델',
                           x=0.5,y=-0.13,xref='paper',yref='paper',showarrow=False)])
    out=Path(__file__).with_name('interactive_wind_route.html')
    fig.write_html(out,include_plotlyjs=True,full_html=True)
    pd.DataFrame(summary,columns=['기압면(hPa)','최적항로 거리(km)','예상 시간(h)']).to_csv(Path(__file__).with_name('interactive_route_summary.csv'),index=False,encoding='utf-8-sig')
    print('\n지도 저장:',out)
    print('고도별 요약 저장: interactive_route_summary.csv')
    print('HTML 파일을 브라우저로 열고 위쪽 300/250/200 hPa 버튼으로 고도를 바꿔 보세요.')
    print('주의: 한 시각의 바람을 비행 내내 고정한 교육용 모델이며 실제 항공 운항에 사용할 수 없습니다.')
    ds.close()

if __name__=='__main__':
    main()
