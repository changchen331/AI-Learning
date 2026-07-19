"""
weather_tools.py — 作业：把原 get_weather 拆成两个独立工具

教学重点：
  原始 src/weather_backend.get_weather 内部串了两次 HTTP 请求：
    1) Geocoding：城市名 → 经纬度
    2) Forecast：经纬度 → 天气
  本文件把它们拆成两个对 LLM 暴露的独立工具：
    - geocode(city)            ：只做"城市名 → 经纬度"，能独立回答"北京的经纬度"
    - get_weather_by_coords(lat, lon)：只做"经纬度 → 天气"，用户给经纬度即可直接答
  两个工具互不依赖，但模型可以把它们链起来：geocode → 拿经纬度 → get_weather_by_coords，
  这正是 agent loop 的价值——模型自己决定调几次、调哪个，宿主只负责按调用循环执行。

依赖：pip install httpx   （Open-Meteo 免费、无需 key）
"""

import httpx

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_CODE_MAP = {
    0: "晴天", 1: "大致晴朗", 2: "局部多云", 3: "阴天",
    45: "雾", 48: "冻雾",
    51: "小毛毛雨", 53: "中毛毛雨", 55: "大毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    71: "小雪", 73: "中雪", 75: "大雪",
    80: "小阵雨", 81: "中阵雨", 82: "大阵雨",
    95: "雷暴", 96: "雷暴伴小冰雹", 99: "雷暴伴大冰雹",
}


def geocode(city: str) -> str:
    """
    工具一：城市名 → 经纬度（Geocoding 接口）。

    返回文字描述，里面明确包含 latitude / longitude，方便模型把这两个数
    传给下一个工具 get_weather_by_coords 完成链式调用，也能独立回答"X 的经纬度"。
    """
    with httpx.Client(timeout=10.0) as client:
        resp = client.get(GEOCODING_URL, params={
            "name": city, "count": 10, "language": "zh", "format": "json",
        })
        resp.raise_for_status()
        results = resp.json().get("results") or []

        # 与原 backend 同样的同名小村庄消歧策略：裸低级行政点且没带"市/县/区"后缀，
        # 就用 city+"市" 重查一次并优先采用。
        def _geocode(name: str):
            r = client.get(GEOCODING_URL, params={
                "name": name, "count": 10, "language": "zh", "format": "json",
            })
            r.raise_for_status()
            return r.json().get("results") or []

        is_low_admin = all(
            str(r.get("feature_code", "")).startswith("PPL")
            and not str(r.get("feature_code", "")).startswith("PPLA")
            for r in results
        ) if results else True
        has_suffix = any(city.endswith(s) for s in ("市", "县", "区", "镇"))
        if is_low_admin and not has_suffix:
            retry = _geocode(city + "市")
            if retry:
                results = retry

        if not results:
            return f"未找到城市 '{city}'，请尝试其他写法（如'宁德市'改'宁德'）"

        def _rank(r):
            fc = str(r.get("feature_code", ""))
            admin_priority = 1 if fc.startswith("PPLA") or fc.startswith("ADM") else 0
            return admin_priority, r.get("population") or 0

        loc = max(results, key=_rank)
        lat = loc["latitude"]
        lon = loc["longitude"]
        location_str = f"{loc.get('country', '')} {loc.get('admin1', '')} {loc.get('name', city)}".strip()
        return (
            f"城市：{location_str}\n"
            f"纬度(latitude)：{lat}\n"
            f"经度(longitude)：{lon}"
        )


def get_weather_by_coords(latitude: float, longitude: float) -> str:
    """
    工具二：经纬度 → 天气（Forecast 接口）。

    只要拿到经纬度就能直接查，不需要城市名。所以用户直接给经纬度也能答，
    模型链式调用时把 geocode 的输出喂进来即可。
    """
    with httpx.Client(timeout=10.0) as client:
        try:
            resp = client.get(WEATHER_URL, params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
                "timezone": "Asia/Shanghai",
                "forecast_days": 3,
            })
            resp.raise_for_status()
        except httpx.RequestError as e:
            return f"天气数据获取失败：{e}"

        data = resp.json()
        cur = data["current"]
        daily = data["daily"]
        weather_desc = WEATHER_CODE_MAP.get(cur["weather_code"], f"代码{cur['weather_code']}")

        lines = [
            f"坐标：{latitude}°N, {longitude}°E",
            "",
            f"当前天气：{weather_desc}",
            f"  温度：{cur['temperature_2m']}°C",
            f"  相对湿度：{cur['relative_humidity_2m']}%",
            f"  风速：{cur['wind_speed_10m']} km/h",
            "",
            "未来3天预报：",
        ]
        for i in range(3):
            day_desc = WEATHER_CODE_MAP.get(daily["weather_code"][i], "")
            lines.append(
                f"  {daily['time'][i]}：{day_desc}，"
                f"{daily['temperature_2m_max'][i]}°C / {daily['temperature_2m_min'][i]}°C，"
                f"降水 {daily['precipitation_sum'][i]} mm"
            )
        return "\n".join(lines)


if __name__ == "__main__":
    # 自测：geocode → 拿经纬度 → get_weather_by_coords，手动演示一遍链式调用
    info = geocode("宁德")
    print(info)
    # 从文本里把经纬度抠出来继续查天气（仅自测用，模型链式调用时自己解析）
    import re

    m_lat = re.search(r"纬度.*?：(-?\d+\.?\d*)", info)
    m_lon = re.search(r"经度.*?：(-?\d+\.?\d*)", info)
    if m_lat and m_lon:
        print("\n--- 链式调用：拿上面经纬度查天气 ---")
        print(get_weather_by_coords(float(m_lat.group(1)), float(m_lon.group(1))))
