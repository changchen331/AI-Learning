"""
数据下载脚本：从巨潮资讯网批量下载上市公司年报

巨潮资讯网（cninfo.com.cn）是中国证监会指定的上市公司信息披露平台
所有年报均为公开信息，下载合法合规
"""

import time
import json
import random
import logging
import requests
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parent.parent / "data" / "raw_pdf"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# 股票代码、所属板块、公司简称
TARGET_STOCKS = [
    ("600519", "sh", "贵州茅台"),   # 上交所主板
    ("000858", "sz", "五粮液"),     # 深交所主板
    ("601318", "sh", "中国平安"),   # 上交所主板
    ("300750", "sz", "宁德时代"),   # 创业板
    ("002415", "sz", "海康威视"),   # 深交所中小板
]

TARGET_YEARS = ["2021", "2022", "2023"]

CNINFO_QUERY_URL = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
CNINFO_BASE_URL  = "http://static.cninfo.com.cn/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "http://www.cninfo.com.cn/",
    "Content-Type": "application/x-www-form-urlencoded",
}


def _do_query(payload: dict) -> list[dict]:
    try:
        resp = requests.post(CNINFO_QUERY_URL, data=payload, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        return resp.json().get("announcements") or []
    except Exception as e:
        logger.debug(f"请求失败: {e}")
        return []


def query_annual_reports(stock_code: str, plate: str, company_name: str, year: str) -> list[dict]:
    """
    搜索年报。
    策略1：用公司名+年份作关键词（适合茅台类标题包含公司名的情况）
    策略2：用纯年份关键词+从结果里按 secCode 过滤（适合五粮液类标题不含公司名的情况）
    """
    pub_year  = str(int(year) + 1)
    plate_col = "sse" if plate == "sh" else "szse"

    base_payload = {
        "stock":     "",
        "tabName":   "fulltext",
        "pageSize":  30,
        "pageNum":   1,
        "column":    plate_col,
        "category":  "category_ndbg_szsh",
        "plate":     plate,
        "seDate":    f"{pub_year}-01-01~{pub_year}-06-30",
        "secid":     "",
        "sortName":  "",
        "sortType":  "",
        "isHLtitle": True,
    }

    # 策略1：公司名+年份关键词
    p1 = {**base_payload, "searchkey": f"{company_name}{year}年年度报告"}
    results = _do_query(p1)
    if results:
        return results

    # 策略2：公司名每字之间加空格（巨潮部分深交所股票 secName 格式为"五 粮 液"）
    logger.debug(f"策略1无结果，尝试策略2（字间加空格）")
    spaced_name = " ".join(list(company_name))   # "五粮液" → "五 粮 液"
    p2 = {**base_payload, "searchkey": spaced_name}
    results2 = _do_query(p2)
    # 只取全报（非摘要/英文），不含特定年份关键词时按年份过滤 announcementTitle
    filtered2 = [
        r for r in results2
        if "年度报告" in r.get("announcementTitle", "")
        and "摘要" not in r.get("announcementTitle", "")
        and "英文" not in r.get("announcementTitle", "")
        and year in r.get("announcementTitle", "")
    ]
    if filtered2:
        return filtered2

    return []


def download_pdf(pdf_url: str, save_path: Path) -> bool:
    if save_path.exists():
        logger.info(f"已存在，跳过: {save_path.name}")
        return True

    for attempt in range(3):
        try:
            resp = requests.get(pdf_url, headers=HEADERS, timeout=120, stream=True)
            resp.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
            size_kb = save_path.stat().st_size // 1024
            logger.info(f"下载成功: {save_path.name}  ({size_kb} KB)")
            return True
        except Exception as e:
            logger.warning(f"第{attempt+1}次失败: {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)

    logger.error(f"下载失败: {pdf_url}")
    return False


def sanitize(name: str) -> str:
    for ch in r'\/:*?"<>|':
        name = name.replace(ch, "_")
    return name.strip()


def main():
    manifest = []

    for stock_code, plate, company_name in TARGET_STOCKS:
        for year in TARGET_YEARS:
            logger.info(f"── {company_name}({stock_code}) {year}年报 ──")
            reports = query_annual_reports(stock_code, plate, company_name, year)

            # 过滤：标题含"年度报告"、不含"摘要"/"英文"，是 PDF
            candidates = [
                r for r in reports
                if "年度报告" in r.get("announcementTitle", "")
                and "摘要" not in r.get("announcementTitle", "")
                and "英文" not in r.get("announcementTitle", "")
                and r.get("adjunctUrl", "").upper().endswith(".PDF")
            ]

            if not candidates:
                logger.warning(f"  未找到匹配年报，跳过")
                # 打印原始结果帮助调试
                for r in reports[:3]:
                    logger.debug(f"    候选: {r.get('announcementTitle')} | {r.get('adjunctUrl','')[:50]}")
                continue

            report   = candidates[0]
            title    = report["announcementTitle"]
            pdf_url  = CNINFO_BASE_URL + report["adjunctUrl"]
            filename = sanitize(f"{stock_code}_{year}_{company_name}_{title}.pdf")
            save_path = RAW_DIR / filename

            success = download_pdf(pdf_url, save_path)
            if success:
                manifest.append({
                    "stock_code":   stock_code,
                    "plate":        plate,
                    "company_name": company_name,
                    "year":         year,
                    "title":        title,
                    "filename":     filename,
                    "source_url":   pdf_url,
                    "announce_id":  report.get("announcementId"),
                })

            time.sleep(random.uniform(1.5, 3.0))

    manifest_path = RAW_DIR.parent / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logger.info(f"\n完成！共下载 {len(manifest)} 份年报")
    for item in manifest:
        logger.info(f"  {item['company_name']} {item['year']}: {item['filename']}")


if __name__ == "__main__":
    main()
