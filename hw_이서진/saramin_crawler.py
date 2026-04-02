"""
사람인 IT 직군 신입•인턴 실시간 공고 동적 크롤링
- 대상: 사람인 홈페이지
- 상세 조건: 서울 전체, 경기 전체, 미국 전체, 일본 전체
- 수집 항목: 회사명, 공고명, 직무, 근무지, 경력, 학력, 고용형태, 공고url, 마감일
- 사용 도구: Playwright, BeautifulSoup
"""
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import time
import re

play = sync_playwright().start()
browser = play.chromium.launch(headless=False,args=["--start-maximized"]) # headless=True 차단
page = browser.new_page(no_viewport=True)

BASE_URL = "https://www.saramin.co.kr/zf_user/jobs/public/list?loc_mcd=101000%2C102000&loc_bcd=220200%2C211200&company_cd=0%2C1%2C2%2C3%2C4%2C5%2C6%2C7%2C9%2C10&cat_mcls=2&panel_type=domestic&search_optional_item=n&search_done=y&panel_count=y&preview=y"

# 상세 페이지 url 수집
url_list = []

for page_num in range(1,4):
    page.goto(f"{BASE_URL}&page={page_num}&isAjaxRequest=y")
    page.wait_for_load_state("networkidle")
    time.sleep(1)
 
    soup = BeautifulSoup(page.content(), "html.parser")
    items = [item for item in soup.select(".list_recruiting .list_item") if item.get("id")]
 
    for item in items:
        url = item.select_one(".col.notification_info .str_tit")["href"] if item.select_one(".col.notification_info .str_tit") else ""
        if url:
            url_list.append("https://www.saramin.co.kr" + url)

# 상세 페이지 정보 수집
data_jobs = []

for i, url in enumerate(url_list):
    page.goto(url)
    page.wait_for_load_state("networkidle")
    time.sleep(1)

    soup = BeautifulSoup(page.content(), "html.parser")

    company = soup.select_one(".jv_header .title_inner a")["title"] if soup.select_one(".jv_header .title_inner a") else ""
    title = soup.select_one(".tit_job").get_text(strip = True) if soup.select_one(".tit_job") else ""
    
    cols = soup.select(".jv_cont.jv_summary .cont .col")
    dls = cols[0].select("dl")
    career = dls[0].select_one("dd strong").get_text(strip=True) if dls[0].select_one("dd strong") else ""
    education = dls[1].select_one("dd strong").get_text(strip=True) if dls[1].select_one("dd strong") else ""
    condition = dls[2].select_one("dd strong").get_text(strip=True) if dls[2].select_one("dd strong") else ""
    location = cols[1].select_one("dl dd").get_text(strip=True) if cols[1].select_one("dl dd") else ""
    
    iframe = page.frame("iframe_content_0")
    iframe_soup = BeautifulSoup(iframe.content(), "html.parser") if iframe else None
    job_desc = {}
    for block in iframe_soup.select(".job-content .info-block"):
        title_tag = block.select_one(".info-block__title")
        list_tag  = block.select_one(".info-block__list")

        if not title_tag:
            continue
        if not list_tag:
            continue

        block_title = re.sub(r'[^\w]', '', title_tag.get_text(strip=True)).strip()

        if block_title == "채용절차":
            lines = []
            for li in list_tag.select("li"):
                value = li.select_one(".value")
                lines.append(value.get_text(strip=True)) if value else ""
            job_desc[block_title] = "\n".join(lines)

        else:
            job_desc[block_title] = list_tag.get_text(separator="\n", strip=True)


    data_jobs.append({
        "url":     url,
        "기업명":  company,
        "공고명":  title,
        "경력":    career,
        "학력":    education,
        "고용형태": condition,
        "근무지":  location,
        "주요업무": job_desc.get("주요업무", ""),
        "자격요건": job_desc.get("자격요건", ""),
        "채용절차": job_desc.get("채용절차", ""),
    })


# excel - 데이터 확인용
import pandas as pd

df = pd.DataFrame(data_jobs)
df.to_excel("saramin_jobs.xlsx", index=False)

# json - Chroma DB 임베딩 작업에 사용
import json

with open("saramin_jobs.json", "w", encoding="utf-8") as f:
    json.dump(data_jobs, f, ensure_ascii=False, indent=2)
