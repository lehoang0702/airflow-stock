"""Collect actual RSS observations locally without publishing or backdating.

Writes immutable CSV/JSON snapshots for future point-in-time research. It does
not manufacture sentiment values or start any scheduler.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from uuid import uuid4
import xml.etree.ElementTree as ET

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'dags'))
from config_shared import SECTOR_MAP, TICKERS_META
from news_data import NEWS_COLUMNS, NEWS_POLICY, normalize_batch, select_eligible_news


def read_feed(url):
    request = Request(url, headers={'User-Agent': 'Mozilla/5.0 (compatible; StockResearch/1.0)'})
    with urlopen(request, timeout=15) as response:
        payload = response.read(2_000_001)
    if len(payload) > 2_000_000:
        raise ValueError('RSS response exceeds size limit')
    root = ET.fromstring(payload)
    return [{'title': item.findtext('title'), 'summary': item.findtext('description'),
             'publisher': item.findtext('source') or 'RSS',
             'pub_date': item.findtext('pubDate'), 'link': item.findtext('link')}
            for item in root.findall('.//item')[:12]]


def collect_ticker(ticker, feed_reader=read_feed):
    company = TICKERS_META[ticker]['name']
    query = urlencode({'q': f'"{company}" stock earnings', 'hl': 'en-US',
                       'gl': 'US', 'ceid': 'US:en'})
    url = 'https://news.google.com/rss/search?' + query
    try:
        articles = feed_reader(url)
        # Actual collection completion; publication is never used as first_seen.
        stamp = datetime.now(timezone.utc)
        frame = normalize_batch(articles, ticker, SECTOR_MAP[ticker], stamp)
        return frame, None
    except Exception as exc:
        return pd.DataFrame(columns=NEWS_COLUMNS), str(exc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tickers', nargs='+', choices=sorted(SECTOR_MAP), default=sorted(SECTOR_MAP))
    args = parser.parse_args()
    frames, failures = [], {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for ticker, (frame, failure) in zip(args.tickers, pool.map(collect_ticker, args.tickers)):
            frames.append(frame)
            if failure:
                failures[ticker] = failure
    stamp = datetime.now(timezone.utc)
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=NEWS_COLUMNS)
    eligible = select_eligible_news(data, stamp)
    report = {'news_policy': NEWS_POLICY, 'collected_at': stamp.isoformat(),
              'rows': len(data), 'eligible_rows': len(eligible),
              'eligible_tickers': sorted(set(eligible.get('ma_co_phieu', []))),
              'missing_tickers': sorted(set(args.tickers) - set(eligible.get('ma_co_phieu', []))),
              'source_failures': failures, 'sentiment_scored_rows': 0,
              'interpretation': 'Local real observations only. No historic backdating or invented sentiment.'}
    args.output.mkdir(parents=True, exist_ok=True)
    name = stamp.strftime('%Y%m%dT%H%M%S%f') + '_' + uuid4().hex
    # Exclusive creation protects observations from being overwritten.
    with (args.output / f'{name}.csv').open('x') as stream:
        data.to_csv(stream, index=False)
    with (args.output / f'{name}.json').open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
