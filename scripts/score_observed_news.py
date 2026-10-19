"""Archive actual FinBERT headline probabilities with their scoring time."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from uuid import uuid4

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'dags'))
from news_data import NEWS_POLICY, SENTIMENT_COLUMNS, select_eligible_news


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    frame = select_eligible_news(pd.read_csv(args.input), datetime.now(timezone.utc))
    scored = []
    model_version = None
    if not frame.empty:
        import numpy as np
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        torch.set_num_threads(2)
        options = {'cache_dir': str(args.cache_dir)} if args.cache_dir else {}
        tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert', **options)
        model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert', **options).eval()
        labels = {str(label).lower(): int(index) for index, label in model.config.id2label.items()}
        if not set(('positive', 'negative', 'neutral')).issubset(labels):
            raise ValueError('FinBERT does not expose the required sentiment labels')
        revision = getattr(model.config, '_commit_hash', None)
        if not revision:
            raise ValueError('The loaded sentiment model revision is unavailable')
        model_version = 'ProsusAI/finbert@' + revision
        for _, row in frame.iterrows():
            text = str(row.tieu_de)
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
            with torch.no_grad():
                p = torch.softmax(model(**inputs).logits, dim=-1)[0].numpy()
            if not np.isfinite(p).all() or not np.isclose(p.sum(), 1, atol=1e-5):
                raise ValueError('Invalid sentiment probabilities')
            scored.append({**row.to_dict(),
                **{f'sentiment_{label}': float(p[index]) for label, index in labels.items()},
                'sentiment_scored_at': datetime.now(timezone.utc).isoformat(),
                'sentiment_model_version': model_version,
                'sentiment_text_field': 'tieu_de',
                'sentiment_text_sha256': hashlib.sha256(text.encode()).hexdigest(),
                'news_source_sha256': hashlib.sha256(raw).hexdigest()})
    stamp = datetime.now(timezone.utc)
    result = pd.DataFrame(scored, columns=list(frame.columns) + SENTIMENT_COLUMNS + [
        'sentiment_scored_at', 'sentiment_model_version', 'sentiment_text_field',
        'sentiment_text_sha256', 'news_source_sha256'])
    eligible = select_eligible_news(result, stamp, require_sentiment=True)
    if len(eligible) != len(result):
        raise ValueError('Scored archive violates point-in-time eligibility')
    report = {'news_policy': NEWS_POLICY, 'scored_at': stamp.isoformat(),
              'input': str(args.input), 'input_sha256': hashlib.sha256(raw).hexdigest(),
              'rows': len(result), 'model_version': model_version,
              'text_field': 'headline only',
              'interpretation': 'Actual sentiment outputs, not probabilities of stock-price increases.'}
    args.output.mkdir(parents=True, exist_ok=True)
    name = stamp.strftime('%Y%m%dT%H%M%S%f') + '_' + uuid4().hex
    with (args.output / f'{name}.csv').open('x') as stream:
        result.to_csv(stream, index=False)
    with (args.output / f'{name}.json').open('x') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
