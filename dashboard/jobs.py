"""Bounded background inference jobs with real, pollable states."""
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import time
from uuid import uuid4

_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix='inference')
_LOCK = Lock()
_JOBS = {}
_ACTIVE = {}
JOB_TTL = 3600
MAX_PENDING = 32

def _run(job_id, ticker, analyze):
    start_mono = time.monotonic()
    with _LOCK:
        job = _JOBS.get(job_id)
        if job:
            job.update(status='processing',
                       started_at=time.time(),
                       stage='Đang chuẩn bị dữ liệu và chạy mô hình đã huấn luyện.')
            q_time = start_mono - job.get('created_monotonic', start_mono)
        else:
            q_time = 0.0
            
    try:
        result = analyze(ticker)
    except Exception as e:
        import logging
        logging.exception(f"Job {job_id} error for {ticker}")
        result = {'status': 'error', 'message': 'Không thể hoàn tất phân tích. Vui lòng thử lại.'}
        
    end_mono = time.monotonic()
    proc_time = end_mono - start_mono
    
    with _LOCK:
        if job_id in _JOBS:
            timings = {
                'queue_seconds': round(q_time, 4),
                'processing_seconds': round(proc_time, 4),
                'total_seconds': round(q_time + proc_time, 4),
            }
            if isinstance(result, dict):
                result['timings'] = timings
            _JOBS[job_id].update(result=result, status='completed', finished_at=time.monotonic(), timings=timings)
        _ACTIVE.pop(ticker, None)


def analysis_job(ticker=None, job_id=None):
    """Repeated requests for an active ticker share one job."""
    from dashboard.inference import analyze_ticker_on_demand
    with _LOCK:
        now = time.monotonic()
        for old_id in list(_JOBS):
            job = _JOBS[old_id]
            if job['status'] == 'completed' and now - job['finished_at'] > JOB_TTL:
                del _JOBS[old_id]
        if job_id is not None:
            job = _JOBS.get(job_id)
            if job is None:
                return {'status': 'error', 'message': 'Lượt phân tích đã hết hạn hoặc không tồn tại.'}, 404
        else:
            if ticker in _ACTIVE:
                job_id = _ACTIVE[ticker]
            else:
                if len(_ACTIVE) >= MAX_PENDING:
                    return {'status': 'error', 'message': 'Hệ thống đang xử lý nhiều yêu cầu. Vui lòng thử lại.'}, 503
                job_id = uuid4().hex
                _JOBS[job_id] = {'status': 'queued', 'ticker': ticker, 'created_at': now,
                                 'created_monotonic': time.monotonic(),
                                 'stage': 'Yêu cầu đang chờ xử lý.'}
                _ACTIVE[ticker] = job_id
                _EXECUTOR.submit(_run, job_id, ticker, analyze_ticker_on_demand)
            job = _JOBS[job_id]
            
        now_mono = time.monotonic()
        elapsed = round(now_mono - job.get('created_monotonic', now_mono), 2)
        if job['status'] == 'completed':
            res = dict(job['result'])
            res['job_id'] = job_id
            if 'timings' in job:
                res.setdefault('timings', job['timings'])
            return res, 200
            
        return {'status': job['status'], 'job_id': job_id, 'ticker': job['ticker'],
                'message': job['stage'], 'stage': job['stage'], 'elapsed': elapsed}, 202
