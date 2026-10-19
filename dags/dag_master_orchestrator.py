"""
DAG: MASTER ORCHESTRATOR — ĐIỀU PHỐI TOÀN BỘ PIPELINE ĐA PHƯƠNG THỨC (1-CLICK RUN)
==================================================================================
Kích hoạt và kiểm soát toàn bộ 8 quy trình từ Airflow Web UI:
  Phase 1→2: Cào dữ liệu & Huấn luyện TUẦN TỰ từng nhánh (XGBoost → LSTM → FinBERT)
  Phase 3: Hợp nhất đa phương thức (Master Ensemble 4 phương pháp)
  Phase 4: Đánh giá khoa học toàn diện (Model Evaluation Pipeline)

CHẾ ĐỘ TIẾT KIỆM TÀI NGUYÊN: Chạy tuần tự từng nhánh một để giảm thiểu
peak CPU/RAM usage. Mỗi thời điểm chỉ có 1 model đang train/predict.
"""

from datetime import datetime, timedelta
import logging

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

try:
    from alert_utils import telegram_failure_callback, send_pipeline_event
except ImportError:
    telegram_failure_callback = None
    def send_pipeline_event(title, message, icon="ℹ️"): pass

try:
    from config_shared import DEFAULT_CRON_SCHEDULE
except ImportError:
    DEFAULT_CRON_SCHEDULE = '0 22 * * 1-5'

default_args = {
    'owner': 'quant_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 0,
    'on_failure_callback': telegram_failure_callback,
    'execution_timeout': timedelta(hours=3),
}


def notify_pipeline_start(**context):
    """Thông báo bắt đầu chạy toàn bộ hệ thống qua Telegram."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = (
        "Hệ thống đã nhận lệnh kích hoạt <b>Master Pipeline</b>.\n"
        "Bắt đầu thực thi TUẦN TỰ từng nhánh (XGBoost → LSTM → FinBERT) để tiết kiệm tài nguyên."
    )
    send_pipeline_event("BẮT ĐẦU CHẠY MASTER PIPELINE", msg, icon="🚀")
    logging.info("🚀 Đã gửi thông báo khởi chạy Master Pipeline về Telegram.")



with DAG(
    dag_id='dag_master_orchestrator',
    default_args=default_args,
    schedule_interval=DEFAULT_CRON_SCHEDULE,
    catchup=False,
    max_active_runs=1,
    tags=['master', 'orchestrator', '1_click', 'pipeline', 'all_models', 'scheduled'],
) as dag:

    dag.doc_md = """
# 👑 DAG: Master Orchestrator — Điều Phối Toàn Bộ Hệ Thống (1-Click Run)
---
### 1. Tổng Quan & Trải Nghiệm Web UI
DAG này là **Trung tâm Điều phối Cấp cao (Orchestrator)** cho phép người dùng hoặc giảng viên chỉ cần bấm **▶ DUY NHẤT 1 NÚT** trên Airflow Web UI để chạy toàn bộ hệ thống từ đầu đến cuối một cách tự động, chuẩn xác và mượt mà.

⚡ **CHẾ ĐỘ TIẾT KIỆM TÀI NGUYÊN:** Pipeline chạy TUẦN TỰ từng nhánh một
(XGBoost → LSTM → FinBERT) để giảm thiểu peak CPU/RAM. Mỗi thời điểm chỉ có
1 model đang crawl/train/predict.

### 2. Sơ Đồ Kiến Trúc Luồng Thực Thi (Tuần Tự)
```mermaid
graph TD
    START([▶ Bấm 1-Click trên Web UI]) --> N_START[🚀 Gửi thông báo bắt đầu qua Telegram]
    
    subgraph Nhánh 1: XGBoost
        N_START --> P1_XGB[Cào dữ liệu & 85+ Features XGBoost]
        P1_XGB --> P2_XGB[Huấn luyện GBDT & Dự báo XGBoost]
    end
    
    subgraph Nhánh 2: LSTM
        P2_XGB --> P1_LSTM[Cào dữ liệu & Chuỗi nến LSTM]
        P1_LSTM --> P2_LSTM[Huấn luyện PyTorch LSTM & Dự báo]
    end
    
    subgraph Nhánh 3: FinBERT
        P2_LSTM --> P1_FIN[Cào tin tức tài chính FinBERT]
        P1_FIN --> P2_FIN[Chạy Transformer NLP Dự báo FinBERT]
    end
    
    subgraph Giai đoạn cuối: Hợp nhất & Đánh giá
        P2_FIN --> P3_ENS[👑 Master Ensemble 4 Phương Pháp]
        P3_ENS --> P4_EVAL[📊 Đánh Giá Khoa Học Toàn Diện]
    end
```

### 3. Các Giai Đoạn Chi Tiết (Tuần Tự)
1. **Nhánh XGBoost:** Cào dữ liệu → Huấn luyện & Dự báo XGBoost
2. **Nhánh LSTM:** Cào dữ liệu → Huấn luyện & Dự báo LSTM
3. **Nhánh FinBERT:** Cào tin tức → Phân tích sentiment FinBERT
4. **Master Ensemble:** Tổng hợp 4 phương pháp và tính Trading Plan
5. **Model Evaluation:** Kiểm định TimeSeriesSplit CV, Sentiment correlation và xuất báo cáo CSV

### 4. Cơ Chế Giám Sát & Báo Động
- **Cảnh báo lỗi tức thời:** Bất kỳ task nào trong chuỗi gặp sự cố, hệ thống `on_failure_callback` sẽ gửi thông báo khẩn cấp kèm chi tiết lỗi và đường link xem log về Telegram.

### 5. Lịch Vận Hành Tự Động (Cron Schedule)
- **Chu kỳ:** `0 22 * * 1-5` (22:00 UTC Thứ 2 – Thứ 6, tức 05:00 sáng Thứ 3 – Thứ 7 giờ VN).
- **Mục đích:** Tự động kích hoạt sau khi phiên giao dịch chứng khoán Mỹ (NYSE/NASDAQ) đóng cửa và dữ liệu nến ngày hoàn tất chốt sổ trên các API tài chính. Đến sáng sớm 06:00 - 07:00, Bot Telegram đã gửi đầy đủ tín hiệu và kế hoạch giao dịch cho ngày mới.
"""

    notify_start = PythonOperator(
        task_id='notify_pipeline_start',
        python_callable=notify_pipeline_start,
        provide_context=True,
    )

    # --- PHASE 1: CÀO DỮ LIỆU SONG SONG ---
    phase1_crawl_xgboost = TriggerDagRunOperator(
        task_id='phase1_crawl_xgboost',
        trigger_dag_id='dag_crawl_xgboost_stock_features',
        wait_for_completion=True,
        poke_interval=10,
        reset_dag_run=False,
        conf={'pipeline_run_id': '{{ run_id }}', 'as_of': '{{ ds_nodash }}'},
    )

    phase1_crawl_lstm = TriggerDagRunOperator(
        task_id='phase1_crawl_lstm',
        trigger_dag_id='dag_crawl_lstm_stock_features',
        wait_for_completion=True,
        poke_interval=10,
        reset_dag_run=False,
        trigger_rule='all_done',
        conf={'pipeline_run_id': '{{ run_id }}', 'as_of': '{{ ds_nodash }}'},
    )

    phase1_crawl_finbert = TriggerDagRunOperator(
        task_id='phase1_crawl_finbert',
        trigger_dag_id='dag_crawl_finbert_news',
        wait_for_completion=True,
        poke_interval=10,
        reset_dag_run=False,
        trigger_rule='all_done',
        conf={'pipeline_run_id': '{{ run_id }}', 'as_of': '{{ ds_nodash }}'},
    )

    # --- PHASE 2: HUẤN LUYỆN & DỰ BÁO SONG SONG ---
    phase2_train_xgboost = TriggerDagRunOperator(
        task_id='phase2_train_xgboost',
        trigger_dag_id='dag_xgboost_train_and_predict',
        wait_for_completion=True,
        poke_interval=15,
        reset_dag_run=False,
        conf={'pipeline_run_id': '{{ run_id }}', 'as_of': '{{ ds_nodash }}'},
    )

    phase2_train_lstm = TriggerDagRunOperator(
        task_id='phase2_train_lstm',
        trigger_dag_id='dag_lstm_train_and_predict',
        wait_for_completion=True,
        poke_interval=15,
        reset_dag_run=False,
        conf={'pipeline_run_id': '{{ run_id }}', 'as_of': '{{ ds_nodash }}'},
    )

    phase2_predict_finbert = TriggerDagRunOperator(
        task_id='phase2_predict_finbert',
        trigger_dag_id='dag_finbert_predict',
        wait_for_completion=True,
        poke_interval=15,
        reset_dag_run=False,
        conf={'pipeline_run_id': '{{ run_id }}', 'as_of': '{{ ds_nodash }}'},
    )

    # --- PHASE 3: MASTER ENSEMBLE ---
    phase3_ensemble = TriggerDagRunOperator(
        task_id='phase3_master_ensemble',
        trigger_dag_id='dag_ensemble_master',
        wait_for_completion=True,
        poke_interval=10,
        reset_dag_run=False,
        trigger_rule='all_done',
        conf={'pipeline_run_id': '{{ run_id }}', 'as_of': '{{ ds_nodash }}'},
    )

    # --- PHASE 4: ĐÁNH GIÁ MÔ HÌNH ---
    phase4_evaluation = TriggerDagRunOperator(
        task_id='phase4_model_evaluation',
        trigger_dag_id='dag_model_evaluation',
        wait_for_completion=True,
        poke_interval=10,
        reset_dag_run=False,
        trigger_rule='all_done',
        conf={'pipeline_run_id': '{{ run_id }}', 'as_of': '{{ ds_nodash }}'},
    )

    def _check_pipeline_status(**context):
        tis = context['dag_run'].get_task_instances()
        failed_tasks = [ti.task_id for ti in tis if ti.state == 'failed']
        if failed_tasks:
            raise ValueError(f"Các nhánh sau đã thất bại: {', '.join(failed_tasks)}. Pipeline chạy degraded.")

    check_status = PythonOperator(
        task_id='check_pipeline_status',
        python_callable=_check_pipeline_status,
        provide_context=True,
        trigger_rule='all_done',
    )

    # --- ĐỊNH NGHĨA LUỒNG THỰC THI TUẦN TỰ (TIẾT KIỆM TÀI NGUYÊN) ---
    # Chạy từng nhánh một: XGBoost → LSTM → FinBERT → Ensemble → Evaluation
    # Mỗi thời điểm chỉ có 1 task nặng đang chạy, giảm peak CPU/RAM.
    (
        notify_start
        >> phase1_crawl_xgboost >> phase2_train_xgboost    # Nhánh 1: XGBoost
        >> phase1_crawl_lstm >> phase2_train_lstm            # Nhánh 2: LSTM
        >> phase1_crawl_finbert >> phase2_predict_finbert    # Nhánh 3: FinBERT
        >> phase3_ensemble                                   # Hợp nhất
        >> phase4_evaluation                                 # Đánh giá
        >> check_status
    )
