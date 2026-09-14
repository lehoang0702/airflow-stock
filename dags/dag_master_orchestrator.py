"""
DAG: MASTER ORCHESTRATOR — ĐIỀU PHỐI TOÀN BỘ PIPELINE ĐA PHƯƠNG THỨC (1-CLICK RUN)
==================================================================================
Kích hoạt và kiểm soát toàn bộ 8 quy trình từ Airflow Web UI:
  Phase 1: Cào dữ liệu song song 3 nhánh (XGBoost, LSTM, FinBERT)
  Phase 2: Huấn luyện & Dự báo song song 3 mô hình (XGBoost, LSTM, FinBERT)
  Phase 3: Hợp nhất đa phương thức (Master Ensemble 4 phương pháp)
  Phase 4: Đánh giá khoa học toàn diện (Model Evaluation Pipeline)
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
}


def notify_pipeline_start(**context):
    """Thông báo bắt đầu chạy toàn bộ hệ thống qua Telegram."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = (
        "Hệ thống đã nhận lệnh kích hoạt <b>Master Pipeline</b>.\n"
        "Bắt đầu thực thi đồng thời 3 nhánh cào dữ liệu (XGBoost, LSTM, FinBERT)."
    )
    send_pipeline_event("BẮT ĐẦU CHẠY MASTER PIPELINE", msg, icon="🚀")
    logging.info("🚀 Đã gửi thông báo khởi chạy Master Pipeline về Telegram.")



with DAG(
    dag_id='dag_master_orchestrator',
    default_args=default_args,
    schedule_interval=DEFAULT_CRON_SCHEDULE,
    catchup=False,
    tags=['master', 'orchestrator', '1_click', 'pipeline', 'all_models', 'scheduled'],
) as dag:

    dag.doc_md = """
# 👑 DAG: Master Orchestrator — Điều Phối Toàn Bộ Hệ Thống (1-Click Run)
---
### 1. Tổng Quan & Trải Nghiệm Web UI
DAG này là **Trung tâm Điều phối Cấp cao (Orchestrator)** cho phép người dùng hoặc giảng viên chỉ cần bấm **▶ DUY NHẤT 1 NÚT** trên Airflow Web UI để chạy toàn bộ hệ thống từ đầu đến cuối một cách tự động, chuẩn xác và mượt mà.

### 2. Sơ Đồ Kiến Trúc Luồng Thực Thi
```mermaid
graph TD
    START([▶ Bấm 1-Click trên Web UI]) --> N_START[🚀 Gửi thông báo bắt đầu qua Telegram]
    
    subgraph Giai đoạn 1: Cào dữ liệu song song 3 nhánh
        N_START --> P1_XGB[Cào dữ liệu & 85+ Features XGBoost]
        N_START --> P1_LSTM[Cào dữ liệu & Chuỗi nến LSTM]
        N_START --> P1_FIN[Cào tin tức tài chính FinBERT]
    end
    
    subgraph Giai đoạn 2: Huấn luyện & Dự báo độc lập
        P1_XGB --> P2_XGB[Huấn luyện GBDT & Dự báo XGBoost]
        P1_LSTM --> P2_LSTM[Huấn luyện PyTorch LSTM & Dự báo]
        P1_FIN --> P2_FIN[Chạy Transformer NLP Dự báo FinBERT]
    end
    
    subgraph Giai đoạn 3: Hợp nhất & Đánh giá khoa học
        P2_XGB --> P3_ENS[👑 Master Ensemble 4 Phương Pháp]
        P2_LSTM --> P3_ENS
        P2_FIN --> P3_ENS
        P3_ENS --> P4_EVAL[📊 Đánh Giá Khoa Học Toàn Diện]
    end
```

### 3. Các Giai Đoạn Chi Tiết
1. **Phase 1 (Data Ingestion & Feature Engineering):**
   - Kích hoạt song song 3 DAGs cào dữ liệu để chuẩn bị dữ liệu đầu vào.
2. **Phase 2 (Model Training & Inference):**
   - Mỗi nhánh khi cào xong sẽ lập tức kích hoạt bước huấn luyện/suy luận tương ứng (không cần đợi 2 nhánh còn lại).
   - Tự động ghi nhận Model Artifact, Scaler và Model Card lên MinIO.
3. **Phase 3 (Master Ensemble Integration):**
   - Đợi cả 3 nhánh hoàn tất ➔ kích hoạt `dag_ensemble_master` để tổng hợp 4 phương pháp và tính Trading Plan.
4. **Phase 4 (Model Evaluation):**
   - Kích hoạt `dag_model_evaluation` để kiểm định TimeSeriesSplit CV, Sentiment correlation và xuất báo cáo CSV.

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
    )

    phase1_crawl_lstm = TriggerDagRunOperator(
        task_id='phase1_crawl_lstm',
        trigger_dag_id='dag_crawl_lstm_stock_features',
        wait_for_completion=True,
        poke_interval=10,
        reset_dag_run=False,
    )

    phase1_crawl_finbert = TriggerDagRunOperator(
        task_id='phase1_crawl_finbert',
        trigger_dag_id='dag_crawl_finbert_news',
        wait_for_completion=True,
        poke_interval=10,
        reset_dag_run=False,
    )

    # --- PHASE 2: HUẤN LUYỆN & DỰ BÁO SONG SONG ---
    phase2_train_xgboost = TriggerDagRunOperator(
        task_id='phase2_train_xgboost',
        trigger_dag_id='dag_xgboost_train_and_predict',
        wait_for_completion=True,
        poke_interval=15,
        reset_dag_run=False,
    )

    phase2_train_lstm = TriggerDagRunOperator(
        task_id='phase2_train_lstm',
        trigger_dag_id='dag_lstm_train_and_predict',
        wait_for_completion=True,
        poke_interval=15,
        reset_dag_run=False,
    )

    phase2_predict_finbert = TriggerDagRunOperator(
        task_id='phase2_predict_finbert',
        trigger_dag_id='dag_finbert_predict',
        wait_for_completion=True,
        poke_interval=15,
        reset_dag_run=False,
    )

    # --- PHASE 3: MASTER ENSEMBLE ---
    phase3_ensemble = TriggerDagRunOperator(
        task_id='phase3_master_ensemble',
        trigger_dag_id='dag_ensemble_master',
        wait_for_completion=True,
        poke_interval=10,
        reset_dag_run=False,
    )

    # --- PHASE 4: ĐÁNH GIÁ MÔ HÌNH ---
    phase4_evaluation = TriggerDagRunOperator(
        task_id='phase4_model_evaluation',
        trigger_dag_id='dag_model_evaluation',
        wait_for_completion=True,
        poke_interval=10,
        reset_dag_run=False,
    )

    # --- ĐỊNH NGHĨA LUỒNG THỰC THI CHUẨN XÁC ---
    notify_start >> [phase1_crawl_xgboost, phase1_crawl_lstm, phase1_crawl_finbert]

    # Đảm bảo Phase 1 (đặc biệt là LSTM chứa dữ liệu giá Close thực 20 mã) hoàn tất trước Phase 2
    [phase1_crawl_xgboost, phase1_crawl_lstm] >> phase2_train_xgboost
    phase1_crawl_lstm >> phase2_train_lstm
    [phase1_crawl_finbert, phase1_crawl_lstm] >> phase2_predict_finbert

    [phase2_train_xgboost, phase2_train_lstm, phase2_predict_finbert] >> phase3_ensemble >> phase4_evaluation

