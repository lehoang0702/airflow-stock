#!/usr/bin/env python3
"""
EXECUTIVE PDF REPORT GENERATOR
Sử dụng ReportLab để biên tập báo cáo PDF chuyên nghiệp chuẩn A4,
đầy đủ tiếng Việt có dấu, sơ đồ hình ảnh sắc nét và bố cục doanh nghiệp.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Thêm thư viện ReportLab cục bộ
sys.path.insert(0, os.path.join(BASE_DIR, "lib"))

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfgen import canvas

# 1. Đăng ký Font chữ hỗ trợ tiếng Việt (Liberation Sans)
FONT_DIR = "/usr/share/fonts/liberation-sans-fonts"
pdfmetrics.registerFont(TTFont('LiberationSans', f'{FONT_DIR}/LiberationSans-Regular.ttf'))
pdfmetrics.registerFont(TTFont('LiberationSans-Bold', f'{FONT_DIR}/LiberationSans-Bold.ttf'))
pdfmetrics.registerFont(TTFont('LiberationSans-Italic', f'{FONT_DIR}/LiberationSans-Italic.ttf'))
pdfmetrics.registerFont(TTFont('LiberationSans-BoldItalic', f'{FONT_DIR}/LiberationSans-BoldItalic.ttf'))

registerFontFamily(
    'LiberationSans',
    normal='LiberationSans',
    bold='LiberationSans-Bold',
    italic='LiberationSans-Italic',
    boldItalic='LiberationSans-BoldItalic'
)

# 2. NumberedCanvas: Tự động đánh số trang "Trang X / Y" và thêm Header/Footer
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("LiberationSans", 8.5)
        self.setFillColor(colors.HexColor("#64748b"))
        
        # Trang 1 (Trang bìa) không vẽ header/footer
        if self._pageNumber > 1:
            # Header
            self.drawString(20 * mm, 282 * mm, "BÁO CÁO DỰ ÁN: QUANTUM MULTI-MODAL STOCK FORECASTING & MLOPS")
            self.setStrokeColor(colors.HexColor("#e2e8f0"))
            self.setLineWidth(0.75)
            self.line(20 * mm, 280 * mm, 190 * mm, 280 * mm)
            
            # Footer
            self.line(20 * mm, 16 * mm, 190 * mm, 16 * mm)
            self.drawString(20 * mm, 11 * mm, "Hệ Thống Định Lượng & Dự Báo Chứng Khoán • Dự án: airflow-stock")
            page_text = f"Trang {self._pageNumber} / {page_count}"
            self.drawRightString(190 * mm, 11 * mm, page_text)
        
        self.restoreState()


def build_pdf():
    pdf_path = os.path.join(BASE_DIR, "BAO_CAO_DU_AN_CHUNG_KHOAN.pdf")
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=22 * mm,
        bottomMargin=22 * mm
    )

    styles = getSampleStyleSheet()
    
    # Custom Typography Styles
    style_cover_badge = ParagraphStyle(
        'CoverBadge',
        fontName='LiberationSans-Bold',
        fontSize=9,
        textColor=colors.HexColor("#2563eb"),
        backColor=colors.HexColor("#eff6ff"),
        borderColor=colors.HexColor("#bfdbfe"),
        borderWidth=1,
        borderPadding=(4, 8, 4, 8),
        spaceAfter=12
    )

    style_cover_title = ParagraphStyle(
        'CoverTitle',
        fontName='LiberationSans-Bold',
        fontSize=24,
        leading=30,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=10
    )

    style_cover_sub = ParagraphStyle(
        'CoverSub',
        fontName='LiberationSans',
        fontSize=12,
        leading=18,
        textColor=colors.HexColor("#475569"),
        spaceAfter=18
    )

    style_h1 = ParagraphStyle(
        'Heading1_Custom',
        fontName='LiberationSans-Bold',
        fontSize=15,
        leading=20,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )

    style_h2 = ParagraphStyle(
        'Heading2_Custom',
        fontName='LiberationSans-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True
    )

    style_body = ParagraphStyle(
        'Body_Custom',
        fontName='LiberationSans',
        fontSize=10,
        leading=14.5,
        textColor=colors.HexColor("#334155"),
        spaceAfter=8
    )

    style_bullet = ParagraphStyle(
        'Bullet_Custom',
        fontName='LiberationSans',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#334155"),
        leftIndent=15,
        spaceAfter=4
    )

    style_callout = ParagraphStyle(
        'Callout',
        fontName='LiberationSans',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#1e40af"),
        backColor=colors.HexColor("#eff6ff"),
        borderColor=colors.HexColor("#3b82f6"),
        borderWidth=1,
        borderPadding=8,
        spaceBefore=8,
        spaceAfter=10
    )

    style_callout_success = ParagraphStyle(
        'CalloutSuccess',
        fontName='LiberationSans',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#065f46"),
        backColor=colors.HexColor("#ecfdf5"),
        borderColor=colors.HexColor("#10b981"),
        borderWidth=1,
        borderPadding=8,
        spaceBefore=8,
        spaceAfter=10
    )

    style_table_header = ParagraphStyle(
        'TableHeader',
        fontName='LiberationSans-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
        alignment=1
    )

    style_table_cell = ParagraphStyle(
        'TableCell',
        fontName='LiberationSans',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1e293b")
    )

    style_table_cell_bold = ParagraphStyle(
        'TableCellBold',
        fontName='LiberationSans-Bold',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#0f172a")
    )

    style_caption = ParagraphStyle(
        'Caption',
        fontName='LiberationSans-Italic',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#64748b"),
        alignment=1,
        spaceBefore=4,
        spaceAfter=12
    )

    story = []

    # =========================================================================
    # TRANG BÌA &amp; TỔNG QUAN DỰ ÁN
    # =========================================================================
    story.append(Paragraph("BÁO CÁO KHOA HỌC &amp; KIẾN TRÚC HỆ THỐNG", style_cover_badge))
    story.append(Paragraph("Quantum Multi-Modal AI Stock Forecasting<br/>& MLOps Platform", style_cover_title))
    story.append(Paragraph(
        "Hệ thống định lượng và dự báo xu hướng chứng khoán đa phương thức kết hợp Trí Tuệ Nhân Tạo "
        "(XGBoost GBDT, PyTorch LSTM, ProsusAI FinBERT) và quy trình quản trị MLOps hoàn chỉnh trên nền tảng "
        "Apache Airflow 2.8, MinIO S3, PostgreSQL và Docker Compose.",
        style_cover_sub
    ))

    # Meta Table
    meta_data = [
        [
            Paragraph("<b>Dự án:</b> airflow-minio", style_table_cell),
            Paragraph("<b>Quy mô:</b> 20 Cổ phiếu S&amp;P 500", style_table_cell),
            Paragraph("<b>Môi trường:</b> Docker 5 Services", style_table_cell),
            Paragraph("<b>Ngày phát hành:</b> 10/09/2026", style_table_cell)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[42*mm, 42*mm, 45*mm, 45*mm])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 14))

    # 1. Tóm Tắt Điều Hành
    story.append(Paragraph("1. Tóm Tắt Điều Hành (Executive Summary)", style_h1))
    story.append(Paragraph(
        "Thị trường chứng khoán là một hệ thống phi tuyến phức tạp, chịu ảnh hưởng đồng thời từ: "
        "<b>(1) Chỉ báo dao động kỹ thuật &amp; vĩ mô</b>, <b>(2) Quán tính chuỗi nến giá lịch sử</b>, và "
        "<b>(3) Tâm lý đám đông từ tin tức báo chí tài chính</b>. Dự án Quantum Multi-Modal giải quyết "
        "triệt để bài toán này bằng cách thu thập độc lập 3 luồng dữ liệu, huấn luyện 3 kiến trúc AI chuyên biệt, "
        "và hợp nhất bằng giải thuật <b>Master Ensemble 4 Phương Pháp</b> kèm hệ thống quản trị rủi ro tự động.",
        style_body
    ))
    story.append(Paragraph(
        "<b>[GHI CHÚ QUAN TRỌNG] Trạng thái làm sạch:</b> Toàn bộ mã độc (Malware Detection) đã được xóa bỏ vĩnh viễn 100%, "
        "giải phóng 1.4 GB dữ liệu đĩa cứng. Hệ thống vận hành độc lập, ổn định tuyệt đối với 16 modules DAGs "
        "và 5 container microservices đã được kiểm thử tính hợp lệ.",
        style_callout_success
    ))
    story.append(Spacer(1, 10))

    # =========================================================================
    # 2. KIẾN TRÚC PHÂN TẦNG TỔNG THỂ
    # =========================================================================
    story.append(Paragraph("2. Sơ Đồ Kiến Trúc Phân Tầng Tổng Thể (5-Layer Architecture)", style_h1))
    story.append(Paragraph(
        "Kiến trúc hệ thống được chia thành 5 tầng độc lập (Decoupled Layers) đảm bảo khả năng mở rộng, "
        "khả năng bảo trì và kiểm thử độc lập cho từng thành phần:",
        style_body
    ))

    arch_img_path = os.path.join(BASE_DIR, "pdf_assets/arch.png")
    if os.path.exists(arch_img_path):
        story.append(Image(arch_img_path, width=174*mm, height=108*mm))
        story.append(Paragraph("Hình 1: Sơ đồ kiến trúc 5 tầng của hệ thống Quantum Multi-Modal Platform", style_caption))

    story.append(Paragraph("<b>Các tầng chức năng cốt lõi:</b>", style_body))
    story.append(Paragraph("• <b>Tầng 1 (Ingestion):</b> Thu thập dữ liệu nến giá từ Yahoo Finance và tin tức tài chính RSS từ Google News cho 20 mã cổ phiếu.", style_bullet))
    story.append(Paragraph("• <b>Tầng 2 (AI Modeling):</b> 3 nhánh mô hình hóa chuyên biệt (XGBoost GBDT, PyTorch LSTM, ProsusAI FinBERT Transformer).", style_bullet))
    story.append(Paragraph("• <b>Tầng 3 (Ensemble &amp; Risk):</b> Hợp nhất tín hiệu theo 4 phương pháp và tính toán kế hoạch giao dịch (Trading Plan Engine).", style_bullet))
    story.append(Paragraph("• <b>Tầng 4 (MLOps &amp; Governance):</b> Kiểm toán chất lượng dữ liệu (Data Quality Gate), Model Registry, TimeSeriesSplit CV, và Paper Trading.", style_bullet))
    story.append(Paragraph("• <b>Tầng 5 (Presentation):</b> Web Dashboard tương tác (Cổng 8050) và hệ thống cảnh báo tức thời qua Telegram Bot API.", style_bullet))

    story.append(PageBreak())

    # =========================================================================
    # 3. CHI TIẾT 3 NHÁNH AI ĐA PHƯƠNG THỨC
    # =========================================================================
    story.append(Paragraph("3. Chi Tiết 3 Nhánh Trí Tuệ Nhân Tạo (Multi-Modal AI)", style_h1))
    story.append(Paragraph(
        "Mỗi nhánh AI phụ trách một miền tri thức độc lập, hỗ trợ bù trừ điểm yếu và tối đa hóa độ tin cậy tín hiệu:",
        style_body
    ))

    model_table_data = [
        [
            Paragraph("Đặc Tính", style_table_header),
            Paragraph("Nhánh 1: XGBoost GBDT", style_table_header),
            Paragraph("Nhánh 2: PyTorch LSTM", style_table_header),
            Paragraph("Nhánh 3: FinBERT NLP", style_table_header)
        ],
        [
            Paragraph("<b>Dạng dữ liệu</b>", style_table_cell_bold),
            Paragraph("Dữ liệu bảng (Tabular): 85+ chỉ báo kỹ thuật (RSI, MACD, Bollinger Bands, ATR, Stoch) + 4 chỉ số vĩ mô (SPY, QQQ, VIX, TNX)", style_table_cell),
            Paragraph("Dữ liệu chuỗi (Sequential): Tensor 3D <code>[Batch, 60, Features]</code> gồm nến Open, High, Low, Close, Volume", style_table_cell),
            Paragraph("Văn bản phi cấu trúc: Tiêu đề &amp; tóm tắt tin tức báo chí tài chính từ Google News RSS", style_table_cell)
        ],
        [
            Paragraph("<b>Kiến trúc mô hình</b>", style_table_cell_bold),
            Paragraph("Gradient Boosted Decision Trees (max_depth=6, lr=0.05, n_estimators=200)", style_table_cell),
            Paragraph("2-Layer PyTorch LSTM (hidden_dim=64, dropout=0.2, Linear Head)", style_table_cell),
            Paragraph("Transformer FinBERT Pre-trained on Financial PhraseBank corpus", style_table_cell)
        ],
        [
            Paragraph("<b>Thế mạnh</b>", style_table_cell_bold),
            Paragraph("Nhận diện tương quan phi tuyến giữa các chỉ báo dao động và vĩ mô", style_table_cell),
            Paragraph("Nắm bắt quán tính xu hướng giá và phụ thuộc thời gian dài hạn", style_table_cell),
            Paragraph("Nhận biết tâm lý hưng phấn/hoảng loạn trước khi giá biến động", style_table_cell)
        ],
        [
            Paragraph("<b>Đầu ra chuẩn hóa</b>", style_table_cell_bold),
            Paragraph("Xác suất tăng: P_XGB trong khoảng [0, 1]", style_table_cell),
            Paragraph("Xác suất tăng: P_LSTM trong khoảng [0, 1]", style_table_cell),
            Paragraph("Điểm cảm xúc [-1, 1] ánh xạ sang P_Fin trong [0, 1]", style_table_cell)
        ]
    ]

    model_table = Table(model_table_data, colWidths=[28*mm, 48*mm, 48*mm, 50*mm])
    model_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e293b")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(model_table)
    story.append(Spacer(1, 14))

    # =========================================================================
    # 4. HỢP NHẤT ENSEMBLE & KẾ HOẠCH GIAO DỊCH
    # =========================================================================
    story.append(Paragraph("4. Cơ Chế Master Ensemble &amp; Kế Hoạch Giao Dịch (Trading Plan)", style_h1))
    story.append(Paragraph(
        "Kết quả dự báo từ 3 nhánh được tổng hợp bằng công thức trọng số tối ưu hóa khoa học "
        "trong <code>dag_ensemble_master.py</code>:",
        style_body
    ))
    story.append(Paragraph(
        "<b>Công thức xác suất tổng hợp:</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>P_Ensemble = 0.40 × P_XGBoost + 0.35 × P_LSTM + 0.25 × P_FinBERT</b>",
        style_callout
    ))

    ens_img_path = os.path.join(BASE_DIR, "pdf_assets/ensemble.png")
    if os.path.exists(ens_img_path):
        story.append(Image(ens_img_path, width=170*mm, height=52*mm))
        story.append(Paragraph("Hình 2: Sơ đồ luồng ra quyết định và phân loại khuyến nghị đầu tư", style_caption))

    story.append(Paragraph("<b>Quy Tắc Quản Trị Vị Thế &amp; Rủi Ro (Trading Plan Engine):</b>", style_h2))
    story.append(Paragraph("• <b>Entry Price (Giá vào lệnh):</b> Giá khớp nến đóng cửa (Close) gần nhất của phiên giao dịch vừa hoàn tất.", style_bullet))
    story.append(Paragraph("• <b>Take Profit (Chốt lời mục tiêu):</b> Đặt tại ngưỡng <b>+3.5%</b> (với tín hiệu MUA) hoặc <b>+5.0%</b> (với MUA MẠNH).", style_bullet))
    story.append(Paragraph("• <b>Stop Loss (Cắt lỗ bảo vệ):</b> Đặt tại ngưỡng <b>-2.0%</b> (với tín hiệu MUA) hoặc <b>-2.5%</b> (với MUA MẠNH).", style_bullet))
    story.append(Paragraph("• <b>Tỷ lệ Lợi Nhuận / Rủi Ro (Risk/Reward):</b> Luôn đảm bảo tối thiểu <b>≥ 1.75 : 1</b> đến <b>2.0 : 1</b>.", style_bullet))
    story.append(Paragraph("• <b>Quy tắc Time Exit (T+5):</b> Vị thế tự động đóng sau 5 phiên giao dịch nếu chưa chạm ngưỡng Take Profit hoặc Stop Loss.", style_bullet))

    story.append(PageBreak())

    # =========================================================================
    # 5. HỆ THỐNG MLOPS TOÀN DIỆN
    # =========================================================================
    story.append(Paragraph("5. Hệ Thống MLOps &amp; Quản Trị Vòng Đời Mô Hình (MLOps Lifecycle)", style_h1))
    story.append(Paragraph(
        "Quy trình MLOps được thiết kế bài bản theo 4 trụ cột khép kín, đảm bảo tính ổn định và minh bạch tuyệt đối:",
        style_body
    ))

    mlops_img_path = os.path.join(BASE_DIR, "pdf_assets/mlops.png")
    if os.path.exists(mlops_img_path):
        story.append(Image(mlops_img_path, width=174*mm, height=52*mm))
        story.append(Paragraph("Hình 3: 4 Trụ cột quản trị vòng đời mô hình trong MLOps Pipeline", style_caption))

    story.append(Paragraph("• <b>1. DataOps &amp; Quality Gate (<code>data_validator.py</code>):</b> Kiểm tra tính hợp lệ của schema, tỷ lệ dữ liệu khuyết thiếu (< 1%), phát hiện ngoại lai bằng khoảng phân vị IQR và theo dõi độ ổn định phân phối dữ liệu (PSI - Population Stability Index). Báo động khẩn qua Telegram nếu có vi phạm.", style_bullet))
    story.append(Paragraph("• <b>2. Model Registry (<code>model_registry.py</code>):</b> Quản lý phiên bản mô hình theo ngày <code>vYYYYMMDD</code>, đóng gói artifacts (.json, .pt, .joblib), sinh tự động Model Cards ghi nhận siêu dữ liệu huấn luyện, siêu tham số và lưu trữ trên MinIO S3.", style_bullet))
    story.append(Paragraph("• <b>3. Kiểm Định Khoa Học (<code>model_evaluator.py</code>):</b> Đánh giá mô hình bằng 5-fold TimeSeriesSplit Cross-Validation, tính toán Accuracy, ROC-AUC, Precision, Recall, F1-Score, Brier Score và đối chiếu mức độ tương quan giữa tin tức FinBERT với giá thực tế.", style_bullet))
    story.append(Paragraph("• <b>4. Paper Trading Engine (<code>portfolio_tracker.py</code>):</b> Giả lập danh mục đầu tư định lượng vốn khởi điểm $100,000 USD, phân bổ vốn 10-15%/mã, thực thi chốt lời/cắt lỗ/T+5 và theo dõi Win Rate %, Sharpe Ratio, Max Drawdown.", style_bullet))
    story.append(Spacer(1, 10))

    # =========================================================================
    # 6. LẬP LỊCH TỰ ĐỘNG CRON
    # =========================================================================
    story.append(Paragraph("6. Lập Lịch Vận Hành Tự Động (Automated Cron Schedule)", style_h1))
    story.append(Paragraph(
        "Hệ thống được thiết lập lịch chạy tự động trên <code>dag_master_orchestrator.py</code>:",
        style_body
    ))
    story.append(Paragraph(
        "<b>Chu kỳ Cron tối ưu: <code>0 22 * * 1-5</code> (22:00 UTC, Thứ Hai đến Thứ Sáu)</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>Giờ Việt Nam tương ứng:</b> Đúng <b>05:00 Sáng</b> từ Thứ Ba đến Thứ Bảy.",
        style_callout_success
    ))
    story.append(Paragraph("<b>Cơ sở kinh tế &amp; kỹ thuật lựa chọn lịch chạy:</b>", style_h2))
    story.append(Paragraph("1. Phiên chứng khoán Mỹ (NYSE/NASDAQ) đóng cửa lúc 16:00 EST/EDT (khoảng 03:00 - 04:00 sáng giờ VN).", style_bullet))
    story.append(Paragraph("2. Sau giờ đóng cửa 60-90 phút, toàn bộ giá đóng cửa chính thức (Close, Volume) và các phiên sau giờ (After-hours) đã chốt sổ hoàn toàn và cập nhật đầy đủ trên Yahoo Finance API.", style_bullet))
    story.append(Paragraph("3. Pipeline chạy trong 15-20 phút. Đến <b>06:00 - 07:00 sáng</b>, Bot Telegram đã gửi đầy đủ bảng tổng hợp tín hiệu, biểu đồ kỹ thuật và kế hoạch giao dịch cho ngày mới.", style_bullet))

    story.append(PageBreak())

    # =========================================================================
    # 7. HẠ TẦNG TRIỂN KHAI DOCKER & DANH MỤC CỔ PHIẾU
    # =========================================================================
    story.append(Paragraph("7. Hạ Tầng Triển Khai Container (Docker Infrastructure)", style_h1))
    
    docker_img_path = os.path.join(BASE_DIR, "pdf_assets/docker.png")
    if os.path.exists(docker_img_path):
        story.append(Image(docker_img_path, width=170*mm, height=50*mm))
        story.append(Paragraph("Hình 4: Sơ đồ mạng nội bộ các dịch vụ trong docker-compose.yaml", style_caption))

    docker_table_data = [
        [
            Paragraph("Dịch Vụ (Service)", style_table_header),
            Paragraph("Container Image", style_table_header),
            Paragraph("Cổng Mạng (Port)", style_table_header),
            Paragraph("Vai Trò &amp; Chức Năng", style_table_header)
        ],
        [
            Paragraph("<b>airflow-webserver</b>", style_table_cell_bold),
            Paragraph("custom-airflow-xgboost", style_table_cell),
            Paragraph("<code>8080:8080</code>", style_table_cell),
            Paragraph("Giao diện quản trị Web UI Airflow, theo dõi trạng thái 8 DAGs", style_table_cell)
        ],
        [
            Paragraph("<b>airflow-scheduler</b>", style_table_cell_bold),
            Paragraph("custom-airflow-xgboost", style_table_cell),
            Paragraph("Nội bộ", style_table_cell),
            Paragraph("Bộ lập lịch ngầm thực thi các DAGs theo Cron 22:00 UTC", style_table_cell)
        ],
        [
            Paragraph("<b>dashboard</b>", style_table_cell_bold),
            Paragraph("custom-airflow-xgboost", style_table_cell),
            Paragraph("<code>8050:8050</code>", style_table_cell),
            Paragraph("Quantum Interactive Web Dashboard xem biểu đồ, tín hiệu, MLOps", style_table_cell)
        ],
        [
            Paragraph("<b>minio</b>", style_table_cell_bold),
            Paragraph("minio/minio:latest", style_table_cell),
            Paragraph("<code>9000 &amp; 9001</code>", style_table_cell),
            Paragraph("Lưu trữ đối tượng S3: datasets, models, metrics và Web Console", style_table_cell)
        ],
        [
            Paragraph("<b>postgres</b>", style_table_cell_bold),
            Paragraph("postgres:13", style_table_cell),
            Paragraph("<code>5432:5432</code>", style_table_cell),
            Paragraph("Cơ sở dữ liệu lưu trữ Metadata vận hành của Apache Airflow", style_table_cell)
        ]
    ]

    docker_table = Table(docker_table_data, colWidths=[38*mm, 42*mm, 28*mm, 66*mm])
    docker_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e293b")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(docker_table)
    story.append(Spacer(1, 14))

    # 8. Danh Mục 20 Mã Cổ Phiếu
    story.append(Paragraph("8. Danh Mục 20 Mã Cổ Phiếu S&amp;P 500 Theo Chuẩn Ngành GICS", style_h1))
    
    tickers_data = [
        [
            Paragraph("Mã", style_table_header),
            Paragraph("Doanh Nghiệp", style_table_header),
            Paragraph("Ngành GICS", style_table_header),
            Paragraph("Mã", style_table_header),
            Paragraph("Doanh Nghiệp", style_table_header),
            Paragraph("Ngành GICS", style_table_header)
        ],
        [
            Paragraph("<b>AAPL</b>", style_table_cell_bold), Paragraph("Apple Inc.", style_table_cell), Paragraph("Technology", style_table_cell),
            Paragraph("<b>JPM</b>", style_table_cell_bold), Paragraph("JPMorgan Chase", style_table_cell), Paragraph("Financials", style_table_cell)
        ],
        [
            Paragraph("<b>MSFT</b>", style_table_cell_bold), Paragraph("Microsoft Corp.", style_table_cell), Paragraph("Technology", style_table_cell),
            Paragraph("<b>V</b>", style_table_cell_bold), Paragraph("Visa Inc.", style_table_cell), Paragraph("Financials", style_table_cell)
        ],
        [
            Paragraph("<b>NVDA</b>", style_table_cell_bold), Paragraph("NVIDIA Corp.", style_table_cell), Paragraph("Technology", style_table_cell),
            Paragraph("<b>UNH</b>", style_table_cell_bold), Paragraph("UnitedHealth", style_table_cell), Paragraph("Healthcare", style_table_cell)
        ],
        [
            Paragraph("<b>GOOGL</b>", style_table_cell_bold), Paragraph("Alphabet Inc.", style_table_cell), Paragraph("Technology", style_table_cell),
            Paragraph("<b>JNJ</b>", style_table_cell_bold), Paragraph("Johnson &amp; Johnson", style_table_cell), Paragraph("Healthcare", style_table_cell)
        ],
        [
            Paragraph("<b>AMZN</b>", style_table_cell_bold), Paragraph("Amazon.com Inc.", style_table_cell), Paragraph("Cons. Discretionary", style_table_cell),
            Paragraph("<b>CAT</b>", style_table_cell_bold), Paragraph("Caterpillar Inc.", style_table_cell), Paragraph("Industrials", style_table_cell)
        ],
        [
            Paragraph("<b>MCD</b>", style_table_cell_bold), Paragraph("McDonald's Corp.", style_table_cell), Paragraph("Cons. Discretionary", style_table_cell),
            Paragraph("<b>BA</b>", style_table_cell_bold), Paragraph("Boeing Co.", style_table_cell), Paragraph("Industrials", style_table_cell)
        ],
        [
            Paragraph("<b>NKE</b>", style_table_cell_bold), Paragraph("Nike Inc.", style_table_cell), Paragraph("Cons. Discretionary", style_table_cell),
            Paragraph("<b>XOM</b>", style_table_cell_bold), Paragraph("ExxonMobil", style_table_cell), Paragraph("Energy", style_table_cell)
        ],
        [
            Paragraph("<b>WMT</b>", style_table_cell_bold), Paragraph("Walmart Inc.", style_table_cell), Paragraph("Consumer Staples", style_table_cell),
            Paragraph("<b>CVX</b>", style_table_cell_bold), Paragraph("Chevron Corp.", style_table_cell), Paragraph("Energy", style_table_cell)
        ],
        [
            Paragraph("<b>PG</b>", style_table_cell_bold), Paragraph("Procter &amp; Gamble", style_table_cell), Paragraph("Consumer Staples", style_table_cell),
            Paragraph("<b>NEE</b>", style_table_cell_bold), Paragraph("NextEra Energy", style_table_cell), Paragraph("Utilities", style_table_cell)
        ],
        [
            Paragraph("<b>KO</b>", style_table_cell_bold), Paragraph("Coca-Cola Co.", style_table_cell), Paragraph("Consumer Staples", style_table_cell),
            Paragraph("<b>LIN</b>", style_table_cell_bold), Paragraph("Linde plc", style_table_cell), Paragraph("Materials", style_table_cell)
        ]
    ]

    tickers_table = Table(tickers_data, colWidths=[16*mm, 35*mm, 36*mm, 16*mm, 35*mm, 36*mm])
    tickers_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0f172a")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(tickers_table)

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"🎉 Executive PDF generated at: {pdf_path}")


if __name__ == "__main__":
    build_pdf()
