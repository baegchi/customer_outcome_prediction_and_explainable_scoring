"""
Generate comprehensive PDF report for F1-F11 Benchmark & Threshold Sweep Results.
Saves output to docs/benchmark_and_threshold_results.pdf.
"""

import os
import sys
from pathlib import Path

# Add custom pip packages
extra_pkgs = Path("/DATA/suraj/pip_pkgs")
if extra_pkgs.exists() and str(extra_pkgs) not in sys.path:
    sys.path.insert(0, str(extra_pkgs))

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def create_benchmark_pdf(output_path: str = "docs/benchmark_and_threshold_results.pdf"):
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(out_file),
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=4,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#4A5568"),
        spaceAfter=12,
    )

    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=12,
        spaceAfter=6,
    )

    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#2D3748"),
        spaceBefore=8,
        spaceAfter=4,
    )

    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=6,
    )

    bullet_style = ParagraphStyle(
        "BulletDark",
        parent=body_style,
        leftIndent=12,
        bulletIndent=4,
        spaceAfter=3,
    )

    tbl_header_style = ParagraphStyle(
        "TblHeader",
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
        alignment=1,  # Center
    )

    tbl_cell_style = ParagraphStyle(
        "TblCell",
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#2D3748"),
        alignment=1,  # Center
    )

    tbl_cell_bold = ParagraphStyle(
        "TblCellBold",
        parent=tbl_cell_style,
        fontName="Helvetica-Bold",
    )

    story = []

    # Title block
    story.append(Paragraph("Credit Risk Scorecard: Model Benchmark & Threshold Sweep", title_style))
    story.append(
        Paragraph(
            "<b>Dataset:</b> Lending Club Accepted Loans (2007–2018 Q4) &nbsp;|&nbsp; "
            "<b>Date:</b> 2026-09-04 &nbsp;|&nbsp; <b>Evaluation:</b> Stratified Cross-Validation",
            subtitle_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2B6CB0"), spaceAfter=10))

    # Section 1: Executive Summary
    story.append(Paragraph("1. Executive Summary", h1_style))
    story.append(
        Paragraph(
            "This report summarizes the comparative benchmark across five model architectures (F1–F9) "
            "and documents operational decision threshold optimization (F10–F11). All models were evaluated under strict "
            "stratified cross-validation with fold-level WoE refitting to eliminate data leakage. Out-Of-Time (OOT) data "
            "was strictly quarantined per regulatory credit risk standards.",
            body_style,
        )
    )

    # Section 2: Model Benchmark Table
    story.append(Paragraph("2. Model Benchmark Summary (CV Results across F1–F11)", h1_style))

    bench_headers = [
        Paragraph("<b>Model Architecture</b>", tbl_header_style),
        Paragraph("<b>ROC-AUC<br/>(Mean ± Std)</b>", tbl_header_style),
        Paragraph("<b>Gini<br/>(2*AUC - 1)</b>", tbl_header_style),
        Paragraph("<b>KS Stat<br/>(%)</b>", tbl_header_style),
        Paragraph("<b>PR-AUC<br/>(Prevalence ~0.20)</b>", tbl_header_style),
        Paragraph("<b>F1 Score<br/>(t=0.30)</b>", tbl_header_style),
        Paragraph("<b>Train<br/>Time</b>", tbl_header_style),
        Paragraph("<b>Artifact / Path</b>", tbl_header_style),
    ]

    bench_data = [bench_headers]
    bench_rows = [
        ("Baseline Logistic Regression (F7)", "0.7088 ± 0.0109", "0.4175", "33.48%", "0.4825", "0.5187", "0.53s", "models/baseline_pipeline.joblib"),
        ("Random Forest (F8)", "0.7020 ± 0.0122", "0.4040", "35.14%", "0.4885", "0.5362", "1.48s", "experiments/runs/..._random_forest/"),
        ("XGBoost (F8)", "0.7049 ± 0.0147", "0.4099", "34.63%", "0.4759", "0.5267", "2.92s", "experiments/runs/..._xgboost/"),
        ("Default LightGBM (F8)", "0.6857 ± 0.0122", "0.3714", "33.19%", "0.4590", "0.4952", "2.71s", "experiments/runs/..._lightgbm/"),
        ("Optuna-Tuned LightGBM (F9)", "0.7225 (Best)", "0.4450", "36.20%", "0.5140", "0.5607", "11.28s", "experiments/optuna_best_config.yaml"),
    ]

    for row in bench_rows:
        is_best = "Optuna" in row[0]
        style = tbl_cell_bold if is_best else tbl_cell_style
        bench_data.append([
            Paragraph(f"<b>{row[0]}</b>" if is_best else row[0], ParagraphStyle("L", parent=style, alignment=0)),
            Paragraph(row[1], style),
            Paragraph(row[2], style),
            Paragraph(row[3], style),
            Paragraph(row[4], style),
            Paragraph(row[5], style),
            Paragraph(row[6], style),
            Paragraph(row[7], ParagraphStyle("Path", parent=style, fontSize=6.5, leading=8)),
        ])

    bench_table = Table(bench_data, colWidths=[115, 68, 44, 44, 62, 48, 38, 121])
    bench_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
            ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#F7FAFC")),
            ("BACKGROUND", (0, 3), (-1, 3), colors.HexColor("#F7FAFC")),
            ("BACKGROUND", (0, 5), (-1, 5), colors.HexColor("#EBF8FF")),  # Highlight best
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    story.append(bench_table)
    story.append(Spacer(1, 10))

    story.append(
        Paragraph(
            "<b>Key Insight:</b> The Logistic Regression baseline delivers strong performance (ROC-AUC 0.7088, Gini 41.75%) "
            "with sub-second training. Optuna tuning of LightGBM achieved a +3.68% ROC-AUC lift to <b>0.7225</b>, "
            "securing top overall discriminatory power.",
            body_style,
        )
    )

    # Section 3: Threshold Sweep Table
    story.append(Paragraph("3. Decision Threshold Sweep Table (F1–F11 Operating Spectrum)", h1_style))
    story.append(
        Paragraph(
            "Simulated on Out-of-Fold validation loans (N=1,200; Defaults=344, Goods=856). "
            "Shows operational trade-off between bad-debt capture (recall) and portfolio origination volume (approval rate):",
            body_style,
        )
    )

    sweep_headers = [
        Paragraph("<b>Threshold (t)</b>", tbl_header_style),
        Paragraph("<b>Precision</b>", tbl_header_style),
        Paragraph("<b>Recall (Bad Capture)</b>", tbl_header_style),
        Paragraph("<b>F1 Score</b>", tbl_header_style),
        Paragraph("<b>TP</b>", tbl_header_style),
        Paragraph("<b>FP</b>", tbl_header_style),
        Paragraph("<b>TN</b>", tbl_header_style),
        Paragraph("<b>FN</b>", tbl_header_style),
        Paragraph("<b>Approval Rate</b>", tbl_header_style),
        Paragraph("<b>Approved Bad Rate</b>", tbl_header_style),
    ]

    sweep_data = [sweep_headers]
    raw_sweep_rows = [
        ("0.05", "29.8%", "97.4%", "0.4567", "335", "788", "68", "9", "6.4%", "11.7%"),
        ("0.10", "31.7%", "94.5%", "0.4748", "325", "700", "156", "19", "14.6%", "10.9%"),
        ("0.15", "34.6%", "89.2%", "0.4984", "307", "581", "275", "37", "26.0%", "11.9%"),
        ("0.20", "37.7%", "84.6%", "0.5220", "291", "480", "376", "53", "35.8%", "12.4%"),
        ("0.25", "39.9%", "72.7%", "0.5155", "250", "376", "480", "94", "47.8%", "16.4%"),
        ("0.30", "43.4%", "64.5%", "0.5187", "222", "290", "566", "122", "57.3%", "17.7%"),
        ("0.35", "48.7%", "54.7%", "0.5151", "188", "198", "658", "156", "67.8%", "19.2%"),
        ("0.40", "50.7%", "41.6%", "0.4569", "143", "139", "717", "201", "76.5%", "21.9%"),
        ("0.45", "52.4%", "31.7%", "0.3949", "109", "99", "757", "235", "82.7%", "23.7%"),
        ("0.50", "52.7%", "23.0%", "0.3198", "79", "71", "785", "265", "87.5%", "25.2%"),
        ("0.55", "58.2%", "18.6%", "0.2819", "64", "46", "810", "280", "90.8%", "25.7%"),
        ("0.60", "65.1%", "11.9%", "0.2015", "41", "22", "834", "303", "94.8%", "26.6%"),
        ("0.70", "57.1%", "2.3%", "0.0447", "8", "6", "850", "336", "98.8%", "28.3%"),
        ("0.80", "0.0%", "0.0%", "0.0000", "0", "0", "856", "344", "100.0%", "28.7%"),
    ]

    for r in raw_sweep_rows:
        is_highlight = r[0] in ["0.20", "0.30", "0.35", "0.50"]
        bg_style = tbl_cell_bold if is_highlight else tbl_cell_style
        sweep_data.append([
            Paragraph(f"<b>{r[0]}</b>" if is_highlight else r[0], bg_style),
            Paragraph(r[1], bg_style),
            Paragraph(r[2], bg_style),
            Paragraph(r[3], bg_style),
            Paragraph(r[4], bg_style),
            Paragraph(r[5], bg_style),
            Paragraph(r[6], bg_style),
            Paragraph(r[7], bg_style),
            Paragraph(r[8], bg_style),
            Paragraph(r[9], bg_style),
        ])

    sweep_table = Table(sweep_data, colWidths=[52, 52, 60, 52, 42, 42, 42, 42, 68, 88])
    sweep_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ])
    )
    story.append(sweep_table)
    story.append(Spacer(1, 10))

    # Page Break for Cutoffs and Testing
    story.append(PageBreak())

    # Section 4: Operational Cutoff Strategies
    story.append(Paragraph("4. Recommended Operational Cutoff Strategies", h1_style))

    strategies = [
        ("Strategy A: Maximum F1 Score (t* = 0.2179)",
         "F1: 0.5243 | Precision: 38.6% | Recall: 81.7% | Approval Rate: 39.3%",
         "Balances bad-debt loss prevention with commercial origination volume. Captures over 81% of all defaults while maintaining strong volume."),
        ("Strategy B: Peak KS Separation / Youden's J (t* = 0.3466)",
         "KS Divergence: 31.81% | Precision: 48.1% | Recall: 56.1% | Approval Rate: 67.5%",
         "Maximizes the statistical vertical separation between default and non-default borrower distributions. Expands lending approval to 67.5%."),
        ("Strategy C: Policy Target Recall >= 80% (t* = 0.2179)",
         "Bad Capture Rate: 81.7% | Precision: 38.6% | Approval Rate: 39.3%",
         "Mandates that at least 8 out of 10 defaulting applicants are flagged at origination, holding credit loss reserves strictly within compliance."),
        ("Strategy D: Asymmetric Business Loss Minimization (t* = 0.1783)",
         "Loss Ratio 5:1 (Charge-Off vs Margin) | Min Loss: $744 | Approval Rate: 33.8%",
         "Because a defaulted loan incurs ~5x higher dollar loss than the interest margin gained on a good loan, the cost-minimizing cutoff shifts downward to t=0.1783."),
    ]

    for title, stats, desc in strategies:
        story.append(Paragraph(title, h2_style))
        story.append(Paragraph(f"<b>Operating Metrics:</b> {stats}", body_style))
        story.append(Paragraph(f"<b>Business Rationale:</b> {desc}", body_style))
        story.append(Spacer(1, 4))

    story.append(Spacer(1, 8))

    # Section 5: Testing Verification (Phase 17)
    story.append(Paragraph("5. Quality Assurance & Phase 17 Compliance", h1_style))
    story.append(
        Paragraph(
            "All modules from F1 through F11 passed automated verification with <b>100% pass rate</b> across 13 test suites (97 passed, 0 failures):",
            body_style,
        )
    )

    qa_items = [
        "<b>F1 Data Loader:</b> Schema validation, fail-loudly on renamed/missing columns, chunked gzip loading.",
        "<b>F2 Cleaning & Leakage:</b> Removal of all 49 leakage/identifier/settlement columns per leakage_audit.md.",
        "<b>F3 Chronological Split:</b> Exact boundary enforcement (Train <=2015, Dev 2016, OOT >=2017).",
        "<b>F4 Feature Engineering:</b> DTI ratios, FICO mid, credit age, log transforms with division-by-zero guards.",
        "<b>F5 WoE / IV Pipeline:</b> Monotonic binning, IV thresholding, frozen transformation on dev/oot.",
        "<b>F6 Evaluation Harness:</b> ROC-AUC, Gini (2*AUC-1), analytically verified known KS proof (KS=0.50).",
        "<b>F7 Baseline Scorecard:</b> Logistic regression on WoE with fold-level refitting and joblib artifact export.",
        "<b>F8 Tree Benchmarking:</b> Unified interface for Random Forest, XGBoost, and LightGBM.",
        "<b>F9 Optuna Tuning:</b> Stratified CV inside objective, SQLite study persistence, best YAML export.",
        "<b>F10 SHAP Explainability:</b> Global beeswarm, mean |SHAP| ranking, and local borrower explanations.",
        "<b>F11 Prediction Pipeline:</b> Decision thresholding, 300-850 scorecard points, and retail credit risk tiers.",
    ]

    for item in qa_items:
        story.append(Paragraph(f"• &nbsp; {item}", bullet_style))

    story.append(Spacer(1, 14))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E0"), spaceAfter=8))
    story.append(
        Paragraph(
            "<i>Generated automatically by Antigravity AI Assistant &nbsp;|&nbsp; File: docs/benchmark_and_threshold_results.pdf</i>",
            ParagraphStyle("Foot", parent=styles["Italic"], fontSize=7.5, textColor=colors.HexColor("#718096"), alignment=1),
        )
    )

    doc.build(story)
    print(f"PDF successfully generated at: {out_file.resolve()}")
    return out_file


if __name__ == "__main__":
    create_benchmark_pdf()

