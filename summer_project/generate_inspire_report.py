# -*- coding: utf-8 -*-
"""
INSPIRE Scholar Format Project Report Generator
Automated Long-Answer Grading using QLoRA Fine-Tuned Qwen2.5-3B-Instruct
"""

from docx import Document
from docx.shared import Pt, Inches, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

doc = Document()

# ── Page margins ────────────────────────────────────────────
for sec in doc.sections:
    sec.top_margin    = Cm(2.5)
    sec.bottom_margin = Cm(2.5)
    sec.left_margin   = Cm(3.0)
    sec.right_margin  = Cm(2.5)

# ── Default style ───────────────────────────────────────────
sty = doc.styles["Normal"]
sty.font.name = "Times New Roman"
sty.font.size = Pt(12)

# ── Helpers ─────────────────────────────────────────────────
def sp(para, bef=0, aft=6):
    para.paragraph_format.space_before = Pt(bef)
    para.paragraph_format.space_after  = Pt(aft)

def add_p(text, bold=False, size=12, align=None, bef=0, aft=6, italic=False, center=False):
    para = doc.add_paragraph()
    run  = para.add_run(text)
    run.bold   = bold
    run.italic = italic
    run.font.size = Pt(size)
    run.font.name = "Times New Roman"
    sp(para, bef, aft)
    if align:
        para.alignment = align
    elif center:
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    return para

def add_h(text, level=1):
    h = doc.add_heading(text, level=level)
    sp(h, 10, 4)
    for r in h.runs:
        r.font.name = "Times New Roman"
    return h

def add_bullet(text):
    p = doc.add_paragraph(style="List Bullet")
    r = p.add_run(text)
    r.font.size = Pt(12)
    r.font.name = "Times New Roman"
    sp(p, 0, 3)

def add_numbered(text):
    p = doc.add_paragraph(style="List Number")
    r = p.add_run(text)
    r.font.size = Pt(12)
    r.font.name = "Times New Roman"
    sp(p, 0, 3)

def hdr_row(table, color="1F4E79"):
    for cell in table.rows[0].cells:
        tc   = cell._tc
        tcPr = tc.get_or_add_tcPr()
        shd  = OxmlElement("w:shd")
        shd.set(qn("w:val"),   "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"),  color)
        tcPr.append(shd)
        for para in cell.paragraphs:
            for run in para.runs:
                run.bold = True
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

def make_table(headers, rows, col_widths=None):
    t = doc.add_table(rows=len(rows)+1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        t.rows[0].cells[i].text = h
    hdr_row(t)
    for ri, row in enumerate(rows, start=1):
        for ci, val in enumerate(row):
            t.rows[ri].cells[ci].text = str(val)
    sp(doc.paragraphs[-1], 0, 6)
    return t

# ════════════════════════════════════════════════════════════
# 1. COVER PAGE
# ════════════════════════════════════════════════════════════
for _ in range(4):
    doc.add_paragraph()

title_p = doc.add_paragraph()
title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
t1 = title_p.add_run("Format for ")
t1.font.size = Pt(16); t1.font.name = "Times New Roman"
t2 = title_p.add_run("Research Project Report")
t2.bold = True; t2.font.size = Pt(18); t2.font.name = "Times New Roman"
sp(title_p, 0, 2)

sub_p = doc.add_paragraph()
sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
sr = sub_p.add_run("(To be filled by INSPIRE Scholar)")
sr.italic = True; sr.font.size = Pt(11); sr.font.name = "Times New Roman"
sp(sub_p, 0, 30)

cover_items = [
    ("Title of Project",
     "Automated Long-Answer Grading using QLoRA Fine-Tuned Qwen2.5-3B-Instruct"),
    ("Name of Scholar(s)",     "Ayan Kumar Batabyal  &  Anirban Nayak"),
    ("IVR No.",                "[To be filled by Scholar]"),
    ("Supervisor",             "Prof. Asit Kumar Das"),
    ("Organization",           "IIEST, Shibpur — Dept. of Computer Science & Technology"),
    ("Duration",               "May – July 2026"),
    ("Date of Submission",     "September 2026"),
]
for label, value in cover_items:
    lp = doc.add_paragraph()
    lp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    lr = lp.add_run(label + ":  ")
    lr.bold = True; lr.font.size = Pt(12); lr.font.name = "Times New Roman"
    vr = lp.add_run(value)
    vr.font.size = Pt(12); vr.font.name = "Times New Roman"
    sp(lp, 0, 5)

doc.add_page_break()

# ════════════════════════════════════════════════════════════
# 2. PROJECT COMPLETION CERTIFICATE
# ════════════════════════════════════════════════════════════
add_h("2. Project Completion Certificate", level=1)
add_p("This is to certify that the research project entitled:", aft=4)
add_p(
    '"Automated Long-Answer Grading using QLoRA Fine-Tuned Qwen2.5-3B-Instruct"',
    bold=True, center=True, aft=4
)
add_p(
    "has been successfully completed by Ayan Kumar Batabyal and Anirban Nayak "
    "under the INSPIRE SHE Scholarship programme during May to July 2026, "
    "under the supervision of Prof. Asit Kumar Das, Department of Computer Science "
    "and Technology, Indian Institute of Engineering Science and Technology (IIEST), Shibpur.",
    aft=14
)
for line in [
    "Supervisor Signature:  ____________________________",
    "Name & Designation:    Prof. Asit Kumar Das, IIEST Shibpur",
    "Date:  ____________________",
    "Place: IIEST, Shibpur",
    "Official Stamp:",
]:
    add_p(line, aft=7)
doc.add_page_break()

# ════════════════════════════════════════════════════════════
# 3. ACKNOWLEDGEMENT
# ════════════════════════════════════════════════════════════
add_h("3. Acknowledgement", level=1)
add_p(
    "We express our sincere gratitude to Prof. Asit Kumar Das, Department of Computer Science "
    "and Technology, IIEST Shibpur, for his invaluable guidance, constant encouragement, and "
    "technical insights throughout this research project. His expertise in machine learning and "
    "natural language processing provided a strong foundation for this work.",
    aft=8
)
add_p(
    "We are grateful to the Department of Science and Technology (DST), Government of India, "
    "for providing the INSPIRE SHE Scholarship that made this research internship possible. "
    "We thank IIEST Shibpur for providing access to computational resources and research infrastructure.",
    aft=8
)
add_p(
    "We also acknowledge the open-source contributions of the HuggingFace community, the Qwen team "
    "at Alibaba Cloud, and the developers of PEFT, TRL, and bitsandbytes libraries, whose tools "
    "formed the technical backbone of this project.",
    aft=8
)
add_p("Finally, we thank our families and friends for their unwavering support and patience.", aft=14)
add_p("Ayan Kumar Batabyal", aft=2)
add_p("Anirban Nayak", aft=4)
add_p("September 2026", aft=0)
doc.add_page_break()

# ════════════════════════════════════════════════════════════
# 4. TITLE OF RESEARCH PROJECT
# ════════════════════════════════════════════════════════════
add_h("4. Title of Research Project", level=1)
add_p(
    "Automated Long-Answer Grading using QLoRA Fine-Tuned Qwen2.5-3B-Instruct",
    bold=True, size=13, center=True, aft=0
)
doc.add_page_break()

# ════════════════════════════════════════════════════════════
# 5. AIM / OBJECTIVES
# ════════════════════════════════════════════════════════════
add_h("5. Aim / Objectives", level=1)
add_p(
    "The primary aim of this project is to develop an automated, rubric-based grading system "
    "for long descriptive student answers using a locally deployable open-weight large language "
    "model fine-tuned with parameter-efficient QLoRA techniques.",
    aft=6
)
add_h("Specific Objectives", level=2)
objectives = [
    "Develop an automated grading pipeline that evaluates each rubric criterion independently "
    "and produces a predicted total score comparable to human judgment.",

    "Prepare and format a domain-specific grading dataset of 1,000 student answer records "
    "across 10 computer science questions.",

    "Fine-tune Qwen2.5-3B-Instruct using QLoRA (4-bit NF4 quantisation + LoRA adapters, rank=16) "
    "on 700 training records.",

    "Compare the fine-tuned model against two baseline configurations: "
    "(a) unquantised FP16 and (b) 4-bit NF4 without fine-tuning.",

    "Evaluate all configurations using six metrics: MAE, RMSE, Exact Score Accuracy, "
    "Rubric-Level Accuracy, Pearson r, and Spearman rho.",

    "Assess generalisation by evaluating on both the original held-out test (n=150) "
    "and a new external test set (n=1,000).",

    "Identify limitations and propose future directions for improvement.",
]
for obj in objectives:
    add_numbered(obj)
doc.add_page_break()

# ════════════════════════════════════════════════════════════
# 6. INTRODUCTION
# ════════════════════════════════════════════════════════════
add_h("6. Introduction of Research Topic", level=1)

add_h("6.1 Background", level=2)
add_p(
    "Evaluating student responses to open-ended or descriptive questions is a fundamental part "
    "of the academic assessment process. Unlike objective questions, long-form answers require "
    "the evaluator to understand the meaning of the student's response, compare it against a "
    "marking rubric, and assign marks to individual criteria. A teacher grading 100 detailed "
    "answers for a single question may spend several hours doing so. Across multiple subjects "
    "and student batches, this workload becomes unmanageable while maintaining consistency."
)
add_p(
    "Manual grading is also inherently subjective. Different evaluators may award different "
    "marks for the same answer depending on their understanding, mood, or familiarity with the "
    "topic. Even the same evaluator may be inconsistent across a long grading session. These "
    "inconsistencies are especially problematic in high-stakes examinations."
)

add_h("6.2 Problem Statement", level=2)
add_p(
    "The task is to build a system that can automatically assign a numerical score to a long "
    "student answer. The system receives a question, a rubric containing multiple grading "
    "criteria, and the student's answer. For each criterion, the system must decide how many "
    "points to award. The sum of criterion-level scores becomes the predicted total score, "
    "compared against the ground-truth score assigned by a human evaluator. The main challenge "
    "is that answers vary widely in length, vocabulary, and quality — a student may be partially "
    "correct, may use different terminology, or may answer only part of the question."
)

add_h("6.3 Motivation for LLM-Based Approach", level=2)
add_p(
    "Early automated grading approaches relied on keyword matching or surface-level text "
    "similarity, failing when students express correct ideas in unexpected words. Modern "
    "instruction-tuned Large Language Models (LLMs), pre-trained on vast corpora, can understand "
    "semantic meaning and context, making them suitable for rubric-based grading. However, "
    "proprietary LLMs (e.g., GPT-4) are expensive and raise data-privacy concerns for "
    "institutions. This motivates the use of smaller, open-weight models that can be hosted "
    "locally and fine-tuned on task-specific data using parameter-efficient QLoRA."
)

add_h("6.4 Scope", level=2)
add_p(
    "This project covers: (1) preparing a grading dataset of 1,000 records across 10 questions "
    "from computer science and data engineering; (2) fine-tuning Qwen2.5-3B-Instruct using QLoRA; "
    "(3) evaluating three model configurations using six metrics on two separate test sets; and "
    "(4) providing an honest assessment of the results. This is a proof-of-concept research "
    "project and is not claimed to be production-ready."
)
doc.add_page_break()

# ════════════════════════════════════════════════════════════
# 7. THEORETICAL FRAMEWORK
# ════════════════════════════════════════════════════════════
add_h("7. Theoretical Framework of Topic", level=1)

add_h("7.1 Automated Essay and Short-Answer Scoring", level=2)
add_p(
    "Automated Essay Scoring (AES) dates back to Page (1966), who used statistical proxies of "
    "writing quality. Landauer et al. (1998) introduced Latent Semantic Analysis (LSA) for "
    "semantic similarity measurement. Modern AES uses deep learning: BERT-based models "
    "(Devlin et al., 2019) achieve strong performance on the ASAP benchmark. Mayfield & Black "
    "(2020) showed fine-tuned BERT outperforms feature-engineered baselines for short-answer "
    "grading. Sung et al. (2019) demonstrated domain-specific BERT pre-training for short "
    "answers. Sonkar et al. (2024) introduced the RiceChem dataset for automated long-answer "
    "grading using LLMs, showing the growing relevance of this task."
)

add_h("7.2 Large Language Models in Education", level=2)
add_p(
    "Brown et al. (2020) showed GPT-3 can perform grading tasks in a few-shot setting. "
    "Kortemeyer (2023) found GPT-4 performs comparably to human teaching assistants for "
    "structured physics exam questions. These studies confirm the potential of LLMs for "
    "grading, but also highlight the need for cost-effective, privacy-preserving alternatives "
    "that can run locally without sending student data to external servers."
)

add_h("7.3 Low-Rank Adaptation (LoRA)", level=2)
add_p(
    "Hu et al. (2022) introduced LoRA, which inserts trainable low-rank matrices into the "
    "attention layers of pre-trained Transformer models. For a weight matrix W of shape d x k, "
    "LoRA decomposes the update as: delta_W = A * B, where A is (d x r) and B is (r x k), "
    "with rank r much smaller than d and k. Only A and B are trained while W remains frozen. "
    "This reduces trainable parameters dramatically while achieving performance close to full "
    "fine-tuning on many tasks."
)

add_h("7.4 QLoRA (Quantised Low-Rank Adaptation)", level=2)
add_p(
    "Dettmers et al. (2023) extended LoRA with 4-bit NF4 (Normal Float 4) quantisation to "
    "create QLoRA. NF4 quantisation maps model weights to a 4-bit normal float format, reducing "
    "GPU memory by approximately 4x compared to FP16. Double quantisation further compresses "
    "the quantisation constants themselves. QLoRA enables fine-tuning of 3B+ parameter models "
    "on a single consumer GPU (8-24 GB VRAM), making it ideal for academic research without "
    "access to large GPU clusters. The trainable LoRA parameters are computed and stored in "
    "float16, while the frozen base model weights remain in 4-bit NF4."
)

add_h("7.5 Qwen2.5-3B-Instruct Base Model", level=2)
add_p(
    "Qwen2.5-3B-Instruct (Qwen Team, 2024) is a 3-billion-parameter instruction-tuned language "
    "model from Alibaba Cloud. Pre-trained on a large multilingual corpus and further aligned "
    "with supervised fine-tuning (SFT) and RLHF for instruction following. It uses grouped-query "
    "attention, has a vocabulary of 151,643 tokens, and supports long context windows. "
    "Its structured output capability makes it well-suited for rubric-based grading, where the "
    "model must produce exactly one score line per rubric criterion in a fixed format."
)

add_h("7.6 Evaluation Metrics Used", level=2)
metric_rows = [
    ("Mean Absolute Error (MAE)",
     "Average point deviation between predicted and actual scores. Lower is better."),
    ("Root Mean Square Error (RMSE)",
     "Like MAE but penalises large errors more due to squaring. Lower is better."),
    ("Exact Score Accuracy",
     "Fraction of predictions matching ground truth exactly. Higher is better."),
    ("Rubric-Level Accuracy",
     "Fraction of per-criterion scores matching human scores exactly. Higher is better."),
    ("Pearson Correlation (r)",
     "Linear correlation between predicted and actual totals [-1, 1]. Higher is better."),
    ("Spearman Correlation (rho)",
     "Rank correlation; robust to outliers. Higher is better."),
]
make_table(["Metric", "Interpretation"], metric_rows)
doc.add_page_break()

# ════════════════════════════════════════════════════════════
# 8. PROFILE OF ORGANISATION
# ════════════════════════════════════════════════════════════
add_h("8. Profile of Organization / Research Lab", level=1)

add_h("8.1 IIEST Shibpur", level=2)
add_p(
    "The Indian Institute of Engineering Science and Technology (IIEST), Shibpur is one of the "
    "oldest technical institutions in Asia, established in 1856. Located in Shibpur, Howrah, "
    "West Bengal, it is an Institute of National Importance under the Ministry of Education, "
    "Government of India. IIEST Shibpur offers B.Tech, M.Tech, and Ph.D. programmes across "
    "engineering, science, and technology disciplines. The institute has a strong tradition of "
    "applied research and industry collaboration."
)

add_h("8.2 Department of Computer Science and Technology", level=2)
add_p(
    "The Department of Computer Science and Technology at IIEST Shibpur conducts research across "
    "machine learning, natural language processing, computer vision, data mining, and distributed "
    "systems. Active research groups work on applied AI, neural network architectures, and "
    "educational technology. The department provides access to GPU-enabled computational servers "
    "for deep learning research, which were used for model training and evaluation in this project."
)

add_h("8.3 Research Supervisor", level=2)
add_p(
    "Prof. Asit Kumar Das is a faculty member in the Department of Computer Science and Technology "
    "at IIEST Shibpur. His research interests include machine learning, data mining, pattern "
    "recognition, and artificial intelligence. Under his supervision, this project was designed, "
    "implemented, and evaluated during the summer internship period of May to July 2026."
)
doc.add_page_break()

# ════════════════════════════════════════════════════════════
# 9. METHODOLOGY
# ════════════════════════════════════════════════════════════
add_h("9. Methodology Followed", level=1)

add_h("9.1 Dataset Preparation", level=2)
add_p(
    "The dataset consists of 1,000 student answer records covering 10 questions from computer "
    "science and data engineering topics. Each question has 100 records, all with maximum score "
    "of 10 points, evaluated by a rubric of 5, 9, or 10 binary criteria (1 point each). "
    "Answer quality spans four categories: fully correct, partially correct, incorrect, and irrelevant."
)
q_rows = [
    ("Q01", "Gradient Descent Variants",          "Batch, SGD, and Mini-Batch gradient descent"),
    ("Q02", "Overfitting Prevention",              "Methods of reducing overfitting in deep learning"),
    ("Q03", "Activation Functions",               "Three activation functions with pros and cons"),
    ("Q04", "Transformer Attention",              "Multi-head self and cross attention"),
    ("Q05", "Apache Kafka",                       "Kafka architecture and components"),
    ("Q06", "Hadoop Architecture",                "HDFS, YARN, and MapReduce"),
    ("Q07", "Batch Norm & Linear Regression",     "Batch Normalisation + 5 assumptions"),
    ("Q08", "NLP Preprocessing",                  "Five text pre-processing techniques"),
    ("Q09", "Ensemble Methods",                   "Bagging, Boosting, and Random Forest"),
    ("Q10", "Recurrent Neural Networks",          "Why RNN beyond CNN and ANN"),
]
make_table(["ID", "Topic", "Question (abbreviated)"], q_rows)

add_h("9.2 Dataset Split", level=2)
make_table(
    ["Split", "Records", "Per Question", "Purpose"],
    [
        ("Training",               "700",   "70",  "QLoRA fine-tuning"),
        ("Validation",             "150",   "15",  "Monitor training loss"),
        ("Original Held-Out Test", "150",   "15",  "Primary in-distribution evaluation"),
        ("New External Test",      "1,000", "100", "Generalisation evaluation"),
    ]
)

add_h("9.3 Data Format and Prompt Construction", level=2)
add_p(
    "Each record is a JSON object with fields: response_id, question_id, question, student_answer, "
    "rubric (list of criterion objects with points and points_awarded), final_score, max_score, "
    "and answer_quality. For training and inference, a structured system/user/assistant chat "
    "prompt is constructed combining the question, rubric (with max points per criterion), and "
    "student answer. The model is instructed to output exactly one score per criterion with no "
    "explanations. During inference, outputs are parsed by regex and clamped to [0, max_points]."
)

add_h("9.4 QLoRA Fine-Tuning Configuration", level=2)
make_table(
    ["Parameter", "Value"],
    [
        ("Base model",                      "Qwen/Qwen2.5-3B-Instruct"),
        ("Quantisation",                    "4-bit NF4 + Double Quantisation"),
        ("Compute dtype",                   "float16"),
        ("LoRA rank / alpha / dropout",     "16 / 32 / 0.05"),
        ("Target modules",                  "q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj"),
        ("Learning rate",                   "2e-4"),
        ("Training epochs",                 "3"),
        ("Effective batch size",            "8  (1 device x 8 gradient accumulation steps)"),
        ("Max sequence length",             "768 tokens"),
        ("Optimizer",                       "paged_adamw_8bit"),
        ("Warmup steps / weight decay",     "20 / 0.01"),
        ("Inference decoding",              "Greedy (do_sample=False), max_new_tokens=300"),
        ("Training time",                   "Approx. 37.5 minutes"),
        ("Final training loss",             "0.2531"),
        ("Final validation loss",           "0.1006"),
        ("Validation token accuracy",       "97.34%"),
    ]
)

add_h("9.5 Training Algorithm (Summary)", level=2)
for step in [
    "Load Qwen2.5-3B-Instruct; apply 4-bit NF4 quantisation with double quantisation enabled.",
    "Call prepare_model_for_kbit_training(); attach LoRA adapters to 7 target modules; freeze all base model weights.",
    "Convert dataset to JSONL with system/user/assistant format. Max sequence length = 768 tokens.",
    "Train 3 epochs using paged_adamw_8bit, lr=2e-4, warmup_steps=20, effective_batch=8.",
    "Evaluate on validation set after each epoch; save checkpoint if validation loss improves.",
    "Load best checkpoint; save final LoRA adapter weights to disk as final_adapter.",
]:
    add_numbered(step)

add_h("9.6 Inference Algorithm (Summary)", level=2)
for step in [
    "Load base model (NF4) + LoRA adapter via PeftModel; set eval() mode.",
    "Build grading prompt; apply Qwen chat template; tokenise input.",
    "Generate response with greedy decoding (max_new_tokens=300).",
    "Parse 'Criterion N: score' lines with regex; clamp each to [0, rubric_max_points].",
    "Sum criterion scores for predicted total; compare with ground truth.",
]:
    add_numbered(step)

add_h("9.7 Baseline Configurations", level=2)
make_table(
    ["Configuration", "Quantisation", "LoRA Adapter", "Purpose"],
    [
        ("Base FP16",          "float16 (none)", "No",       "Unquantised upper-bound reference"),
        ("Base NF4",           "4-bit NF4",      "No",       "Shows quantisation-only effect"),
        ("Fine-tuned QLoRA",   "4-bit NF4",      "Yes (r=16)", "Primary experiment"),
    ]
)
doc.add_page_break()

# ════════════════════════════════════════════════════════════
# 10. ANALYSIS AND INTERPRETATION
# ════════════════════════════════════════════════════════════
add_h("10. Analysis and Interpretation of Research Project", level=1)

add_h("10.1 Results: Original Held-Out Test Set (n = 150)", level=2)
add_p("15 records per question drawn from the same distribution as training (random seed = 42).")
make_table(
    ["Model", "MAE", "RMSE", "Exact Acc.", "Rubric Acc.", "Pearson r", "Spearman rho"],
    [
        ("Base FP16",          "2.2333", "3.3397", "23.3%", "72.6%", "0.6785", "0.6534"),
        ("Base NF4",           "2.4600", "3.5637", "22.7%", "71.7%", "0.6369", "0.6146"),
        ("Fine-tuned (QLoRA)", "1.3467", "2.4413", "45.3%", "85.4%", "0.8186", "0.8527"),
    ]
)
add_p(
    "The fine-tuned QLoRA model leads on all metrics. Compared to Base NF4 (same quantised base, "
    "most controlled comparison): MAE improved by 45.3% (2.46 to 1.35), exact accuracy doubled "
    "(22.7% to 45.3%), rubric accuracy rose from 71.7% to 85.4%, and Pearson r improved from "
    "0.637 to 0.819. NF4 quantisation alone slightly degrades the untuned baseline, but QLoRA "
    "fine-tuning on top of NF4 more than compensates, even surpassing the FP16 baseline."
)

add_h("10.2 Results: New External Test Set (n = 1,000)", level=2)
add_p("100 records per question, collected independently. Uniform 10-criterion rubric structure.")
make_table(
    ["Model", "MAE", "RMSE", "Exact Acc.", "Rubric Acc.", "Pearson r", "Spearman rho"],
    [
        ("Base FP16",          "2.3880", "3.2168", "22.6%", "73.8%", "0.7272", "0.7284"),
        ("Base NF4",           "2.9920", "3.9764", "21.1%", "69.5%", "0.6666", "0.6632"),
        ("Fine-tuned (QLoRA)", "1.3790", "2.2034", "38.9%", "83.8%", "0.8570", "0.8534"),
    ]
)
add_p(
    "The fine-tuned model again outperforms both baselines by a large margin. "
    "Pearson r improves to 0.857 on the new test (vs 0.819 on original), indicating good "
    "generalisation. Exact accuracy drops from 45.3% to 38.9% — expected for a larger, "
    "more uniformly distributed test set. Base NF4 degrades more on the new test "
    "(MAE: 2.46 to 2.99) than Base FP16 (2.23 to 2.39), confirming quantisation alone "
    "reduces robustness to distribution shift."
)

add_h("10.3 Cross-Test Stability of the Fine-Tuned Model", level=2)
make_table(
    ["Metric", "Original Test (n=150)", "New External Test (n=1,000)", "Change"],
    [
        ("MAE",                "1.3467", "1.3790", "+0.032  (stable)"),
        ("Pearson r",          "0.8186", "0.8570", "+0.038  (improved)"),
        ("Rubric Accuracy",    "85.4%",  "83.8%",  "-1.6%   (stable)"),
        ("Exact Score Acc.",   "45.3%",  "38.9%",  "-6.4%"),
    ]
)
add_p(
    "The near-constant MAE and improving Pearson r across both tests confirms that the "
    "fine-tuned model generalised well and did not overfit the original training distribution."
)

add_h("10.4 Error Analysis (Fine-Tuned Model, Original Test)", level=2)
for item in [
    "68 / 150 predictions (45.3%) are exactly correct.",
    "115 / 150 predictions (76.7%) are within +/-1 point of the ground truth.",
    "125 / 150 predictions (83.3%) are within +/-2 points.",
    "21 predictions (14.0%) have an error greater than 3 points.",
    "Worst cases: errors of 8-9 points — model either awards full marks to a weak answer "
    "or completely fails to score a strong one, likely due to unusual answer styles.",
    "6 records had at least one missing criterion output (counted as 0).",
]:
    add_bullet(item)
doc.add_page_break()

# ════════════════════════════════════════════════════════════
# 11. CONCLUSION & SUGGESTION
# ════════════════════════════════════════════════════════════
add_h("11. Conclusion & Suggestion of Research Project", level=1)

add_h("11.1 Conclusion", level=2)
add_p(
    "This project developed a rubric-based automated grading system for long descriptive student "
    "answers using QLoRA fine-tuning of Qwen2.5-3B-Instruct. The fine-tuned model achieves "
    "MAE=1.35, Pearson r=0.82, and rubric-level accuracy=85.4% on the original held-out test "
    "(n=150), and similar performance (MAE=1.38, Pearson r=0.86) on a new external test "
    "(n=1,000). It outperforms both baselines on all six metrics, confirming task-specific "
    "fine-tuning is necessary for reliable automated grading."
)
add_p(
    "Generalisation across both test sets is encouraging. However, exact accuracy below 50% "
    "and occasional extreme prediction errors mean human review remains essential. The system "
    "is best treated as a proof-of-concept establishing a viable direction for further "
    "development — not a production-ready grading tool."
)

add_h("11.2 Suggestions for Future Work", level=2)
for sug in [
    "Expand training data to 5,000+ records across multiple subjects and question types.",
    "Evaluate larger base models (7B, 14B parameters) to assess the impact of model scale.",
    "Add score calibration and outlier detection to suppress extreme prediction errors automatically.",
    "Introduce a confidence estimate to flag uncertain predictions for human review.",
    "Measure human inter-rater agreement to establish a realistic performance ceiling.",
    "Explore DPO or RLHF to better align model scoring with human evaluator preferences.",
    "Test cross-subject generalisation: train on some domains, evaluate on entirely unseen domains.",
    "Compare QLoRA with other PEFT methods (prefix tuning, prompt tuning, IA3).",
]:
    add_numbered(sug)
doc.add_page_break()

# ════════════════════════════════════════════════════════════
# 12. RESULTS ACHIEVED
# ════════════════════════════════════════════════════════════
add_h("12. Result(s) Achieved", level=1)

add_p("Key results achieved during this research project:", aft=4)
for result in [
    "Successfully built an end-to-end automated grading pipeline: dataset preparation, "
    "prompt construction, QLoRA fine-tuning, inference, score parsing, and evaluation.",

    "Fine-tuned Qwen2.5-3B-Instruct in approximately 37.5 minutes with training loss=0.2531, "
    "validation loss=0.1006, and validation token accuracy=97.34%.",

    "Original held-out test (n=150): MAE=1.35, RMSE=2.44, Exact Score Accuracy=45.3%, "
    "Rubric Accuracy=85.4%, Pearson r=0.819, Spearman rho=0.853.",

    "New external test (n=1,000): MAE=1.38, RMSE=2.20, Exact Score Accuracy=38.9%, "
    "Rubric Accuracy=83.8%, Pearson r=0.857, Spearman rho=0.853.",

    "MAE reduced by 45.3% and exact accuracy doubled compared to the untuned NF4 baseline.",

    "76.7% of predictions fall within 1 point of ground truth on the original test.",

    "Stable generalisation confirmed: MAE changes by only 0.032 between original and new test.",

    "All code, evaluation scripts, comparison charts, and trained LoRA adapter produced and documented.",
]:
    add_bullet(result)
doc.add_page_break()

# ════════════════════════════════════════════════════════════
# 13. REFERENCES
# ════════════════════════════════════════════════════════════
add_h("13. References", level=1)
references = [
    "[1] Shashank Sonkar, Kangqi Ni, Lesa Tran Lu, Kristi Kincaid, John S. Hutchinson, "
    "Richard G. Baraniuk (2024). Automated Long Answer Grading with RiceChem Dataset.",

    "[2] Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). "
    "QLoRA: Efficient Finetuning of Quantized LLMs. arXiv preprint arXiv:2305.14314.",

    "[3] Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. "
    "(2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022. arXiv:2106.09685.",

    "[4] Qwen Team. (2024). Qwen2.5 Technical Report. Alibaba Cloud. "
    "Available: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct",

    "[5] Shermis, M. D., & Burstein, J. (Eds.). (2013). "
    "Handbook of Automated Essay Evaluation: Current Applications and New Directions. Routledge.",

    "[6] Mayfield, E., & Black, A. W. (2020). Should You Fine-Tune BERT for Automated Essay Scoring? "
    "15th Workshop on Innovative Use of NLP for Building Educational Applications, ACL 2020.",

    "[7] Sung, C., Dhamecha, T., Saha, S., Ma, T., Reddy, V., & Arora, R. (2019). "
    "Pre-Training BERT on Domain Resources for Short Answer Grading. EMNLP-IJCNLP 2019.",

    "[8] Brown, T. et al. (2020). Language Models are Few-Shot Learners. "
    "Advances in Neural Information Processing Systems 33 (NeurIPS 2020). arXiv:2005.14165.",

    "[9] Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). "
    "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL-HLT 2019.",

    "[10] Kortemeyer, G. (2023). Could an artificial intelligence agent pass an introductory physics course? "
    "Physical Review Physics Education Research, 19(1), 010132.",
]
for ref in references:
    rp = doc.add_paragraph(ref)
    rp.paragraph_format.first_line_indent = Pt(-18)
    rp.paragraph_format.left_indent       = Pt(18)
    for r in rp.runs:
        r.font.size = Pt(11)
    sp(rp, 0, 4)
doc.add_page_break()

# ════════════════════════════════════════════════════════════
# 14. DECLARATION
# ════════════════════════════════════════════════════════════
add_h("14. Declaration by the Scholar", level=1)
add_p(
    "I / We,  Ayan Kumar Batabyal  and  Anirban Nayak  "
    "hereby declare that the details / facts mentioned above are true to the best of "
    "my / our knowledge and I / we solely be held responsible in case of any "
    "discrepancies found in the details mentioned above.",
    aft=12
)
add_p(
    "The project titled 'Automated Long-Answer Grading using QLoRA Fine-Tuned "
    "Qwen2.5-3B-Instruct' was carried out by us independently during the summer "
    "internship period of May to July 2026 under the supervision of "
    "Prof. Asit Kumar Das, IIEST Shibpur. All results reported are genuine and no "
    "fabrication or plagiarism has been involved.",
    aft=24
)
add_p("Date:   ____________________", aft=4)
add_p("Place:  IIEST, Shibpur, West Bengal", aft=30)
add_p("Signature of Scholar(s):", bold=True, aft=12)
add_p("1.  Ayan Kumar Batabyal        _______________________", aft=10)
add_p("2.  Anirban Nayak               _______________________", aft=0)

# ════════════════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════════════════
output = "INSPIRE_Project_Report.docx"
doc.save(output)
print("Saved:", output)
