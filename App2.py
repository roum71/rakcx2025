
# -*- coding: utf-8 -*-
# Arabic CX Dashboard (3 Dimensions) — Streamlit
# Files expected in the same folder:
#   - MUN.csv                          ← raw survey data
#   - Digital_Data_tables.xlsx         ← lookup/metadata tables
#
# Run:
#   streamlit run Arabic_Dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io, re
from datetime import datetime
from pathlib import Path

# =========================================================
# إعداد الصفحة + اتجاه RTL
# =========================================================
st.set_page_config(page_title="لوحة تجربة المتعاملين — نسخة عربية", layout="wide")
PASTEL = px.colors.qualitative.Pastel

# شعار أعلى الصفحة (استبدل بالرابط المناسب إذا رغبت)
LOGO_URL = "https://raw.githubusercontent.com/roum71/rakcx2025/main/assets/mini_header2.png"
st.markdown(f"""
    <div style="text-align:center; margin-top:-40px;">
        <img src="{LOGO_URL}" alt="Logo" style="width:950px; max-width:95%; height:auto;">
    </div>
    <hr style="margin-top:20px; margin-bottom:10px;">
""", unsafe_allow_html=True)

# اتجاه عربي وخط مناسب
st.markdown("""
    <style>
        html, body, [class*="css"] {direction:rtl; text-align:right; font-family:"Tajawal","Cairo","Segoe UI";}
        .stTabs [data-baseweb="tab-list"] {flex-direction: row-reverse;}
        .stDownloadButton, .stButton > button {font-weight:600;}
    </style>
""", unsafe_allow_html=True)

# =========================================================
# تحميل البيانات
# =========================================================
@st.cache_data(show_spinner=False)
def load_data():
    # البيانات الرئيسية
    df = pd.read_csv("MUN.csv", encoding="utf-8", low_memory=False)
    # الجداول الوصفية
    lookup_catalog = {}
    xls_path = Path("Digital_Data_tables2.xlsx")
    if xls_path.exists():
        xls = pd.ExcelFile(xls_path)
        for sheet in xls.sheet_names:
            tbl = pd.read_excel(xls, sheet_name=sheet)
            tbl.columns = [str(c).strip().upper() for c in tbl.columns]
            lookup_catalog[sheet.strip().upper()] = tbl
    return df, lookup_catalog

def series_to_percent(vals: pd.Series):
    vals = pd.to_numeric(vals, errors="coerce").dropna()
    if len(vals) == 0:
        return np.nan
    mx = vals.max()
    if mx <= 5:   # سلم 1-5
        return ((vals - 1) / 4 * 100).mean()
    elif mx <= 10:  # سلم 1-10
        return ((vals - 1) / 9 * 100).mean()
    else:        # بيانات جاهزة كنسب
        return vals.mean()

def detect_nps(df: pd.DataFrame):
    cand_cols = [c for c in df.columns if ("NPS" in c.upper()) or ("RECOMMEND" in c.upper()) or ("NETPROMOTER" in c.upper())]
    if not cand_cols:
        return np.nan, 0, 0, 0, None
    col = cand_cols[0]
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(s) == 0:
        return np.nan, 0, 0, 0, col
    promoters = (s >= 9).sum()
    passives  = ((s >= 7) & (s <= 8)).sum()
    detract   = (s <= 6).sum()
    total     = len(s)
    promoters_pct = promoters / total * 100
    passives_pct  = passives  / total * 100
    detract_pct   = detract   / total * 100
    nps = promoters_pct - detract_pct
    return nps, promoters_pct, passives_pct, detract_pct, col

def autodetect_metric_cols(df: pd.DataFrame):
    # نحاول التعرف على أعمدة CSAT و CES (قد تكون Dim6.1/Dim6.2 أو CSAT/CES أو FEES)
    cols_upper = {c.upper(): c for c in df.columns}
    # CSAT
    csat_candidates = [c for c in df.columns if "CSAT" in c.upper()] 
                      
    csat_col = csat_candidates[0] if csat_candidates else None

    #  Fees
    ces_candidates = [c for c in df.columns if "FEES" in c.upper()]
    ces_col = ces_candidates[0] if ces_candidates else None

    # NPS
    nps_candidates = [c for c in df.columns if "NPS" in c.upper()] 
    nps_col = nps_candidates[0] if nps_candidates else None

    return csat_col, ces_col, nps_col

df, lookup_catalog = load_data()


st.sidebar.header("🎛️ الفلاتر")
# نحاول تطبيق ترجمة للأبعاد/المتغيرات باستخدام جداول الـ lookup إذا وجدت
df_filtered = df.copy()

# سنعرض فلاتر لأكثر الحقول شيوعًا؛ ويمكن التوسع تلقائيًا إذا وُجدت جداول مطابقة في الـ lookup
candidate_filter_cols = []
# أبعاد ديموغرافية أو وصفية شائعة
common_keys = ["Language", "SERVICE", "AGE", "PERIOD", "CHANNEL"]
candidate_filter_cols = [c for c in df.columns if any(k in c.upper() for k in common_keys)]

# وظيفة لتطبيق جدول lookup إذا توفّر باسم العمود

# وظيفة لتطبيق جدول lookup (تربط تلقائيًا بين الأكواد والأسماء العربية)
def apply_lookup(column_name: str, s: pd.Series) -> pd.Series:
    key = column_name.strip().upper()
    # نحاول إيجاد جدول مطابق جزئياً في ملفات الوصف
    match_key = next((k for k in lookup_catalog.keys() if key in k or k in key), None)
    if not match_key:
        return s

    tbl = lookup_catalog[match_key].copy()
    tbl.columns = [str(c).strip().upper() for c in tbl.columns]
    if len(tbl.columns) < 2:
        return s

    code_col = tbl.columns[0]
    name_col = tbl.columns[1]
    map_dict = dict(zip(tbl[code_col].astype(str), tbl[name_col].astype(str)))
    return s.astype(str).map(map_dict).fillna(s)

# نُحضّر نسخة مترجمة للعرض في الفلاتر
df_filtered_display = df_filtered.copy()
for col in candidate_filter_cols:
    df_filtered_display[col] = apply_lookup(col, df_filtered[col])

with st.sidebar.expander("تطبيق/إزالة الفلاتر"):
    applied_filters = {}
    for col in candidate_filter_cols:
        # طبّق الترجمة العربية إن وُجدت
        df_filtered[col] = apply_lookup(col, df_filtered[col])
        options = df_filtered_display[col].dropna().unique().tolist()
        options_sorted = sorted(options, key=lambda x: str(x))
        default = options_sorted  # افتراضيًا: الكل
        sel = st.multiselect(f"{col}", options_sorted, default=default)
        applied_filters[col] = sel

# تطبيق الفلاتر
for col, selected in applied_filters.items():
    if selected:
        df_filtered = df_filtered[df_filtered[col].isin(selected)]

# البيانات النهائية للعرض
df_view = df_filtered.copy()

# =========================================================
# التبويبات
# =========================================================
tab_data, tab_sample, tab_kpis, tab_dimensions, tab_services, tab_pareto = st.tabs([
    "📁 البيانات",
    "📈 توزيع العينة",
    "📊 المؤشرات",
    "🧩 الأبعاد",
    "📋 الخدمات",
    "💬 الملاحظات (Pareto)"
])

# =========================================================
# تبويب البيانات + تنزيل
# =========================================================
with tab_data:
    st.subheader("📁 البيانات (بعد الفلترة)")
    st.dataframe(df_view, use_container_width=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_view.to_excel(writer, index=False, sheet_name="Filtered_Data")
    st.download_button("📥 تنزيل البيانات (Excel)", data=buf.getvalue(),
                       file_name=f"Filtered_Data_{ts}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# =========================================================
# تبويب توزيع العينة
# =========================================================
with tab_sample:
    st.subheader("📈 توزيع العينة")
    total = len(df_view)
    st.markdown(f"### 🧮 إجمالي الردود: {total:,}")
    chart_type = st.radio("📊 نوع الرسم", ["مخطط أعمدة", "مخطط دائري"], index=0, horizontal=True)

    for col in candidate_filter_cols:
        if col not in df_view.columns:
            continue
        counts = df_view[col].value_counts(dropna=True).reset_index()
        counts.columns = [col, "Count"]
        if counts.empty:
            continue
        counts["Percentage"] = counts["Count"] / counts["Count"].sum() * 100

        if chart_type == "مخطط أعمدة":
            fig = px.bar(counts, x=col, y="Count", text="Count", color=col,
                         color_discrete_sequence=PASTEL, title=f"توزيع — {col}")
            fig.update_traces(textposition="outside")
            fig.update_layout(xaxis_title="الفئة", yaxis_title="العدد")
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.pie(counts, names=col, values="Count", hole=0.3,
                         color=col, color_discrete_sequence=PASTEL,
                         title=f"التوزيع النسبي — {col}")
            fig.update_traces(textposition="inside",
                              texttemplate="%{label}<br>%{percent:.1%}")
            st.plotly_chart(fig, use_container_width=True)

# =========================================================
# تبويب المؤشرات (CSAT / CES / NPS)
# =========================================================
with tab_kpis:
    st.subheader("📊 مؤشرات الأداء الرئيسية")
    csat_col, ces_col, nps_col = autodetect_metric_cols(df_view)

    # حساب CSAT
    csat = series_to_percent(df_view.get(csat_col, pd.Series(dtype=float))) if csat_col else np.nan
    # حساب CES/Value
    ces  = series_to_percent(df_view.get(ces_col,  pd.Series(dtype=float))) if ces_col else np.nan
    # حساب NPS
    nps, p_pct, s_pct, d_pct, nps_col = detect_nps(df_view)

    def color_label(score, metric_type):
        if metric_type in ["CSAT", "CES"]:
            if pd.isna(score):           return "#bdc3c7", "غير متاح"
            if score < 70:               return "#FF6B6B", "ضعيف جدًا"
            elif score < 80:             return "#FFD93D", "بحاجة إلى تحسين"
            elif score < 90:             return "#6BCB77", "جيد"
            else:                        return "#4D96FF", "ممتاز"
        else:  # NPS
            if pd.isna(score):           return "#bdc3c7", "غير متاح"
            if score < 0:                return "#FF6B6B", "ضعيف جدًا"
            elif score < 30:             return "#FFD93D", "ضعيف"
            elif score < 60:             return "#6BCB77", "جيد"
            else:                        return "#4D96FF", "ممتاز"

    def gauge(score, title, metric_type):
        color, label = color_label(score, metric_type)
        axis_range = [0, 100] if metric_type in ["CSAT", "CES"] else [-100, 100]
        steps = (
            [{'range': [0, 70], 'color': '#FF6B6B'},
             {'range': [70, 80], 'color': '#FFD93D'},
             {'range': [80, 90], 'color': '#6BCB77'},
             {'range': [90, 100], 'color': '#4D96FF'}]
            if metric_type in ["CSAT", "CES"]
            else [{'range': [-100, 0], 'color': '#FF6B6B'},
                  {'range': [0, 30], 'color': '#FFD93D'},
                  {'range': [30, 60], 'color': '#6BCB77'},
                  {'range': [60, 100], 'color': '#4D96FF'}]
        )
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=0 if pd.isna(score) else float(score),
            number={'suffix': "٪" if metric_type != "NPS" else ""},
            title={'text': title, 'font': {'size': 18}},
            gauge={
                'axis': {'range': axis_range},
                'bar': {'color': color},
                'steps': steps
            }
        ))
        fig.update_layout(height=300, margin=dict(l=30, r=30, t=60, b=30))
        return fig, label

    c1, c2, c3 = st.columns(3)
    fig1, lab1 = gauge(csat, "السعادة العامة (CSAT)", "CSAT")
    fig2, lab2 = gauge(ces,  "القيمة مقابل الجهد/التكلفة (CES/Value)", "CES")
    fig3, lab3 = gauge(nps,  "صافي نقاط الترويج (NPS)", "NPS")
    c1.plotly_chart(fig1, use_container_width=True)
    c1.markdown(f"**التفسير:** {lab1}")
    if csat_col: c1.caption(f"المصدر: {csat_col}")
    c2.plotly_chart(fig2, use_container_width=True)
    c2.markdown(f"**التفسير:** {lab2}")
    if ces_col: c2.caption(f"المصدر: {ces_col}")
    c3.plotly_chart(fig3, use_container_width=True)
    c3.markdown(f"**التفسير:** {lab3}")
    if nps_col: c3.caption(f"المصدر: {nps_col}")
    c3.markdown(f"المروجون: {p_pct:.1f}% | المحايدون: {s_pct:.1f}% | المعارضون: {d_pct:.1f}%", unsafe_allow_html=True)

# =========================================================
# تبويب الأبعاد (3 أبعاد فقط)
# =========================================================
with tab_dimensions:
    st.subheader("🧩 تحليل الأبعاد")
    # نبحث عن الأعمدة التي تبدأ بـ "DimX." لمستوى السؤال الفرعي
    dim_subcols = [c for c in df_view.columns if re.match(r"Dim\d+\.", str(c).strip())]
    if not dim_subcols:
        st.info("لا توجد أعمدة فرعية للأبعاد (مثل Dim1.1 أو Dim2.3).")
    else:
        # نبني متوسط لكل بعد رئيسي (نفترض الآن ثلاثة أبعاد فعالة Dim1/Dim2/Dim3 — أو الموجود فقط)
        main_dim_map = {}
        for i in range(1, 6):  # نلتقط ما هو موجود حتى لو أقل من 5
            sub = [c for c in df_view.columns if str(c).startswith(f"Dim{i}.")]
            if sub:
                main_dim_map[f"Dim{i}"] = df_view[sub].apply(pd.to_numeric, errors="coerce").mean(axis=1)

        # نكون ملخصًا
        summary = []
        for dim, series in main_dim_map.items():
            score = series_to_percent(series)
            summary.append({"Dimension": dim, "Score": score})
        dims = pd.DataFrame(summary).dropna()
        if dims.empty:
            st.info("لا توجد نتائج كافية للأبعاد.")
        else:
            dims["Order"] = dims["Dimension"].str.extract(r"(\d+)").astype(float)
            dims = dims.sort_values("Order")

# 🔄 استبدال أسماء الأبعاد برموزها العربية من ورقة Question إذا وُجدت
if "QUESTION" in lookup_catalog:
    qtbl = lookup_catalog["QUESTION"].copy()
    qtbl.columns = [str(c).strip().upper() for c in qtbl.columns]
    
    # نحاول تحديد عمود يحتوي أسماء الأكواد (مثل DIM أو CODE)
    code_col = next((c for c in qtbl.columns if "DIM" in c or "CODE" in c), None)
    name_col = next((c for c in qtbl.columns if "ARABIC" in c or "NAME" in c or "LABEL" in c), None)
    
    if code_col and name_col:
        map_dict = dict(zip(qtbl[code_col].astype(str), qtbl[name_col].astype(str)))
        dims["Dimension"] = dims["Dimension"].astype(str).map(map_dict).fillna(dims["Dimension"])

            
            def cat(score):
                if score < 70:  return "🔴 ضعيف"
                elif score < 80: return "🟡 متوسط"
                elif score < 90: return "🟢 جيد"
                else:            return "🔵 ممتاز"
            dims["Category"] = dims["Score"].apply(cat)

            fig = px.bar(
                dims, x="Dimension", y="Score", text="Score", color="Category",
                color_discrete_map={
                    "🔴 ضعيف": "#FF6B6B",
                    "🟡 متوسط": "#FFD93D",
                    "🟢 جيد":   "#6BCB77",
                    "🔵 ممتاز": "#4D96FF"
                },
                title="تحليل متوسط الأبعاد"
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(yaxis=dict(range=[0, 100]), xaxis_title="البعد", yaxis_title="النسبة المئوية (%)")
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                dims[["Dimension", "Score"]].rename(columns={"Dimension":"البعد","Score":"النسبة (%)"})
                .style.format({"النسبة (%)":"{:.1f}%"}),
                use_container_width=True, hide_index=True
            )

# =========================================================
# تبويب الخدمات
# =========================================================
with tab_services:
    st.subheader("📋 تحليل الخدمات")
    if "SERVICE" not in df_view.columns:
        st.warning("⚠️ لا توجد بيانات خاصة بالخدمات (SERVICE).")
    else:
        csat_col, ces_col, _ = autodetect_metric_cols(df_view)
        work = df_view.copy()
        if csat_col:
            work["سعادة (%)"] = (pd.to_numeric(work[csat_col], errors="coerce") - 1) * 25
        if ces_col:
            work["قيمة (%)"] = (pd.to_numeric(work[ces_col], errors="coerce") - 1) * 25

        # NPS لكل خدمة إن وُجد عمود NPS
        nps_cols = [c for c in df_view.columns if "NPS" in c.upper() or "RECOMMEND" in c.upper()]
        if nps_cols:
            work["NPS_VAL"] = pd.to_numeric(work[nps_cols[0]], errors="coerce")
            nps_summary = []
            for svc, g in work.groupby("SERVICE"):
                s = g["NPS_VAL"].dropna()
                if len(s) == 0:
                    nps_summary.append((svc, np.nan))
                    continue
                promoters = (s >= 9).sum()
                detractors = (s <= 6).sum()
                total = len(s)
                nps_value = ((promoters - detractors) / total) * 100
                nps_summary.append((svc, nps_value))
            nps_df = pd.DataFrame(nps_summary, columns=["SERVICE", "NPS (%)"])
        else:
            nps_df = pd.DataFrame(columns=["SERVICE", "NPS (%)"])

        # حساب المتوسط وعدد الردود
        agg_dict = {}
        if "سعادة (%)" in work.columns: agg_dict["سعادة (%)"] = "mean"
        if "قيمة (%)" in work.columns:  agg_dict["قيمة (%)"]  = "mean"
        if csat_col:                   agg_dict[csat_col]    = "count"

        if not agg_dict:
            st.info("لا توجد أعمدة كافية لحساب مؤشرات الخدمة.")
        else:
            summary = work.groupby("SERVICE").agg(agg_dict).reset_index()
            if csat_col and csat_col in summary.columns:
                summary.rename(columns={csat_col: "عدد الردود"}, inplace=True)

            # دمج NPS
            if not nps_df.empty:
                summary = summary.merge(nps_df, on="SERVICE", how="left")

            # ترجمة اسم الخدمة عبر lookup (إن وجد sheet باسم SERVICE)
            if "SERVICE" in lookup_catalog:
                tbl = lookup_catalog["SERVICE"].copy()
                tbl.columns = [str(c).strip().upper() for c in tbl.columns]
                code_col = next((c for c in tbl.columns if "CODE" in c or "SERVICE" in c), None)
                ar_col   = next((c for c in tbl.columns if ("ARABIC" in c) or ("SERVICE2" in c)), None)
                if code_col and ar_col:
                    name_map = dict(zip(tbl[code_col].astype(str), tbl[ar_col].astype(str)))
                    summary["SERVICE"] = summary["SERVICE"].astype(str).map(name_map).fillna(summary["SERVICE"])

            # فلترة إلى خدمات بعدد ردود كافٍ (اختياري: 30)
            if "عدد الردود" in summary.columns:
                summary = summary[summary["عدد الردود"] >= 30]

            # ترتيب
            sort_key = "سعادة (%)" if "سعادة (%)" in summary.columns else ("قيمة (%)" if "قيمة (%)" in summary.columns else None)
            if sort_key:
                summary = summary.sort_values(sort_key, ascending=False)

            # عرض الجدول
            fmt = {}
            if "سعادة (%)" in summary.columns: fmt["سعادة (%)"] = "{:.1f}%"
            if "قيمة (%)"  in summary.columns: fmt["قيمة (%)"]  = "{:.1f}%"
            if "NPS (%)"   in summary.columns: fmt["NPS (%)"]   = "{:.1f}%"
            if "عدد الردود" in summary.columns: fmt["عدد الردود"] = "{:,.0f}"

            st.dataframe(summary.style.format(fmt), use_container_width=True, hide_index=True)

            # رسم مقارنة (سعادة/قيمة)
            if "سعادة (%)" in summary.columns or "قيمة (%)" in summary.columns:
                melted = summary.melt(id_vars=["SERVICE"], value_vars=[v for v in ["سعادة (%)","قيمة (%)"] if v in summary.columns],
                                      var_name="المؤشر", value_name="القيمة")
                fig = px.bar(melted, x="SERVICE", y="القيمة", color="المؤشر", barmode="group",
                             text="القيمة", color_discrete_sequence=PASTEL,
                             title="مقارنة مؤشرات الخدمة")
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig.update_layout(yaxis=dict(range=[0, 100]), xaxis_title="الخدمة", yaxis_title="النسبة (%)")
                st.plotly_chart(fig, use_container_width=True)

# =========================================================
# تبويب الملاحظات النوعية (Pareto)
# =========================================================
with tab_pareto:
    st.subheader("💬 تحليل الملاحظات (Pareto)")
    # نحاول العثور على عمود نصي مناسب
    text_cols = [c for c in df_view.columns if any(k in c.lower() for k in ["comment", "ملاحظ", "unsat", "reason", "feedback"])]
    if not text_cols:
        st.info("⚠️ لا يوجد عمود نصي مناسب لتحليل Pareto.")
    else:
        col = text_cols[0]
        work = df_view[[col]].copy().rename(columns={col: "text"})
        work["text"] = work["text"].astype(str).str.lower()
        work["text"] = work["text"].replace(r"[^\u0600-\u06FFA-Za-z0-9\s]", " ", regex=True)
        work["text"] = work["text"].replace(r"\s+", " ", regex=True).str.strip()

        empty_terms = {"", " ", "لا يوجد", "لايوجد", "لا شيء", "no", "none", "nothing", "جيد", "ممتاز", "ok"}
        work = work[~work["text"].isin(empty_terms)]
        work = work[work["text"].apply(lambda x: len(x.split()) >= 3)]
        if work.empty:
            st.info("لا توجد ملاحظات نصية كافية بعد التنظيف.")
        else:
            themes = {
                "المواقف":    ["موقف", "مواقف", "parking"],
                "الانتظار":   ["انتظار", "بطء", "تأخير", "delay", "slow"],
                "الموظفون":   ["موظف", "تعامل", "سلوك", "staff"],
                "الرسوم":     ["رسوم", "دفع", "fee"],
                "الإجراءات":  ["اجراء", "إجراء", "process", "إنجاز", "انجاز"],
                "المنصة":     ["تطبيق", "منصة", "app", "system"],
                "المكان":     ["مكان", "نظافة", "ازدحام"],
                "التواصل":    ["رد", "تواصل", "اتصال"]
            }
            def classify(t):
                for th, ws in themes.items():
                    if any(w in t for w in ws):
                        return th
                return "أخرى"

            work["Theme"] = work["text"].apply(classify)
            work = work[work["Theme"] != "أخرى"]
            counts = work["Theme"].value_counts().reset_index()
            counts.columns = ["Theme", "Count"]
            if counts.empty:
                st.info("لا توجد محاور قابلة للتحليل.")
            else:
                counts["%"] = counts["Count"] / counts["Count"].sum() * 100
                counts["Cum%"] = counts["%"].cumsum()
                counts["Color"] = np.where(counts["Cum%"] <= 80, "#e74c3c", "#95a5a6")
                if not counts[counts["Cum%"] > 80].empty:
                    first_above_80 = counts[counts["Cum%"] > 80].index[0]
                    counts.loc[first_above_80, "Color"] = "#e74c3c"

                # جدول
                tbl = counts[["Theme","Count","%","Cum%"]].rename(columns={
                    "Theme":"المحور",
                    "Count":"عدد الملاحظات",
                    "%":"النسبة %",
                    "Cum%":"النسبة التراكمية %"
                })
                st.dataframe(tbl.style.format({"النسبة %":"{:.1f}", "النسبة التراكمية %":"{:.1f}"}),
                            use_container_width=True, hide_index=True)

                # رسم Pareto
                fig = go.Figure()
                fig.add_bar(x=counts["Theme"], y=counts["Count"], marker_color=counts["Color"], name="عدد الملاحظات")
                fig.add_scatter(x=counts["Theme"], y=counts["Cum%"], yaxis="y2",
                                name="النسبة التراكمية", mode="lines+markers")
                fig.update_layout(
                    title="تحليل باريتو — المحاور الرئيسية",
                    yaxis=dict(title="عدد الملاحظات"),
                    yaxis2=dict(title="النسبة التراكمية (%)", overlaying="y", side="right"),
                    height=550, bargap=0.25, legend=dict(orientation="h", y=-0.2)
                )
                st.plotly_chart(fig, use_container_width=True)

                # تنزيل النتائج
                pbuf = io.BytesIO()
                with pd.ExcelWriter(pbuf, engine="openpyxl") as writer:
                    tbl.to_excel(writer, index=False, sheet_name="Pareto")
                st.download_button("📥 تنزيل نتائج Pareto (Excel)", data=pbuf.getvalue(),
                                   file_name=f"Pareto_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# =========================================================
# تحسينات شكلية
# =========================================================
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer, [data-testid="stFooter"] {opacity: 0.03 !important; height: 1px !important; overflow: hidden !important;}
    </style>
""", unsafe_allow_html=True)
