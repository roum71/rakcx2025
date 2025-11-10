# -*- coding: utf-8 -*-
# Arabic CX Dashboard (3 Dimensions) — Streamlit
# Files expected in the same folder:
#   - MUN.csv
#   - Digital_Data_tables2.xlsx
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

# شعار أعلى الصفحة
LOGO_URL = "https://raw.githubusercontent.com/roum71/rakcx2025/main/assets/mini_header2.png"
st.markdown(f"""
    <div style="text-align:center; margin-top:-40px;">
        <img src="{LOGO_URL}" alt="Logo" style="width:950px; max-width:95%; height:auto;">
    </div>
    <hr style="margin-top:20px; margin-bottom:10px;">
""", unsafe_allow_html=True)

# اتجاه عربي وخط جميل
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
    df = pd.read_csv("MUN.csv", encoding="utf-8", low_memory=False)
    df.columns = [c.strip().upper() for c in df.columns]
    df.columns = [c.replace('DIM', 'Dim') for c in df.columns]

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
    if mx <= 5:
        return ((vals - 1) / 4 * 100).mean()
    elif mx <= 10:
        return ((vals - 1) / 9 * 100).mean()
    else:
        return vals.mean()

def detect_nps(df: pd.DataFrame):
    cand_cols = [c for c in df.columns if ("NPS" in c.upper()) or ("RECOMMEND" in c.upper())]
    if not cand_cols:
        return np.nan, 0, 0, 0, None
    col = cand_cols[0]
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(s) == 0:
        return np.nan, 0, 0, 0, col
    promoters = (s >= 9).sum()
    passives = ((s >= 7) & (s <= 8)).sum()
    detract = (s <= 6).sum()
    total = len(s)
    promoters_pct = promoters / total * 100
    detract_pct = detract / total * 100
    nps = promoters_pct - detract_pct
    return nps, promoters_pct, passives, detract, col

def autodetect_metric_cols(df: pd.DataFrame):
    csat = next((c for c in df.columns if "CSAT" in c.upper()), None)
    ces = next((c for c in df.columns if "FEES" in c.upper()), None)
    nps = next((c for c in df.columns if "NPS" in c.upper()), None)
    return csat, ces, nps

df, lookup_catalog = load_data()

# =========================================================
# الفلاتر
# =========================================================
st.sidebar.header("🎛️ الفلاتر")
df_filtered = df.copy()
common_keys = ["LANGUAGE", "SERVICE", "AGE", "PERIOD", "CHANNEL"]
candidate_filter_cols = [c for c in df.columns if any(k in c.upper() for k in common_keys)]

def apply_lookup(column_name: str, s: pd.Series) -> pd.Series:
    key = column_name.strip().upper()
    match_key = next((k for k in lookup_catalog.keys() if key in k or k in key), None)
    if not match_key: return s
    tbl = lookup_catalog[match_key].copy()
    tbl.columns = [str(c).strip().upper() for c in tbl.columns]
    if len(tbl.columns) < 2: return s
    map_dict = dict(zip(tbl.iloc[:,0].astype(str), tbl.iloc[:,1].astype(str)))
    return s.astype(str).map(map_dict).fillna(s)

df_filtered_display = df_filtered.copy()
for col in candidate_filter_cols:
    df_filtered_display[col] = apply_lookup(col, df_filtered[col])

with st.sidebar.expander("تطبيق/إزالة الفلاتر"):
    applied_filters = {}
    for col in candidate_filter_cols:
        df_filtered[col] = apply_lookup(col, df_filtered[col])
        options = sorted(df_filtered_display[col].dropna().unique().tolist())
        sel = st.multiselect(f"{col}", options, default=options)
        applied_filters[col] = sel
for col, selected in applied_filters.items():
    df_filtered = df_filtered[df_filtered[col].isin(selected)]
df_view = df_filtered.copy()

# =========================================================
# التبويبات
# =========================================================
tab_data, tab_sample, tab_kpis, tab_dimensions, tab_services, tab_unsat, tab_pareto = st.tabs([
    "📁 البيانات",
    "📈 توزيع العينة",
    "📊 المؤشرات",
    "🧩 الأبعاد",
    "📋 الخدمات",
    "💬 عدم الرضا (Pareto)",
    "💬 الملاحظات العامة (Pareto)"
])

# =========================================================
# تبويب البيانات
# =========================================================
with tab_data:
    st.subheader("📁 البيانات (بعد الفلترة)")
    st.dataframe(df_view, use_container_width=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_view.to_excel(writer, index=False)
    st.download_button("📥 تنزيل البيانات (Excel)", data=buf.getvalue(),
                       file_name=f"Filtered_Data_{ts}.xlsx")

# =========================================================
# تبويب توزيع العينة
# =========================================================
with tab_sample:
    st.subheader("📈 توزيع العينة")
    total = len(df_view)
    st.markdown(f"### 🧮 إجمالي الردود: {total:,}")
    chart_type = st.radio("📊 نوع الرسم", ["مخطط أعمدة", "مخطط دائري"], index=0, horizontal=True)
    for col in candidate_filter_cols:
        counts = df_view[col].value_counts().reset_index()
        counts.columns = [col, "Count"]
        counts["Percentage"] = counts["Count"]/counts["Count"].sum()*100
        if chart_type == "مخطط أعمدة":
            fig = px.bar(counts, x=col, y="Count", text="Count", color=col,
                         color_discrete_sequence=PASTEL)
        else:
            fig = px.pie(counts, names=col, values="Count", hole=0.3)
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 💬 تبويب تحليل أسباب عدم الرضا (Pareto)
# =========================================================
with tab_unsat:
    st.subheader("💬 تحليل أسباب عدم الرضا في الخدمات الرقمية (Pareto)")

    unsat_col = next((c for c in df_view.columns if "MOST_UNSAT" in c.upper()), None)
    if not unsat_col:
        st.warning("⚠️ لم يتم العثور على العمود Most_Unsat.")
    else:
        data_unsat = df_view[[unsat_col]].copy()
        data_unsat.columns = ["Comment"]
        data_unsat["Comment"] = data_unsat["Comment"].astype(str).str.strip()

        exclude_terms = ["", " ", "لا يوجد", "لايوجد", "لاشيء", "لا شيء",
                         "none", "no", "nothing", "nil", "جيد", "ممتاز", "ok", "تمام", "great"]
        data_unsat = data_unsat[~data_unsat["Comment"].str.lower().isin([t.lower() for t in exclude_terms])]
        data_unsat = data_unsat[data_unsat["Comment"].apply(lambda x: len(x.split()) >= 2)]

        if data_unsat.empty:
            st.info("لا توجد ملاحظات نصية كافية بعد التنظيف.")
        else:
            themes = {
                "السرعة / الأداء": ["بطء", "تأخير", "انتظار", "delay", "slow", "زمن", "وقت"],
                "التطبيق / المنصة": ["تطبيق", "app", "منصة", "system", "موقع", "بوابة", "صفحة"],
                "الإجراءات / الخطوات": ["إجراء", "اجراء", "عملية", "process", "خطوات", "مراحل"],
                "الرسوم / الدفع": ["رسوم", "دفع", "fee", "تكلفة", "سداد", "pay"],
                "التواصل / الدعم الفني": ["رد", "تواصل", "اتصال", "support", "response", "مساعدة"],
                "الوضوح / المعلومات": ["معلومة", "إيضاح", "clarity", "instructions", "بيانات", "شرح"],
                "الأمان / الدخول": ["كلمة مرور", "دخول", "login", "تحقق", "أمان"]
            }

            def classify_text(txt):
                t = txt.lower()
                for theme, keys in themes.items():
                    if any(k in t for k in keys):
                        return theme
                return "غير مصنّف"

            data_unsat["المحور"] = data_unsat["Comment"].apply(classify_text)
            data_unsat = data_unsat[data_unsat["المحور"] != "غير مصنّف"]

            summary = data_unsat.groupby("المحور").agg({
                "Comment": lambda x: " / ".join(x.tolist())
            }).reset_index()
            summary["عدد الملاحظات"] = summary["Comment"].apply(lambda x: len(x.split("/")))
            summary = summary.sort_values("عدد الملاحظات", ascending=False).reset_index(drop=True)
            summary["النسبة (%)"] = summary["عدد الملاحظات"]/summary["عدد الملاحظات"].sum()*100
            summary["النسبة التراكمية (%)"] = summary["النسبة (%)"].cumsum()
            summary["اللون"] = np.where(summary["النسبة التراكمية (%)"] <= 80, "#E74C3C", "#BDC3C7")
            if not summary[summary["النسبة التراكمية (%)"] > 80].empty:
                first_above = summary[summary["النسبة التراكمية (%)"] > 80].index[0]
                summary.loc[first_above, "اللون"] = "#E74C3C"

            st.dataframe(summary[["المحور","عدد الملاحظات","النسبة (%)","النسبة التراكمية (%)","Comment"]]
                         .rename(columns={"Comment":"التعليقات (مجمعة)"}).style.format({"النسبة (%)":"{:.1f}%", "النسبة التراكمية (%)":"{:.1f}%"}),
                         use_container_width=True, hide_index=True)

            fig = go.Figure()
            fig.add_bar(x=summary["المحور"], y=summary["عدد الملاحظات"], marker_color=summary["اللون"], name="عدد الملاحظات")
            fig.add_scatter(x=summary["المحور"], y=summary["النسبة التراكمية (%)"], yaxis="y2",
                            name="النسبة التراكمية (%)", mode="lines+markers+text",
                            text=[f"{v:.1f}%" for v in summary["النسبة التراكمية (%)"]],
                            textposition="top center", line=dict(color="#2E86DE", width=3))
            fig.update_layout(
                title="📊 تحليل Pareto لأسباب عدم الرضا في الخدمات الرقمية",
                xaxis=dict(title="المحور", tickangle=-15),
                yaxis=dict(title="عدد الملاحظات"),
                yaxis2=dict(title="النسبة التراكمية (%)", overlaying="y", side="right", range=[0,110]),
                height=600, bargap=0.3, legend=dict(orientation="h", y=-0.2)
            )
            st.plotly_chart(fig, use_container_width=True)

# =========================================================
# تحسينات شكلية
# =========================================================
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer, [data-testid="stFooter"] {opacity: 0.03 !important; height: 1px !important; overflow: hidden !important;}
    </style>
""", unsafe_allow_html=True)
