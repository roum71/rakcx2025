# -*- coding: utf-8 -*-
# Arabic CX Dashboard (3 Dimensions) — Streamlit
# Files required in the same folder:
#   - MUN.csv
#   - Digital_Data_tables2.xlsx
#
# Run using:
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
# إعداد الصفحة + الاتجاه العربي
# =========================================================
st.set_page_config(page_title="لوحة تجربة المتعاملين — نسخة عربية", layout="wide")
PASTEL = px.colors.qualitative.Pastel

LOGO_URL = "https://raw.githubusercontent.com/roum71/rakcx2025/main/assets/mini_header2.png"
st.markdown(f"""
    <div style="text-align:center; margin-top:-40px;">
        <img src="{LOGO_URL}" style="width:950px; max-width:95%; height:auto;">
    </div>
    <hr style="margin-top:20px; margin-bottom:10px;">
""", unsafe_allow_html=True)

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
    df.columns = [c.strip() for c in df.columns]  # ⚙️ نحافظ على حالة الحروف الأصلية
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
    if len(vals) == 0: return np.nan
    mx = vals.max()
    if mx <= 5: return ((vals - 1) / 4 * 100).mean()
    elif mx <= 10: return ((vals - 1) / 9 * 100).mean()
    else: return vals.mean()

def autodetect_metric_cols(df: pd.DataFrame):
    csat = next((c for c in df.columns if "CSAT" in c.upper()), None)
    ces = next((c for c in df.columns if "FEES" in c.upper()), None)
    nps = next((c for c in df.columns if "NPS" in c.upper()), None)
    return csat, ces, nps

def detect_nps(df: pd.DataFrame):
    nps_col = next((c for c in df.columns if "NPS" in c.upper()), None)
    if not nps_col: return np.nan, 0, 0, 0, None
    s = pd.to_numeric(df[nps_col], errors="coerce").dropna()
    if len(s) == 0: return np.nan, 0, 0, 0, nps_col
    promoters = (s >= 9).sum()
    detractors = (s <= 6).sum()
    total = len(s)
    nps = (promoters - detractors) / total * 100
    return nps, promoters / total * 100, 0, detractors / total * 100, nps_col

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
    map_dict = dict(zip(tbl.iloc[:, 0].astype(str), tbl.iloc[:, 1].astype(str)))
    return s.astype(str).map(map_dict).fillna(s)

df_display = df_filtered.copy()
for col in candidate_filter_cols:
    df_display[col] = apply_lookup(col, df_filtered[col])

with st.sidebar.expander("تطبيق/إزالة الفلاتر"):
    applied_filters = {}
    for col in candidate_filter_cols:
        df_filtered[col] = apply_lookup(col, df_filtered[col])
        options = sorted(df_display[col].dropna().unique().tolist())
        sel = st.multiselect(f"{col}", options, default=options)
        applied_filters[col] = sel
for col, selected in applied_filters.items():
    df_filtered = df_filtered[df_filtered[col].isin(selected)]
df_view = df_filtered.copy()

# =========================================================
# التبويبات
# =========================================================
tab_data, tab_sample, tab_kpis, tab_dimensions, tab_services, tab_unsat = st.tabs([
    "📁 البيانات",
    "📈 توزيع العينة",
    "📊 المؤشرات",
    "🧩 الأبعاد",
    "📋 الخدمات",
    "💬 تحليل المزعجات (Pareto)"
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
    st.download_button("📥 تنزيل البيانات", data=buf.getvalue(),
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
        if col not in df_view: continue
        counts = df_view[col].value_counts().reset_index()
        counts.columns = [col, "Count"]
        counts["Percentage"] = counts["Count"] / counts["Count"].sum() * 100
        if chart_type == "مخطط أعمدة":
            fig = px.bar(counts, x=col, y="Count", text="Count", color=col,
                         color_discrete_sequence=PASTEL)
        else:
            fig = px.pie(counts, names=col, values="Count", hole=0.3)
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# تبويب المؤشرات
# =========================================================
with tab_kpis:
    st.subheader("📊 مؤشرات الأداء الرئيسية")
    csat_col, ces_col, nps_col = autodetect_metric_cols(df_view)
    csat = series_to_percent(df_view.get(csat_col, pd.Series(dtype=float))) if csat_col else np.nan
    ces = series_to_percent(df_view.get(ces_col, pd.Series(dtype=float))) if ces_col else np.nan
    nps, p_pct, s_pct, d_pct, nps_col = detect_nps(df_view)

    def gauge(value, title):
        color = "#bdc3c7" if pd.isna(value) else (
            "#FF6B6B" if value < 70 else "#FFD93D" if value < 80 else "#6BCB77" if value < 90 else "#4D96FF"
        )
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=0 if pd.isna(value) else float(value),
            number={'suffix': "%"},
            title={'text': title},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': color}}
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        return fig

    c1, c2, c3 = st.columns(3)
    c1.plotly_chart(gauge(csat, "رضا المتعامل (CSAT)"), use_container_width=True)
    c2.plotly_chart(gauge(ces, "سهولة/قيمة الخدمة (FEES)"), use_container_width=True)
    c3.plotly_chart(gauge(nps, "صافي المروجين (NPS)"), use_container_width=True)

# =========================================================
# تبويب الأبعاد
# =========================================================
with tab_dimensions:
    st.subheader("🧩 تحليل الأبعاد")
    dim_cols = [c for c in df_view.columns if re.match(r"Dim\d+\.", str(c))]
    if not dim_cols:
        st.info("⚠️ لا توجد أعمدة للأبعاد مثل Dim1.1 أو Dim2.3.")
    else:
        summary = []
        for i in range(1, 6):
            sub = [c for c in dim_cols if c.startswith(f"Dim{i}.")]
            if sub:
                score = series_to_percent(df_view[sub].mean(axis=1))
                summary.append({"البعد": f"Dim{i}", "النسبة (%)": score})
        dims = pd.DataFrame(summary)
        st.dataframe(dims.style.format({"النسبة (%)": "{:.1f}%"}), use_container_width=True)
        fig = px.bar(dims, x="البعد", y="النسبة (%)", text="النسبة (%)", color="البعد",
                     color_discrete_sequence=PASTEL)
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 💬 تبويب تحليل المزعجات (Pareto)
# =========================================================
with tab_unsat:
    st.subheader("💬 تحليل المزعجات في الخدمات الرقمية (Pareto)")

    text_cols = [c for c in df_view.columns if any(k in c.upper() for k in ["MOST_UNSAT", "UNSAT", "COMMENT", "ملاح"])]
    if not text_cols:
        st.warning("⚠️ لم يتم العثور على عمود نصي للملاحظات.")
    else:
        col = text_cols[0]
        data = df_view[[col]].copy()
        data.columns = ["Comment"]
        data["Comment"] = data["Comment"].astype(str).str.strip()

        exclude = ["", " ", "لا يوجد", "لايوجد", "none", "no", "nothing", "nil", "ok", "جيد", "ممتاز", "تمام"]
        data = data[~data["Comment"].str.lower().isin([t.lower() for t in exclude])]
        data = data[data["Comment"].apply(lambda x: len(x.split()) >= 2)]

        if data.empty:
            st.info("لا توجد ملاحظات نصية كافية.")
        else:
            themes = {
                "السرعة / الأداء": ["بطء", "تأخير", "انتظار", "delay", "slow", "وقت"],
                "التطبيق / المنصة": ["تطبيق", "app", "منصة", "system", "موقع", "بوابة"],
                "الإجراءات / الخطوات": ["إجراء", "اجراء", "process", "خطوة", "نموذج"],
                "الرسوم / الدفع": ["رسوم", "دفع", "fee", "pay", "سداد"],
                "التواصل / الدعم الفني": ["رد", "تواصل", "support", "response", "مساعدة"],
                "الوضوح / المعلومات": ["معلومة", "إيضاح", "clarity", "شرح"],
                "الأمان / الدخول": ["دخول", "login", "تحقق", "كلمة مرور"]
            }

            def classify(txt):
                t = txt.lower()
                for theme, words in themes.items():
                    if any(w in t for w in words):
                        return theme
                return "أخرى"

            data["المحور"] = data["Comment"].apply(classify)
            summary = data.groupby("المحور").agg({
                "Comment": lambda x: " / ".join(x.tolist())
            }).reset_index()
            summary["عدد الملاحظات"] = summary["Comment"].apply(lambda x: len(x.split("/")))
            summary = summary.sort_values("عدد الملاحظات", ascending=False)
            summary["النسبة (%)"] = summary["عدد الملاحظات"] / summary["عدد الملاحظات"].sum() * 100
            summary["النسبة التراكمية (%)"] = summary["النسبة (%)"].cumsum()
            summary["اللون"] = np.where(summary["النسبة التراكمية (%)"] <= 80, "#E74C3C", "#BDC3C7")
            if not summary[summary["النسبة التراكمية (%)"] > 80].empty:
                first_above = summary[summary["النسبة التراكمية (%)"] > 80].index[0]
                summary.loc[first_above, "اللون"] = "#E74C3C"

            st.dataframe(summary[["المحور", "عدد الملاحظات", "النسبة (%)", "النسبة التراكمية (%)", "Comment"]]
                         .rename(columns={"Comment": "التعليقات (مجمعة)"})
                         .style.format({"النسبة (%)": "{:.1f}%", "النسبة التراكمية (%)": "{:.1f}%"}),
                         use_container_width=True, hide_index=True)

            fig = go.Figure()
            fig.add_bar(x=summary["المحور"], y=summary["عدد الملاحظات"],
                        marker_color=summary["اللون"], name="عدد الملاحظات")
            fig.add_scatter(x=summary["المحور"], y=summary["النسبة التراكمية (%)"], yaxis="y2",
                            mode="lines+markers+text", name="النسبة التراكمية (%)",
                            text=[f"{v:.1f}%" for v in summary["النسبة التراكمية (%)"]],
                            textposition="top center", line=dict(color="#2E86DE", width=3))
            fig.update_layout(
                title="📊 تحليل Pareto لأهم المزعجات في الخدمات الرقمية",
                xaxis=dict(title="المحور", tickangle=-15),
                yaxis=dict(title="عدد الملاحظات"),
                yaxis2=dict(title="النسبة التراكمية (%)", overlaying="y", side="right", range=[0,110]),
                height=600, bargap=0.3, legend=dict(orientation="h", y=-0.2)
            )
            st.plotly_chart(fig, use_container_width=True)

# =========================================================
# إخفاء التذييل
# =========================================================
st.markdown("""
    <style>
    #MainMenu {visibility:hidden;}
    footer, [data-testid="stFooter"] {display:none;}
    </style>
""", unsafe_allow_html=True)
