#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Customer  Experience  Dashboard — v10.7 
Unified | Secure | Multi-Center | Lookup | KPI Gauges | Dimensions | Pareto | Services Overview
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io, re
from datetime import datetime
from pathlib import Path

# =========================================================
# 🔐 USERS
# =========================================================
import streamlit as st

USER_KEYS = {
    "Public Services Department": {
        "password": st.secrets["users"]["Public_Services_Department"],
        "role": "center",
        "file": "Center_Public_Services.csv"
    },
    "Ras Al Khaimah Municipality": {
        "password": st.secrets["users"]["Ras_Al_Khaimah_Municipality"],
        "role": "center",
        "file": "Center_RAK_Municipality.csv"
    },
    "Sheikh Saud Center-Ras Al Khaimah Courts": {
        "password": st.secrets["users"]["Sheikh_Saud_Center"],
        "role": "center",
        "file": "Center_Sheikh_Saud_Courts.csv"
    },
    "Sheikh Saqr Center-Ras Al Khaimah Courts": {
        "password": st.secrets["users"]["Sheikh_Saqr_Center"],
        "role": "center",
        "file": "Center_Sheikh_Saqr_Courts.csv"
    },
    "Executive Council": {
        "password": st.secrets["users"]["Executive_Council"],
        "role": "admin",
        "file": "Centers_Master.csv"
    }
}


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="لوحة تجربة المتعاملين 2025 — رأس الخيمة", layout="wide")
PASTEL = px.colors.qualitative.Pastel

#=========================================================
# 🏛️ HEADER — شعار الأمانة العامة + عنوان التقرير الرسمي (Full Width + GitHub Link)
# =========================================================

# 🔗 ضع هنا رابط الصورة من GitHub (raw)
#logo_url = "https://raw.githubusercontent.com/roum71/rakcx2025/main/assets/logo_gsec_full.png"
logo_url = "https://raw.githubusercontent.com/roum71/rakcx2025/main/assets/mini_header2.png"

st.markdown(f"""
    <div style="text-align:center; margin-top:-40px;">
        <img src="{logo_url}" alt="RAK Executive Council Logo" style="width:950px; max-width:95%; height:auto;">
    </div>

    <hr style="margin-top:20px; margin-bottom:10px;">
""", unsafe_allow_html=True)

# =========================================================
# LANGUAGE
# =========================================================
lang = st.sidebar.radio("🌍 اللغة / Language", ["العربية", "English"], index=0)
if lang == "العربية":
    st.markdown("""
        <style>
        html, body, [class*="css"] {direction:rtl;text-align:right;font-family:"Tajawal","Cairo","Segoe UI";}
        </style>
    """, unsafe_allow_html=True)


# =========================================================
# 🌍 BILINGUAL TEXT FUNCTION
# =========================================================
def bi_text(ar_text, en_text):
    """عرض النص بالعربية أو الإنجليزية بناءً على اختيار المستخدم"""
    return ar_text if lang == "العربية" else en_text

# =========================================================
# LOGIN (ثنائي اللغة)
# =========================================================
params = st.query_params
center_from_link = params.get("center", [None])[0]

# 🗂️ إعداد قائمة المراكز بالعربية والإنجليزية
center_names_ar = {
    "Public Services Department": "دائرة الخدمات العامة",
    "Ras Al Khaimah Municipality": "بلدية رأس الخيمة",
    "Sheikh Saud Center-Ras Al Khaimah Courts": "مركز الشيخ سعود - محاكم رأس الخيمة",
    "Sheikh Saqr Center-Ras Al Khaimah Courts": "مركز الشيخ صقر - محاكم رأس الخيمة",
    "Executive Council": "الأمانة العامة للمجلس التنفيذي"
}

# ✅ اختيار الاسم حسب اللغة
if lang == "العربية":
    center_options = [center_names_ar.get(k, k) for k in USER_KEYS.keys()]
else:
    center_options = list(USER_KEYS.keys())

# 🏢 اختيار المركز
st.sidebar.header(bi_text("🏢 اختر المركز", "🏢 Select Center"))

# ⚙️ إنشاء خريطة عكسية عند استخدام اللغة العربية
reverse_map = {v: k for k, v in center_names_ar.items()}

# إذا تم التمرير عبر الرابط
if center_from_link and center_from_link in USER_KEYS:
    selected_center = center_from_link
else:
    selected_center = st.sidebar.selectbox(
        bi_text("اختر المركز", "Select Center"),
        center_options
    )

# 🔁 تحويل الاسم العربي إلى الاسم الأصلي (المفتاح الحقيقي)
if lang == "العربية":
    selected_center = reverse_map.get(selected_center, selected_center)

# حفظ حالة الجلسة
if "authorized" not in st.session_state:
    st.session_state.update({"authorized": False, "center": None, "role": None})

# التحقق من كلمة المرور
if not st.session_state["authorized"] or st.session_state["center"] != selected_center:
    st.sidebar.subheader(bi_text("🔑 كلمة المرور", "🔑 Password"))
    password = st.sidebar.text_input(bi_text("كلمة المرور", "Password"), type="password")
    
    if password == USER_KEYS[selected_center]["password"]:
        st.session_state.update({
            "authorized": True,
            "center": selected_center,
            "role": USER_KEYS[selected_center]["role"],
            "file": USER_KEYS[selected_center]["file"]
        })
        st.success(bi_text(f"✅ تم تسجيل الدخول كمركز: {center_names_ar.get(selected_center, selected_center)}",
                           f"✅ Logged in as: {selected_center}"))
        st.rerun()
    elif password:
        st.error(bi_text("🚫 كلمة المرور غير صحيحة.", "🚫 Incorrect password."))
        st.stop()
    else:
        st.warning(bi_text("يرجى إدخال كلمة المرور.", "Please enter the password."))
        st.stop()

center, role = st.session_state["center"], st.session_state["role"]

# =========================================================
# LOAD DATA
# =========================================================
def safe_read(file):
    try:
        return pd.read_csv(file, encoding="utf-8", low_memory=False)
    except Exception:
        return None

file_path = USER_KEYS[center]["file"]
df = safe_read(file_path)
if df is None:
    st.error(f"❌ تعذر تحميل الملف: {file_path}")
    st.stop()




# =========================================================
# LOOKUP TABLES
# =========================================================
lookup_path = Path("Data_tables.xlsx")
lookup_catalog = {}
if lookup_path.exists():
    xls = pd.ExcelFile(lookup_path)
    for sheet in xls.sheet_names:
        tbl = pd.read_excel(xls, sheet_name=sheet)
        tbl.columns = [c.strip().upper() for c in tbl.columns]
        lookup_catalog[sheet.upper()] = tbl


# =========================================================
# UTILS
# =========================================================
def series_to_percent(vals):
    vals = pd.to_numeric(vals, errors="coerce").dropna()
    if len(vals) == 0:
        return np.nan
    mx = vals.max()
    if mx <= 5: return ((vals - 1)/4*100).mean()
    elif mx <= 10: return ((vals - 1)/9*100).mean()
    else: return vals.mean()

def detect_nps(df):
    cands = [c for c in df.columns if "nps" in c.lower() or "recommend" in c.lower()]
    if not cands: return np.nan, 0, 0, 0
    s = pd.to_numeric(df[cands[0]], errors="coerce").dropna()
    if len(s)==0: return np.nan, 0, 0, 0
    promoters = (s>=9).sum()
    passives = ((s>=7)&(s<=8)).sum()
    detractors = (s<=6).sum()
    total = len(s)
    promoters_pct = promoters/total*100
    passives_pct = passives/total*100
    detractors_pct = detractors/total*100
    nps = promoters_pct - detractors_pct
    return nps, promoters_pct, passives_pct, detractors_pct
# =========================================================
# 🎛️ FILTERS — الفلاتر (تتغير اللغة تلقائيًا)
# =========================================================
filter_cols = [c for c in df.columns if any(k in c.upper() for k in ["GENDER", "SERVICE", "SECTOR", "NATIONALITY", "ACADEMIC"])]
filters = {}
df_filtered = df.copy()

with st.sidebar.expander("🎛️ الفلاتر / Filters"):
    for col in filter_cols:
        lookup_name = col.strip().upper()
        mapped = False

        # 🔍 البحث عن جدول المطابقة في ملف Data_tables.xlsx
        if lookup_name in lookup_catalog:
            tbl = lookup_catalog[lookup_name]
            tbl.columns = [c.strip().upper() for c in tbl.columns]

            # تحديد الأعمدة في جدول الـ Lookup
            ar_col = next((c for c in tbl.columns if "ARABIC" in c or "SERVICE2" in c), None)
            en_col = next((c for c in tbl.columns if "ENGLISH" in c), None)
            code_col = next((c for c in tbl.columns if "CODE" in c or lookup_name in c), None)

            # تطبيق الترجمة على القيم
            if code_col and ((lang == "العربية" and ar_col) or (lang == "English" and en_col)):
                name_col = ar_col if lang == "العربية" else en_col
                name_map = dict(zip(tbl[code_col].astype(str), tbl[name_col].astype(str)))
                df_filtered[col] = df_filtered[col].astype(str).map(name_map).fillna(df_filtered[col])
                mapped = True

        if not mapped:
            st.sidebar.warning(f"⚠️ Lookup not applied for {col}")

        # 🏷️ تسمية الفلتر بالعربية أو الإنجليزية
        if lang == "العربية":
            if "GENDER" in col.upper():
                label = "النوع"
            elif "NATIONALITY" in col.upper():
                label = "الجنسية"
            elif "ACADEMIC" in col.upper():
                label = "المستوى الأكاديمي"
            elif "SERVICE" in col.upper():
                label = "الخدمة"
            elif "SECTOR" in col.upper():
                label = "القطاع"
            else:
                label = col
        else:
            if "GENDER" in col.upper():
                label = "Gender"
            elif "NATIONALITY" in col.upper():
                label = "Nationality"
            elif "ACADEMIC" in col.upper():
                label = "Academic Level"
            elif "SERVICE" in col.upper():
                label = "Service"
            elif "SECTOR" in col.upper():
                label = "Sector"
            else:
                label = col

        # 🧩 إنشاء الفلتر
        options = df_filtered[col].dropna().unique().tolist()
        selection = st.multiselect(label, options, default=options)
        filters[col] = selection

# 🔽 تطبيق الفلاتر على البيانات
for col, values in filters.items():
    df_filtered = df_filtered[df_filtered[col].isin(values)]

df = df_filtered.copy()

# =========================================================
# 📈 TABS
# =========================================================
tab_data, tab_sample, tab_kpis, tab_dimensions, tab_services, tab_pareto = st.tabs([
    bi_text("📁 البيانات", "Data"),
    bi_text("📈 توزيع العينة", "Sample Distribution"),
    bi_text("📊 المؤشرات", "KPIs"),
    bi_text("🧩 الأبعاد", "Dimensions"),
    bi_text("📋 الخدمات", "Services"),
    bi_text("💬المزعجات", "Pain Points")
])

# =========================================================
# 📁 DATA TAB — Multi-language headers
# =========================================================
with tab_data:
#    st.subheader("📁 البيانات الخام /Raw Data")

    questions_map_ar, questions_map_en = {}, {}
    if "QUESTIONS" in lookup_catalog:
        qtbl = lookup_catalog["QUESTIONS"]
        qtbl.columns = [c.strip().upper() for c in qtbl.columns]
        code_col = next((c for c in qtbl.columns if "CODE" in c or "DIMENSION" in c), None)
        ar_col = next((c for c in qtbl.columns if "ARABIC" in c or c == "ARABIC"), None)
        en_col = next((c for c in qtbl.columns if "ENGLISH" in c or c == "ENGLISH"), None)

        if code_col and ar_col and en_col:
            qtbl["CODE_NORM"] = qtbl[code_col].astype(str).str.strip().str.upper()
            questions_map_ar = dict(zip(qtbl["CODE_NORM"], qtbl[ar_col]))
            questions_map_en = dict(zip(qtbl["CODE_NORM"], qtbl[en_col]))

    df_display = df.copy()
    df_display.columns = [c.strip() for c in df_display.columns]
    ar_row = [questions_map_ar.get(c.strip().upper(), "") for c in df_display.columns]
    en_row = [questions_map_en.get(c.strip().upper(), "") for c in df_display.columns]
    df_final = pd.concat([pd.DataFrame([ar_row, en_row], columns=df_display.columns), df_display], ignore_index=True)

    st.dataframe(df_final, use_container_width=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_final.to_excel(writer, index=False)
    st.download_button("📥 تنزيل البيانات", buffer.getvalue(), file_name=f"Filtered_Data_{ts}.xlsx")

# =========================================================
# 📈 SAMPLE TAB — توزيع العينة (ثنائي اللغة مع عناوين ديناميكية)
# =========================================================
with tab_sample:
    st.subheader(bi_text("📈 توزيع العينة", "Sample Distribution"))

    # 🧮 إجمالي الردود
    total = len(df)
    st.markdown(f"### 🧮 {bi_text('إجمالي الردود:', 'Total Responses:')} {total:,}")

    # 🟩 نوع الرسم البياني
    chart_type = st.radio(
        bi_text("📊 نوع الرسم البياني", "📊 Chart Type"),
        [bi_text("مخطط دائري (Pie Chart)", "Pie Chart"),
         bi_text("مخطط أعمدة (Bar Chart)", "Bar Chart"),
         bi_text("شبكي / مصفوفة (Grid / Matrix)", "Grid / Matrix")],
        index=1,
        horizontal=True
    )

    # 🟨 طريقة العرض
    value_type = st.radio(
        bi_text("📏 طريقة العرض", "📏 Display Mode"),
        [bi_text("الأعداد (Numbers)", "Numbers"),
         bi_text("النسب المئوية (Percentages)", "Percentages")],
        index=1,
        horizontal=True
    )

    # 🧩 تنفيذ الرسم حسب الأعمدة المختارة
    for col in filter_cols:
        counts = df[col].value_counts().reset_index()
        counts.columns = [col, "Count"]
        counts["Percentage"] = counts["Count"] / total * 100

        value_col = "Count" if "Numbers" in value_type else "Percentage"

        # 🏷️ اختيار التسمية بناءً على اللغة
        if lang == "العربية":
            if col.upper() == "GENDER":
                col_label = "النوع"
            elif col.upper() == "NATIONALITY":
                col_label = "الجنسية"
            elif "ACADEMIC" in col.upper():
                col_label = "المستوى الأكاديمي"
            elif "SECTOR" in col.upper():
                col_label = "القطاع"
            elif "SERVICE" in col.upper():
                col_label = "الخدمة"
            else:
                col_label = col
            st.markdown(f"### {col_label} — {total:,} ردود")
            graph_title = f"توزيع {col_label}"
            x_title = "الفئة"
            y_title = "النسبة المئوية (%)" if value_col == "Percentage" else "العدد"

        else:  # English
            if col.upper() == "GENDER":
                col_label = "Gender"
            elif col.upper() == "NATIONALITY":
                col_label = "Nationality"
            elif "ACADEMIC" in col.upper():
                col_label = "Academic Level"
            elif "SERVICE" in col.upper():
                col_label = "Service"
            else:
                col_label = col
            st.markdown(f"### {col_label} — {total:,} Responses")
            graph_title = f"Distribution of {col_label}"
            x_title = "Category"
            y_title = "Percentage (%)" if value_col == "Percentage" else "Count"

        # 🥧 Pie Chart
        if "Pie" in chart_type:
            fig = px.pie(
                counts,
                names=col,
                values=value_col,
                hole=0.3,
                title=graph_title,
                color_discrete_sequence=PASTEL
            )
            fig.update_traces(
                texttemplate="%{label}<br>%{percent:.1%}" if value_col == "Percentage" else "%{label}<br>%{value}",
                textposition="inside",
                textfont_size=14
            )
            fig.update_layout(title_x=0.5, title_font=dict(size=20))
            st.plotly_chart(fig, use_container_width=True)

        # 📊 Bar Chart
        elif "Bar" in chart_type:
            fig = px.bar(
                counts,
                x=col,
                y=value_col,
                text=value_col,
                color=col,
                color_discrete_sequence=PASTEL,
                title=graph_title
            )
            fig.update_traces(
                texttemplate="%{text:.1f}" if value_col == "Percentage" else "%{text}",
                textposition="outside"
            )
            fig.update_layout(
                xaxis_title=x_title,
                yaxis_title=y_title,
                title_x=0.5,
                title_font=dict(size=20)
            )
            st.plotly_chart(fig, use_container_width=True)

        # 🧩 Grid / Matrix View
        else:
            st.write(f"### 🧩 {bi_text('عرض شبكي —', 'Grid View —')} {col_label}")
            matrix = counts[[col, "Count", "Percentage"]].copy()
            matrix.columns = [
                bi_text("القيمة", "Value"),
                bi_text("العدد", "Count"),
                bi_text("النسبة المئوية", "Percentage")
            ]
            st.dataframe(
                matrix.style.format({bi_text("النسبة المئوية", "Percentage"): "{:.1f}%"}),
                use_container_width=True
            )
# =========================================================
# 📊 KPIs TAB — السعادة / القيمة / صافي نقاط الترويج
# =========================================================
with tab_kpis:
    st.subheader(bi_text("📊 مؤشرات الأداء الرئيسية (السعادة / القيمة / صافي نقاط الترويج)",
                         "Key Performance Indicators (Happiness / Value / NPS)"))
    st.info(bi_text(
        "يعرض هذا القسم نتائج المؤشرات الثلاثة.",
        "This section shows the three key indicators ."
    ))

    # 🧮 حساب المؤشرات من البيانات
    csat = series_to_percent(df.get("Dim6.1", pd.Series(dtype=float)))   # Happiness
    ces = series_to_percent(df.get("Dim6.2", pd.Series(dtype=float)))    # Value
    nps, prom, passv, detr = detect_nps(df)                              # NPS

    # =========================================================
    # 🎨 تدرج الألوان والأوصاف حسب اللغة
    # =========================================================
    def get_color_and_label(score, metric_type, lang="العربية"):
        if metric_type in ["CSAT", "CES"]:
            if score < 70:
                color, label = "#FF6B6B", ("ضعيف جدًا" if lang == "العربية" else "Very Poor")
            elif score < 80:
                color, label = "#FFD93D", ("بحاجة إلى تحسين" if lang == "العربية" else "Needs Improvement")
            elif score < 90:
                color, label = "#6BCB77", ("جيد" if lang == "العربية" else "Good")
            else:
                color, label = "#4D96FF", ("ممتاز" if lang == "العربية" else "Excellent")
        else:  # NPS logic
            if score < 0:
                color, label = "#FF6B6B", ("ضعيف جدًا" if lang == "العربية" else "Very Poor")
            elif score < 30:
                color, label = "#FFD93D", ("ضعيف" if lang == "العربية" else "Fair")
            elif score < 60:
                color, label = "#6BCB77", ("جيد" if lang == "العربية" else "Good")
            else:
                color, label = "#4D96FF", ("ممتاز" if lang == "العربية" else "Excellent")
        return color, label

    # =========================================================
    # 🧭 دالة إنشاء الرسم Gauge
    # =========================================================
    def create_gauge(score, metric_type, lang="العربية"):
        color, label = get_color_and_label(score, metric_type, lang)
        if metric_type in ["CSAT", "CES"]:
            title = "السعادة عموما / Overall Happiness" if metric_type == "CSAT" else " القيمة مقابل الجهد والتكلفة / Value"
            axis_range = [0, 100]
            steps = [
                {'range': [0, 70], 'color': '#FF6B6B'},
                {'range': [70, 80], 'color': '#FFD93D'},
                {'range': [80, 90], 'color': '#6BCB77'},
                {'range': [90, 100], 'color': '#4D96FF'}
            ]
        else:
            title = "صافي نقاط الترويج / NPS"
            axis_range = [-100, 100]
            steps = [
                {'range': [-100, 0], 'color': '#FF6B6B'},
                {'range': [0, 30], 'color': '#FFD93D'},
                {'range': [30, 60], 'color': '#6BCB77'},
                {'range': [60, 100], 'color': '#4D96FF'}
            ]

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score if not np.isnan(score) else 0,
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

    # =========================================================
    # 📈 عرض المؤشرات الثلاثة (السعادة / القيمة / NPS)
    # =========================================================
    c1, c2, c3 = st.columns(3)
    for col, val, mtype in zip([c1, c2, c3], [csat, ces, nps], ["CSAT", "CES", "NPS"]):
        fig, label = create_gauge(val, mtype, lang)
        col.plotly_chart(fig, use_container_width=True)

        # 🧮 تحديد اللون الخاص بالتفسير النصي
        color, _ = get_color_and_label(val, mtype, lang)
        text_color = f"color:{color};font-weight:bold;"

        if mtype == "NPS":
            # 🔎 تفسير مخصص للـ NPS
            if lang == "العربية":
                if val < 0:
                    detail = "نتيجة سلبية تشير إلى أن عدد المعارضين يفوق عدد المروجين."
                elif val < 30:
                    detail = "نتيجة ضعيفة — رضا العملاء محدود وعدد المروجين منخفض."
                elif val < 60:
                    detail = "نتيجة جيدة — أغلب العملاء راضون والمروجون أكثر من المعارضين."
                else:
                    detail = "نتيجة ممتازة — ولاء العملاء مرتفع جدًا ومعظمهم مروجون للخدمة."
                col.markdown(
                    f"<p style='{text_color}'>🔎 التفسير: {label}<br>{detail}<br>"
                    f"المروجون: {prom:.1f}% | المحايدون: {passv:.1f}% | المعارضون: {detr:.1f}%</p>",
                    unsafe_allow_html=True
                )
            else:
                if val < 0:
                    detail = "Negative score — more detractors than promoters."
                elif val < 30:
                    detail = "Low score — limited satisfaction and few promoters."
                elif val < 60:
                    detail = "Good score — most customers are satisfied, promoters exceed detractors."
                else:
                    detail = "Excellent score — strong loyalty and many promoters."
                col.markdown(
                    f"<p style='{text_color}'>🔎 Interpretation: {label}<br>{detail}<br>"
                    f"Promoters: {prom:.1f}% | Passives: {passv:.1f}% | Detractors: {detr:.1f}%</p>",
                    unsafe_allow_html=True
                )
        else:
            # 🔎 تفسير للسعادة والقيمة
            text = "🔎 التفسير: " + label if lang == "العربية" else "🔎 Interpretation: " + label
            col.markdown(f"<p style='{text_color}'>{text}</p>", unsafe_allow_html=True)

    # =========================================================
    # 🎨 وسيلتا الإيضاح (Legends)
    # =========================================================
    if lang == "العربية":
        legend_html = """
        <div style='background-color:#f9f9f9;border:1px solid #ddd;border-radius:10px;padding:12px;margin-top:15px;'>
          <h4 style='margin-bottom:8px;'>🎨 وسيلة الإيضاح — السعادة / القيمة</h4>
          🔴 أقل من 70٪ — ضعيف جدًا<br>
          🟡 من 70 إلى أقل من 80٪ — بحاجة إلى تحسين<br>
          🟢 من 80 إلى أقل من 90٪ — جيد<br>
          🔵 90٪ فأكثر — ممتاز
        </div>

        <div style='background-color:#f9f9f9;border:1px solid #ddd;border-radius:10px;padding:12px;margin-top:10px;'>
          <h4 style='margin-bottom:8px;'>🎯 وسيلة الإيضاح — صافي نقاط الترويج (NPS)</h4>
          🔴 أقل من 0 — ضعيف جدًا (عدد المعارضين أكبر من المروجين)<br>
          🟡 من 0 إلى أقل من 30 — ضعيف (رضا محدود)<br>
          🟢 من 30 إلى أقل من 60 — جيد (رضا عام)<br>
          🔵 60 فأكثر — ممتاز (ولاء مرتفع جدًا)
        </div>
        """
    else:
        legend_html = """
        <div style='background-color:#f9f9f9;border:1px solid #ddd;border-radius:10px;padding:12px;margin-top:15px;'>
          <h4 style='margin-bottom:8px;'>🎨 Legend — Happiness / Value</h4>
          🔴 Below 70% — Very Poor<br>
          🟡 70–80% — Needs Improvement<br>
          🟢 80–90% — Good<br>
          🔵 90%+ — Excellent
        </div>

        <div style='background-color:#f9f9f9;border:1px solid #ddd;border-radius:10px;padding:12px;margin-top:10px;'>
          <h4 style='margin-bottom:8px;'>🎯 Legend — NPS (Net Promoter Score)</h4>
          🔴 Below 0 — Very Poor (More detractors than promoters)<br>
          🟡 0–30 — Fair (Limited satisfaction)<br>
          🟢 30–60 — Good (Majority satisfied)<br>
          🔵 60+ — Excellent (Strong loyalty)
        </div>
        """
    st.markdown(legend_html, unsafe_allow_html=True)
# =========================================================
# 🧩 DIMENSIONS TAB — تحليل الأبعاد (تنسيق + ثنائية اللغة)
# =========================================================
# =========================================================
# 🧩 DIMENSIONS TAB — تحليل الأبعاد (ألوان مطابقة للوسيلة)
# =========================================================
with tab_dimensions:
    all_dim_cols = [c for c in df.columns if re.match(r"Dim\d+\.", c.strip())]

    if not all_dim_cols:
        st.warning("⚠️ لا توجد أعمدة فرعية للأبعاد (مثل Dim1.1 أو Dim2.3).")
    else:
        # حساب المتوسط لكل بعد رئيسي
        main_dims = {}
        for i in range(1, 6):
            sub_cols = [c for c in df.columns if c.startswith(f"Dim{i}.")]
            if sub_cols:
                main_dims[f"Dim{i}"] = df[sub_cols].mean(axis=1)
                df[f"Dim{i}"] = main_dims[f"Dim{i}"]

        summary = []
        for dim in [f"Dim{i}" for i in range(1, 6)]:
            if dim in df.columns:
                avg = series_to_percent(df[dim])
                summary.append({"Dimension": dim, "Score": avg})
        dims = pd.DataFrame(summary).dropna()

        # إضافة أسماء الأبعاد
        if "QUESTIONS" in lookup_catalog:
            qtbl = lookup_catalog["QUESTIONS"]
            qtbl.columns = [c.strip().upper() for c in qtbl.columns]
            code_col = next((c for c in qtbl.columns if "CODE" in c or "DIMENSION" in c), None)
            ar_col = next((c for c in qtbl.columns if "ARABIC" in c), None)
            en_col = next((c for c in qtbl.columns if "ENGLISH" in c), None)
            if code_col and ar_col and en_col:
                qtbl["CODE_NORM"] = qtbl[code_col].astype(str).str.strip()
                name_map = dict(zip(qtbl["CODE_NORM"], qtbl[ar_col if lang == "العربية" else en_col]))
                dims["Dimension_name"] = dims["Dimension"].map(name_map)

        # الحفاظ على الترتيب
        dims["Order"] = dims["Dimension"].str.extract(r"(\d+)").astype(float)
        dims = dims.sort_values("Order")

        # تصنيف الأداء حسب النسبة
        def get_category(score):
            if score < 70:
                return "🔴 ضعيف الأداء / Weak"
            elif score < 80:
                return "🟡 متوسط / Average"
            elif score < 90:
                return "🟢 جيد / Good"
            else:
                return "🔵 ممتاز / Excellent"

        dims["Category"] = dims["Score"].apply(get_category)

        # خريطة الألوان حسب الفئة
        color_map = {
            "🔴 ضعيف الأداء / Weak": "#FF6B6B",
            "🟡 متوسط / Average": "#FFD93D",
            "🟢 جيد / Good": "#6BCB77",
            "🔵 ممتاز / Excellent": "#4D96FF"
        }

        chart_title = "📊 تحليل متوسط الأبعاد / Average Dimensions Analysis"
        x_axis_title = "الأبعاد / Dimensions"
        y_axis_title = "النسبة المئوية (%) / Percentage (%)"

        x_col = "Dimension_name" if "Dimension_name" in dims.columns else "Dimension"
        category_order = dims.sort_values("Order")[x_col].tolist()

        # رسم الأعمدة بالألوان الصحيحة
        fig = px.bar(
            dims,
            x=x_col,
            y="Score",
            text="Score",
            color="Category",  # ← نستخدم الفئة بدلاً من الأكواد
            color_discrete_map=color_map,
            title=chart_title,
            category_orders={x_col: category_order}
        )

        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")

        fig.update_layout(
            title=dict(
                text=chart_title,
                x=0.5,
                xanchor="center",
                font=dict(size=18, family="Cairo, sans-serif", color="#333")
            ),
            xaxis_title=x_axis_title,
            yaxis_title=y_axis_title,
            yaxis=dict(range=[0, 100]),
            showlegend=True,  # ← تفعيل وسيلة الإيضاح
            margin=dict(t=60, b=40),
            uniformtext_minsize=8,
            uniformtext_mode="hide"
        )

        st.plotly_chart(fig, use_container_width=True)

        # وسيلة الإيضاح ثنائية اللغة
        st.markdown(bi_text(
            """
            **🗂️ وسيلة الإيضاح:**
            - 🔴 أقل من 70٪ — ضعيف الأداء  
            - 🟡 من 70٪ إلى أقل من 80٪ — متوسط  
            - 🟢 من 80٪ إلى أقل من 90٪ — جيد  
            - 🔵 90٪ فأكثر — ممتاز  
            """,
            """
            **🗂️ Color Legend:**
            - 🔴 Below 70% — Weak Performance  
            - 🟡 70% to <80% — Average  
            - 🟢 80% to <90% — Good  
            - 🔵 90% and above — Excellent  
            """
        ), unsafe_allow_html=True)

        # عرض الجدول
        display_cols = ["Dimension", "Dimension_name", "Score"]
        dims = dims[display_cols]

        if lang == "العربية":
            dims.columns = ["البعد", "اسم البعد", "النسبة (%)"]
        else:
            dims.columns = ["Dimension", "Dimension Name", "Score (%)"]

        st.dataframe(
            dims.style.format({"النسبة (%)": "{:.1f}%", "Score (%)": "{:.1f}%"}),
            use_container_width=True,
            hide_index=True
        )

# =========================================================
# 📋 SERVICES TAB — تحليل الخدمات (Happiness / Value / NPS)
# =========================================================
with tab_services:
    st.subheader(bi_text("📋 تحليل الخدمات", "Service Analysis"))
    st.info(bi_text(
        "مقارنة مستويات السعادة والقيمة وصافي نقاط الترويج حسب الخدمة.",
        "Comparison of Happiness, Value, and NPS levels by service."
    ))

    if "SERVICE" not in df.columns:
        st.warning("⚠️ لا توجد بيانات خاصة بالخدمات.")
    else:
        df_services = df.copy()

        # 🔍 تحديد الأعمدة الخاصة بالسعادة (CSAT) والقيمة (CES) وNPS
        csat_col = next((c for c in df_services.columns if c.upper().startswith("DIM6.1")), None)
        ces_col = next((c for c in df_services.columns if c.upper().startswith("DIM6.2")), None)
        nps_col = next((c for c in df_services.columns if "NPS" in c.upper()), None)

        if not csat_col or not ces_col:
            st.warning("⚠️ لم يتم العثور على الأعمدة Dim6.1 أو Dim6.2 في البيانات.")
        else:
            # 🧮 تحويل القيم من 1–5 إلى 0–100
            df_services["Happiness / سعادة (٪)"] = (df_services[csat_col] - 1) * 25
            df_services["Value / قيمة (٪)"] = (df_services[ces_col] - 1) * 25

            # 🧮 حساب NPS (إن وجد)
            if nps_col:
                df_services["NPS_SCORE"] = pd.to_numeric(df_services[nps_col], errors="coerce")
                nps_summary = []
                for svc, subdf in df_services.groupby("SERVICE"):
                    valid = subdf["NPS_SCORE"].dropna()
                    if len(valid) == 0:
                        nps_summary.append((svc, np.nan))
                        continue
                    promoters = (valid >= 9).sum()
                    detractors = (valid <= 6).sum()
                    total = len(valid)
                    nps_value = ((promoters - detractors) / total) * 100
                    nps_summary.append((svc, nps_value))
                nps_df = pd.DataFrame(nps_summary, columns=["SERVICE", "NPS / صافي نقاط الترويج (٪)"])
            else:
                nps_df = pd.DataFrame(columns=["SERVICE", "NPS / صافي نقاط الترويج (٪)"])

            # 🧾 حساب المتوسط وعدد الردود لكل خدمة
            summary = (
                df_services.groupby("SERVICE")
                .agg({
                    "Happiness / سعادة (٪)": "mean",
                    "Value / قيمة (٪)": "mean",
                    csat_col: "count"
                })
                .reset_index()
                .rename(columns={csat_col: "عدد الردود / Responses"})
            )

            # دمج نتائج NPS
            summary = summary.merge(nps_df, on="SERVICE", how="left")

            # 🌐 استبدال أسماء الخدمات بالعربية / الإنجليزية من lookup
            if "SERVICE" in lookup_catalog:
                tbl = lookup_catalog["SERVICE"]
                tbl.columns = [c.strip().upper() for c in tbl.columns]
                ar_col = next((c for c in tbl.columns if "ARABIC" in c or "SERVICE2" in c), None)
                en_col = next((c for c in tbl.columns if "ENGLISH" in c), None)
                code_col = next((c for c in tbl.columns if "CODE" in c or "SERVICE" in c), None)
                if ar_col and en_col and code_col:
                    name_map = dict(zip(tbl[code_col], tbl[ar_col if lang == "العربية" else en_col]))
                    summary["SERVICE"] = summary["SERVICE"].map(name_map).fillna(summary["SERVICE"])

            # 🧹 تنسيق الأعمدة
            summary.rename(columns={"SERVICE": "الخدمة / Service"}, inplace=True)

            # 🚫 عرض فقط الخدمات التي بها 30 ردًا أو أكثر
            summary = summary[summary["عدد الردود / Responses"] >= 30]

            # 🧭 ترتيب الجدول تنازليًا حسب السعادة
            summary = summary.sort_values("Happiness / سعادة (٪)", ascending=False)

            # ✅ تلوين الخلايا في الجدول (السعادة والقيمة فقط)
            def color_cells(val):
                try:
                    v = float(val)
                    if v < 70:
                        color = "#FF6B6B"  # أحمر
                    elif v < 80:
                        color = "#FFD93D"  # أصفر
                    elif v < 90:
                        color = "#6BCB77"  # أخضر
                    else:
                        color = "#4D96FF"  # أزرق
                    return f"background-color:{color};color:black"
                except:
                    return ""

            # 📋 عرض الجدول
            styled_table = (
                summary.style
                .format({
                    "Happiness / سعادة (٪)": "{:.1f}%",
                    "Value / قيمة (٪)": "{:.1f}%",
                    "NPS / صافي نقاط الترويج (٪)": "{:.1f}%",
                    "عدد الردود / Responses": "{:,.0f}"
                })
                .map(color_cells, subset=["Happiness / سعادة (٪)", "Value / قيمة (٪)"])
            )
            st.dataframe(styled_table, use_container_width=True)

            # 🛈 ملاحظة توضيحية باللغتين
            st.markdown(bi_text(
                """
                **ℹ️ ملاحظة:**  
                يتم عرض الخدمات التي تحتوي على **30 ردًا أو أكثر فقط** لضمان دقة النتائج.  
                """,
                """
                **ℹ️ Note:**  
                Only **services with 30 or more responses** are shown to ensure result accuracy.
                """
            ))

            # 🎨 الرسم البياني (السعادة والقيمة فقط)
            if not summary.empty:
                df_melted = summary.melt(
                    id_vars=["الخدمة / Service", "عدد الردود / Responses"],
                    value_vars=["Happiness / سعادة (٪)", "Value / قيمة (٪)"],
                    var_name="المؤشر / Indicator",
                    value_name="القيمة / Value"
                )

                chart_title = "📊 مقارنة مؤشري السعادة والقيمة حسب الخدمة / Comparison of Happiness and Value by Service"

                fig = px.bar(
                    df_melted,
                    x="الخدمة / Service",
                    y="القيمة / Value",
                    color="المؤشر / Indicator",
                    barmode="group",
                    text="القيمة / Value",
                    title=chart_title,
                    color_discrete_sequence=PASTEL
                )

                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")

                # 🎯 خط مستهدف عند 80%
                fig.add_shape(
                    type="line",
                    x0=-0.5, x1=len(summary) - 0.5,
                    y0=80, y1=80,
                    line=dict(color="green", dash="dash", width=2)
                )
                fig.add_annotation(
                    xref="paper", x=1.02, y=80,
                    text=bi_text("🎯 الحد المستهدف (80%)", "🎯 Target Threshold (80%)"),
                    showarrow=False,
                    font=dict(color="green")
                )

                fig.update_layout(
                    title=dict(
                        text=chart_title,
                        x=0.5,  # 📍 العنوان في المنتصف
                        xanchor="center",
                        font=dict(size=18, family="Cairo, sans-serif", color="#333")
                    ),
                    yaxis_title=bi_text("النسبة المئوية (%)", "Percentage (%)"),
                    xaxis_title=bi_text("الخدمة / Service", "Service"),
                    legend_title=bi_text("المؤشر", "Indicator"),
                    yaxis=dict(range=[0, 100])
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(bi_text(
                    "ℹ️ لا توجد خدمات تحتوي على 30 ردًا أو أكثر.",
                    "ℹ️ No services with 30 or more responses found."
                ))
# =========================================================
# 💬 PARETO TAB — تحليل الملاحظات النوعية
# =========================================================
with tab_pareto:
    st.subheader(bi_text("💬 تحليل المزعجات ", "Customer Comments )"))
    st.info(bi_text(
        "تحليل الملاحظات النوعية لتحديد أكثر الأسباب شيوعًا لعدم الرضا",
        "Qualitative analysis of comments to identify top dissatisfaction reasons."
    ))

    # 🔍 البحث عن العمود النصي المناسب
    text_cols = [c for c in df.columns if any(k in c.lower() for k in ["comment", "ملاحظ", "unsat", "reason"])]
    if not text_cols:
        st.warning("⚠️ لا يوجد عمود نصي لتحليل Pareto.")
    else:
        col = text_cols[0]

        # 🧹 تنظيف النصوص
        df["__clean"] = df[col].astype(str).str.lower()
        df["__clean"] = df["__clean"].replace(r"[^\u0600-\u06FFA-Za-z0-9\s]", " ", regex=True)
        df["__clean"] = df["__clean"].replace(r"\s+", " ", regex=True).str.strip()

        empty_terms = {"", " ", "لا يوجد", "لايوجد", "لا شيء", "no", "none", "nothing", "جيد", "ممتاز", "ok"}
        df = df[~df["__clean"].isin(empty_terms)]
        df = df[df["__clean"].str.split().str.len() >= 3]

        # 🧭 تصنيف الملاحظات إلى محاور رئيسية
        themes = {
            "Parking / مواقف السيارات": ["موقف", "مواقف", "parking"],
            "Waiting / الانتظار": ["انتظار", "بطء", "delay", "slow"],
            "Staff / الموظفون": ["موظف", "تعامل", "staff"],
            "Fees / الرسوم": ["رسوم", "دفع", "fee"],
            "Process / الإجراءات": ["اجراء", "process", "انجاز"],
            "Platform / المنصة": ["تطبيق", "app", "system"],
            "Facility / المكان": ["مكان", "نظافة", "ازدحام"],
            "Communication / التواصل": ["رد", "تواصل", "اتصال"]
        }

        def classify_theme(t):
            for th, ws in themes.items():
                if any(w in t for w in ws):
                    return th
            return "Other / أخرى"

        df["Theme"] = df["__clean"].apply(classify_theme)
        df = df[df["Theme"] != "Other / أخرى"]

        # 📊 حساب عدد الملاحظات والنسب
        counts = df["Theme"].value_counts().reset_index()
        counts.columns = ["Theme", "Count"]
        counts["%"] = counts["Count"] / counts["Count"].sum() * 100
        counts["Cum%"] = counts["%"].cumsum()

        # 🎨 تلوين الأعمدة:
        # - أحمر حتى 80٪
        # - وأيضًا العمود الأول الذي يتجاوز 80٪ يُلون بالأحمر
        # - الرمادي للبقية
        counts["Color"] = np.where(counts["Cum%"] <= 80, "#e74c3c", "#95a5a6")
        if not counts[counts["Cum%"] > 80].empty:
            first_above_80_index = counts[counts["Cum%"] > 80].index[0]
            counts.loc[first_above_80_index, "Color"] = "#e74c3c"

        # 🗂️ تجميع الإجابات النصية لكل محور
        all_answers = df.groupby("Theme")["__clean"].apply(lambda x: " / ".join(x.astype(str))).reset_index()
        counts = counts.merge(all_answers, on="Theme", how="left")
        counts.rename(columns={"__clean": "جميع الإجابات / All Responses"}, inplace=True)

        # 📋 تجهيز الجدول وعناوين الأعمدة ثنائية اللغة
        pareto_display = counts.drop(columns=["Color"], errors="ignore").reset_index(drop=True)
        pareto_display.rename(columns={
            "Theme": "المحور / Theme",
            "Count": "عدد الملاحظات / Count",
            "%": "النسبة / %",
            "Cum%": "النسبة التراكمية / Cum%",
            "جميع الإجابات / All Responses": "جميع الإجابات / All Responses"
        }, inplace=True)

        # 🧾 عرض الجدول
        st.dataframe(
            pareto_display[
                ["المحور / Theme", "عدد الملاحظات / Count", "النسبة / %", "النسبة التراكمية / Cum%", "جميع الإجابات / All Responses"]
            ].style.format({"النسبة / %": "{:.1f}", "النسبة التراكمية / Cum%": "{:.1f}"}),
            use_container_width=True,
            hide_index=True
        )

        # 📈 رسم باريتو بالألوان المعدّلة
        fig = go.Figure()
        fig.add_bar(
            x=counts["Theme"],
            y=counts["Count"],
            marker_color=counts["Color"],
            name=bi_text("عدد الملاحظات", "Count")
        )
        fig.add_scatter(
            x=counts["Theme"],
            y=counts["Cum%"],
            name=bi_text("النسبة التراكمية", "Cumulative %"),
            yaxis="y2",
            mode="lines+markers",
            marker=dict(color="#2c3e50")
        )

        # 🎨 تصميم الرسم البياني
        fig.update_layout(
            title=dict(
                text=bi_text("📊 تحليل باريتو — المحاور الرئيسية", "📊 Pareto Analysis — Key Themes"),
                x=0.5,  # العنوان في المنتصف
                xanchor="center",
                font=dict(size=18, color="#333")
            ),
            yaxis=dict(title=bi_text("عدد الملاحظات", "Number of Comments")),
            yaxis2=dict(title=bi_text("النسبة التراكمية (%)", "Cumulative Percentage (%)"), overlaying="y", side="right"),
            bargap=0.25,
            height=600,
            legend=dict(orientation="h", y=-0.2)
        )

        st.plotly_chart(fig, use_container_width=True)

        # 🧠 تعليق تفسيري بسيط
        top80 = counts[counts["Cum%"] <= 80]
        if not top80.empty:
            top_themes = "، ".join(top80["Theme"].tolist())
            st.markdown(
                f"✅ **{bi_text('تمثل المحاور التالية نحو 80٪ من أسباب عدم الرضا:', 'These themes represent about 80% of dissatisfaction reasons:')}**<br>{top_themes}",
                unsafe_allow_html=True
            )

        # 📥 زر تنزيل النتائج
        pareto_buffer = io.BytesIO()
        with pd.ExcelWriter(pareto_buffer, engine="openpyxl") as writer:
            pareto_display.to_excel(writer, index=False, sheet_name="Pareto_Results")

        st.download_button(
            "📥 تنزيل جدول Pareto (Excel)",
            data=pareto_buffer.getvalue(),
            file_name=f"Pareto_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# =========================================================
# 🎨 جعل شعار "Hosted with Streamlit" شفافًا (إخفاء بصري)
# =========================================================
st.markdown("""
    <style>
    /* إخفاء القائمة العلوية */
    #MainMenu {visibility: hidden;}

    /* جعل الفوتر شبه شفاف وغير ظاهر */
    footer, [data-testid="stFooter"] {
        opacity: 0.03 !important;     /* شفافية شبه كاملة */
        height: 1px !important;       /* تقليص الارتفاع */
        overflow: hidden !important;  /* منع النص من الظهور */
    }

    /* إخفاء الأزرار الجانبية مثل Manage app */
    [data-testid="stActionButtonIcon"],
    .stAppDeployButton, 
    .viewerBadge_link__1S137,
    .stDeployButton {
        opacity: 0 !important;
        height: 0 !important;
        visibility: hidden !important;
    }
    </style>
""", unsafe_allow_html=True)












