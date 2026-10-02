import sys
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PROJECT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Feedback Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# IMPORT PIPELINE
# ============================================================

try:
    from src.pipeline.pipeline import run_pipeline
except Exception as e:
    st.error("Could not load the NLP pipeline.")
    st.code(str(e))
    st.stop()


# ============================================================
# CUSTOM HTML RENDERER
# ============================================================

def render_html(html):
    """
    Render raw HTML using Streamlit's native HTML renderer.
    This prevents HTML from appearing as Markdown code blocks.
    """
    st.html(html)


# ============================================================
# DARK UI CSS
# ============================================================

st.markdown(
    """
    <style>

    /* =====================================================
       GLOBAL
       ===================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 100% 0%,
                rgba(79, 70, 229, 0.16),
                transparent 28%
            ),
            radial-gradient(
                circle at 0% 0%,
                rgba(14, 165, 233, 0.10),
                transparent 25%
            ),
            #080d1a;
        color: #f8fafc;
    }

    .main {
        background: transparent;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }


    /* =====================================================
       SIDEBAR
       ===================================================== */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #0b1220 0%,
                #080d18 100%
            );
        border-right: 1px solid rgba(148, 163, 184, 0.10);
    }

    section[data-testid="stSidebar"] * {
        color: #e2e8f0;
    }


    /* =====================================================
       TEXT
       ===================================================== */

    h1, h2, h3, h4, h5 {
        color: #f8fafc !important;
    }

    p {
        color: #cbd5e1;
    }


    /* =====================================================
       FILE UPLOADER
       ===================================================== */

    [data-testid="stFileUploader"] {
        background: rgba(15, 23, 42, 0.65);
        border: 1px dashed rgba(148, 163, 184, 0.30);
        border-radius: 16px;
        padding: 10px;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: rgba(99, 102, 241, 0.70);
    }


    /* =====================================================
       BUTTON
       ===================================================== */

    .stButton > button {
        width: 100%;
        border-radius: 12px;
        border: 1px solid rgba(99, 102, 241, 0.45);
        background:
            linear-gradient(
                135deg,
                #4f46e5,
                #6366f1
            );
        color: white;
        font-weight: 700;
        padding: 0.7rem 1rem;
        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow:
            0 10px 30px rgba(79, 70, 229, 0.30);
    }


    /* =====================================================
       METRICS
       ===================================================== */

    [data-testid="stMetric"] {
        background:
            rgba(15, 23, 42, 0.75);
        border:
            1px solid rgba(148, 163, 184, 0.10);
        border-radius: 16px;
        padding: 18px;
        box-shadow:
            0 10px 30px rgba(0, 0, 0, 0.16);
    }

    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }

    [data-testid="stMetricValue"] {
        color: #f8fafc !important;
    }


    /* =====================================================
       DATAFRAME
       ===================================================== */

    [data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
    }


    /* =====================================================
       TABS
       ===================================================== */

    button[data-baseweb="tab"] {
        color: #94a3b8 !important;
        font-weight: 600;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #818cf8 !important;
    }


    /* =====================================================
       ALERTS
       ===================================================== */

    [data-testid="stAlert"] {
        border-radius: 14px;
    }


    /* =====================================================
       FOOTER
       ===================================================== */

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

render_html(
    """
    <div style="
        padding: 1.5rem 0 0.5rem 0;
    ">

        <div style="
            display:flex;
            align-items:center;
            gap:12px;
            margin-bottom:8px;
        ">

            <div style="
                width:48px;
                height:48px;
                border-radius:14px;
                display:flex;
                align-items:center;
                justify-content:center;
                background:
                    linear-gradient(
                        135deg,
                        #4f46e5,
                        #7c3aed
                    );
                font-size:25px;
                box-shadow:
                    0 10px 30px rgba(79,70,229,0.30);
            ">
                🧠
            </div>

            <div>

                <div style="
                    font-size:2rem;
                    font-weight:800;
                    color:#f8fafc;
                    line-height:1.1;
                ">
                    Customer Feedback Intelligence
                </div>

                <div style="
                    color:#64748b;
                    font-size:0.9rem;
                    margin-top:5px;
                ">
                    NLP / ML Portfolio Project
                </div>

            </div>

        </div>

    </div>
    """
)


# ============================================================
# HERO SECTION
# ============================================================

render_html(
    """
    <div style="
        margin-top:20px;
        padding:32px;
        border-radius:22px;
        background:
            linear-gradient(
                135deg,
                rgba(30,41,59,0.85),
                rgba(15,23,42,0.72)
            );
        border:
            1px solid rgba(99,102,241,0.18);
        box-shadow:
            0 20px 60px rgba(0,0,0,0.20);
    ">

        <div style="
            font-size:0.78rem;
            color:#818cf8;
            font-weight:700;
            letter-spacing:0.08em;
            text-transform:uppercase;
            margin-bottom:10px;
        ">
            AI-Powered Customer Analytics
        </div>

        <h1 style="
            font-size:2.1rem;
            margin:0 0 12px 0;
            color:#f8fafc;
        ">
            Turn customer reviews into actionable intelligence.
        </h1>

        <p style="
            max-width:850px;
            color:#94a3b8;
            font-size:1rem;
            line-height:1.7;
            margin:0;
        ">
            Transform unstructured customer feedback into structured
            business insights using Sentiment Analysis, Emotion
            Classification, Topic Modeling, Aspect Extraction,
            and Aspect-Based Sentiment Analysis.
        </p>

    </div>
    """
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    render_html(
        """
        <div style="
            padding:8px 0 18px 0;
        ">

            <div style="
                font-size:1.15rem;
                font-weight:800;
                color:#f8fafc;
            ">
                📁 Input
            </div>

            <div style="
                color:#64748b;
                font-size:0.82rem;
                margin-top:4px;
            ">
                Upload customer reviews
            </div>

        </div>
        """
    )

    uploaded_file = st.file_uploader(
        "Upload customer reviews",
        type=["xlsx", "xls", "pdf"],
        help="Maximum file size: 200MB",
        label_visibility="collapsed",
    )

    render_html(
        """
        <div style="
            margin-top:14px;
            padding:16px;
            border-radius:14px;
            background:rgba(15,23,42,0.70);
            border:
                1px solid rgba(148,163,184,0.10);
        ">

            <div style="
                color:#f8fafc;
                font-weight:700;
                margin-bottom:10px;
            ">
                📄 Supported Formats
            </div>

            <div style="
                color:#94a3b8;
                font-size:0.78rem;
                line-height:1.8;
            ">

                <b style="color:#cbd5e1;">
                    Excel
                </b>

                <br>
                • .xlsx
                <br>
                • .xls

                <br><br>

                <b style="color:#cbd5e1;">
                    PDF
                </b>

                <br>
                Structured format:
                <br>

                <code style="color:#818cf8;">
                    Review 1: ...
                </code>

                <br>

                <code style="color:#818cf8;">
                    Review 2: ...
                </code>

            </div>

        </div>
        """
    )

    render_html(
        """
        <div style="
            margin-top:16px;
            padding:16px;
            border-radius:14px;
            background:
                rgba(15,23,42,0.55);
            border:
                1px solid rgba(148,163,184,0.08);
        ">

            <div style="
                color:#f8fafc;
                font-weight:700;
                margin-bottom:10px;
            ">
                🧠 NLP Components
            </div>

            <div style="
                color:#94a3b8;
                font-size:0.80rem;
                line-height:2;
            ">

                <div>✓ Sentiment Analysis</div>
                <div>✓ Emotion Classification</div>
                <div>✓ Topic Modeling</div>
                <div>✓ Aspect Extraction</div>
                <div>✓ Aspect-Based Sentiment</div>
                <div>✓ Business Insights</div>

            </div>

        </div>
        """
    )

    st.markdown("---")

    render_html(
        """
        <div style="
            color:#64748b;
            font-size:0.72rem;
            line-height:1.6;
        ">
            Built with Python, PyTorch,
            Transformers, BERTopic,
            Sentence Transformers and Streamlit.
        </div>
        """
    )


# ============================================================
# EMPTY STATE
# ============================================================

if uploaded_file is None:

    render_html(
        """
        <div style="
            margin-top:28px;
            padding:30px;
            border-radius:20px;
            background:
                rgba(15,23,42,0.55);
            border:
                1px solid rgba(148,163,184,0.10);
            text-align:center;
        ">

            <div style="
                font-size:3rem;
                margin-bottom:12px;
            ">
                📊
            </div>

            <h2 style="
                margin:0;
                color:#f8fafc;
            ">
                Ready to analyze your customer feedback?
            </h2>

            <p style="
                color:#64748b;
                margin-top:10px;
            ">
                Upload an Excel or PDF file from the sidebar
                to start the analysis.
            </p>

        </div>
        """
    )

    st.markdown("### 🔎 What this application analyzes")

    st.caption(
        "An end-to-end NLP pipeline that converts customer feedback "
        "into structured insights."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        render_html(
            """
            <div style="
                padding:22px;
                min-height:150px;
                border-radius:16px;
                background:rgba(15,23,42,0.65);
                border:
                    1px solid rgba(148,163,184,0.10);
            ">

                <div style="
                    font-size:2.2rem;
                    margin-bottom:10px;
                ">
                    😊
                </div>

                <div style="
                    font-size:1.1rem;
                    font-weight:700;
                    color:#f8fafc;
                ">
                    Sentiment
                </div>

                <div style="
                    color:#94a3b8;
                    font-size:0.82rem;
                    margin-top:5px;
                ">
                    Positive / Negative
                </div>

            </div>
            """
        )

    with col2:

        render_html(
            """
            <div style="
                padding:22px;
                min-height:150px;
                border-radius:16px;
                background:rgba(15,23,42,0.65);
                border:
                    1px solid rgba(148,163,184,0.10);
            ">

                <div style="
                    font-size:2.2rem;
                    margin-bottom:10px;
                ">
                    🎭
                </div>

                <div style="
                    font-size:1.1rem;
                    font-weight:700;
                    color:#f8fafc;
                ">
                    Emotion
                </div>

                <div style="
                    color:#94a3b8;
                    font-size:0.82rem;
                    margin-top:5px;
                ">
                    Customer emotions
                </div>

            </div>
            """
        )

    with col3:

        render_html(
            """
            <div style="
                padding:22px;
                min-height:150px;
                border-radius:16px;
                background:rgba(15,23,42,0.65);
                border:
                    1px solid rgba(148,163,184,0.10);
            ">

                <div style="
                    font-size:2.2rem;
                    margin-bottom:10px;
                ">
                    🏷️
                </div>

                <div style="
                    font-size:1.1rem;
                    font-weight:700;
                    color:#f8fafc;
                ">
                    Topics
                </div>

                <div style="
                    color:#94a3b8;
                    font-size:0.82rem;
                    margin-top:5px;
                ">
                    Main themes
                </div>

            </div>
            """
        )

    col4, col5, col6 = st.columns(3)

    with col4:

        render_html(
            """
            <div style="
                margin-top:16px;
                padding:22px;
                min-height:150px;
                border-radius:16px;
                background:rgba(15,23,42,0.65);
                border:
                    1px solid rgba(148,163,184,0.10);
            ">

                <div style="
                    font-size:2.2rem;
                    margin-bottom:10px;
                ">
                    🔍
                </div>

                <div style="
                    font-size:1.1rem;
                    font-weight:700;
                    color:#f8fafc;
                ">
                    Aspects
                </div>

                <div style="
                    color:#94a3b8;
                    font-size:0.82rem;
                    margin-top:5px;
                ">
                    Product attributes
                </div>

            </div>
            """
        )

    with col5:

        render_html(
            """
            <div style="
                margin-top:16px;
                padding:22px;
                min-height:150px;
                border-radius:16px;
                background:rgba(15,23,42,0.65);
                border:
                    1px solid rgba(148,163,184,0.10);
            ">

                <div style="
                    font-size:2.2rem;
                    margin-bottom:10px;
                ">
                    🎯
                </div>

                <div style="
                    font-size:1.1rem;
                    font-weight:700;
                    color:#f8fafc;
                ">
                    ABSA
                </div>

                <div style="
                    color:#94a3b8;
                    font-size:0.82rem;
                    margin-top:5px;
                ">
                    Aspect sentiment
                </div>

            </div>
            """
        )

    with col6:

        render_html(
            """
            <div style="
                margin-top:16px;
                padding:22px;
                min-height:150px;
                border-radius:16px;
                background:rgba(15,23,42,0.65);
                border:
                    1px solid rgba(148,163,184,0.10);
            ">

                <div style="
                    font-size:2.2rem;
                    margin-bottom:10px;
                ">
                    💼
                </div>

                <div style="
                    font-size:1.1rem;
                    font-weight:700;
                    color:#f8fafc;
                ">
                    Business Insights
                </div>

                <div style="
                    color:#94a3b8;
                    font-size:0.82rem;
                    margin-top:5px;
                ">
                    Actionable intelligence
                </div>

            </div>
            """
        )

    st.stop()


# ============================================================
# FILE INFORMATION
# ============================================================

st.markdown("## 📁 Uploaded Dataset")

file_name = uploaded_file.name
file_size_mb = uploaded_file.size / (1024 * 1024)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "File",
        file_name
    )

with col2:
    st.metric(
        "Format",
        Path(file_name).suffix.upper().replace(".", "")
    )

with col3:
    st.metric(
        "Size",
        f"{file_size_mb:.2f} MB"
    )


# ============================================================
# SAVE TEMPORARY INPUT
# ============================================================

TEMP_DIR = PROJECT_ROOT / "data" / "raw"

TEMP_DIR.mkdir(
    parents=True,
    exist_ok=True
)

input_path = TEMP_DIR / uploaded_file.name

with open(input_path, "wb") as f:
    f.write(uploaded_file.getbuffer())


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.markdown("")

analyze_button = st.button(
    "🚀 Analyze Customer Feedback",
    use_container_width=True
)


# ============================================================
# RUN PIPELINE
# ============================================================

if analyze_button:

    progress_container = st.empty()

    progress_container.info(
        "🧠 Running the NLP pipeline. "
        "This may take a little while depending on the number of reviews."
    )

    try:

        with st.spinner(
            "Running Sentiment, Emotion, Topics, Aspects and ABSA..."
        ):

            pipeline_output = run_pipeline(
                input_path
            )

        progress_container.empty()

        st.success(
            "✅ Analysis completed successfully."
        )


        # ----------------------------------------------------
        # NORMALIZE OUTPUT
        # ----------------------------------------------------

        if isinstance(pipeline_output, tuple):

            if len(pipeline_output) >= 1:
                results_df = pipeline_output[0]
            else:
                results_df = pd.DataFrame()

            if len(pipeline_output) >= 2:
                insights = pipeline_output[1]
            else:
                insights = {}

        elif isinstance(pipeline_output, pd.DataFrame):

            results_df = pipeline_output
            insights = {}

        elif isinstance(pipeline_output, dict):

            results_df = pipeline_output.get(
                "results",
                pipeline_output.get(
                    "data",
                    pd.DataFrame()
                )
            )

            insights = pipeline_output.get(
                "insights",
                {}
            )

        else:

            results_df = pd.DataFrame()
            insights = {}

        if not isinstance(results_df, pd.DataFrame):
            results_df = pd.DataFrame(results_df)

        if not isinstance(insights, dict):
            insights = {}


        # ====================================================
        # EXECUTIVE OVERVIEW
        # ====================================================

        st.markdown("---")
        st.markdown("## 📊 Executive Overview")

        total_reviews = len(results_df)

        positive_count = 0
        negative_count = 0

        if "sentiment" in results_df.columns:

            sentiment_counts = (
                results_df["sentiment"]
                .astype(str)
                .str.lower()
                .value_counts()
            )

            positive_count = int(
                sentiment_counts.get("positive", 0)
            )

            negative_count = int(
                sentiment_counts.get("negative", 0)
            )

        positive_rate = (
            positive_count / total_reviews * 100
            if total_reviews
            else 0
        )

        negative_rate = (
            negative_count / total_reviews * 100
            if total_reviews
            else 0
        )

        aspect_count = 0

        if "aspects" in results_df.columns:

            for value in results_df["aspects"]:

                if isinstance(value, list):

                    aspect_count += len(value)

                elif pd.notna(value):

                    try:

                        parsed = json.loads(value)

                        if isinstance(parsed, list):
                            aspect_count += len(parsed)

                    except Exception:
                        pass


        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Reviews",
                f"{total_reviews:,}"
            )

        with col2:
            st.metric(
                "Positive",
                f"{positive_rate:.1f}%"
            )

        with col3:
            st.metric(
                "Negative",
                f"{negative_rate:.1f}%"
            )

        with col4:
            st.metric(
                "Detected Aspects",
                f"{aspect_count:,}"
            )


        # ====================================================
        # TABS
        # ====================================================

        (
            tab_overview,
            tab_sentiment,
            tab_topics,
            tab_aspects,
            tab_reviews
        ) = st.tabs(
            [
                "📊 Overview",
                "😊 Sentiment & Emotion",
                "🏷️ Topics",
                "🔍 Aspects & ABSA",
                "📄 Reviews",
            ]
        )


        # ====================================================
        # OVERVIEW TAB
        # ====================================================

        with tab_overview:

            st.markdown("### 🧠 NLP Intelligence Pipeline")

            col1, col2, col3 = st.columns(3)

            with col1:

                render_html(
                    """
                    <div style="
                        padding:20px;
                        border-radius:16px;
                        background:rgba(15,23,42,0.65);
                        border:1px solid rgba(148,163,184,0.10);
                    ">

                        <div style="
                            font-size:2rem;
                        ">
                            😊
                        </div>

                        <div style="
                            font-weight:700;
                            color:#f8fafc;
                            margin-top:8px;
                        ">
                            Sentiment
                        </div>

                        <div style="
                            color:#64748b;
                            font-size:0.82rem;
                            margin-top:5px;
                        ">
                            Customer polarity
                        </div>

                    </div>
                    """
                )

            with col2:

                render_html(
                    """
                    <div style="
                        padding:20px;
                        border-radius:16px;
                        background:rgba(15,23,42,0.65);
                        border:1px solid rgba(148,163,184,0.10);
                    ">

                        <div style="
                            font-size:2rem;
                        ">
                            🎭
                        </div>

                        <div style="
                            font-weight:700;
                            color:#f8fafc;
                            margin-top:8px;
                        ">
                            Emotion
                        </div>

                        <div style="
                            color:#64748b;
                            font-size:0.82rem;
                            margin-top:5px;
                        ">
                            Emotional signals
                        </div>

                    </div>
                    """
                )

            with col3:

                render_html(
                    """
                    <div style="
                        padding:20px;
                        border-radius:16px;
                        background:rgba(15,23,42,0.65);
                        border:1px solid rgba(148,163,184,0.10);
                    ">

                        <div style="
                            font-size:2rem;
                        ">
                            💼
                        </div>

                        <div style="
                            font-weight:700;
                            color:#f8fafc;
                            margin-top:8px;
                        ">
                            Business Insights
                        </div>

                        <div style="
                            color:#64748b;
                            font-size:0.82rem;
                            margin-top:5px;
                        ">
                            Actionable findings
                        </div>

                    </div>
                    """
                )


            st.markdown("### 💡 Business Insights")

            if insights:

                for key, value in insights.items():

                    if isinstance(value, dict):

                        st.markdown(
                            f"#### {str(key).replace('_', ' ').title()}"
                        )

                        insight_rows = []

                        for sub_key, sub_value in value.items():

                            insight_rows.append(
                                {
                                    "Metric": str(sub_key)
                                    .replace("_", " ")
                                    .title(),
                                    "Value": sub_value
                                }
                            )

                        if insight_rows:

                            st.dataframe(
                                pd.DataFrame(insight_rows),
                                use_container_width=True,
                                hide_index=True
                            )

                    elif isinstance(value, list):

                        st.markdown(
                            f"#### {str(key).replace('_', ' ').title()}"
                        )

                        for item in value:
                            st.write(f"• {item}")

                    else:

                        st.markdown(
                            f"**{str(key).replace('_', ' ').title()}:** "
                            f"{value}"
                        )

            else:

                st.info(
                    "Business insights were generated from the analysis, "
                    "but no additional structured summary was returned."
                )


        # ====================================================
        # SENTIMENT TAB
        # ====================================================

        with tab_sentiment:

            st.markdown("## 😊 Sentiment Analysis")

            if "sentiment" in results_df.columns:

                sentiment_counts = (
                    results_df["sentiment"]
                    .astype(str)
                    .str.lower()
                    .value_counts()
                )

                sentiment_df = (
                    sentiment_counts
                    .rename_axis("Sentiment")
                    .reset_index(name="Reviews")
                )

                sentiment_df["Percentage"] = (
                    sentiment_df["Reviews"]
                    / total_reviews
                    * 100
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.markdown("### Distribution")

                    fig_sentiment = px.pie(
                        sentiment_df,
                        names="Sentiment",
                        values="Reviews",
                        hole=0.55,
                        title="Sentiment Distribution"
                    )

                    fig_sentiment.update_traces(
                        texttemplate="%{percent}",
                        textposition="inside"
                    )

                    fig_sentiment.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(
                            l=20,
                            r=20,
                            t=60,
                            b=20
                        ),
                        legend_title_text="Sentiment"
                    )

                    st.plotly_chart(
                        fig_sentiment,
                        use_container_width=True
                    )

                with col2:

                    st.markdown("### Summary")

                    for _, row in sentiment_df.iterrows():

                        label = str(
                            row["Sentiment"]
                        ).title()

                        count = int(
                            row["Reviews"]
                        )

                        percentage = (
                            row["Percentage"]
                        )

                        st.write(
                            f"**{label}** — "
                            f"{count} reviews "
                            f"({percentage:.1f}%)"
                        )

            else:

                st.warning(
                    "Sentiment results are not available."
                )


            st.markdown("---")

            st.markdown("## 🎭 Emotion Classification")

            if "emotion" in results_df.columns:

                emotion_counts = (
                    results_df["emotion"]
                    .astype(str)
                    .value_counts()
                )

                emotion_df = (
                    emotion_counts
                    .rename_axis("Emotion")
                    .reset_index(name="Reviews")
                )

                emotion_df["Percentage"] = (
                    emotion_df["Reviews"]
                    / total_reviews
                    * 100
                )

                col1, col2 = st.columns(2)

                with col1:

                    fig_emotion = px.pie(
                        emotion_df,
                        names="Emotion",
                        values="Reviews",
                        hole=0.55,
                        title="Emotion Distribution"
                    )

                    fig_emotion.update_traces(
                        texttemplate="%{percent}",
                        textposition="inside"
                    )

                    fig_emotion.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(
                            l=20,
                            r=20,
                            t=60,
                            b=20
                        ),
                        legend_title_text="Emotion"
                    )

                    st.plotly_chart(
                        fig_emotion,
                        use_container_width=True
                    )

                with col2:

                    st.dataframe(
                        emotion_df,
                        use_container_width=True,
                        hide_index=True
                    )

            else:

                st.warning(
                    "Emotion results are not available."
                )


        # ====================================================
        # TOPICS TAB
        # ====================================================

        with tab_topics:

            st.markdown("## 🏷️ Topic Modeling")

            if "topic" in results_df.columns:

                topic_counts = (
                    results_df["topic"]
                    .astype(str)
                    .value_counts()
                )

                topic_df = (
                    topic_counts
                    .rename_axis("Topic")
                    .reset_index(name="Reviews")
                )

                topic_df["Percentage"] = (
                    topic_df["Reviews"]
                    / total_reviews
                    * 100
                )

                fig_topics = px.pie(
                    topic_df,
                    names="Topic",
                    values="Reviews",
                    hole=0.45,
                    title="Topic Distribution"
                )

                fig_topics.update_traces(
                    texttemplate="%{percent}",
                    textposition="inside"
                )

                fig_topics.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(
                        l=20,
                        r=20,
                        t=60,
                        b=20
                    ),
                    legend_title_text="Topic"
                )

                st.plotly_chart(
                    fig_topics,
                    use_container_width=True
                )

                st.markdown("### Topic Distribution")

                st.dataframe(
                    topic_df,
                    use_container_width=True,
                    hide_index=True
                )

            elif "topic_id" in results_df.columns:

                topic_counts = (
                    results_df["topic_id"]
                    .value_counts()
                )

                topic_df = (
                    topic_counts
                    .rename_axis("Topic ID")
                    .reset_index(name="Reviews")
                )

                topic_df["Percentage"] = (
                    topic_df["Reviews"]
                    / total_reviews
                    * 100
                )

                fig_topics = px.pie(
                    topic_df,
                    names="Topic ID",
                    values="Reviews",
                    hole=0.45,
                    title="Topic Distribution"
                )

                fig_topics.update_traces(
                    texttemplate="%{percent}",
                    textposition="inside"
                )

                fig_topics.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(
                        l=20,
                        r=20,
                        t=60,
                        b=20
                    )
                )

                st.plotly_chart(
                    fig_topics,
                    use_container_width=True
                )

                st.dataframe(
                    topic_df,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.warning(
                    "Topic modeling results are not available."
                )


        # ====================================================
        # ASPECTS TAB
        # ====================================================

        with tab_aspects:

            st.markdown("## 🔍 Aspect Intelligence")

            if "aspects" in results_df.columns:

                aspect_rows = []

                for _, row in results_df.iterrows():

                    aspects = row["aspects"]

                    if isinstance(aspects, list):

                        for aspect in aspects:

                            aspect_rows.append(
                                {
                                    "Aspect": aspect
                                }
                            )

                    elif pd.notna(aspects):

                        try:

                            parsed = json.loads(aspects)

                            if isinstance(parsed, list):

                                for aspect in parsed:

                                    aspect_rows.append(
                                        {
                                            "Aspect": aspect
                                        }
                                    )

                        except Exception:
                            pass


                if aspect_rows:

                    aspect_df = pd.DataFrame(
                        aspect_rows
                    )

                    aspect_counts = (
                        aspect_df["Aspect"]
                        .astype(str)
                        .value_counts()
                    )

                    aspect_summary = (
                        aspect_counts
                        .rename_axis("Aspect")
                        .reset_index(name="Mentions")
                    )

                    aspect_summary["Percentage"] = (
                        aspect_summary["Mentions"]
                        / aspect_summary["Mentions"].sum()
                        * 100
                    )

                    fig_aspects = px.bar(
                        aspect_summary,
                        x="Percentage",
                        y="Aspect",
                        orientation="h",
                        text="Percentage",
                        title="Aspect Mentions (%)"
                    )

                    fig_aspects.update_traces(
                        texttemplate="%{text:.1f}%",
                        textposition="outside"
                    )

                    fig_aspects.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        xaxis_title="Percentage",
                        yaxis_title="Aspect",
                        margin=dict(
                            l=20,
                            r=60,
                            t=60,
                            b=40
                        )
                    )

                    st.plotly_chart(
                        fig_aspects,
                        use_container_width=True
                    )

                    st.markdown(
                        "### Aspect Mentions"
                    )

                    st.dataframe(
                        aspect_summary,
                        use_container_width=True,
                        hide_index=True
                    )

                else:

                    st.info(
                        "No aspects were detected."
                    )

            else:

                st.warning(
                    "Aspect extraction results are not available."
                )


            st.markdown("---")

            st.markdown(
                "## 🎯 Aspect-Based Sentiment Analysis"
            )

            if "aspect_sentiments" in results_df.columns:

                absa_rows = []

                for _, row in results_df.iterrows():

                    value = row[
                        "aspect_sentiments"
                    ]

                    if isinstance(value, list):

                        for item in value:

                            if isinstance(item, dict):

                                absa_rows.append(item)

                    elif pd.notna(value):

                        try:

                            parsed = json.loads(value)

                            if isinstance(parsed, list):

                                for item in parsed:

                                    if isinstance(item, dict):

                                        absa_rows.append(item)

                        except Exception:
                            pass


                if absa_rows:

                    absa_df = pd.DataFrame(
                        absa_rows
                    )

                    preferred_columns = [
                        "aspect",
                        "sentiment",
                        "confidence",
                        "confidence_level",
                    ]

                    existing_columns = [
                        c
                        for c in preferred_columns
                        if c in absa_df.columns
                    ]

                    if existing_columns:

                        st.dataframe(
                            absa_df[existing_columns],
                            use_container_width=True,
                            hide_index=True
                        )

                    else:

                        st.dataframe(
                            absa_df,
                            use_container_width=True,
                            hide_index=True
                        )


                    # ------------------------------------------------
                    # ABSA PERCENTAGE CHART
                    # ------------------------------------------------

                    if (
                        "aspect" in absa_df.columns
                        and "sentiment" in absa_df.columns
                    ):

                        absa_chart_df = (
                            absa_df[
                                ["aspect", "sentiment"]
                            ]
                            .copy()
                        )

                        absa_chart_df["aspect"] = (
                            absa_chart_df["aspect"]
                            .astype(str)
                        )

                        absa_chart_df["sentiment"] = (
                            absa_chart_df["sentiment"]
                            .astype(str)
                            .str.lower()
                        )

                        absa_counts = (
                            absa_chart_df
                            .groupby(
                                ["aspect", "sentiment"]
                            )
                            .size()
                            .reset_index(
                                name="Count"
                            )
                        )

                        absa_counts["Percentage"] = (
                            absa_counts
                            .groupby("aspect")["Count"]
                            .transform(
                                lambda x:
                                x / x.sum() * 100
                            )
                        )

                        fig_absa = px.bar(
                            absa_counts,
                            x="aspect",
                            y="Percentage",
                            color="sentiment",
                            text="Percentage",
                            barmode="stack",
                            title="Aspect Sentiment Distribution (%)"
                        )

                        fig_absa.update_traces(
                            texttemplate="%{text:.1f}%",
                            textposition="inside"
                        )

                        fig_absa.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            xaxis_title="Aspect",
                            yaxis_title="Percentage",
                            legend_title_text="Sentiment",
                            margin=dict(
                                l=20,
                                r=20,
                                t=60,
                                b=80
                            )
                        )

                        st.plotly_chart(
                            fig_absa,
                            use_container_width=True
                        )

                else:

                    st.info(
                        "No aspect sentiment predictions available."
                    )

            else:

                st.info(
                    "Aspect-based sentiment results are not available."
                )


        # ====================================================
        # REVIEWS TAB
        # ====================================================

        with tab_reviews:

            st.markdown("## 📄 Review-Level Results")

            display_df = results_df.copy()

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                height=500
            )

            st.markdown("---")

            st.markdown(
                "### ⬇️ Export Results"
            )

            col1, col2 = st.columns(2)

            with col1:

                csv_data = results_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv_data,
                    file_name="customer_feedback_analysis.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            with col2:

                json_data = results_df.to_json(
                    orient="records",
                    force_ascii=False,
                    indent=2
                )

                st.download_button(
                    label="⬇️ Download JSON",
                    data=json_data,
                    file_name="customer_feedback_analysis.json",
                    mime="application/json",
                    use_container_width=True,
                )


        # ====================================================
        # FINAL BUSINESS SUMMARY
        # ====================================================

        st.markdown("---")

        render_html(
            """
            <div style="
                padding:24px;
                border-radius:18px;
                background:
                    linear-gradient(
                        135deg,
                        rgba(30,41,59,0.85),
                        rgba(15,23,42,0.70)
                    );
                border:
                    1px solid rgba(99,102,241,0.18);
            ">

                <div style="
                    color:#818cf8;
                    font-size:0.75rem;
                    font-weight:700;
                    letter-spacing:0.08em;
                    text-transform:uppercase;
                ">
                    BUSINESS INTELLIGENCE
                </div>

                <div style="
                    font-size:1.35rem;
                    font-weight:800;
                    color:#f8fafc;
                    margin-top:6px;
                ">
                    Customer feedback has been transformed
                    into structured business intelligence.
                </div>

                <div style="
                    color:#94a3b8;
                    margin-top:8px;
                    line-height:1.6;
                ">
                    Use sentiment, emotions, topics and aspect-level
                    signals to identify customer pain points,
                    product strengths and areas requiring attention.
                </div>

            </div>
            """
        )


    except Exception as e:

        progress_container.empty()

        st.error(
            "❌ An error occurred while running the NLP pipeline."
        )

        with st.expander(
            "Show technical error"
        ):
            st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

render_html(
    """
    <div style="
        text-align:center;
        padding:20px 0 10px 0;
        color:#475569;
        font-size:0.75rem;
    ">
        Customer Feedback Intelligence
        &nbsp;•&nbsp;
        NLP / ML Portfolio Project
        &nbsp;•&nbsp;
        Built with Python & Streamlit
    </div>
    """
)
#streamlit run app/app.py