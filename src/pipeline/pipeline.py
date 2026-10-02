from pathlib import Path
import json
import re

import pandas as pd
import pymupdf
import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SENTIMENT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "sentiment"
    / "bert_binary_weighted"
)

EMOTION_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "emotion"
    / "bert_emotion"
)

TOPIC_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "topics"
    / "bertopic_final"
)

ASPECT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "aspects"
    / "bert_aspect_extraction"
)

ABSA_MODEL_NAME = "yangheng/deberta-v3-base-absa-v1.1"


# ============================================================
# 2. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

PIPELINE_DEVICE = (
    0 if torch.cuda.is_available() else -1
)


# ============================================================
# 3. LOAD SENTIMENT MODEL
# ============================================================

sentiment_tokenizer = AutoTokenizer.from_pretrained(
    SENTIMENT_MODEL_PATH
)

sentiment_model = AutoModelForSequenceClassification.from_pretrained(
    SENTIMENT_MODEL_PATH
)

sentiment_model.to(DEVICE)
sentiment_model.eval()


# ============================================================
# 4. LOAD EMOTION MODEL
# ============================================================

emotion_tokenizer = AutoTokenizer.from_pretrained(
    EMOTION_MODEL_PATH
)

emotion_model = AutoModelForSequenceClassification.from_pretrained(
    EMOTION_MODEL_PATH
)

emotion_model.to(DEVICE)
emotion_model.eval()


# ============================================================
# 5. LOAD BERTopic
# ============================================================

from bertopic import BERTopic

topic_model = BERTopic.load(
    str(TOPIC_MODEL_PATH)
)

topic_embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device="cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 6. TOPIC NAME MAP
# ============================================================

topic_name_map = {
    0: "0_bag_great_quality_watch",
    1: "1_sound_work_charge_battery",
    2: "2_size_comfortable_fit_shoes",
    3: "3_light_bright_led",
    4: "4_wall_stand_mount",
    5: "5_received_item_box_arrived",
    6: "6_water_bottle_hose_filter",
    7: "7_dog_collar",
    8: "8_chair_seat",
    9: "9_phone_case_screen",
    10: "10_easy_camera_install_use",
}


# ============================================================
# 7. LOAD ASPECT EXTRACTION MODEL
# ============================================================

aspect_tokenizer = AutoTokenizer.from_pretrained(
    ASPECT_MODEL_PATH
)

aspect_model = AutoModelForSequenceClassification.from_pretrained(
    ASPECT_MODEL_PATH
)

aspect_model.to(DEVICE)
aspect_model.eval()


# ============================================================
# 8. ASPECT VOCABULARY
# ============================================================

ASPECT_TERMS = {
    "delivery": [
        "delivery",
        "shipping",
        "shipment",
        "arrived",
        "arrival",
        "delivered",
        "deliver"
    ],
    "quality": [
        "quality",
        "durable",
        "durability",
        "material"
    ],
    "price": [
        "price",
        "cost",
        "value",
        "expensive",
        "cheap"
    ],
    "size": [
        "size",
        "sizing",
        "small",
        "large",
        "bigger",
        "smaller",
        "length",
        "width"
    ],
    "fit": [
        "fit",
        "fits",
        "fitting",
        "comfortable",
        "comfort"
    ],
    "battery": [
        "battery",
        "charging",
        "charge",
        "charger"
    ],
    "sound": [
        "sound",
        "audio",
        "volume",
        "bass",
        "noise"
    ],
    "customer_service": [
        "service",
        "support",
        "customer service"
    ],
    "packaging": [
        "packaging",
        "package",
        "box"
    ],
    "design": [
        "design",
        "style",
        "appearance",
        "look"
    ],
    "performance": [
        "performance",
        "works",
        "working",
        "function",
        "speed"
    ],
    "ease_of_use": [
        "easy",
        "easier",
        "simple",
        "use",
        "install",
        "installation"
    ],
}


# ============================================================
# 9. LOAD ABSA MODEL
# ============================================================

absa_classifier = pipeline(
    "text-classification",
    model=ABSA_MODEL_NAME,
    tokenizer=ABSA_MODEL_NAME,
    device=PIPELINE_DEVICE
)


# ============================================================
# 10. SENTIMENT PREDICTION
# ============================================================

def predict_sentiment(text):

    inputs = sentiment_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    ).to(DEVICE)

    with torch.no_grad():

        outputs = sentiment_model(
            **inputs
        )

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )

    predicted_id = probabilities.argmax(
        dim=-1
    ).item()

    return {
        "label": sentiment_model.config.id2label[
            predicted_id
        ],
        "confidence": round(
            probabilities[
                0,
                predicted_id
            ].item(),
            4
        )
    }


# ============================================================
# 11. EMOTION PREDICTION
# ============================================================

def predict_emotion(text):

    inputs = emotion_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    ).to(DEVICE)

    with torch.no_grad():

        outputs = emotion_model(
            **inputs
        )

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )

    predicted_id = probabilities.argmax(
        dim=-1
    ).item()

    return {
        "label": emotion_model.config.id2label[
            predicted_id
        ],
        "confidence": round(
            probabilities[
                0,
                predicted_id
            ].item(),
            4
        )
    }


# ============================================================
# 12. ASPECT EXTRACTION
# ============================================================

def extract_review_aspects(text):

    text_lower = text.lower()

    detected_aspects = []

    for aspect, terms in ASPECT_TERMS.items():

        for term in terms:

            pattern = (
                r"\b"
                + re.escape(term)
                + r"\b"
            )

            if re.search(
                pattern,
                text_lower
            ):

                detected_aspects.append(
                    aspect
                )

                break

    return detected_aspects


# ============================================================
# 13. ASPECT SENTIMENT
# ============================================================

def predict_aspect_sentiment(
    text,
    aspect
):

    result = absa_classifier(
        text,
        text_pair=aspect
    )[0]

    confidence = result["score"]

    if confidence >= 0.90:

        confidence_level = "high"

    elif confidence >= 0.70:

        confidence_level = "medium"

    else:

        confidence_level = "low"

    return {
        "aspect": aspect,
        "sentiment": result["label"],
        "confidence": round(
            confidence,
            4
        ),
        "confidence_level": confidence_level
    }


def predict_aspect_sentiments(
    text,
    aspects
):

    results = []

    for aspect in aspects:

        result = predict_aspect_sentiment(
            text,
            aspect
        )

        results.append(
            result
        )

    return results


# ============================================================
# 14. EXCEL LOADER
# ============================================================

def load_reviews_from_excel(file_path):

    file_path = Path(file_path)

    if not file_path.exists():

        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    # --------------------------------------------------------
    # Read Excel
    # --------------------------------------------------------

    df = pd.read_excel(
        file_path
    )

    if df.empty:

        raise ValueError(
            "The Excel file is empty."
        )

    # Keep original columns for
    # useful error messages.

    original_columns = list(
        df.columns
    )

    # --------------------------------------------------------
    # Normalize column names
    # --------------------------------------------------------

    normalized_columns = {
        str(column)
        .strip()
        .lower(): column

        for column in df.columns
    }

    # --------------------------------------------------------
    # Supported review column names
    # --------------------------------------------------------

    supported_names = [

        "review",

        "review_body",

        "review_text",

        "review text",

        "text",

        "feedback",

        "customer_feedback",

        "customer feedback",

        "comment",

        "comments",

        "customer_review",

        "customer review",

    ]

    review_column = None

    # --------------------------------------------------------
    # Try exact supported names
    # --------------------------------------------------------

    for name in supported_names:

        if name in normalized_columns:

            review_column = (
                normalized_columns[name]
            )

            break

    # --------------------------------------------------------
    # Fallback:
    #
    # If there is exactly one text-like
    # column, use it as the review column.
    # --------------------------------------------------------

    if review_column is None:

        text_candidates = []

        for column in df.columns:

            column_dtype = str(
                df[column].dtype
            ).lower()

            if (
                column_dtype == "object"
                or "string" in column_dtype
            ):

                non_null_values = (
                    df[column]
                    .dropna()
                    .astype(str)
                    .str.strip()
                )

                if len(
                    non_null_values
                ) > 0:

                    text_candidates.append(
                        column
                    )

        if len(
            text_candidates
        ) == 1:

            review_column = (
                text_candidates[0]
            )

    # --------------------------------------------------------
    # Could not detect review column
    # --------------------------------------------------------

    if review_column is None:

        raise ValueError(
            "Could not detect the review column.\n\n"

            f"Available columns: "
            f"{original_columns}\n\n"

            "Supported review column names include:\n"

            "review, review_body, review_text, review text,\n"

            "text, feedback, customer_feedback,\n"

            "comment, comments, customer_review."
        )

    # --------------------------------------------------------
    # Rename detected column
    # to the standard name: review
    # --------------------------------------------------------

    df = df.rename(
        columns={
            review_column: "review"
        }
    )

    # --------------------------------------------------------
    # Clean review values
    # --------------------------------------------------------

    df["review"] = (
        df["review"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Remove empty reviews
    # --------------------------------------------------------

    df = df[
        df["review"].str.len() > 0
    ].reset_index(
        drop=True
    )

    if df.empty:

        raise ValueError(
            "No valid reviews were found "
            "in the Excel file."
        )

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    print(
        f"Detected review column: "
        f"'{review_column}'"
    )

    print(
        f"Loaded {len(df)} reviews "
        f"from Excel. ✅"
    )

    print(
        "Columns:",
        list(df.columns)
    )

    return df


# ============================================================
# 15. PDF LOADER
# ============================================================

def load_reviews_from_pdf(
    file_path
):

    file_path = Path(
        file_path
    )

    if not file_path.exists():

        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    document = pymupdf.open(
        file_path
    )

    pages_text = []

    for page in document:

        text = page.get_text(
            "text"
        ).strip()

        if text:

            pages_text.append(
                text
            )

    document.close()

    full_text = "\n".join(
        pages_text
    )

    if not full_text.strip():

        raise ValueError(
            "No text could be extracted "
            "from the PDF."
        )

    return full_text


# ============================================================
# 16. PARSE PDF REVIEWS
# ============================================================

def parse_pdf_reviews(
    pdf_text
):

    pattern = (
        r"Review\s+(\d+):\s*(.*?)"
        r"(?=Review\s+\d+:|$)"
    )

    matches = re.findall(
        pattern,
        pdf_text,
        flags=(
            re.IGNORECASE
            | re.DOTALL
        )
    )

    if not matches:

        raise ValueError(
            "Could not detect reviews in "
            "the PDF. Expected format like: "
            "Review 1: ..."
        )

    reviews = pd.DataFrame(
        matches,
        columns=[
            "review_id",
            "review"
        ]
    )

    reviews["review_id"] = (
        reviews["review_id"]
        .astype(int)
    )

    reviews["review"] = (
        reviews["review"]
        .str.replace(
            r"\s+",
            " ",
            regex=True
        )
        .str.strip()
    )

    reviews = reviews[
        reviews["review"].str.len() > 0
    ].reset_index(
        drop=True
    )

    return reviews


# ============================================================
# 17. PREPARE REVIEWS
# ============================================================

def prepare_reviews(
    df
):

    df = df.copy()

    if "review" not in df.columns:

        raise ValueError(
            "The input file must contain "
            "a 'review' column."
        )

    # --------------------------------------------------------
    # Create review ID if not available
    # --------------------------------------------------------

    if "review_id" not in df.columns:

        df.insert(
            0,
            "review_id",
            range(
                1,
                len(df) + 1
            )
        )

    # --------------------------------------------------------
    # Keep only required columns
    # --------------------------------------------------------

    df = df[
        [
            "review_id",
            "review"
        ]
    ]

    # --------------------------------------------------------
    # Clean text
    # --------------------------------------------------------

    df["review"] = (
        df["review"]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Remove empty reviews
    # --------------------------------------------------------

    df = df[
        df["review"].str.len() > 0
    ].reset_index(
        drop=True
    )

    return df


# ============================================================
# 18. UNIFIED FILE LOADER
# ============================================================

def load_reviews(
    file_path
):

    file_path = Path(
        file_path
    )

    if not file_path.exists():

        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    suffix = (
        file_path.suffix.lower()
    )

    # --------------------------------------------------------
    # Excel
    # --------------------------------------------------------

    if suffix in [
        ".xlsx",
        ".xls"
    ]:

        df = load_reviews_from_excel(
            file_path
        )

        df = prepare_reviews(
            df
        )

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    elif suffix == ".pdf":

        pdf_text = load_reviews_from_pdf(
            file_path
        )

        df = parse_pdf_reviews(
            pdf_text
        )

        df = prepare_reviews(
            df
        )

    # --------------------------------------------------------
    # Unsupported
    # --------------------------------------------------------

    else:

        raise ValueError(
            "Unsupported file format. "
            "Supported formats: "
            ".xlsx, .xls, .pdf"
        )

    print(
        f"Loaded {len(df)} reviews successfully. ✅"
    )

    return df


# ============================================================
# 19. TEXT ANALYSIS
# ============================================================

def run_text_analysis(
    df
):

    results = df.copy()

    # --------------------------------------------------------
    # Sentiment
    # --------------------------------------------------------

    results["sentiment_result"] = (
        results["review"]
        .apply(
            predict_sentiment
        )
    )

    results["sentiment"] = (
        results[
            "sentiment_result"
        ]
        .apply(
            lambda x: x["label"]
        )
    )

    results["sentiment_confidence"] = (
        results[
            "sentiment_result"
        ]
        .apply(
            lambda x: x["confidence"]
        )
    )

    # --------------------------------------------------------
    # Emotion
    # --------------------------------------------------------

    results["emotion_result"] = (
        results["review"]
        .apply(
            predict_emotion
        )
    )

    results["emotion"] = (
        results[
            "emotion_result"
        ]
        .apply(
            lambda x: x["label"]
        )
    )

    results["emotion_confidence"] = (
        results[
            "emotion_result"
        ]
        .apply(
            lambda x: x["confidence"]
        )
    )

    # --------------------------------------------------------
    # Remove temporary columns
    # --------------------------------------------------------

    results = results.drop(
        columns=[
            "sentiment_result",
            "emotion_result"
        ]
    )

    return results


# ============================================================
# 20. TOPIC ANALYSIS
# ============================================================

def run_topic_analysis(
    df
):

    results = df.copy()

    texts = results[
        "review"
    ].tolist()

    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    embeddings = (
        topic_embedding_model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True
        )
    )

    # --------------------------------------------------------
    # Predict topics
    # --------------------------------------------------------

    topic_ids, _ = (
        topic_model.transform(
            texts,
            embeddings=embeddings
        )
    )

    results["topic_id"] = (
        topic_ids
    )

    results["topic"] = (
        results["topic_id"]
        .map(topic_name_map)
        .fillna("unknown")
    )

    return results


# ============================================================
# 21. ASPECT ANALYSIS
# ============================================================

def run_aspect_analysis(
    df
):

    results = df.copy()

    # --------------------------------------------------------
    # Extract aspects
    # --------------------------------------------------------

    results["aspects"] = (
        results["review"]
        .apply(
            extract_review_aspects
        )
    )

    # --------------------------------------------------------
    # Aspect sentiment
    # --------------------------------------------------------

    results["aspect_sentiments"] = (
        results.apply(
            lambda row:
            predict_aspect_sentiments(
                row["review"],
                row["aspects"]
            ),
            axis=1
        )
    )

    return results


# ============================================================
# 22. BUSINESS INSIGHTS
# ============================================================

def generate_business_insights(
    df
):

    aspect_rows = []

    # --------------------------------------------------------
    # Flatten aspect sentiment results
    # --------------------------------------------------------

    for _, row in df.iterrows():

        for aspect_result in row[
            "aspect_sentiments"
        ]:

            aspect_rows.append(
                {
                    "aspect": (
                        aspect_result[
                            "aspect"
                        ]
                    ),
                    "sentiment": (
                        aspect_result[
                            "sentiment"
                        ]
                    ),
                    "confidence": (
                        aspect_result[
                            "confidence"
                        ]
                    )
                }
            )

    aspect_df = pd.DataFrame(
        aspect_rows
    )

    # --------------------------------------------------------
    # Overall sentiment
    # --------------------------------------------------------

    positive_count = int(
        (
            df["sentiment"]
            == "positive"
        ).sum()
    )

    negative_count = int(
        (
            df["sentiment"]
            == "negative"
        ).sum()
    )

    total_reviews = len(df)

    if total_reviews > 0:

        positive_percentage = round(
            positive_count
            / total_reviews
            * 100,
            2
        )

        negative_percentage = round(
            negative_count
            / total_reviews
            * 100,
            2
        )

    else:

        positive_percentage = 0.0
        negative_percentage = 0.0

    positive_aspects = {}
    negative_aspects = {}

    insights = []

    # --------------------------------------------------------
    # Overall insight
    # --------------------------------------------------------

    insights.append(
        f"Overall sentiment: "
        f"{positive_percentage:.1f}% positive and "
        f"{negative_percentage:.1f}% negative reviews."
    )

    # --------------------------------------------------------
    # Aspect insights
    # --------------------------------------------------------

    if not aspect_df.empty:

        positive_aspects_series = (
            aspect_df.loc[
                aspect_df[
                    "sentiment"
                ]
                .astype(str)
                .str.lower()
                == "positive",
                "aspect"
            ]
            .value_counts()
        )

        negative_aspects_series = (
            aspect_df.loc[
                aspect_df[
                    "sentiment"
                ]
                .astype(str)
                .str.lower()
                == "negative",
                "aspect"
            ]
            .value_counts()
        )

        positive_aspects = {
            aspect: int(count)

            for aspect, count
            in positive_aspects_series.items()
        }

        negative_aspects = {
            aspect: int(count)

            for aspect, count
            in negative_aspects_series.items()
        }

        # ----------------------------------------------------
        # Positive aspect insights
        # ----------------------------------------------------

        for aspect, count in (
            positive_aspects.items()
        ):

            insights.append(
                f"Positive feedback was detected "
                f"for {aspect} "
                f"({count} mention"
                f"{'s' if count != 1 else ''})."
            )

        # ----------------------------------------------------
        # Negative aspect insights
        # ----------------------------------------------------

        for aspect, count in (
            negative_aspects.items()
        ):

            insights.append(
                f"Negative feedback was detected "
                f"for {aspect} "
                f"({count} mention"
                f"{'s' if count != 1 else ''})."
            )

    # --------------------------------------------------------
    # Final business insight object
    # --------------------------------------------------------

    return {
        "summary": {
            "total_reviews": total_reviews,
            "positive_reviews": positive_count,
            "negative_reviews": negative_count,
            "positive_percentage": (
                positive_percentage
            ),
            "negative_percentage": (
                negative_percentage
            )
        },

        "positive_aspects": (
            positive_aspects
        ),

        "negative_aspects": (
            negative_aspects
        ),

        "insights": insights
    }


# ============================================================
# 23. COMPLETE PIPELINE
# ============================================================

def run_pipeline(
    file_path
):

    # --------------------------------------------------------
    # Load reviews
    # --------------------------------------------------------

    reviews_df = load_reviews(
        file_path
    )

    # --------------------------------------------------------
    # Sentiment + Emotion
    # --------------------------------------------------------

    results = run_text_analysis(
        reviews_df
    )

    # --------------------------------------------------------
    # Topics
    # --------------------------------------------------------

    results = run_topic_analysis(
        results
    )

    # --------------------------------------------------------
    # Aspects + ABSA
    # --------------------------------------------------------

    results = run_aspect_analysis(
        results
    )

    # --------------------------------------------------------
    # Business insights
    # --------------------------------------------------------

    business_insights = (
        generate_business_insights(
            results
        )
    )

    return (
        results,
        business_insights
    )


# ============================================================
# 24. SAVE PIPELINE OUTPUT
# ============================================================

def save_pipeline_results(
    results,
    business_insights
):

    processed_dir = (
        PROJECT_ROOT
        / "data"
        / "processed"
    )

    processed_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    results_to_save = (
        results.copy()
    )

    # --------------------------------------------------------
    # Convert aspects to JSON
    # --------------------------------------------------------

    results_to_save[
        "aspects"
    ] = (
        results_to_save[
            "aspects"
        ]
        .apply(
            lambda x:
            json.dumps(
                x,
                ensure_ascii=False
            )
        )
    )

    # --------------------------------------------------------
    # Convert ABSA results to JSON
    # --------------------------------------------------------

    results_to_save[
        "aspect_sentiments"
    ] = (
        results_to_save[
            "aspect_sentiments"
        ]
        .apply(
            lambda x:
            json.dumps(
                x,
                ensure_ascii=False
            )
        )
    )

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    results_path = (
        processed_dir
        / "final_pipeline_results.csv"
    )

    insights_path = (
        processed_dir
        / "business_insights.json"
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    results_to_save.to_csv(
        results_path,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    with open(
        insights_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            business_insights,
            f,
            indent=2,
            ensure_ascii=False
        )

    return (
        results_path,
        insights_path
    )