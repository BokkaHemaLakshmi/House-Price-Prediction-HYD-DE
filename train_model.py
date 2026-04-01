from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
MODEL_PATH = BASE_DIR / "model.pkl"

FEATURE_COLUMNS = [
    "City",
    "Location",
    "Area",
    "No. of Bedrooms",
    "Furnishing",
    "Gymnasium",
    "24X7Security",
    "PowerBackup",
]
TARGET_COLUMN = "Price"
CATEGORICAL_COLUMNS = ["City", "Location", "Furnishing"]
NUMERIC_COLUMNS = ["Area", "No. of Bedrooms", "Gymnasium", "24X7Security", "PowerBackup"]

FURNISHING_MAP = {
    "semi-furnished": "Semi-Furnished",
    "furnished": "Furnished",
    "unfurnished": "Unfurnished",
}
DELHI_AMENITY_PATTERNS = {
    "Gymnasium": [
        r"\bgym(?:nasium)?s?\b",
        r"\bfitness\b",
        r"\bclub\s*house\b",
        r"\bclubhouse\b",
    ],
    "24X7Security": [
        r"\b24\s*x\s*7\s*security\b",
        r"\b24x7\s*security\b",
        r"\b24\s*hours?\s*security\b",
        r"\bsecurity\b",
        r"\bgated\b",
    ],
    "PowerBackup": [
        r"\bpower\s*backup\b",
        r"\bfull\s*power\s*backup\b",
        r"\bbackup\s*power\b",
        r"\binvert(?:e|o)r\b",
        r"\bgenerator\b",
    ],
}


def normalize_label(value, mapping, default="Unknown"):
    if pd.isna(value):
        return default
    normalized = str(value).strip()
    if not normalized:
        return default
    return mapping.get(normalized.lower(), normalized.replace("_", " ").title())


def normalize_binary(series: pd.Series) -> pd.Series:
    normalized = (
        series.fillna(0)
        .astype(str)
        .str.strip()
        .replace({"True": "1", "False": "0", "Yes": "1", "No": "0", "": "0"})
    )
    return pd.to_numeric(normalized, errors="coerce").fillna(0).clip(0, 1).astype(int)


def infer_amenity_flags(text_series: pd.Series, patterns: list[str]) -> pd.Series:
    normalized_text = text_series.fillna("").astype(str).str.lower()
    regex = "|".join(f"(?:{pattern})" for pattern in patterns)
    return normalized_text.str.contains(regex, regex=True, na=False).astype(int)


def load_and_prepare_data() -> pd.DataFrame:
    delhi = pd.read_csv(DATA_DIR / "Delhi.csv").rename(
        columns={"Locality": "Location", "BHK": "No. of Bedrooms"}
    )
    delhi["City"] = "Delhi"
    delhi["Furnishing"] = delhi["Furnishing"].map(
        lambda value: normalize_label(value, FURNISHING_MAP)
    )
    for amenity, patterns in DELHI_AMENITY_PATTERNS.items():
        delhi[amenity] = infer_amenity_flags(delhi["Location"], patterns)

    hyderabad = pd.read_csv(DATA_DIR / "Hyderabad.csv")
    hyderabad["City"] = "Hyderabad"
    hyderabad["Furnishing"] = "Unknown"
    hyderabad["Gymnasium"] = normalize_binary(hyderabad["Gymnasium"])
    hyderabad["24X7Security"] = normalize_binary(hyderabad["24X7Security"])
    hyderabad["PowerBackup"] = normalize_binary(hyderabad["PowerBackup"])

    merged = pd.concat([delhi, hyderabad], ignore_index=True, sort=False)
    merged = merged.rename(columns=lambda column: column.strip())

    merged["City"] = merged["City"].astype(str).str.strip()
    merged["Location"] = merged["Location"].astype(str).str.strip()
    merged["Area"] = pd.to_numeric(merged["Area"], errors="coerce")
    merged["No. of Bedrooms"] = pd.to_numeric(
        merged["No. of Bedrooms"], errors="coerce"
    )
    merged["Price"] = pd.to_numeric(merged["Price"], errors="coerce")

    for column in ["Furnishing"]:
        merged[column] = merged[column].map(lambda value: normalize_label(value, {}))

    for column in ["Gymnasium", "24X7Security", "PowerBackup"]:
        merged[column] = normalize_binary(merged[column])

    merged = merged.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()
    merged = merged[merged["Area"] > 0]
    merged["No. of Bedrooms"] = merged["No. of Bedrooms"].astype(int)

    return merged


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLUMNS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ]
    )

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", model),
        ]
    )


def train_and_save_model():
    data = load_and_prepare_data()
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    metrics = {
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }

    joblib.dump(pipeline, MODEL_PATH)

    return {
        "model_path": str(MODEL_PATH),
        "metrics": metrics,
        "feature_columns": FEATURE_COLUMNS,
        "data": data,
    }


if __name__ == "__main__":
    artifacts = train_and_save_model()
    print("Saved model to:", artifacts["model_path"])
    print("Features:", artifacts["feature_columns"])
    print("Metrics:", artifacts["metrics"])
