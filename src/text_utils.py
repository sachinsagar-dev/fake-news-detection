import re
import pandas as pd


def clean_text(text: str) -> str:
    """Normalize article text while retaining useful word boundaries."""

    # Handle missing values
    text = "" if pd.isna(text) else str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Keep only English letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def combine_title_text(df: pd.DataFrame) -> pd.Series:
    """Combine headline and article body into one text field."""

    title = df["title"] if "title" in df.columns else ""
    body = df["text"] if "text" in df.columns else ""

    return (
        title.fillna("").astype(str)
        + " "
        + body.fillna("").astype(str)
    ).map(clean_text)


def normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert WELFake dataset into the format used by this project.

    WELFake:
        0 = Fake
        1 = Real

    Our project:
        0 = Real
        1 = Fake

    Therefore, labels are inverted.
    """

    df = df.copy()

    # Normalize column names
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "label" not in df.columns:
        raise ValueError(
            "Dataset must contain a 'label' column."
        )

    # Create standardized dataframe
    out = pd.DataFrame()

    # Get title and text
    out["title"] = (
        df["title"]
        if "title" in df.columns
        else ""
    )

    out["text"] = (
        df["text"]
        if "text" in df.columns
        else ""
    )

    # Convert labels to numbers
    original_label = pd.to_numeric(
        df["label"],
        errors="coerce"
    )

    # WELFake:
    # 0 = Fake
    # 1 = Real
    #
    # Our application:
    # 0 = Real
    # 1 = Fake
    #
    # Therefore:
    # 1 - original_label
    out["label"] = 1 - original_label

    # Remove rows with invalid labels
    out = out.dropna(subset=["label"])

    # Convert labels to integer
    out["label"] = out["label"].astype(int)

    return out[["title", "text", "label"]]


def load_fake_true(fake_path, true_path):
    """
    Load datasets where Fake.csv contains fake articles
    and True.csv contains real articles.
    """

    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)

    # Project convention:
    # 1 = Fake
    # 0 = Real
    fake["label"] = 1
    true["label"] = 0

    # Make sure title/text columns exist
    fake["title"] = fake.get("title", "")
    true["title"] = true.get("title", "")

    fake["text"] = fake.get("text", "")
    true["text"] = true.get("text", "")

    # Combine datasets
    df = pd.concat(
        [fake, true],
        ignore_index=True
    )

    return df[["title", "text", "label"]]


def load_dataset(path):
    """Load a CSV dataset."""

    path = str(path)

    if not path.lower().endswith(".csv"):
        raise ValueError(
            "Only CSV input is supported."
        )

    df = pd.read_csv(path)

    return normalize_dataset(df)