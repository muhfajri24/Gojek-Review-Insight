from __future__ import annotations

import re
from pathlib import Path
import pandas as pd
from src.preprocessing import basic_clean_text

THEME_KEYWORDS: dict[str, tuple[str, ...]] = {
    "application errors": ("error", "bug", "lemot", "lambat", "lag", "crash", "gangguan", "server", "loading"),
    "payment problems": ("gopay", "bayar", "pembayaran", "saldo", "transfer", "topup", "top up", "potong saldo"),
    "driver availability": ("driver", "pengemudi", "susah driver", "cari driver", "tidak dapat driver"),
    "GPS or map issues": ("gps", "maps", "map", "lokasi", "titik jemput", "alamat"),
    "account and login": ("login", "akun", "otp", "verifikasi", "nomor hp", "masuk akun"),
    "promotions and vouchers": ("promo", "voucher", "diskon", "kupon", "cashback"),
    "customer service": ("customer service", "cs", "layanan pelanggan", "komplain", "laporan", "bantuan"),
    "cancellation": ("cancel", "batal", "dibatalkan", "pembatalan"),
    "pricing": ("mahal", "harga", "tarif", "ongkir", "biaya"),
}


def match_theme(text: str, keywords: tuple[str, ...]) -> bool:
    normalized = basic_clean_text(text)
    pattern = "|".join(rf"(?<!\w){re.escape(keyword)}(?!\w)" for keyword in keywords)
    return bool(re.search(pattern, normalized))


def categorize_complaint_themes(df: pd.DataFrame, output_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    negative = df.loc[df["sentiment"] == "negative", ["review_text"]].copy()
    matched_any = pd.Series(False, index=negative.index)
    rows: list[dict[str, object]] = []
    for theme, keywords in THEME_KEYWORDS.items():
        mask = negative["review_text"].map(lambda text: match_theme(text, keywords))
        matched_any |= mask
        examples = negative.loc[mask, "review_text"].drop_duplicates().head(3).tolist()
        rows.append({
            "theme": theme, "review_count": int(mask.sum()),
            "percentage_of_negative_reviews": float(mask.mean() * 100),
            "representative_keywords": ", ".join(keywords),
            **{f"example_review_{i + 1}": examples[i] if i < len(examples) else "" for i in range(3)},
        })
    unmatched = negative.loc[~matched_any, ["review_text"]].copy()
    examples = unmatched["review_text"].head(3).tolist()
    rows.append({
        "theme": "Other / uncategorized", "review_count": len(unmatched),
        "percentage_of_negative_reviews": float((~matched_any).mean() * 100),
        "representative_keywords": "",
        **{f"example_review_{i + 1}": examples[i] if i < len(examples) else "" for i in range(3)},
    })
    themes = pd.DataFrame(rows).sort_values("review_count", ascending=False)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        themes.to_csv(output_dir / "complaint_themes.csv", index=False)
        unmatched.to_csv(output_dir / "uncategorized_negative_reviews.csv", index=False)
    return themes, unmatched
