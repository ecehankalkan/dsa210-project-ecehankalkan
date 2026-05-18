#!/usr/bin/env python3
from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARCHIVE_DIR = BASE_DIR / "archive"

app = FastAPI(title="Recipe Rating Explorer")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def _safe_list_len(value) -> float:
    try:
        if pd.isna(value):
            return float("nan")
        if isinstance(value, list):
            return float(len(value))
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, list):
            return float(len(parsed))
    except Exception:
        return float("nan")
    return float("nan")


def _parse_list(value) -> List[str]:
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except Exception:
        return []
    return []


def _extract_calories(value) -> float:
    try:
        if pd.isna(value):
            return float("nan")
        text = str(value).strip().strip("[]")
        parts = [p.strip() for p in text.split(",")]
        return float(parts[0]) if parts and parts[0] else float("nan")
    except Exception:
        return float("nan")


def _load_interactions() -> pd.DataFrame:
    raw_path = DATA_DIR / "RAW_interactions.csv"
    if raw_path.exists():
        return pd.read_csv(raw_path, low_memory=False)
    sample_path = DATA_DIR / "RAW_interactions_sample.csv"
    if sample_path.exists():
        return pd.read_csv(sample_path, low_memory=False)
    archive_path = ARCHIVE_DIR / "interactions_train.csv"
    if archive_path.exists():
        return pd.read_csv(archive_path, low_memory=False)
    raise FileNotFoundError("No interactions file found in data/ or archive/")


def load_data() -> pd.DataFrame:
    recipes_path = DATA_DIR / "RAW_recipes.csv"
    if not recipes_path.exists():
        recipes_path = DATA_DIR / "RAW_recipes_sample.csv"
    if not recipes_path.exists():
        raise FileNotFoundError("Recipe file not found in data/")

    recipes = pd.read_csv(recipes_path, low_memory=False)
    interactions = _load_interactions()

    if "id" not in recipes.columns or "recipe_id" not in interactions.columns:
        raise ValueError("Missing merge keys in recipes or interactions")

    ratings = interactions[["recipe_id", "rating"]].dropna(subset=["rating"]).copy()
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings = ratings.dropna(subset=["rating"])

    grouped = (
        ratings.groupby("recipe_id")
        .agg(avg_rating=("rating", "mean"), rating_count=("rating", "size"))
        .reset_index()
    )

    df = recipes.merge(grouped, left_on="id", right_on="recipe_id", how="inner")

    if "name" not in df.columns:
        df["name"] = "Unknown"
    else:
        df["name"] = df["name"].fillna("Unknown")
    if "minutes" not in df.columns:
        df["minutes"] = pd.NA
    if "n_ingredients" not in df.columns:
        if "ingredients" in df.columns:
            df["n_ingredients"] = df["ingredients"].apply(_safe_list_len)
        else:
            df["n_ingredients"] = pd.NA
    if "n_steps" not in df.columns:
        if "steps" in df.columns:
            df["n_steps"] = df["steps"].apply(_safe_list_len)
        else:
            df["n_steps"] = pd.NA
    if "calories" not in df.columns and "nutrition" in df.columns:
        df["calories"] = df["nutrition"].apply(_extract_calories)

    for col in ["minutes", "n_ingredients", "n_steps", "avg_rating", "rating_count", "calories"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[
        df["minutes"].notna()
        & (df["minutes"] > 0)
        & df["n_ingredients"].notna()
        & (df["n_ingredients"] > 0)
        & df["n_steps"].notna()
        & (df["n_steps"] > 0)
    ]

    df["name_lower"] = df["name"].str.lower()
    return df


DATA = load_data()

TIME_FILTERS: Dict[str, Tuple[float | None, float | None]] = {
    "any": (None, None),
    "under_15": (None, 15),
    "under_30": (None, 30),
    "under_60": (None, 60),
    "one_two_hours": (60, 120),
    "over_2_hours": (120, None),
}

ING_FILTERS: Dict[str, Tuple[float | None, float | None]] = {
    "any": (None, None),
    "1_5": (1, 5),
    "6_10": (6, 10),
    "11_15": (11, 15),
    "over_15": (15, None),
}

SORT_OPTIONS = {
    "rating_desc": ("avg_rating", False),
    "minutes_asc": ("minutes", True),
    "ingredients_asc": ("n_ingredients", True),
    "reviews_desc": ("rating_count", False),
}


def _apply_range_filter(df: pd.DataFrame, column: str, bounds: Tuple[float | None, float | None]) -> pd.DataFrame:
    min_val, max_val = bounds
    if min_val is not None:
        df = df[df[column] >= min_val]
    if max_val is not None:
        df = df[df[column] < max_val]
    return df


def apply_filters(df: pd.DataFrame, params: Dict[str, str]) -> pd.DataFrame:
    filtered = df.copy()

    keyword = params.get("q", "").strip().lower()
    if keyword:
        filtered = filtered[filtered["name_lower"].str.contains(keyword, na=False)]

    min_rating = params.get("min_rating", "any")
    if min_rating != "any":
        try:
            rating_value = float(min_rating)
            filtered = filtered[filtered["avg_rating"] >= rating_value]
        except ValueError:
            pass

    time_key = params.get("time", "any")
    if time_key in TIME_FILTERS:
        filtered = _apply_range_filter(filtered, "minutes", TIME_FILTERS[time_key])

    ing_key = params.get("ingredients", "any")
    if ing_key in ING_FILTERS:
        filtered = _apply_range_filter(filtered, "n_ingredients", ING_FILTERS[ing_key])

    sort_key = params.get("sort", "rating_desc")
    if sort_key in SORT_OPTIONS:
        sort_col, ascending = SORT_OPTIONS[sort_key]
        filtered = filtered.sort_values(sort_col, ascending=ascending, na_position="last")

    return filtered


def summary_stats(df: pd.DataFrame) -> Dict[str, str]:
    if df.empty:
        return {
            "count": "0",
            "avg_rating": "N/A",
            "avg_minutes": "N/A",
            "avg_ingredients": "N/A",
            "avg_steps": "N/A",
        }

    return {
        "count": f"{len(df):,}",
        "avg_rating": f"{df['avg_rating'].mean():.2f}",
        "avg_minutes": f"{df['minutes'].mean():.1f}",
        "avg_ingredients": f"{df['n_ingredients'].mean():.1f}",
        "avg_steps": f"{df['n_steps'].mean():.1f}",
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    params = dict(request.query_params)
    filtered = apply_filters(DATA, params)
    stats = summary_stats(filtered)
    results = filtered.head(200).to_dict(orient="records")

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "results": results,
            "stats": stats,
            "filters": {
                "q": params.get("q", ""),
                "min_rating": params.get("min_rating", "any"),
                "time": params.get("time", "any"),
                "ingredients": params.get("ingredients", "any"),
                "sort": params.get("sort", "rating_desc"),
            },
        },
    )


@app.get("/recipe/{recipe_id}", response_class=HTMLResponse)
async def recipe_detail(request: Request, recipe_id: int) -> HTMLResponse:
    match = DATA[DATA["id"] == recipe_id]
    if match.empty:
        raise HTTPException(status_code=404, detail="Recipe not found")

    row = match.iloc[0]
    ingredients = _parse_list(row.get("ingredients"))
    steps = _parse_list(row.get("steps"))
    servings = row.get("servings")
    if pd.isna(servings):
        servings = None
    description = row.get("description")
    if pd.isna(description):
        description = None

    return templates.TemplateResponse(
        request,
        "recipe.html",
        {
            "recipe": {
                "id": int(row.get("id")),
                "name": row.get("name"),
                "avg_rating": row.get("avg_rating"),
                "rating_count": row.get("rating_count"),
                "minutes": row.get("minutes"),
                "n_ingredients": row.get("n_ingredients"),
                "n_steps": row.get("n_steps"),
                "calories": row.get("calories"),
                "servings": servings,
                "description": description,
                "ingredients": ingredients,
                "steps": steps,
            }
        },
    )
