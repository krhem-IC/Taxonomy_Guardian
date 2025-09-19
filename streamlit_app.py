# streamlit_app.py
# Taxonomy Guardian - Enhanced version with improved category hierarchy and description logic
# Brand Accuracy with master mapping, cleaned phrases, heuristics,
# UPC safety-net (OpenFoodFacts + optional UPCItemDB), and "(UPC Lookup)" strength label.

import io
import re
import os
import json
import time
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import pandas as pd

# =========================
# Streamlit import with graceful fallback
# =========================
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    # Create mock streamlit for testing
    class MockStreamlit:
        def __init__(self):
            self.sidebar = self
            self.session_state = {}
            self.secrets = {}
        
        def set_page_config(self, **kwargs): pass
        def title(self, text): print(f"TITLE: {text}")
        def caption(self, text): pass
        def header(self, text): pass
        def subheader(self, text): pass
        def markdown(self, text): pass
        def info(self, text): print(f"INFO: {text}")
        def warning(self, text): print(f"WARNING: {text}")
        def error(self, text): print(f"ERROR: {text}")
        def success(self, text): print(f"SUCCESS: {text}")
        def dataframe(self, df, **kwargs): pass
        def json(self, data): pass
        def download_button(self, *args, **kwargs): pass
        def file_uploader(self, *args, **kwargs): return None
        def selectbox(self, label, options, **kwargs): return options[0] if options else None
        def checkbox(self, label, **kwargs): return kwargs.get('value', False)
        def number_input(self, label, **kwargs): return kwargs.get('value', 200)
        def slider(self, label, min_val, max_val, default, step): return default
        def button(self, label, **kwargs): return False
        def spinner(self, text): return self
        def expander(self, label, **kwargs): return self
        def columns(self, spec): return [self] * (spec if isinstance(spec, int) else len(spec))
        def cache_data(self, **kwargs):
            def decorator(func):
                return func
            return decorator
        def image(self, *args, **kwargs): pass
        def metric(self, label, value): print(f"METRIC {label}: {value}")
        def progress(self, value): pass
        def rerun(self): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
    
    st = MockStreamlit()

# =========================
# App config
# =========================
if STREAMLIT_AVAILABLE:
    st.set_page_config(page_title="Taxonomy Guardian", page_icon="🛡️", layout="wide")

    try:
        c1, c2 = st.columns([0.18, 0.82])
        with c1:
            # Use a placeholder if image doesn't exist
            try:
                st.image("Taxonomy_Guardian.png", use_container_width=True)
            except:
                st.markdown("🛡️")  # Fallback emoji
        with c2:
            st.title("Taxonomy Guardian")
            st.caption("Ensuring the Fetch Taxonomy remains our source of truth")
    except Exception:
        st.title("Taxonomy Guardian")
        st.caption("Ensuring the Fetch Taxonomy remains our source of truth")

# =========================
# Data Classes for Better Structure
# =========================
@dataclass
class CleanupResult:
    success: bool
    message: str
    data: Optional[pd.DataFrame] = None
    stats: Optional[Dict] = None

@dataclass
class CategorySuggestion:
    category_1: str
    category_2: str
    category_3: str
    category_4: str
    confidence: str
    reason: str

# =========================
# Logging System
# =========================
if "logs" not in st.session_state:
    st.session_state["logs"] = []

def get_timestamp() -> str:
    """Get current timestamp in ISO format"""
    return datetime.now(timezone.utc).isoformat()

def log_event(level: str, msg: str, **data):
    """Log an event with structured data"""
    st.session_state["logs"].append({
        "timestamp": get_timestamp(),
        "level": level.upper(),
        "message": msg,
        **data
    })

def clear_logs():
    """Clear all logs"""
    st.session_state["logs"] = []

# =========================
# Constants and Configuration
# =========================
REQUIRED_COLUMNS = [
    "FIDO", "BARCODE", "CATEGORY_HIERARCHY", "CATEGORY_1", "CATEGORY_2", 
    "CATEGORY_3", "CATEGORY_4", "DESCRIPTION", "MANUFACTURER", "BRAND", "FIDO_TYPE"
]

OUTPUT_COLUMNS = [
    "Correct Brand?", "Correct Categories?", "Vague Description?", "Suggested Brand",
    "Suggested Category 1", "Suggested Category 2", "Suggested Category 3", 
    "Suggested New Description", "Match Strength"
]

GENERIC_BRAND_TERMS = {"generic", "unknown", "n/a", "na", "misc", "private label", "store brand"}

# Enhanced noise words for cleaning
UNITS = {
    "ml", "oz", "fl", "floz", "ct", "count", "pack", "pk", "pk.", "pcs", "pieces",
    "3pk", "6pk", "12pk", "24pk", "case", "box", "g", "kg", "lb", "lbs", "gram", "grams"
}

FORM_WORDS = {
    "frozen", "fresh", "organic", "natural", "roasted", "grilled", "baked", "fried",
    "garlic", "onion", "powder", "liquid", "solid", "capsules", "tablet", "tablets", 
    "softgels", "gummy", "gummies", "chewable", "ravioli", "pasta", "sauce", "dressing",
    "mix", "blend", "assorted", "variety", "mixed", "single", "multi", "double", "triple",
    "bottle", "bottles", "can", "cans", "jar", "jars", "bag", "bags", "pods", "cups",
    "k-cup", "kcup", "kcups", "keurig", "ground", "instant", "whole", "bean", "beans"
}

SIZE_WORDS = {
    "small", "medium", "large", "xl", "xxl", "mini", "micro", "giant", "jumbo",
    "regular", "standard", "premium", "deluxe", "classic", "original", "new", "improved"
}

COLOR_WORDS = {
    "black", "white", "red", "blue", "green", "yellow", "orange", "purple", "pink",
    "brown", "gray", "grey", "silver", "gold", "clear", "transparent"
}

GENERIC_SINGLE_BRAND_TOKENS = {"garden", "fitness", "beauty", "baby", "wine", "sports", "snacks", "home", "kitchen"}

STOPWORDS = {
    "pack", "size", "flavor", "flavour", "assorted", "variety", "brand", "product", "item", 
    "mix", "mixed", "selection", "collection", "set", "kit", "bundle",
    "cabernet", "sauvignon", "merlot", "pinot", "noir", "chardonnay", "red", "white", "rose",
    "sports", "exercise", "workout", "fitness", "treadmill", "bike", "pro", "max", "ultra",
    "premium", "deluxe", "classic", "original", "new", "improved", "enhanced", "advanced",
    "chips", "crisps", "crackers", "nuts", "seeds", "trail", "snack", "snacks", "treats"
}

# Enhanced category mapping for intelligent suggestions
CATEGORY_PATTERNS = {
    # Food categories
    ("Food", "Beverages", "Alcoholic", "Wine"): [
        "wine", "merlot", "cabernet", "pinot", "chardonnay", "sauvignon", "syrah", "rioja", 
        "tempranillo", "moscato", "riesling", "prosecco", "champagne"
    ],
    ("Food", "Beverages", "Non-Alcoholic", "Coffee"): [
        "coffee", "espresso", "cappuccino", "latte", "mocha", "keurig", "k-cup", "kcup", 
        "ground coffee", "coffee beans", "instant coffee"
    ],
    ("Food", "Beverages", "Non-Alcoholic", "Tea"): [
        "tea", "green tea", "black tea", "herbal tea", "chamomile", "earl grey", "oolong"
    ],
    ("Food", "Snacks", "Nuts & Seeds", "Mixed Nuts"): [
        "nuts", "almonds", "peanuts", "cashews", "walnuts", "trail mix", "mixed nuts"
    ],
    ("Food", "Snacks", "Chips & Crisps", "Potato Chips"): [
        "chips", "potato chips", "crisps", "kettle chips", "baked chips"
    ],
    
    # Health & Beauty
    ("Health & Beauty", "Personal Care", "Oral Care", "Toothpaste"): [
        "toothpaste", "toothbrush", "mouthwash", "dental", "oral care"
    ],
    ("Health & Beauty", "Supplements", "Vitamins", "Multivitamins"): [
        "vitamins", "multivitamin", "supplements", "capsules", "tablets", "gummy vitamins"
    ],
    
    # Household
    ("Household", "Cleaning", "Kitchen", "Dish Soap"): [
        "dish soap", "dishwashing", "kitchen cleaner", "degreaser"
    ],
    ("Household", "Paper Products", "Toilet Paper", "Bath Tissue"): [
        "toilet paper", "bath tissue", "bathroom tissue", "tp"
    ],
    
    # Electronics
    ("Electronics", "Audio", "Headphones", "Wireless"): [
        "headphones", "earbuds", "wireless", "bluetooth", "airpods"
    ],
    
    # Home & Garden
    ("Home & Garden", "Decor", "Seasonal", "Halloween"): [
        "halloween", "decoration", "ornament", "witch", "pumpkin", "spooky", "costume"
    ],
    ("Home & Garden", "Tools", "Hand Tools", "Screwdrivers"): [
        "screwdriver", "hammer", "wrench", "pliers", "drill"
    ]
}

# Description improvement patterns
DESCRIPTION_IMPROVEMENTS = {
    # Vague terms to specific alternatives
    "assorted": "Mixed Variety",
    "variety pack": "Multi-Pack Selection",
    "mixed": "Assorted Selection",
    "misc": "Miscellaneous Items",
    "pack of": "Multi-Pack",
    "various": "Multiple Varieties",
    "different": "Assorted Types",
    "several": "Multiple Items"
}

# =========================
# Enhanced Utility Functions
# =========================
def normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    return str(text).strip().lower()

def safe_str(value) -> str:
    """Safely convert value to string"""
    if pd.isna(value):
        return ""
    return str(value).strip()

def tokenize(text: str) -> List[str]:
    """Enhanced tokenization for analysis"""
    text = safe_str(text).lower()
    # Remove special characters but keep spaces and hyphens
    text = re.sub(r'[^\w\s\-]', ' ', text)
    # Split on whitespace and filter empty tokens
    return [token for token in re.split(r'\s+', text) if token and len(token) > 1]

def levenshtein_distance(s1: str, s2: str, max_distance: int = 2) -> int:
    """Optimized Levenshtein distance with early termination"""
    if s1 == s2:
        return 0
    if abs(len(s1) - len(s2)) > max_distance:
        return max_distance + 1
    
    m, n = len(s1), len(s2)
    if m > n:
        s1, s2, m, n = s2, s1, n, m
    
    prev_row = list(range(n + 1))
    
    for i in range(1, m + 1):
        current_row = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            current_row[j] = min(
                prev_row[j] + 1,      # deletion
                current_row[j-1] + 1, # insertion
                prev_row[j-1] + cost  # substitution
            )
        prev_row = current_row
        # Early termination if minimum possible distance exceeded
        if min(prev_row) > max_distance:
            return max_distance + 1
    
    return prev_row[n]

def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts using multiple metrics"""
    if not text1 or not text2:
        return 0.0
    
    text1, text2 = text1.lower().strip(), text2.lower().strip()
    
    if text1 == text2:
        return 1.0
    
    # Jaccard similarity on tokens
    tokens1, tokens2 = set(tokenize(text1)), set(tokenize(text2))
    if not tokens1 or not tokens2:
        return 0.0
    
    jaccard = len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))
    
    # Length similarity bonus
    len_similarity = min(len(text1), len(text2)) / max(len(text1), len(text2))
    
    # Combined score
    return (jaccard * 0.7) + (len_similarity * 0.3)

def extract_meaningful_tokens(text: str, exclude_sets: List[Set[str]] = None) -> List[str]:
    """Extract meaningful tokens excluding noise words"""
    if exclude_sets is None:
        exclude_sets = [UNITS, FORM_WORDS, SIZE_WORDS, COLOR_WORDS, STOPWORDS]
    
    tokens = tokenize(text)
    meaningful_tokens = []
    
    for token in tokens:
        # Skip if token is in any exclude set
        if any(token in exclude_set for exclude_set in exclude_sets):
            continue
        
        # Skip very short tokens unless they're important
        if len(token) < 3 and token not in {"oz", "ml", "xl", "tv", "pc"}:
            continue
        
        meaningful_tokens.append(token)
    
    return meaningful_tokens

# =========================
# Enhanced Category Processing
# =========================
class CategoryIntelligence:
    """Enhanced category suggestion engine"""
    
    def __init__(self):
        self.pattern_cache = {}
        self._build_pattern_index()
    
    def _build_pattern_index(self):
        """Build reverse index for fast pattern matching"""
        self.keyword_to_categories = {}
        
        for category_path, keywords in CATEGORY_PATTERNS.items():
            for keyword in keywords:
                if keyword not in self.keyword_to_categories:
                    self.keyword_to_categories[keyword] = []
                self.keyword_to_categories[keyword].append(category_path)
    
    def suggest_categories(self, description: str, brand: str = "", current_categories: Tuple = None) -> CategorySuggestion:
        """Suggest categories based on description and brand"""
        desc_lower = description.lower()
        brand_lower = brand.lower()
        combined_text = f"{desc_lower} {brand_lower}".strip()
        
        # Find matching patterns
        category_scores = {}
        
        for keyword, category_paths in self.keyword_to_categories.items():
            if keyword in combined_text:
                for category_path in category_paths:
                    if category_path not in category_scores:
                        category_scores[category_path] = 0
                    
                    # Score based on keyword specificity and match quality
                    specificity_bonus = len(keyword.split()) * 2  # Multi-word keywords get bonus
                    exact_match_bonus = 5 if keyword in desc_lower else 0
                    brand_match_bonus = 3 if keyword in brand_lower else 0
                    
                    category_scores[category_path] += specificity_bonus + exact_match_bonus + brand_match_bonus
        
        if not category_scores:
            # Fallback: try to infer from existing categories
            if current_categories and any(current_categories):
                return self._infer_from_existing(current_categories, "Low")
            
            # Final fallback: generic food category
            return CategorySuggestion("Food", "Other", "Miscellaneous", "General", "Low", "Generic fallback")
        
        # Get best matching category
        best_category = max(category_scores.items(), key=lambda x: x[1])
        category_path, score = best_category
        
        confidence = "High" if score >= 10 else "Medium" if score >= 5 else "Low"
        reason = f"Keyword match (score: {score})"
        
        return CategorySuggestion(*category_path, confidence, reason)
    
    def _infer_from_existing(self, current_categories: Tuple, confidence: str) -> CategorySuggestion:
        """Infer missing categories from existing ones"""
        cat1, cat2, cat3, cat4 = current_categories
        
        # Fill in missing categories with logical defaults
        if not cat1:
            cat1 = "Food"  # Most common category
        if not cat2:
            cat2 = "Other"
        if not cat3:
            cat3 = "Miscellaneous"
        if not cat4:
            cat4 = "General"
        
        return CategorySuggestion(cat1, cat2, cat3, cat4, confidence, "Inferred from existing")
    
    def validate_hierarchy(self, cat1: str, cat2: str, cat3: str, cat4: str) -> Tuple[bool, str]:
        """Validate if category hierarchy makes logical sense"""
        # Check for obvious inconsistencies
        categories = [safe_str(c).strip() for c in [cat1, cat2, cat3, cat4]]
        
        # All must be filled
        if not all(categories):
            missing_count = sum(1 for c in categories if not c)
            return False, f"Missing {missing_count} category level(s)"
        
        # Check for duplicates
        if len(set(categories)) < len(categories):
            return False, "Duplicate categories in hierarchy"
        
        # Check for logical inconsistencies (basic)
        if "food" in cat1.lower() and "electronics" in cat2.lower():
            return False, "Inconsistent category hierarchy (Food -> Electronics)"
        
        return True, "Valid hierarchy"

# =========================
# Enhanced Description Processing
# =========================
class DescriptionIntelligence:
    """Enhanced description analysis and improvement engine"""
    
    def __init__(self):
        self.vague_patterns = self._compile_vague_patterns()
        self.improvement_cache = {}
    
    def _compile_vague_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for detecting vague descriptions"""
        patterns = [
            r'\b(assorted|variety pack|mixed|misc|various|different|several)\b',
            r'\b(pack of|set of|bunch of|lot of)\s*\d*\b',
            r'\b(item|product|thing|stuff|material)\b',
            r'\b(brand|flavor|size|color)\s*(unknown|varies|mixed)\b',
            r'^\s*[a-z\s]{1,10}\s*$',  # Very short descriptions
            r'\bn/?a\b|\bunknown\b|\btbd\b|\btba\b',  # Not available, unknown, etc.
        ]
        
        return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def analyze_description(self, description: str) -> Tuple[bool, List[str]]:
        """Analyze if description is vague and identify issues"""
        desc = safe_str(description).strip()
        
        if len(desc) < 5:
            return True, ["Too short"]
        
        issues = []
        
        # Check against vague patterns
        for i, pattern in enumerate(self.vague_patterns):
            if pattern.search(desc):
                if i == 0:
                    issues.append("Contains generic terms")
                elif i == 1:
                    issues.append("Uses vague quantity terms")
                elif i == 2:
                    issues.append("Uses generic item terms")
                elif i == 3:
                    issues.append("Contains unknown/varies terms")
                elif i == 4:
                    issues.append("Too short/minimal")
                elif i == 5:
                    issues.append("Contains unavailable/unknown terms")
        
        # Check token quality
        meaningful_tokens = extract_meaningful_tokens(desc)
        if len(meaningful_tokens) < 2:
            issues.append("Lacks descriptive content")
        
        # Check for repeated words
        words = desc.lower().split()
        if len(set(words)) < len(words) * 0.7:  # More than 30% repeated words
            issues.append("Excessive word repetition")
        
        return len(issues) > 0, issues
    
    def suggest_description_improvements(self, description: str, brand: str = "", category_info: Tuple = None) -> str:
        """Suggest improved description"""
        desc = safe_str(description).strip()
        
        if not desc:
            return "Product description needed"
        
        # Start with original description
        improved = desc
        
        # Apply improvement patterns
        for vague_term, improvement in DESCRIPTION_IMPROVEMENTS.items():
            improved = re.sub(r'\b' + re.escape(vague_term) + r'\b', improvement, improved, flags=re.IGNORECASE)
        
        # Enhance with brand information if missing
        if brand and brand.lower() not in improved.lower():
            # Check if we should add brand
            meaningful_tokens = extract_meaningful_tokens(improved)
            if len(meaningful_tokens) < 3:
                improved = f"{brand} {improved}"
        
        # Enhance with category information if available and useful
        if category_info and len(meaningful_tokens) < 2:
            cat4 = safe_str(category_info[3]) if len(category_info) > 3 else ""
            if cat4 and cat4.lower() not in improved.lower():
                improved = f"{improved} - {cat4}"
        
        # Clean up formatting
        improved = re.sub(r'\s+', ' ', improved).strip()
        improved = improved[0].upper() + improved[1:] if improved else improved
        
        # Ensure it's actually improved
        if len(improved) <= len(desc) and calculate_text_similarity(improved, desc) > 0.8:
            # Minimal improvement, try to add more context
            tokens = extract_meaningful_tokens(desc)
            if len(tokens) >= 2:
                improved = f"{' '.join(tokens[:3]).title()} Product"
            else:
                improved = f"Enhanced {improved}"
        
        return improved

# =========================
# Enhanced UPC Lookup with Better Performance
# =========================
class UPCLookupService:
    """Enhanced UPC lookup with caching and rate limiting"""
    
    def __init__(self):
        self.session_cache = {}
        self.disk_cache = self._load_disk_cache()
        self.rate_limiter = {}
    
    def _load_disk_cache(self) -> Dict:
        """Load cached UPC data from disk"""
        try:
            cache_file = ".upc_cache.json"
            if os.path.exists(cache_file):
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            log_event("WARNING", f"Failed to load UPC cache: {str(e)}")
        return {}
    
    def _save_disk_cache(self):
        """Save UPC cache to disk"""
        try:
            with open(".upc_cache.json", "w", encoding="utf-8") as f:
                json.dump(self.disk_cache, f)
        except Exception as e:
            log_event("WARNING", f"Failed to save UPC cache: {str(e)}")
    
    def _check_rate_limit(self, service: str, limit_per_minute: int = 60) -> bool:
        """Check if we're within rate limits for a service"""
        now = time.time()
        
        if service not in self.rate_limiter:
            self.rate_limiter[service] = []
        
        # Remove old entries (older than 1 minute)
        self.rate_limiter[service] = [
            timestamp for timestamp in self.rate_limiter[service] 
            if now - timestamp < 60
        ]
        
        # Check if under limit
        if len(self.rate_limiter[service]) >= limit_per_minute:
            return False
        
        # Add current request
        self.rate_limiter[service].append(now)
        return True
    
    def batch_lookup(self, barcodes: List[str], max_concurrent: int = 5) -> Dict[str, Dict]:
        """Perform batch UPC lookup with concurrency"""
        if not barcodes:
            return {}
        
        # Filter valid barcodes
        valid_barcodes = [
            bc for bc in barcodes 
            if bc and str(bc).strip().isdigit() and len(str(bc).strip()) >= 8
        ]
        
        if not valid_barcodes:
            return {}
        
        log_event("INFO", f"Starting UPC lookup for {len(valid_barcodes)} barcodes")
        
        results = {}
        
        # Check caches first
        uncached_barcodes = []
        for bc in valid_barcodes:
            if bc in self.session_cache:
                results[bc] = self.session_cache[bc]
            elif bc in self.disk_cache:
                results[bc] = self.disk_cache[bc]
                self.session_cache[bc] = self.disk_cache[bc]
            else:
                uncached_barcodes.append(bc)
        
        if not uncached_barcodes:
            return results
        
        log_event("INFO", f"Looking up {len(uncached_barcodes)} uncached barcodes")
        
        # Lookup remaining barcodes with concurrency
        try:
            with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                future_to_barcode = {
                    executor.submit(self._lookup_single_barcode, bc): bc 
                    for bc in uncached_barcodes[:50]  # Limit batch size
                }
                
                for future in as_completed(future_to_barcode, timeout=30):
                    barcode = future_to_barcode[future]
                    try:
                        result = future.result()
                        if result:
                            results[barcode] = result
                            self.session_cache[barcode] = result
                            self.disk_cache[barcode] = result
                    except Exception as e:
                        log_event("WARNING", f"UPC lookup failed for {barcode}: {str(e)}")
        
        except Exception as e:
            log_event("ERROR", f"Batch UPC lookup failed: {str(e)}")
        
        # Save cache
        self._save_disk_cache()
        
        return results
    
    def _lookup_single_barcode(self, barcode: str) -> Optional[Dict]:
        """Lookup single barcode from multiple sources"""
        import requests
        
        # Try OpenFoodFacts first
        if self._check_rate_limit("openfoodfacts", 100):
            try:
                response = requests.get(
                    f"https://world.openfoodfacts.org/api/v2/product/{barcode}.json",
                    timeout=5
                )
                
                if response.status_code == 200:
                    data = response.json()
                    product = data.get("product", {})
                    
                    brand = None
                    if "brands_tags" in product and product["brands_tags"]:
                        brand = product["brands_tags"][0]
                    elif "brands" in product:
                        brand = product["brands"]
                    
                    if brand:
                        return {"source": "OpenFoodFacts", "brand": brand}
                
                time.sleep(0.1)  # Rate limiting
            
            except Exception as e:
                log_event("DEBUG", f"OpenFoodFacts lookup failed for {barcode}: {str(e)}")
        
        # Try UPCItemDB if API key is available
        try:
            api_key = st.secrets.get("upcitemdb_api_key")
            if api_key and self._check_rate_limit("upcitemdb", 30):
                endpoint = st.secrets.get("upcitemdb_endpoint", "https://api.upcitemdb.com/prod/trial/lookup")
                headers = {
                    "user_key": api_key,
                    "key_type": "free" if "trial" in endpoint else "paid"
                }
                
                response = requests.get(
                    endpoint,
                    params={"upc": barcode},
                    headers=headers,
                    timeout=5
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("code") == "OK" and data.get("total", 0) > 0:
                        items = data.get("items", [])
                        if items and items[0].get("brand"):
                            return {"source": "UPCItemDB", "brand": items[0]["brand"]}
                
                time.sleep(0.2)  # Rate limiting
        
        except Exception as e:
            log_event("DEBUG", f"UPCItemDB lookup failed for {barcode}: {str(e)}")
        
        return None

# =========================
# Progress Tracking
# =========================
class ProgressTracker:
    """Track and display progress for long-running operations"""
    
    def __init__(self, total_items: int, operation_name: str):
        self.total_items = total_items
        self.operation_name = operation_name
        self.current_item = 0
        self.start_time = time.time()
        
        if STREAMLIT_AVAILABLE:
            self.progress_bar.progress(progress)
            
            elapsed = time.time() - self.start_time
            if progress > 0:
                eta = (elapsed / progress) * (1 - progress)
                eta_str = f"ETA: {int(eta)}s" if eta < 120 else f"ETA: {int(eta/60)}m"
            else:
                eta_str = ""
            
            status_msg = f"{self.operation_name}: {self.current_item}/{self.total_items} {status} {eta_str}"
            self.status_text.text(status_msg)
    
    def complete(self):
        """Mark operation as complete"""
        if STREAMLIT_AVAILABLE:
            self.progress_bar.progress(1.0)
            elapsed = time.time() - self.start_time
            self.status_text.text(f"{self.operation_name} completed in {elapsed:.1f}s")

# =========================
# File I/O Functions
# =========================
@st.cache_data(show_spinner=False)
def read_excel_file(file_path_or_buffer) -> Optional[pd.DataFrame]:
    """Read Excel file with error handling"""
    try:
        df = pd.read_excel(file_path_or_buffer)
        # Normalize column names
        df.columns = [col.strip().upper() for col in df.columns]
        
        # Clean string columns
        string_columns = ["BRAND", "MANUFACTURER", "WEBSITE", "ALLOWED_PRODUCT_TYPES"]
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        return df
    except Exception as e:
        log_event("ERROR", f"Failed to read Excel file: {str(e)}")
        return None

@st.cache_data(show_spinner=False)
def read_user_file(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    """Read user uploaded file"""
    from io import BytesIO, StringIO
    
    try:
        file_name = file_name.lower()
        
        if file_name.endswith('.csv'):
            content = file_bytes.decode('utf-8', errors='ignore')
            df = pd.read_csv(StringIO(content))
        else:
            df = pd.read_excel(BytesIO(file_bytes))
        
        log_event("INFO", f"Successfully loaded file: {file_name}", rows=len(df), columns=len(df.columns))
        return df
        
    except Exception as e:
        log_event("ERROR", f"Failed to read user file: {str(e)}")
        raise

def normalize_column_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column headers to match required format"""
    df = df.copy()
    
    # Create mapping from incoming columns to required columns
    column_mapping = {}
    normalized_incoming = {col: col.strip().upper() for col in df.columns}
    
    for incoming_col, normalized_col in normalized_incoming.items():
        if normalized_col in REQUIRED_COLUMNS:
            column_mapping[incoming_col] = normalized_col
    
    # Rename columns
    if column_mapping:
        df = df.rename(columns=column_mapping)
    
    # Add missing required columns as empty strings
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    
    return df

def ensure_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all output columns exist with default values"""
    df = df.copy()
    
    # Add missing output columns
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            if col in ["Correct Brand?", "Correct Categories?", "Vague Description?"]:
                df[col] = "N"  # Default to No
            else:
                df[col] = ""   # Empty string for suggestions
    
    return df

# =========================
# Brand Reference Management
# =========================
def load_brand_reference(show_upload: bool = True) -> Optional[pd.DataFrame]:
    """Load brand-manufacturer reference file"""
    # Try to load bundled reference file first
    brand_df = read_excel_file("All_Brands_Manufacturers.xlsx")
    
    if brand_df is None and show_upload and STREAMLIT_AVAILABLE:
        st.sidebar.markdown("##### Brand-Manufacturer Reference")
        uploaded_ref = st.sidebar.file_uploader(
            "Upload All_Brands_Manufacturers.xlsx",
            type=["xlsx", "xls"],
            key="brand_reference_uploader"
        )
        
        if uploaded_ref is not None:
            brand_df = read_excel_file(uploaded_ref)
            if brand_df is None:
                st.sidebar.error("Failed to read reference file.")
    
    if brand_df is not None:
        log_event("INFO", "Brand reference loaded", brands=len(brand_df))
    
    return brand_df

@st.cache_data(show_spinner=False)
def get_manufacturer_brand_lists(brand_df: pd.DataFrame) -> Tuple[List[str], Dict[str, List[str]]]:
    """Extract manufacturer and brand lists from reference data"""
    if brand_df is None or brand_df.empty:
        return [], {}
    
    # Get top 50 manufacturers by frequency
    manufacturer_counts = brand_df["MANUFACTURER"].value_counts()
    top_50_manufacturers = manufacturer_counts.head(50).index.tolist()
    
    # Group brands by manufacturer
    brands_by_manufacturer = (
        brand_df.dropna(subset=["MANUFACTURER", "BRAND"])
        .groupby("MANUFACTURER")["BRAND"]
        .apply(lambda x: sorted(x.unique()))
        .to_dict()
    )
    
    return top_50_manufacturers, brands_by_manufacturer

def get_master_brand_set(brand_df: Optional[pd.DataFrame]) -> set:
    """Get set of all master brands in lowercase"""
    if brand_df is None or "BRAND" not in brand_df.columns:
        return set()
    
    return set(brand_df["BRAND"].astype(str).str.strip().str.lower())

def get_allowed_product_types(brand_df: pd.DataFrame, brand_name: str) -> List[str]:
    """Get allowed product types for a specific brand"""
    if brand_df is None or "ALLOWED_PRODUCT_TYPES" not in brand_df.columns:
        return []
    
    brand_rows = brand_df[brand_df["BRAND"].str.lower() == brand_name.lower()]
    if brand_rows.empty:
        return []
    
    allowed_types_raw = safe_str(brand_rows.iloc[0].get("ALLOWED_PRODUCT_TYPES", ""))
    if not allowed_types_raw:
        return []
    
    return [t.strip().lower() for t in allowed_types_raw.split(",") if t.strip()]

# =========================
# Enhanced Brand Processing
# =========================
def map_to_master_brand(raw_brand: str, master_brands: set) -> Optional[str]:
    """Map a raw brand to master brand list with fuzzy matching"""
    if not raw_brand:
        return None
    
    # Clean and normalize
    clean_brand = re.sub(r'[^a-z0-9 ]+', ' ', raw_brand.lower()).strip()
    clean_brand = re.sub(r'\s+', ' ', clean_brand)
    
    if not clean_brand:
        return None
    
    # Exact match
    if clean_brand in master_brands:
        return clean_brand
    
    # Fuzzy match (only check brands starting with same letter for efficiency)
    if clean_brand:
        candidates = [brand for brand in master_brands if brand[0] == clean_brand[0]]
        
        best_match = None
        best_distance = 2
        
        for candidate in candidates:
            distance = levenshtein_distance(clean_brand, candidate, max_distance=1)
            if distance < best_distance:
                best_match = candidate
                best_distance = distance
                if distance == 0:  # Perfect match found
                    break
        
        return best_match if best_distance <= 1 else None
    
    return None

def extract_brand_from_description(description: str, master_brands: set) -> Tuple[Optional[str], str]:
    """Extract potential brand from description"""
    if not description:
        return None, "Low"
    
    desc_lower = description.lower()
    
    # Look for exact master brand matches
    best_brand = None
    best_length = 0
    
    for brand in master_brands:
        if len(brand) > 2 and brand in desc_lower:
            # Check if it's a whole word match
            pattern = r'\b' + re.escape(brand) + r'\b'
            if re.search(pattern, desc_lower) and len(brand) > best_length:
                best_brand = brand
                best_length = len(brand)
    
    if best_brand:
        return best_brand, "High"
    
    # Try to extract potential new brands
    # Look for capitalized words that could be brands
    words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', description)
    
    for word_group in words:
        cleaned = word_group.strip()
        if len(cleaned) >= 3 and cleaned.lower() not in STOPWORDS:
            # Check if it maps to a master brand
            mapped = map_to_master_brand(cleaned, master_brands)
            if mapped:
                return mapped, "Medium"
    
    return None, "Low"

# =========================
# Main Cleanup Functions - Enhanced
# =========================
def brand_accuracy_cleanup(
    df: pd.DataFrame,
    brand_df: Optional[pd.DataFrame],
    selected_manufacturer: str,
    selected_brand: str,
    log_changes_only: bool = True,
    max_logs: int = 200
) -> pd.DataFrame:
    """Enhanced brand accuracy cleanup with UPC integration"""
    
    if not selected_manufacturer or not selected_brand:
        log_event("ERROR", "No manufacturer or brand selected")
        return df
    
    log_event("INFO", "Starting enhanced brand accuracy cleanup", 
              manufacturer=selected_manufacturer, 
              brand=selected_brand, 
              rows=len(df))
    
    df = df.copy()
    df = ensure_output_columns(df)
    
    # Initialize progress tracker
    progress = ProgressTracker(len(df), "Brand Analysis")
    
    # Get master brand set and allowed product types
    master_brands = get_master_brand_set(brand_df)
    allowed_types = get_allowed_product_types(brand_df, selected_brand)
    
    selected_brand_lower = selected_brand.lower()
    
    # Initialize results
    df["Correct Brand?"] = "N"
    df["Suggested Brand"] = ""
    df["Match Strength"] = ""
    
    # Collect barcodes for batch UPC lookup
    upc_service = UPCLookupService()
    barcodes_needed = []
    
    # First pass: analyze each row
    changes_logged = 0
    rows_needing_upc = []
    
    for idx, row in df.iterrows():
        current_brand = safe_str(row.get("BRAND", "")).lower()
        description = safe_str(row.get("DESCRIPTION", ""))
        barcode = safe_str(row.get("BARCODE", ""))
        
        progress.update(1, f"Analyzing row {idx + 1}")
        
        # Check if current brand is generic/missing
        is_generic = (
            not current_brand or 
            current_brand in {"", "none", "null", "nan", "n/a", "na"} or
            current_brand in GENERIC_BRAND_TERMS
        )
        
        # Check if description belongs to selected brand
        belongs_to_selected = False
        if allowed_types:
            # Use allowed product types if available
            for product_type in allowed_types:
                if product_type in description.lower():
                    belongs_to_selected = True
                    break
        else:
            # Fallback to simple brand name matching
            belongs_to_selected = selected_brand_lower in description.lower()
        
        # Set correct brand status
        if not is_generic and belongs_to_selected:
            df.at[idx, "Correct Brand?"] = "Y"
            df.at[idx, "Match Strength"] = "High"
            continue
        
        # Try to find brand suggestions
        suggested_brand, confidence = extract_brand_from_description(description, master_brands)
        
        if suggested_brand and suggested_brand != selected_brand_lower:
            df.at[idx, "Suggested Brand"] = suggested_brand.title()
            df.at[idx, "Match Strength"] = confidence
        else:
            # Mark for UPC lookup if barcode is available
            if barcode and barcode.isdigit() and len(barcode) >= 8:
                rows_needing_upc.append(idx)
                barcodes_needed.append(barcode)
        
        # Log changes
        if log_changes_only and changes_logged < max_logs and df.at[idx, "Correct Brand?"] == "N":
            log_event("INFO", "Brand correction needed",
                      fido=safe_str(row.get("FIDO")),
                      current_brand=current_brand,
                      suggested_brand=df.at[idx, "Suggested Brand"],
                      strength=df.at[idx, "Match Strength"])
            changes_logged += 1
    
    progress.update(0, "Performing UPC lookups...")
    
    # Batch UPC lookup for remaining items
    if barcodes_needed:
        upc_results = upc_service.batch_lookup(list(set(barcodes_needed)))
        
        for idx in rows_needing_upc:
            barcode = safe_str(df.iloc[idx]["BARCODE"])
            if barcode in upc_results:
                upc_data = upc_results[barcode]
                raw_brand = upc_data.get("brand")
                
                if raw_brand:
                    # Try to map to master brands
                    mapped_brand = map_to_master_brand(raw_brand, master_brands)
                    
                    if mapped_brand and mapped_brand != selected_brand_lower:
                        df.at[idx, "Suggested Brand"] = mapped_brand.title()
                        df.at[idx, "Match Strength"] = f"High (UPC Lookup)"
                    elif raw_brand.lower() != selected_brand_lower:
                        df.at[idx, "Suggested Brand"] = f"{raw_brand.title()} (NEW)"
                        df.at[idx, "Match Strength"] = f"Medium (UPC Lookup)"
    
    # Final pass: handle remaining unclear cases
    for idx, row in df.iterrows():
        if not df.at[idx, "Suggested Brand"] and df.at[idx, "Correct Brand?"] == "N":
            # Try family-based suggestion
            description = safe_str(row.get("DESCRIPTION", ""))
            family = detect_product_family(description)
            
            if family:
                df.at[idx, "Suggested Brand"] = f"{selected_brand} {family}"
                df.at[idx, "Match Strength"] = "Low"
            else:
                df.at[idx, "Suggested Brand"] = "Unclear"
                df.at[idx, "Match Strength"] = "Low"
    
    progress.complete()
    
    # Summary stats
    correct_count = len(df[df["Correct Brand?"] == "Y"])
    incorrect_count = len(df[df["Correct Brand?"] == "N"])
    suggestions_count = len(df[df["Suggested Brand"] != ""])
    
    log_event("INFO", "Enhanced brand cleanup completed",
              correct_brands=correct_count,
              incorrect_brands=incorrect_count,
              suggestions_made=suggestions_count,
              total_rows=len(df))
    
    return df

def category_hierarchy_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced category hierarchy cleanup with intelligent suggestions"""
    log_event("INFO", "Starting enhanced category hierarchy cleanup", rows=len(df))
    
    df = df.copy()
    df = ensure_output_columns(df)
    
    # Initialize category intelligence
    category_ai = CategoryIntelligence()
    progress = ProgressTracker(len(df), "Category Analysis")
    
    for idx, row in df.iterrows():
        progress.update(1, f"Analyzing row {idx + 1}")
        
        # Get current categories
        current_cats = (
            safe_str(row.get("CATEGORY_1", "")),
            safe_str(row.get("CATEGORY_2", "")),
            safe_str(row.get("CATEGORY_3", "")),
            safe_str(row.get("CATEGORY_4", ""))
        )
        
        # Check if hierarchy is valid
        is_valid, validation_message = category_ai.validate_hierarchy(*current_cats)
        
        if is_valid:
            df.at[idx, "Correct Categories?"] = "Y"
        else:
            df.at[idx, "Correct Categories?"] = "N"
            
            # Generate suggestions
            description = safe_str(row.get("DESCRIPTION", ""))
            brand = safe_str(row.get("BRAND", ""))
            
            suggestion = category_ai.suggest_categories(description, brand, current_cats)
            
            df.at[idx, "Suggested Category 1"] = suggestion.category_1
            df.at[idx, "Suggested Category 2"] = suggestion.category_2
            df.at[idx, "Suggested Category 3"] = suggestion.category_3
            df.at[idx, "Match Strength"] = suggestion.confidence
    
    progress.complete()
    
    correct_count = len(df[df["Correct Categories?"] == "Y"])
    log_event("INFO", "Enhanced category cleanup completed", 
              correct_categories=correct_count,
              total_rows=len(df))
    
    return df

def vague_description_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced vague description detection and improvement"""
    log_event("INFO", "Starting enhanced description cleanup", rows=len(df))
    
    df = df.copy()
    df = ensure_output_columns(df)
    
    # Initialize description intelligence
    desc_ai = DescriptionIntelligence()
    progress = ProgressTracker(len(df), "Description Analysis")
    
    for idx, row in df.iterrows():
        progress.update(1, f"Analyzing row {idx + 1}")
        
        description = safe_str(row.get("DESCRIPTION", ""))
        brand = safe_str(row.get("BRAND", ""))
        
        # Analyze description
        is_vague, issues = desc_ai.analyze_description(description)
        
        df.at[idx, "Vague Description?"] = "Y" if is_vague else "N"
        
        if is_vague:
            # Get category info for context
            category_info = (
                safe_str(row.get("CATEGORY_1", "")),
                safe_str(row.get("CATEGORY_2", "")),
                safe_str(row.get("CATEGORY_3", "")),
                safe_str(row.get("CATEGORY_4", ""))
            )
            
            # Generate improvement suggestion
            improved_desc = desc_ai.suggest_description_improvements(
                description, brand, category_info
            )
            
            df.at[idx, "Suggested New Description"] = improved_desc
            
            # Set strength based on improvement quality
            if calculate_text_similarity(description, improved_desc) < 0.5:
                df.at[idx, "Match Strength"] = "High"  # Significant improvement
            else:
                df.at[idx, "Match Strength"] = "Medium"  # Minor improvement
    
    progress.complete()
    
    vague_count = len(df[df["Vague Description?"] == "Y"])
    improved_count = len(df[df["Suggested New Description"] != ""])
    
    log_event("INFO", "Enhanced description cleanup completed",
              vague_descriptions=vague_count,
              improvements_suggested=improved_count,
              total_rows=len(df))
    
    return df

def detect_product_family(description: str) -> Optional[str]:
    """Enhanced product family detection"""
    desc_lower = description.lower()
    
    # Wine family
    wine_terms = ["wine", "merlot", "cabernet", "pinot", "chardonnay", "sauvignon", 
                  "syrah", "rioja", "tempranillo", "moscato", "riesling", "prosecco"]
    if any(term in desc_lower for term in wine_terms):
        return "Wine"
    
    # Coffee family
    coffee_terms = ["coffee", "espresso", "cappuccino", "latte", "k-cup", "keurig", 
                   "ground coffee", "instant coffee", "coffee beans"]
    if any(term in desc_lower for term in coffee_terms):
        return "Coffee"
    
    # Supplement family
    supplement_terms = ["vitamin", "supplement", "capsule", "tablet", "gummy", 
                       "protein", "probiotic", "omega"]
    if any(term in desc_lower for term in supplement_terms):
        return "Supplements"
    
    # Electronics family
    electronics_terms = ["headphones", "speaker", "charger", "cable", "bluetooth", 
                         "wireless", "battery"]
    if any(term in desc_lower for term in electronics_terms):
        return "Electronics"
    
    # Home decor family
    decor_terms = ["decoration", "ornament", "candle", "picture frame", "vase", 
                   "pillow", "throw", "curtain"]
    if any(term in desc_lower for term in decor_terms):
        return "Home Decor"
    
    return None

# =========================
# UI Functions (Unchanged to maintain UI)
# =========================
def render_sidebar_controls(brand_df: Optional[pd.DataFrame]):
    """Render sidebar controls and return user selections"""
    st.sidebar.header("📁 File Upload")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload data file from Snowflake export",
        type=["xlsx", "xls", "csv"],
        help="Supported formats: Excel (.xlsx, .xls) and CSV (.csv)"
    )
    
    # Handle file upload
    raw_df = None
    if uploaded_file is not None:
        try:
            file_bytes = uploaded_file.getvalue()
            file_key = f"{uploaded_file.name}:{hash(file_bytes)}"
            
            # Cache file to avoid re-reading
            if st.session_state.get("_last_file_key") != file_key:
                st.session_state["_last_file_key"] = file_key
                st.session_state["_raw_df"] = read_user_file(uploaded_file.name, file_bytes)
                st.sidebar.success(f"✅ File loaded: {uploaded_file.name}")
            
            raw_df = st.session_state.get("_raw_df")
            
        except Exception as e:
            st.sidebar.error(f"❌ Error reading file: {str(e)}")
    
    st.sidebar.header("🔧 Cleanup Options")
    
    cleanup_type = st.sidebar.selectbox(
        "Select cleanup type:",
        ["Brand Accuracy", "Category Hierarchy Cleanup", "Vague Description Cleanup"],
        help="Choose the type of data cleanup to perform"
    )
    
    # Brand-specific controls
    selected_manufacturer = None
    selected_brand = None
    
    if cleanup_type == "Brand Accuracy":
        st.sidebar.subheader("Brand Settings")
        
        if brand_df is not None and not brand_df.empty:
            show_top_only = st.sidebar.checkbox("Show only top 50 manufacturers", value=True)
            
            top_manufacturers, brands_by_manufacturer = get_manufacturer_brand_lists(brand_df)
            
            manufacturer_options = top_manufacturers if show_top_only else sorted(brands_by_manufacturer.keys())
            
            selected_manufacturer = st.sidebar.selectbox(
                "Select manufacturer:",
                options=manufacturer_options,
                help="Choose the manufacturer to focus cleanup on"
            )
            
            if selected_manufacturer:
                brand_options = brands_by_manufacturer.get(selected_manufacturer, [])
                selected_brand = st.sidebar.selectbox(
                    "Select brand:",
                    options=brand_options,
                    help="Choose the specific brand for cleanup"
                )
        else:
            st.sidebar.warning("⚠️ No brand reference data available. Upload the brand-manufacturer reference file.")
    
    st.sidebar.header("⚙️ Advanced Settings")
    
    log_changes_only = st.sidebar.checkbox(
        "Log only changes",
        value=True,
        help="Only log rows that were flagged or changed"
    )
    
    max_logs = st.sidebar.number_input(
        "Maximum log entries:",
        min_value=50,
        max_value=1000,
        value=200,
        step=50,
        help="Limit the number of log entries to prevent performance issues"
    )
    
    run_cleanup = st.sidebar.button(
        "🚀 Run Cleanup",
        use_container_width=True,
        type="primary"
    )
    
    return (uploaded_file, raw_df, cleanup_type, selected_manufacturer, 
            selected_brand, log_changes_only, max_logs, run_cleanup)

def render_results(df: pd.DataFrame):
    """Render results and download options"""
    st.subheader("📊 Results")
    
    # Show summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", len(df))
    
    with col2:
        if "Correct Brand?" in df.columns:
            correct_brands = len(df[df["Correct Brand?"] == "Y"])
            st.metric("Correct Brands", correct_brands)
    
    with col3:
        if "Correct Categories?" in df.columns:
            correct_categories = len(df[df["Correct Categories?"] == "Y"])
            st.metric("Correct Categories", correct_categories)
    
    with col4:
        if "Suggested Brand" in df.columns:
            suggestions = len(df[df["Suggested Brand"] != ""])
            st.metric("Suggestions Made", suggestions)
    
    # Show detailed breakdown
    with st.expander("📈 Detailed Statistics", expanded=False):
        stats_col1, stats_col2 = st.columns(2)
        
        with stats_col1:
            if "Match Strength" in df.columns:
                strength_counts = df["Match Strength"].value_counts()
                st.write("**Match Strength Distribution:**")
                for strength, count in strength_counts.items():
                    if strength:
                        st.write(f"• {strength}: {count} rows")
        
        with stats_col2:
            if "Vague Description?" in df.columns:
                vague_count = len(df[df["Vague Description?"] == "Y"])
                clear_count = len(df[df["Vague Description?"] == "N"])
                st.write("**Description Quality:**")
                st.write(f"• Clear: {clear_count} rows")
                st.write(f"• Vague: {vague_count} rows")
    
    # Show preview with pagination
    st.subheader("📋 Data Preview")
    
    # Add filters for better data exploration
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        if "Correct Brand?" in df.columns:
            brand_filter = st.selectbox(
                "Filter by Brand Status:",
                options=["All", "Correct", "Incorrect"],
                help="Filter rows by brand correctness"
            )
        else:
            brand_filter = "All"
    
    with filter_col2:
        if "Match Strength" in df.columns:
            strength_filter = st.selectbox(
                "Filter by Strength:",
                options=["All"] + sorted(df["Match Strength"].unique().tolist()),
                help="Filter by match strength"
            )
        else:
            strength_filter = "All"
    
    with filter_col3:
        show_suggestions_only = st.checkbox(
            "Show only suggestions",
            help="Show only rows with suggestions"
        )
    
    # Apply filters
    filtered_df = df.copy()
    
    if brand_filter != "All" and "Correct Brand?" in df.columns:
        filter_value = "Y" if brand_filter == "Correct" else "N"
        filtered_df = filtered_df[filtered_df["Correct Brand?"] == filter_value]
    
    if strength_filter != "All" and "Match Strength" in df.columns:
        filtered_df = filtered_df[filtered_df["Match Strength"] == strength_filter]
    
    if show_suggestions_only:
        suggestion_cols = ["Suggested Brand", "Suggested Category 1", "Suggested New Description"]
        available_cols = [col for col in suggestion_cols if col in filtered_df.columns]
        if available_cols:
            mask = pd.Series(False, index=filtered_df.index)
            for col in available_cols:
                mask |= (filtered_df[col] != "")
            filtered_df = filtered_df[mask]
    
    st.info(f"Showing {len(filtered_df)} of {len(df)} rows")
    
    # Display filtered data
    if not filtered_df.empty:
        # Show only most relevant columns for preview
        display_columns = [
            "FIDO", "BARCODE", "DESCRIPTION", "BRAND", "MANUFACTURER",
            "Correct Brand?", "Suggested Brand", "Match Strength"
        ]
        available_display_cols = [col for col in display_columns if col in filtered_df.columns]
        
        st.dataframe(
            filtered_df[available_display_cols].head(100), 
            use_container_width=True,
            height=400
        )
    else:
        st.warning("No data matches the current filters.")
    
    # Download cleaned file
    st.subheader("📥 Download Results")
    
    output_buffer = io.BytesIO()
    with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Cleaned_Data')
        
        # Add summary sheet
        summary_data = {
            "Metric": ["Total Rows", "Processing Date", "Cleanup Type"],
            "Value": [
                len(df),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Multiple" if len([col for col in OUTPUT_COLUMNS if col in df.columns]) > 3 else "Single"
            ]
        }
        
        # Add specific metrics based on available columns
        if "Correct Brand?" in df.columns:
            summary_data["Metric"].extend(["Correct Brands", "Incorrect Brands"])
            summary_data["Value"].extend([
                len(df[df["Correct Brand?"] == "Y"]),
                len(df[df["Correct Brand?"] == "N"])
            ])
        
        if "Correct Categories?" in df.columns:
            summary_data["Metric"].extend(["Correct Categories", "Incorrect Categories"])
            summary_data["Value"].extend([
                len(df[df["Correct Categories?"] == "Y"]),
                len(df[df["Correct Categories?"] == "N"])
            ])
        
        if "Vague Description?" in df.columns:
            summary_data["Metric"].extend(["Clear Descriptions", "Vague Descriptions"])
            summary_data["Value"].extend([
                len(df[df["Vague Description?"] == "N"]),
                len(df[df["Vague Description?"] == "Y"])
            ])
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, index=False, sheet_name='Summary')
    
    download_button = st.download_button(
        "📥 Download Cleaned File (.xlsx)",
        data=output_buffer.getvalue(),
        file_name=f"taxonomy_guardian_cleaned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
    
    if download_button:
        st.success("✅ File downloaded successfully!")
    
    # Export NEW brand suggestions separately
    if "Suggested Brand" in df.columns:
        new_suggestions = df[df["Suggested Brand"].str.endswith("(NEW)", na=False)]
        
        if not new_suggestions.empty:
            st.subheader("🆕 New Brand Suggestions")
            st.write(f"Found {len(new_suggestions)} products with potential new brands:")
            
            display_cols = ["FIDO", "BARCODE", "DESCRIPTION", "BRAND", "MANUFACTURER", "Suggested Brand", "Match Strength"]
            available_cols = [col for col in display_cols if col in new_suggestions.columns]
            
            st.dataframe(new_suggestions[available_cols], use_container_width=True)
            
            # Download new suggestions
            new_buffer = io.BytesIO()
            with pd.ExcelWriter(new_buffer, engine='openpyxl') as writer:
                new_suggestions[available_cols].to_excel(writer, index=False, sheet_name='New_Brand_Suggestions')
            
            new_download = st.download_button(
                "📥 Download New Brand Suggestions (.xlsx)",
                data=new_buffer.getvalue(),
                file_name=f"taxonomy_guardian_new_brands_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            
            if new_download:
                st.success("✅ New brand suggestions downloaded!")

def render_logs():
    """Render application logs with improved formatting"""
    with st.expander("📋 Application Logs", expanded=False):
        logs = st.session_state.get("logs", [])
        
        if logs:
            # Show controls for log filtering
            log_col1, log_col2, log_col3 = st.columns(3)
            
            with log_col1:
                log_levels = ["All"] + sorted(list(set(log.get("level", "INFO") for log in logs)))
                selected_level = st.selectbox("Filter by level:", log_levels, key="log_level_filter")
            
            with log_col2:
                num_logs = st.selectbox("Number of logs:", [10, 25, 50, 100], index=2, key="num_logs")
            
            with log_col3:
                if st.button("🗑️ Clear Logs"):
                    clear_logs()
                    st.rerun()
            
            # Filter and display logs
            filtered_logs = logs
            if selected_level != "All":
                filtered_logs = [log for log in logs if log.get("level") == selected_level]
            
            # Show recent logs (newest first)
            recent_logs = filtered_logs[-num_logs:]
            
            if recent_logs:
                for log_entry in reversed(recent_logs):
                    timestamp = log_entry.get("timestamp", "")
                    level = log_entry.get("level", "INFO")
                    message = log_entry.get("message", "")
                    
                    # Format timestamp
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        formatted_time = dt.strftime("%H:%M:%S")
                    except:
                        formatted_time = timestamp[:8] if len(timestamp) > 8 else timestamp
                    
                    # Create expandable log entry for detailed info
                    extra_data = {k: v for k, v in log_entry.items() 
                                 if k not in ["timestamp", "level", "message"]}
                    
                    if extra_data:
                        with st.expander(f"[{formatted_time}] {level}: {message}", expanded=False):
                            st.json(extra_data)
                    else:
                        # Color code by level
                        if level == "ERROR":
                            st.error(f"[{formatted_time}] {message}")
                        elif level == "WARNING":
                            st.warning(f"[{formatted_time}] {message}")
                        elif level == "SUCCESS":
                            st.success(f"[{formatted_time}] {message}")
                        else:
                            st.info(f"[{formatted_time}] {message}")
            else:
                st.info("No logs match the current filter")
        else:
            st.info("No logs available")

# =========================
# Main Application
# =========================
def main():
    """Enhanced main application function"""
    
    if not STREAMLIT_AVAILABLE:
        print("Streamlit not available. Running in test mode.")
        return
    
    # Load brand reference
    brand_df = load_brand_reference(show_upload=True)
    
    # Render sidebar controls
    (uploaded_file, raw_df, cleanup_type, selected_manufacturer, 
     selected_brand, log_changes_only, max_logs, run_cleanup) = render_sidebar_controls(brand_df)
    
    # Main content area
    st.header("🛡️ Taxonomy Guardian - Data Cleanup Status")
    
    # Show file status
    if uploaded_file is None:
        st.info("👆 Upload a data file in the sidebar to get started")
        
        # Show some helpful information
        with st.expander("ℹ️ How to Use Taxonomy Guardian", expanded=True):
            st.markdown("""
            **Step 1:** Upload your data file (Excel or CSV) from Snowflake export
            
            **Step 2:** Select the type of cleanup you need:
            - **Brand Accuracy**: Validates and suggests brand corrections
            - **Category Hierarchy**: Ensures all 4 category levels are properly filled
            - **Vague Description**: Identifies and improves unclear product descriptions
            
            **Step 3:** For Brand Accuracy, select your target manufacturer and brand
            
            **Step 4:** Click "Run Cleanup" to process your data
            
            **Step 5:** Review results and download the cleaned file
            """)
        
        render_logs()
        return
    
    if raw_df is None:
        st.error("❌ Failed to load the uploaded file. Please check the file format and try again.")
        render_logs()
        return
    
    # Show file info with enhanced stats
    st.success(f"✅ File loaded successfully: **{uploaded_file.name}**")
    
    file_col1, file_col2, file_col3, file_col4 = st.columns(4)
    with file_col1:
        st.metric("Rows", f"{len(raw_df):,}")
    with file_col2:
        st.metric("Columns", len(raw_df.columns))
    with file_col3:
        # Calculate file size
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.metric("File Size", f"{file_size_mb:.1f} MB")
    with file_col4:
        # Show data completeness
        completeness = (raw_df.count().sum() / (len(raw_df) * len(raw_df.columns))) * 100
        st.metric("Data Completeness", f"{completeness:.1f}%")
    
    # Check for required columns and show warnings
    try:
        processed_df = normalize_column_headers(raw_df)
        missing_columns = [col for col in REQUIRED_COLUMNS if col not in processed_df.columns]
        
        if missing_columns:
            st.warning(f"⚠️ Missing required columns (will be added as empty): {', '.join(missing_columns)}")
        else:
            st.success("✅ All required columns are present")
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        render_logs()
        return
    
    # Show cleanup selection with more details
    st.subheader(f"🔧 Selected Cleanup: {cleanup_type}")
    
    cleanup_descriptions = {
        "Brand Accuracy": "Validates brand names against master reference and suggests corrections using fuzzy matching and UPC lookups.",
        "Category Hierarchy Cleanup": "Ensures all 4 category levels are populated and logically consistent with intelligent suggestions.",
        "Vague Description Cleanup": "Identifies unclear product descriptions and suggests improvements for better clarity."
    }
    
    st.info(cleanup_descriptions.get(cleanup_type, "Unknown cleanup type"))
    
    if cleanup_type == "Brand Accuracy":
        if not selected_manufacturer or not selected_brand:
            st.warning("⚠️ Please select both manufacturer and brand in the sidebar to proceed")
            render_logs()
            return
        else:
            st.success(f"🎯 Target Configuration: **{selected_manufacturer}** → **{selected_brand}**")
            
            # Show additional brand info if available
            if brand_df is not None:
                allowed_types = get_allowed_product_types(brand_df, selected_brand)
                if allowed_types:
                    st.info(f"📋 Allowed Product Types: {', '.join(allowed_types)}")
    
    # Run cleanup if requested
    if not run_cleanup:
        st.info("👆 Click **Run Cleanup** in the sidebar to process your data")
        render_logs()
        return
    
    # Execute cleanup with enhanced error handling and progress tracking
    try:
        log_event("INFO", f"Starting {cleanup_type} cleanup", 
                  rows=len(processed_df),
                  cleanup_type=cleanup_type)
        
        with st.spinner(f"🔄 Running {cleanup_type.lower()}..."):
            
            if cleanup_type == "Brand Accuracy":
                cleaned_df = brand_accuracy_cleanup(
                    processed_df, 
                    brand_df, 
                    selected_manufacturer, 
                    selected_brand,
                    log_changes_only, 
                    max_logs
                )
            
            elif cleanup_type == "Category Hierarchy Cleanup":
                cleaned_df = category_hierarchy_cleanup(processed_df)
            
            elif cleanup_type == "Vague Description Cleanup":
                cleaned_df = vague_description_cleanup(processed_df)
            
            else:
                st.error("❌ Unknown cleanup type selected")
                return
        
        st.success("✅ Cleanup completed successfully!")
        log_event("SUCCESS", f"{cleanup_type} cleanup completed successfully")
        
        render_results(cleaned_df)
    
    except Exception as e:
        error_msg = f"Error during {cleanup_type.lower()}: {str(e)}"
        st.error(f"❌ {error_msg}")
        log_event("ERROR", error_msg)
        
        # Show helpful error recovery suggestions
        with st.expander("🔧 Troubleshooting Tips", expanded=True):
            st.markdown("""
            **Common issues and solutions:**
            
            1. **File format errors**: Ensure your file is a valid Excel (.xlsx) or CSV format
            2. **Memory errors**: Try processing smaller batches of data (< 10,000 rows)
            3. **Network timeouts**: UPC lookups may fail due to network issues - this is normal
            4. **Missing reference data**: Ensure the brand-manufacturer reference file is uploaded
            5. **Column mapping**: Check that your data has the required column structure
            
            **If issues persist:** Check the application logs below for detailed error information.
            """)
    
    # Always show logs at the bottom
    render_logs()

# =========================
# Entry Point
# =========================
if __name__ == "__main__":
    main()_bar = st.progress(0)
            self.status_text = st.empty()
    
    def update(self, increment: int = 1, status: str = ""):
        """Update progress"""
        self.current_item += increment
        progress = min(self.current_item / self.total_items, 1.0)
        
        if STREAMLIT_AVAILABLE:
            self.progress
