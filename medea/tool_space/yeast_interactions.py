"""
Yeast Interaction Query Tools

Provides direct access to curated interaction and annotation databases for
Saccharomyces cerevisiae, covering three fundamentally different interaction
types plus functional annotations:

    physical  = proteins physically interact / bind in the cell
    genetic   = double knockout phenotype differs from single-knockout expectation
    functional = genes share similar functions (integrated evidence)

Tools:
    - query_sgd_interactions:          Query SGD (BioGRID-sourced) curated
      genetic AND physical interactions between two yeast genes.
    - query_string_yeast_interactions: Query STRING-DB for functional
      association scores between two yeast genes.
    - query_costanzo_sga_dataset:      Query local Costanzo SGA quantitative
      genetic interaction dataset (epsilon scores).
    - query_yeast_go_annotations:      Query SGD for GO annotations
      (molecular function, biological process, cellular component).
"""

import os
import json
import time
import logging
import tempfile
import requests
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — no magic numbers buried in functions
# ---------------------------------------------------------------------------

# SGD REST API (BioGRID-sourced curated interactions)
SGD_API_BASE = "https://www.yeastgenome.org/backend"
SGD_API_TIMEOUT = 60          # seconds; hub genes like SGS1 return large payloads

# STRING-DB API (functional associations)
STRING_API_URL = "https://string-db.org/api/json/network"
STRING_SPECIES_YEAST = 4932   # NCBI taxonomy ID for S. cerevisiae
STRING_API_TIMEOUT = 20


# HTTP retry settings
HTTP_MAX_RETRIES = 3
HTTP_RETRY_BACKOFF_BASE = 2   # seconds, multiplied by attempt number
HTTP_RATE_LIMIT_BACKOFF = 5   # seconds, multiplied by attempt number

# Known hub genes with very large interaction counts (>2000 total interactions
# in SGD). When querying pairs involving these, query the OTHER gene first to
# avoid downloading a huge JSON payload.
_KNOWN_HUB_GENES = frozenset({
    "SGS1", "RPD3", "SIN3", "RAD52", "BIM1", "MMS4", "SLX4", "RAD51",
    "TEL1", "MEC1", "RAD9", "CHK1", "DUN1", "RSC2", "RSC1", "IRA1",
    "CDC73", "DST1", "RAD54", "MSH6", "CDC4", "ELG1", "BLM10",
})

# Evidence classification for structured tool returns.
# Downstream consumers (PA agent, Evidence Auditor, Hypo panel) use these to
# distinguish informative negatives from uninformative search failures.
#
# evidence_type:  did the tool find data?  (coverage / availability)
# sl_relevance:   what does the data mean for the SL hypothesis?  (direction)
EVIDENCE_HIT = "hit"                         # Found records in the database
EVIDENCE_NO_HIT = "no_hit"                   # Searched correctly, no records found
EVIDENCE_UNINFORMATIVE = "uninformative"     # Tool scope doesn't cover the request
EVIDENCE_ERROR = "error"                     # Tool failed (API down, parse error, etc.)

# sl_relevance values (only meaningful when evidence_type == "hit")
SL_AGGRAVATING = "aggravating"   # Negative GI / SL reported — supports SL hypothesis
SL_ALLEVIATING = "alleviating"   # Only positive GI reported — opposes SL hypothesis
SL_MIXED = "mixed"               # Both aggravating and alleviating interactions
SL_NONE = "none"                 # No data to assess direction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SGD_CACHE: Dict[str, Any] = {}
_STRING_CACHE: Dict[str, Any] = {}

def _log(tool: str, msg: str, level: str = "INFO"):
    print(f"[{tool}] {level}: {msg}", flush=True)


def _atomic_write_bytes(filepath, data: bytes):
    """Write bytes to a file atomically (temp file + os.replace).
    Safe for concurrent processes writing to the same path."""
    parent = filepath if isinstance(filepath, Path) else Path(filepath)
    parent = parent.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp_path, str(filepath))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _safe_request(url: str, params: dict = None, timeout: int = 30,
                  tool: str = "HTTP") -> Optional[requests.Response]:
    """HTTP GET with retry and structured logging."""
    for attempt in range(HTTP_MAX_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.exceptions.Timeout:
            _log(tool, f"Timeout (attempt {attempt+1}/{HTTP_MAX_RETRIES})", "WARNING")
            time.sleep(HTTP_RETRY_BACKOFF_BASE * (attempt + 1))
        except requests.exceptions.HTTPError as e:
            if r.status_code == 429:
                _log(tool, f"Rate limited, backing off (attempt {attempt+1}/{HTTP_MAX_RETRIES})", "WARNING")
                time.sleep(HTTP_RATE_LIMIT_BACKOFF * (attempt + 1))
            elif r.status_code == 404:
                _log(tool, f"Gene not found (404): {url}", "WARNING")
                return None
            else:
                _log(tool, f"HTTP {r.status_code}: {e}", "ERROR")
                return None
        except requests.exceptions.RequestException as e:
            _log(tool, f"Request error: {e}", "ERROR")
            return None
    _log(tool, "All retries exhausted", "ERROR")
    return None


# ===========================================================================
# Tool 1: SGD Interaction Query (Genetic + Physical)
# ===========================================================================

def query_sgd_interactions(
    gene_a: str,
    gene_b: str,
) -> dict:
    """
    SPECIES: Yeast (Saccharomyces cerevisiae).
    Query the Saccharomyces Genome Database (SGD) for curated genetic AND
    physical interactions between two yeast genes. SGD aggregates
    BioGRID-curated high-throughput and manually curated interaction data.

    Genetic interactions: Synthetic Lethality, Negative Genetic, Positive
    Genetic, Synthetic Growth Defect, Phenotypic Enhancement, etc.
    Physical interactions: Affinity Capture, Two-hybrid, Co-purification,
    Reconstituted Complex, PCA, etc.

    These are fundamentally different: genetic = double knockout phenotype
    differs from expectation; physical = proteins bind / interact in the cell.

    Parameters
    ----------
    gene_a : str
        First yeast gene standard name or systematic name (e.g. "SGS1",
        "YMR190C").
    gene_b : str
        Second yeast gene standard name or systematic name.

    Returns
    -------
    dict
        {
            "success": bool,
            "gene_a": str,
            "gene_b": str,
            "genetic_interactions": [...],
            "physical_interactions": [...],
            "num_genetic_interactions": int,
            "num_physical_interactions": int,
            "has_negative_interaction": bool,
            "has_synthetic_lethality": bool,
            "has_physical_interaction": bool,
            "summary": str,
            "evidence_type": str,
            "sl_relevance": str,
            "error": str or None
        }
    """
    TOOL = "SGD"
    gene_a = gene_a.strip().upper()
    gene_b = gene_b.strip().upper()
    _log(TOOL, f"Querying interactions: {gene_a} × {gene_b}")

    # Check cache
    cache_key = tuple(sorted([gene_a, gene_b]))
    if cache_key in _SGD_CACHE:
        _log(TOOL, "Cache hit", "INFO")
        return _SGD_CACHE[cache_key]

    result = {
        "success": False, "gene_a": gene_a, "gene_b": gene_b,
        "genetic_interactions": [], "physical_interactions": [],
        "num_genetic_interactions": 0, "num_physical_interactions": 0,
        "summary": "", "has_negative_interaction": False,
        "has_synthetic_lethality": False, "has_physical_interaction": False,
        "error": None,
        "evidence_type": EVIDENCE_ERROR,
        "sl_relevance": SL_NONE,
        "coverage": {
            "database": "SGD (BioGRID-sourced curated interactions)",
            "scope": "All published genetic and physical interactions for S. cerevisiae",
            "conditions": "Standard laboratory conditions (not condition-specific)",
            "limitations": "Does not include unpublished screens or condition-specific SGA data",
        },
    }

    # Choose which gene to query first: prefer the non-hub gene (smaller
    # response payload) to avoid downloading thousands of interactions.
    # E.g. querying EAR1 (71 genetic interactions) is much faster than
    # querying SGS1 (4113 interactions) when we only need the pair.
    primary, secondary = gene_a, gene_b
    if gene_a in _KNOWN_HUB_GENES and gene_b not in _KNOWN_HUB_GENES:
        primary, secondary = gene_b, gene_a
    elif gene_b in _KNOWN_HUB_GENES and gene_a not in _KNOWN_HUB_GENES:
        primary, secondary = gene_a, gene_b

    url = f"{SGD_API_BASE}/locus/{primary}/interaction_details"
    resp = _safe_request(url, timeout=SGD_API_TIMEOUT, tool=TOOL)

    if resp is None:
        # Try the other gene (in case primary name didn't resolve)
        _log(TOOL, f"Retrying with {secondary} as primary query", "INFO")
        url = f"{SGD_API_BASE}/locus/{secondary}/interaction_details"
        resp = _safe_request(url, timeout=SGD_API_TIMEOUT, tool=TOOL)
        if resp is None:
            result["error"] = f"SGD API unreachable for both {gene_a} and {gene_b}"
            _log(TOOL, result["error"], "ERROR")
            return result

    try:
        all_interactions = resp.json()
    except (json.JSONDecodeError, ValueError) as e:
        result["error"] = f"Failed to parse SGD response: {e}"
        _log(TOOL, result["error"], "ERROR")
        return result

    # Filter interactions involving the target pair — separate genetic vs physical
    matched_genetic = []
    matched_physical = []
    for rec in all_interactions:
        l1 = rec.get("locus1", {}).get("display_name", "").upper()
        l2 = rec.get("locus2", {}).get("display_name", "").upper()
        l1_sys = rec.get("locus1", {}).get("format_name", "").upper()
        l2_sys = rec.get("locus2", {}).get("format_name", "").upper()
        pair_names = {l1, l2, l1_sys, l2_sys}
        if gene_a in pair_names and gene_b in pair_names:
            itype = rec.get("interaction_type", "")
            if itype == "Genetic":
                matched_genetic.append(rec)
            elif itype == "Physical":
                matched_physical.append(rec)

    # Deduplicate and extract records
    def _extract_records(matched):
        seen_ids = set()
        records = []
        for rec in matched:
            rid = rec.get("id")
            if rid in seen_ids:
                continue
            seen_ids.add(rid)
            exp = rec.get("experiment", {}).get("display_name", "Unknown")
            pheno_raw = rec.get("phenotype")
            pheno = pheno_raw.get("display_name", "") if isinstance(pheno_raw, dict) else str(pheno_raw or "")
            ann = rec.get("annotation_type", "")
            src = rec.get("source", {}).get("display_name", "")
            ref_obj = rec.get("reference", {})
            ref = ref_obj.get("display_name", "") if ref_obj else ""
            pmid = ref_obj.get("pubmed_id") if ref_obj else None
            records.append({
                "experiment": exp, "phenotype": pheno,
                "annotation_type": ann, "source": src,
                "reference": ref, "pubmed_id": pmid,
            })
        return records

    genetic = _extract_records(matched_genetic)
    physical = _extract_records(matched_physical)

    # Classify genetic interactions
    NEGATIVE_TYPES = {
        "Synthetic Lethality", "Negative Genetic",
        "Synthetic Growth Defect", "Dosage Lethality",
        "Phenotypic Enhancement",
    }
    POSITIVE_TYPES = {
        "Positive Genetic", "Synthetic Rescue",
        "Phenotypic Suppression", "Dosage Rescue",
    }

    neg = [i for i in genetic if i["experiment"] in NEGATIVE_TYPES]
    pos = [i for i in genetic if i["experiment"] in POSITIVE_TYPES]
    sl = [i for i in genetic if i["experiment"] == "Synthetic Lethality"]
    curated = [i for i in genetic if i["annotation_type"] == "manually curated"]

    result["success"] = True
    result["genetic_interactions"] = genetic
    result["physical_interactions"] = physical
    result["num_genetic_interactions"] = len(genetic)
    result["num_physical_interactions"] = len(physical)
    result["has_negative_interaction"] = len(neg) > 0
    result["has_synthetic_lethality"] = len(sl) > 0
    result["has_physical_interaction"] = len(physical) > 0

    # Build summary with evidence classification
    has_any = genetic or physical
    if not has_any:
        result["evidence_type"] = EVIDENCE_NO_HIT
        result["sl_relevance"] = SL_NONE
        result["summary"] = (
            f"NO HIT: No curated interactions (genetic or physical) found between "
            f"{gene_a} and {gene_b} in SGD/BioGRID. Note: SGD covers published "
            f"literature only — absence may mean the pair was never screened, "
            f"not that no interaction exists."
        )
    else:
        result["evidence_type"] = EVIDENCE_HIT
        # sl_relevance based on GENETIC interactions only (physical ≠ genetic)
        if neg and pos:
            result["sl_relevance"] = SL_MIXED
        elif neg:
            result["sl_relevance"] = SL_AGGRAVATING
        elif pos:
            result["sl_relevance"] = SL_ALLEVIATING
        else:
            result["sl_relevance"] = SL_NONE

        parts = []

        # Genetic interaction summary
        if genetic:
            exp_counts = {}
            for i in genetic:
                exp_counts[i["experiment"]] = exp_counts.get(i["experiment"], 0) + 1
            exp_str = ", ".join(f"{v}× {k}" for k, v in sorted(exp_counts.items(), key=lambda x: -x[1]))
            pmids = sorted(set(str(i["pubmed_id"]) for i in genetic if i["pubmed_id"]))
            pmid_str = f" (PMIDs: {', '.join(pmids[:5])})" if pmids else ""
            curated_str = f" ({len(curated)} manually curated)" if curated else ""
            parts.append(
                f"GENETIC: {len(genetic)} interaction(s): {exp_str}{curated_str}{pmid_str}."
            )
            if sl:
                parts.append(
                    f"Synthetic Lethality directly reported ({len(sl)} record(s)). "
                    f"[sl_relevance=aggravating]"
                )
            elif neg:
                parts.append(
                    f"Negative/aggravating interactions ({len(neg)} record(s)), "
                    f"synthetic sickness but not full lethality. [sl_relevance=aggravating]"
                )
            if pos and not neg:
                parts.append(
                    f"Only positive/alleviating interactions ({len(pos)} record(s)). "
                    f"Evidence AGAINST synthetic lethality. [sl_relevance=alleviating]"
                )
            if neg and pos:
                parts.append(
                    f"[sl_relevance=mixed: {len(neg)} aggravating + {len(pos)} alleviating]"
                )

        # Physical interaction summary
        if physical:
            phys_exp_counts = {}
            for i in physical:
                phys_exp_counts[i["experiment"]] = phys_exp_counts.get(i["experiment"], 0) + 1
            phys_str = ", ".join(f"{v}× {k}" for k, v in sorted(phys_exp_counts.items(), key=lambda x: -x[1]))
            parts.append(
                f"PHYSICAL: {len(physical)} interaction(s): {phys_str}. "
                f"Proteins physically bind — indicates functional proximity "
                f"but does not directly indicate genetic interaction direction."
            )

        if not genetic:
            parts.append(
                f"No genetic interactions found. Physical interaction alone "
                f"does not determine SL. [sl_relevance=none]"
            )

        result["summary"] = f"{gene_a} × {gene_b}: " + " | ".join(parts)

    _log(TOOL, f"[evidence_type={result['evidence_type']}] {result['summary']}")
    _SGD_CACHE[cache_key] = result
    return result


# ===========================================================================
# Tool 2: STRING-DB Yeast Interaction Query
# ===========================================================================

def query_string_yeast_interactions(
    gene_a: str,
    gene_b: str,
    required_score: int = 0,
) -> dict:
    """
    SPECIES: Yeast (Saccharomyces cerevisiae).
    Query STRING-DB for functional association evidence between two yeast
    genes. STRING integrates multiple evidence channels: experimental
    (physical/genetic), co-expression, text mining, database imports, and
    computational predictions. Returns a combined score (0-1) and per-channel
    scores.

    Useful as a complement to SGD genetic interactions: STRING captures
    broader functional associations (protein-protein interactions,
    co-expression, pathway co-membership) that can support or contextualize
    genetic interaction predictions.

    Parameters
    ----------
    gene_a : str
        First yeast gene standard name (e.g. "SGS1").
    gene_b : str
        Second yeast gene standard name (e.g. "RPD3").
    required_score : int
        Minimum combined score threshold (0-1000). Default 0 returns all.

    Returns
    -------
    dict
        {
            "success": bool,
            "gene_a": str,
            "gene_b": str,
            "combined_score": float,  # 0-1 scale
            "experimental_score": float,
            "database_score": float,
            "textmining_score": float,
            "coexpression_score": float,
            "neighborhood_score": float,
            "fusion_score": float,
            "cooccurrence_score": float,
            "summary": str,
            "error": str or None
        }
    """
    TOOL = "STRING_DB"
    gene_a = gene_a.strip()
    gene_b = gene_b.strip()
    _log(TOOL, f"Querying yeast functional associations: {gene_a} × {gene_b}")

    cache_key = tuple(sorted([gene_a.upper(), gene_b.upper()]))
    if cache_key in _STRING_CACHE:
        _log(TOOL, "Cache hit")
        return _STRING_CACHE[cache_key]

    result = {
        "success": False, "gene_a": gene_a, "gene_b": gene_b,
        "combined_score": 0.0, "experimental_score": 0.0,
        "database_score": 0.0, "textmining_score": 0.0,
        "coexpression_score": 0.0, "neighborhood_score": 0.0,
        "fusion_score": 0.0, "cooccurrence_score": 0.0,
        "summary": "", "error": None,
        "evidence_type": EVIDENCE_ERROR,
        "sl_relevance": SL_NONE,
        "coverage": {
            "database": "STRING-DB (functional association network)",
            "scope": "Integrated functional associations from experimental, co-expression, text mining, database, and computational channels",
            "conditions": "Condition-agnostic (aggregated across all available data)",
            "limitations": "Scores are integrated predictions, not direct experimental measurements for specific conditions",
        },
    }

    # Query STRING network endpoint
    # Use literal newline (\r) as separator — requests will URL-encode it properly
    params = {
        "identifiers": f"{gene_a}\r{gene_b}",
        "species": STRING_SPECIES_YEAST,
        "required_score": required_score,
        "caller_identity": "medea_agent",
    }
    resp = _safe_request(
        STRING_API_URL,
        params=params, timeout=STRING_API_TIMEOUT, tool=TOOL,
    )

    if resp is None:
        result["error"] = "STRING-DB API unreachable"
        _log(TOOL, result["error"], "ERROR")
        return result

    try:
        data = resp.json()
    except (json.JSONDecodeError, ValueError) as e:
        result["error"] = f"Failed to parse STRING response: {e}"
        return result

    if not data:
        result["success"] = True
        result["evidence_type"] = EVIDENCE_NO_HIT
        result["sl_relevance"] = SL_NONE
        result["summary"] = (
            f"NO HIT: No functional association found between "
            f"{gene_a} and {gene_b} in STRING-DB (S. cerevisiae). STRING "
            f"integrates experimental, co-expression, text mining, and database "
            f"evidence — absence here means no known functional link across all channels."
        )
        _log(TOOL, f"[evidence_type={result['evidence_type']}] {result['summary']}")
        _STRING_CACHE[cache_key] = result
        return result

    # Extract scores from the first (should be only) result
    entry = data[0]
    result["success"] = True
    result["evidence_type"] = EVIDENCE_HIT
    result["combined_score"] = entry.get("score", 0)
    result["experimental_score"] = entry.get("escore", 0)
    result["database_score"] = entry.get("dscore", 0)
    result["textmining_score"] = entry.get("tscore", 0)
    result["coexpression_score"] = entry.get("ascore", 0)
    result["neighborhood_score"] = entry.get("nscore", 0)
    result["fusion_score"] = entry.get("fscore", 0)
    result["cooccurrence_score"] = entry.get("pscore", 0)

    # Build summary
    score = result["combined_score"]
    strength = (
        "very high" if score >= 0.9 else
        "high" if score >= 0.7 else
        "medium" if score >= 0.4 else
        "low" if score > 0 else "none"
    )

    channels = []
    if result["experimental_score"] > 0.1:
        channels.append(f"experimental={result['experimental_score']:.3f}")
    if result["database_score"] > 0.1:
        channels.append(f"database={result['database_score']:.3f}")
    if result["textmining_score"] > 0.1:
        channels.append(f"text_mining={result['textmining_score']:.3f}")
    if result["coexpression_score"] > 0.1:
        channels.append(f"coexpression={result['coexpression_score']:.3f}")

    channel_str = f" ({', '.join(channels)})" if channels else ""
    # STRING provides undirected functional association — it cannot distinguish
    # aggravating from alleviating interactions, so sl_relevance stays "none".
    result["sl_relevance"] = SL_NONE
    result["summary"] = (
        f"STRING-DB functional association between {gene_a} and {gene_b}: "
        f"combined score = {score:.3f} ({strength} confidence){channel_str}. "
        f"Note: STRING scores reflect functional association strength, not "
        f"genetic interaction direction — use SGD or Costanzo for SL-specific evidence."
    )

    _log(TOOL, f"[evidence_type={result['evidence_type']}] {result['summary']}")
    _STRING_CACHE[cache_key] = result
    return result


# ===========================================================================
# Tool 3: CostanzoSGA / Condition-Specific SGA Local Dataset Query
# ===========================================================================
#
# Data sources:
#   - Costanzo et al. 2016 (Science): Standard-condition SGA (~23M pairs)
#     Download: https://boonelab.ccbr.utoronto.ca/supplement/costanzo2016/
#   - Costanzo et al. 2021 (Science): Condition-specific SGA (14 conditions
#     × ~30K representative gene pairs). NOTE: BLEO is NOT among the 14.
#     Download: https://boonelab.ccbr.utoronto.ca/condition_sga/supplement/
#     File S3 (132 MB xlsx) = raw interaction dataset across all conditions.
#
# Auto-download:
#   If data files are missing, the tool will attempt to download them
#   automatically to ~/.medea/cache/sga/ (condition-specific data, ~133 MB).
# ===========================================================================

from pathlib import Path

# Module-level dataset cache
_COSTANZO_DATA = None

# Costanzo 2021 condition-specific: indexed parquet path + available conditions
_CONDITION_SGA_PARQUET: Optional[str] = None  # path to indexed parquet
_CONDITION_SGA_AVAILABLE: set = set()          # conditions found in data

# Download URLs
_CONDITION_SGA_URL = (
    "https://boonelab.ccbr.utoronto.ca/condition_sga/static/files/"
    "Costanzo%20et%20al_Data%20File%20S3_Raw%20interaction%20dataset.xlsx"
)
_CONDITION_SGA_FILENAME = "costanzo_2021_condition_sga_S3.xlsx"

_SGA_CACHE_DIR = Path.home() / ".medea" / "cache" / "sga"


def _auto_download_condition_sga(dest_dir: Path) -> Optional[str]:
    """Download Costanzo 2021 condition-specific SGA dataset if not present."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    local_path = dest_dir / _CONDITION_SGA_FILENAME

    if local_path.exists():
        age_days = (time.time() - local_path.stat().st_mtime) / 86400
        if age_days < 365:  # Re-download annually at most
            _log("SGA_DOWNLOAD", f"Using cached {_CONDITION_SGA_FILENAME} ({age_days:.0f} days old)")
            return str(local_path)

    _log("SGA_DOWNLOAD",
         f"Downloading Costanzo 2021 condition-specific SGA dataset (~133 MB)... "
         f"Source: Costanzo et al. 2021, Science (DOI: 10.1126/science.abf8424)")

    for attempt in range(3):
        try:
            resp = requests.get(_CONDITION_SGA_URL, timeout=300, stream=True)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))

            # Atomic download: write to temp file, then rename
            fd, tmp_path = tempfile.mkstemp(dir=str(dest_dir), suffix=".xlsx.tmp")
            try:
                with os.fdopen(fd, "wb") as f:
                    downloaded = 0
                    for chunk in resp.iter_content(chunk_size=1024 * 256):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total and downloaded % (10 * 1024 * 1024) < 1024 * 256:
                            _log("SGA_DOWNLOAD", f"  {downloaded / 1e6:.0f}/{total / 1e6:.0f} MB")
                os.replace(tmp_path, str(local_path))
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

            _log("SGA_DOWNLOAD",
                 f"Downloaded {_CONDITION_SGA_FILENAME} ({local_path.stat().st_size / 1e6:.1f} MB)")
            return str(local_path)

        except requests.RequestException as e:
            if attempt < 2:
                wait = 5 * (attempt + 1)
                _log("SGA_DOWNLOAD", f"Download failed, retrying in {wait}s: {e}", "WARNING")
                time.sleep(wait)
            else:
                _log("SGA_DOWNLOAD", f"Download failed after 3 attempts: {e}", "ERROR")
                return None

    return None


_INDEXED_PARQUET_SUFFIX = ".indexed.parquet"


def _find_condition_sga_source(data_dir: str) -> Optional[str]:
    """Locate raw Costanzo 2021 source file (xlsx/tsv) on disk or auto-download."""
    candidates = [
        _CONDITION_SGA_FILENAME,
        "costanzo_2021_condition_sga.xlsx",
        "condition_sga_S3.xlsx",
        "kuzmin_2018_condition_gi.tsv.gz",
        "kuzmin_2018_condition_gi.tsv",
        "condition_specific_gi.tsv.gz",
        "condition_specific_gi.tsv",
    ]

    search_dirs = [data_dir] if data_dir and os.path.isdir(data_dir) else []
    search_dirs.append(str(_SGA_CACHE_DIR))

    for d in search_dirs:
        for c in candidates:
            fp = os.path.join(d, c)
            if os.path.exists(fp):
                return fp

    # Auto-download
    _log("CONDITION_SGA", "Dataset not found locally — attempting auto-download...")
    result = _auto_download_condition_sga(_SGA_CACHE_DIR)
    if result is None:
        _log("CONDITION_SGA",
             "Auto-download failed. Manually download from: "
             "https://boonelab.ccbr.utoronto.ca/condition_sga/supplement/", "ERROR")
    return result


def _ensure_indexed_parquet(data_dir: str) -> Optional[str]:
    """
    Ensure an indexed parquet exists for Costanzo 2021 condition-specific SGA.

    The indexed parquet has pre-computed columns (pair_min, pair_max,
    condition_upper) that enable pyarrow predicate pushdown — each query
    reads only matching rows from disk instead of loading the full dataset.

    Returns path to the indexed parquet, or None on failure.
    """
    global _CONDITION_SGA_PARQUET, _CONDITION_SGA_AVAILABLE

    if _CONDITION_SGA_PARQUET is not None:
        return _CONDITION_SGA_PARQUET

    import pandas as pd
    import pyarrow.parquet as pq

    filepath = _find_condition_sga_source(data_dir)
    if filepath is None:
        return None

    # Check for existing indexed parquet
    indexed_path = Path(filepath).with_suffix(_INDEXED_PARQUET_SUFFIX)
    # Also check in cache dir (source might be user-placed, index in cache)
    indexed_cache = _SGA_CACHE_DIR / (
        Path(filepath).stem + _INDEXED_PARQUET_SUFFIX)

    for candidate in (indexed_path, indexed_cache):
        if candidate.exists() and candidate.stat().st_mtime >= os.path.getmtime(filepath):
            _log("CONDITION_SGA", f"Using indexed parquet: {candidate.name}")
            # Read available conditions from parquet schema metadata (instant)
            pf = pq.ParquetFile(str(candidate))
            meta = pf.schema_arrow.metadata or {}
            conds_json = meta.get(b"medea_conditions")
            if conds_json:
                _CONDITION_SGA_AVAILABLE = set(json.loads(conds_json))
            else:
                # Fallback: scan the condition column
                cond_table = pf.read(columns=["condition_upper"])
                _CONDITION_SGA_AVAILABLE = set(
                    cond_table.column("condition_upper").to_pylist())
            _CONDITION_SGA_PARQUET = str(candidate)
            _log("CONDITION_SGA",
                 f"Available conditions: {sorted(_CONDITION_SGA_AVAILABLE)}")
            return _CONDITION_SGA_PARQUET

    # --- Need to build the indexed parquet from source ---
    _log("CONDITION_SGA", f"Building indexed parquet from {filepath} (one-time cost)...")

    # Load raw data
    plain_parquet = Path(filepath).with_suffix(".parquet")
    if plain_parquet.exists() and plain_parquet.stat().st_mtime >= os.path.getmtime(filepath):
        _log("CONDITION_SGA", f"Reading plain parquet cache: {plain_parquet.name}")
        df = pd.read_parquet(plain_parquet)
    elif filepath.endswith((".xlsx", ".xls")):
        xls = pd.ExcelFile(filepath)
        sheet_name = None
        for name in xls.sheet_names:
            nl = name.lower()
            if any(kw in nl for kw in ("diagnostic", "interaction", "raw", "data", "gi")):
                sheet_name = name
                break
        if sheet_name is None:
            sheet_name = xls.sheet_names[0]

        _log("CONDITION_SGA",
             f"Reading sheet '{sheet_name}' from xlsx (one-time, ~3 min)...")
        df = pd.read_excel(filepath, sheet_name=sheet_name)
    else:
        sep = "\t" if ".tsv" in filepath else ","
        df = pd.read_csv(filepath, sep=sep, low_memory=False)

    _log("CONDITION_SGA", f"Loaded {len(df)} rows, columns: {list(df.columns)[:10]}")

    # --- Detect columns (same flexible logic as before) ---
    gene_a_col = gene_b_col = cond_col = None

    for col in df.columns:
        cl = col.lower().strip()
        if gene_a_col is None and any(kw in cl for kw in
                ("query_common", "query allele name", "query_allele", "query_gene")):
            gene_a_col = col
        elif gene_b_col is None and any(kw in cl for kw in
                ("array_common", "array allele name", "array_allele", "array_gene")):
            gene_b_col = col

    if gene_a_col is None or gene_b_col is None:
        for col in df.columns:
            cl = col.lower().strip()
            if gene_a_col is None and any(kw in cl for kw in
                    ("query orf", "query_orf", "gene_a", "query")):
                gene_a_col = col
            elif gene_b_col is None and any(kw in cl for kw in
                    ("array orf", "array_orf", "gene_b", "array")):
                gene_b_col = col

    for col in df.columns:
        cl = col.lower().strip()
        if cond_col is None and any(kw in cl for kw in
                ("condition", "treatment", "drug", "environment", "medium")):
            cond_col = col

    if gene_a_col is None or gene_b_col is None:
        _log("CONDITION_SGA",
             f"Could not identify gene columns. Found columns: {list(df.columns)}",
             "ERROR")
        return None

    _log("CONDITION_SGA",
         f"Detected columns: gene_a={gene_a_col}, gene_b={gene_b_col}, "
         f"condition={cond_col}")

    # --- Add pre-computed index columns for predicate pushdown ---
    ga = df[gene_a_col].astype(str).str.strip().str.upper()
    gb = df[gene_b_col].astype(str).str.strip().str.upper()
    if cond_col:
        cu = df[cond_col].fillna("STANDARD").astype(str).str.strip().str.upper()
    else:
        cu = pd.Series("STANDARD", index=df.index)

    df["pair_min"] = ga.where(ga <= gb, gb)
    df["pair_max"] = gb.where(ga <= gb, ga)
    df["condition_upper"] = cu

    # Sort by index columns for row group locality
    df = df.sort_values(["pair_min", "pair_max", "condition_upper"])

    # Save atomically — embed available conditions in parquet metadata
    # so we can read them back without scanning the condition column.
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(df, preserve_index=False)
    available_conds = sorted(set(cu.unique()))
    metadata = table.schema.metadata or {}
    metadata[b"medea_conditions"] = json.dumps(available_conds).encode()
    table = table.replace_schema_metadata(metadata)

    dest = indexed_cache
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_pq = tempfile.mkstemp(dir=str(dest.parent), suffix=".parquet.tmp")
    try:
        os.close(fd)
        pq.write_table(table, tmp_pq)
        os.replace(tmp_pq, str(dest))
        _log("CONDITION_SGA",
             f"Saved indexed parquet ({dest.stat().st_size / 1e6:.1f} MB)")
    except Exception as e:
        try:
            os.unlink(tmp_pq)
        except OSError:
            pass
        _log("CONDITION_SGA", f"Could not save indexed parquet: {e}", "ERROR")
        return None

    _CONDITION_SGA_AVAILABLE = set(cu.unique())
    _CONDITION_SGA_PARQUET = str(dest)
    _log("CONDITION_SGA",
         f"Available conditions: {sorted(_CONDITION_SGA_AVAILABLE)}")
    return _CONDITION_SGA_PARQUET


def _query_condition_parquet(
    parquet_path: str, pair_min: str, pair_max: str,
    condition_variants: List[str],
) -> List[dict]:
    """Query the indexed parquet using pyarrow predicate pushdown.

    Reads only rows matching the gene pair + condition from disk.
    Typically <100ms per query on SSD — no full-dataset load needed.
    """
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    # Build filter: pair_min == X AND pair_max == Y AND condition_upper IN [variants]
    pair_filter = (
        (ds.field("pair_min") == pair_min) &
        (ds.field("pair_max") == pair_max) &
        ds.field("condition_upper").isin(condition_variants)
    )

    table = pq.read_table(parquet_path, filters=pair_filter)

    if table.num_rows == 0:
        return []

    return table.to_pandas().to_dict("records")


def _load_costanzo_dataset(data_dir: str) -> dict:
    """Load Costanzo et al. 2016 standard-condition SGA dataset into a lookup dict."""
    global _COSTANZO_DATA
    if _COSTANZO_DATA is not None:
        return _COSTANZO_DATA

    import pandas as pd

    _log("COSTANZO_SGA", f"Loading Costanzo 2016 SGA dataset from {data_dir}")

    # Try multiple possible file names
    candidates = [
        "costanzo_2016_sga.tsv.gz",
        "costanzo_2016_sga.tsv",
        "SGA_ExE_NxN.tsv.gz",
        "SGA_ExE_NxN.tsv",
        "SGA_NxN.txt.gz",
        "SGA_NxN.txt",
        "costanzo_sga.tsv.gz",
        "costanzo_sga.tsv",
    ]

    # Also search the auto-download cache dir
    search_dirs = [data_dir] if data_dir and os.path.isdir(data_dir) else []
    search_dirs.append(str(_SGA_CACHE_DIR))

    filepath = None
    for d in search_dirs:
        for c in candidates:
            fp = os.path.join(d, c)
            if os.path.exists(fp):
                filepath = fp
                break
        if filepath:
            break

    if filepath is None:
        _log("COSTANZO_SGA", f"No standard-condition SGA dataset found. "
             f"Download from: https://boonelab.ccbr.utoronto.ca/supplement/costanzo2016/ "
             f"(Data File S1, ~430 MB). Place in {data_dir} or {_SGA_CACHE_DIR}", "WARNING")
        return {}

    try:
        sep = "\t" if (".tsv" in filepath or ".txt" in filepath) else ","
        df = pd.read_csv(filepath, sep=sep, low_memory=False)
        _log("COSTANZO_SGA", f"Loaded {len(df)} rows from {os.path.basename(filepath)}")

        # Build lookup: (gene_a, gene_b) -> row dict
        lookup = {}
        gene_a_col = None
        gene_b_col = None
        for col in df.columns:
            cl = col.lower().strip()
            if gene_a_col is None and any(kw in cl for kw in
                    ("query allele name", "query_allele", "query_gene", "query orf",
                     "gene_a", "query")):
                gene_a_col = col
            elif gene_b_col is None and any(kw in cl for kw in
                    ("array allele name", "array_allele", "array_gene", "array orf",
                     "gene_b", "array")):
                gene_b_col = col

        if gene_a_col is None or gene_b_col is None:
            _log("COSTANZO_SGA",
                 f"Could not identify gene columns. Columns: {list(df.columns)[:10]}",
                 "ERROR")
            return {}

        # Vectorized indexing (much faster than iterrows for ~23M rows)
        _log("COSTANZO_SGA", "Building lookup index (vectorized)...")
        ga_vec = df[gene_a_col].astype(str).str.strip().str.upper()
        gb_vec = df[gene_b_col].astype(str).str.strip().str.upper()
        pair_min = ga_vec.where(ga_vec <= gb_vec, gb_vec)
        pair_max = gb_vec.where(ga_vec <= gb_vec, ga_vec)

        groups = df.groupby([pair_min, pair_max], sort=False)
        for (g1, g2), grp in groups:
            lookup[(g1, g2)] = grp.to_dict("records")

        _COSTANZO_DATA = lookup
        _log("COSTANZO_SGA", f"Indexed {len(lookup)} unique gene pairs")
        return lookup

    except Exception as e:
        _log("COSTANZO_SGA", f"Failed to load dataset: {e}", "ERROR")
        return {}


def get_condition_sga_available() -> set:
    """Return the set of available conditions in Costanzo 2021 SGA dataset.

    Initializes the indexed parquet if needed (one-time cost ~4s).
    Called by condition_availability_checker in id_checkers.py.
    """
    if _CONDITION_SGA_AVAILABLE:
        return set(_CONDITION_SGA_AVAILABLE)

    # Trigger initialization
    try:
        from .env_utils import get_medeadb_path
        data_dir = os.path.join(get_medeadb_path(), "sga")
    except Exception:
        data_dir = os.path.join(os.environ.get("MEDEADB_PATH", ""), "sga")

    _ensure_indexed_parquet(data_dir)
    return set(_CONDITION_SGA_AVAILABLE)


def query_costanzo_sga_dataset(
    gene_a: str,
    gene_b: str,
    condition: str = "standard",
    data_dir: str = None,
) -> dict:
    """
    SPECIES: Yeast (Saccharomyces cerevisiae).
    Query the Costanzo SGA genetic interaction datasets:

    1. Costanzo et al. 2016 (Science): Standard-condition SGA with ~23 million
       gene pairs scored for genetic interactions. Gold-standard dataset.
    2. Costanzo et al. 2021 (Science): Condition-specific SGA across 14
       conditions (bleomycin, MMS, hydroxyurea, etc.) × ~30K gene pairs.

    The condition-specific dataset (133 MB) will be auto-downloaded if missing.
    The standard dataset (430 MB) must be manually placed due to size.

    Parameters
    ----------
    gene_a : str
        First yeast gene standard name or systematic ORF name.
    gene_b : str
        Second yeast gene standard name or systematic ORF name.
    condition : str
        Growth condition (default "standard"). Use "BLEO" or "bleomycin" for
        bleomycin treatment, "MMS" for methyl methanesulfonate, etc.
    data_dir : str, optional
        Path to SGA dataset directory. If None, uses $MEDEADB_PATH/sga/.

    Returns
    -------
    dict
        {
            "success": bool,
            "gene_a": str,
            "gene_b": str,
            "condition": str,
            "num_records": int,
            "gi_score": float or None,       # Genetic interaction score (epsilon)
            "gi_p_value": float or None,
            "gi_type": str or None,          # "negative" / "positive" / "neutral"
            "records": [...],
            "summary": str,
            "error": str or None,
            "dataset_available": bool,
        }
    """
    TOOL = "COSTANZO_SGA"
    gene_a = gene_a.strip().upper()
    gene_b = gene_b.strip().upper()
    condition = condition.strip().upper()
    _log(TOOL, f"Querying SGA dataset: {gene_a} × {gene_b} (condition={condition})")

    if data_dir is None:
        try:
            from .env_utils import get_medeadb_path
            data_dir = os.path.join(get_medeadb_path(), "sga")
        except Exception:
            data_dir = os.path.join(os.environ.get("MEDEADB_PATH", ""), "sga")

    result = {
        "success": False, "gene_a": gene_a, "gene_b": gene_b,
        "condition": condition, "num_records": 0,
        "gi_score": None, "gi_p_value": None, "gi_type": None,
        "records": [], "summary": "", "error": None,
        "dataset_available": False,
        "evidence_type": EVIDENCE_ERROR,
        "sl_relevance": SL_NONE,
        "coverage": {
            "database": "Costanzo SGA datasets (2016 standard + 2021 condition-specific)",
            "condition_requested": condition,
            "condition_available": None,       # set after loading data
            "available_conditions": [],        # populated from loaded data
            "dataset_searched": [],            # which datasets were actually queried
        },
    }

    pair_key = tuple(sorted([gene_a, gene_b]))

    # --- Helper to extract GI score and p-value from records ---
    def _extract_scores(records: list) -> None:
        for rec in records:
            # Costanzo 2021 uses mean_condition_epsilon / condition_p_value
            # Costanzo 2016 uses "Genetic interaction score (epsilon)" / "P-value"
            for col_name in ("mean_condition_epsilon", "Genetic interaction score",
                             "Genetic interaction score (epsilon)", "GI_score",
                             "epsilon", "gi_score",
                             "Double mutant fitness - Expected double mutant fitness"):
                val = rec.get(col_name)
                if val is not None and result["gi_score"] is None:
                    try:
                        import math
                        v = float(val)
                        if not math.isnan(v):
                            result["gi_score"] = v
                            break
                    except (ValueError, TypeError):
                        pass
            for col_name in ("condition_p_value", "P-value", "p_value",
                             "p-value", "pvalue"):
                val = rec.get(col_name)
                if val is not None and result["gi_p_value"] is None:
                    try:
                        import math
                        v = float(val)
                        if not math.isnan(v):
                            result["gi_p_value"] = v
                            break
                    except (ValueError, TypeError):
                        pass

        if result["gi_score"] is not None:
            if result["gi_score"] < -0.08:
                result["gi_type"] = "negative"
            elif result["gi_score"] > 0.08:
                result["gi_type"] = "positive"
            else:
                result["gi_type"] = "neutral"

    # --- Try condition-specific dataset first ---
    if condition not in ("STANDARD", ""):
        parquet_path = _ensure_indexed_parquet(data_dir)
        if parquet_path:
            result["dataset_available"] = True
            result["coverage"]["dataset_searched"].append("Costanzo 2021 condition-specific")
            result["coverage"]["available_conditions"] = sorted(_CONDITION_SGA_AVAILABLE)

            # Normalise condition name variants
            cond_variants = [condition]
            cond_map = {
                "BLEO": ["BLEO", "BLEOMYCIN", "BLE"],
                "BLEOMYCIN": ["BLEO", "BLEOMYCIN", "BLE"],
                "MMS": ["MMS"],
                "HU": ["HU", "HYDROXYUREA"],
                "HYDROXYUREA": ["HU", "HYDROXYUREA"],
                "ACTINOMYCIND": ["ACTINOMYCIND", "ACTINOMYCIN_D", "ACTINOMYCIN D"],
                "BENOMYL": ["BENOMYL"],
                "BORTEZOMIB": ["BORTEZOMIB"],
                "CASPOFUNGIN": ["CASPOFUNGIN"],
                "CONCANAMYCINA": ["CONCANAMYCINA", "CONCANAMYCIN_A", "CONCANAMYCIN A"],
                "CYCLOHEXIMIDE": ["CYCLOHEXIMIDE", "CHX"],
                "FLUCONAZOLE": ["FLUCONAZOLE"],
                "GALACTOSE": ["GALACTOSE", "GAL"],
                "GELDENAMYCIN": ["GELDENAMYCIN", "GELDANAMYCIN"],
                "MONENSIN": ["MONENSIN"],
                "RAPAMYCIN": ["RAPAMYCIN", "RAP"],
                "SORBITOL": ["SORBITOL"],
                "TUNICAMYCIN": ["TUNICAMYCIN"],
            }
            cond_variants = cond_map.get(condition, [condition])

            # Check if the requested condition is in the dataset at all
            condition_covered = any(
                cv in _CONDITION_SGA_AVAILABLE for cv in cond_variants
            )
            result["coverage"]["condition_available"] = condition_covered

            if not condition_covered:
                # CRITICAL DISTINCTION: this is NOT a negative finding —
                # the tool simply doesn't cover this condition.
                _log(TOOL,
                     f"UNINFORMATIVE: '{condition}' is NOT among the conditions "
                     f"in Costanzo 2021. Available: {sorted(_CONDITION_SGA_AVAILABLE)}. "
                     f"Cannot assess {gene_a} × {gene_b} under {condition} from this dataset.",
                     "WARNING")
                # Don't return yet — fall through to try standard-condition data
            else:
                # PyArrow predicate pushdown: read only matching rows (~100ms)
                p_min, p_max = pair_key
                records = _query_condition_parquet(
                    parquet_path, p_min, p_max, cond_variants)

                if records:
                    result["success"] = True
                    result["evidence_type"] = EVIDENCE_HIT
                    result["records"] = records
                    result["num_records"] = len(records)
                    _extract_scores(records)

                    # Determine sl_relevance from gi_type
                    if result["gi_type"] == "negative":
                        result["sl_relevance"] = SL_AGGRAVATING
                    elif result["gi_type"] == "positive":
                        result["sl_relevance"] = SL_ALLEVIATING
                    else:
                        result["sl_relevance"] = SL_NONE

                    result["summary"] = (
                        f"Costanzo 2021 condition-specific SGA: {gene_a} × {gene_b} "
                        f"under {condition}: GI score = {result['gi_score']}, "
                        f"p-value = {result['gi_p_value']}, type = {result['gi_type']}. "
                        f"[sl_relevance={result['sl_relevance']}]"
                    )
                    _log(TOOL, f"[evidence_type={result['evidence_type']}] {result['summary']}")
                    return result
                else:
                    # Condition IS covered but pair not found → meaningful negative
                    result["evidence_type"] = EVIDENCE_NO_HIT
                    result["sl_relevance"] = SL_NONE
                    result["summary"] = (
                        f"NO HIT: {gene_a} × {gene_b} was NOT found in "
                        f"Costanzo 2021 SGA under {condition}. This condition IS in the "
                        f"dataset (~30K pairs tested), so absence means the pair was "
                        f"either not tested or showed no significant interaction."
                    )
                    _log(TOOL, f"[evidence_type={result['evidence_type']}] {result['summary']}")
                    return result

    # --- Fall back to standard-condition Costanzo 2016 dataset ---
    costanzo = _load_costanzo_dataset(data_dir)
    if costanzo:
        result["dataset_available"] = True
        result["coverage"]["dataset_searched"].append("Costanzo 2016 standard-condition")
        if pair_key in costanzo:
            records = costanzo[pair_key]
            result["success"] = True
            result["records"] = records
            result["num_records"] = len(records)
            _extract_scores(records)

            result["evidence_type"] = EVIDENCE_HIT

            # Determine sl_relevance from gi_type
            if result["gi_type"] == "negative":
                result["sl_relevance"] = SL_AGGRAVATING
            elif result["gi_type"] == "positive":
                result["sl_relevance"] = SL_ALLEVIATING
            else:
                result["sl_relevance"] = SL_NONE

            # If we got here after a condition-specific miss, note the fallback
            cond_note = ""
            if condition not in ("STANDARD", "") and not result["coverage"].get("condition_available"):
                cond_note = (
                    f" Note: {condition}-specific data NOT available in Costanzo 2021 "
                    f"(available: {', '.join(result['coverage'].get('available_conditions', []))}). "
                    f"This result is from STANDARD conditions only — it cannot confirm "
                    f"or deny a {condition}-specific genetic interaction."
                )
                result["coverage"]["condition_match"] = "standard_fallback"
            else:
                result["coverage"]["condition_match"] = "exact"

            result["summary"] = (
                f"Costanzo 2016 SGA (standard conditions): {gene_a} × {gene_b}: "
                f"GI score = {result['gi_score']}, type = {result['gi_type']}. "
                f"[sl_relevance={result['sl_relevance']}]{cond_note}"
            )
            _log(TOOL, f"[evidence_type={result['evidence_type']}] {result['summary']}")
            return result

    # --- Check if we have any dataset at all ---
    if not result["dataset_available"]:
        result["evidence_type"] = EVIDENCE_ERROR
        if not os.path.isdir(data_dir) and not _SGA_CACHE_DIR.is_dir():
            result["error"] = (
                f"No SGA dataset directory found. The condition-specific dataset "
                f"(Costanzo 2021, ~133 MB) will be auto-downloaded on first use. "
                f"For the larger standard dataset (Costanzo 2016, ~430 MB), "
                f"manually download from: "
                f"https://boonelab.ccbr.utoronto.ca/supplement/costanzo2016/"
            )
        else:
            result["error"] = (
                f"SGA dataset loaded but no records for {gene_a} × {gene_b}."
            )
        _log(TOOL, result.get("error", ""), "WARNING")
        return result

    # Nothing found in either dataset
    result["success"] = True
    # Determine evidence type based on whether condition was covered
    if condition not in ("STANDARD", "") and not result["coverage"].get("condition_available"):
        result["evidence_type"] = EVIDENCE_UNINFORMATIVE
        result["summary"] = (
            f"UNINFORMATIVE for {condition}: {gene_a} × {gene_b} not found, AND "
            f"'{condition}' is NOT a condition in the Costanzo 2021 dataset. "
            f"Available conditions: {', '.join(result['coverage'].get('available_conditions', []))}. "
            f"Standard-condition (Costanzo 2016) data also not found for this pair. "
            f"This tool CANNOT provide evidence about {gene_a} × {gene_b} under {condition}."
        )
    else:
        result["evidence_type"] = EVIDENCE_NO_HIT
        result["sl_relevance"] = SL_NONE
        result["summary"] = (
            f"NO HIT: No SGA records for {gene_a} × {gene_b} "
            f"in Costanzo datasets (condition={condition}). "
            f"The pair was not tested or showed no significant interaction."
        )
    _log(TOOL, f"[evidence_type={result['evidence_type']}] {result['summary']}")
    return result


# ===========================================================================
# Tool 4: SGD GO Functional Annotations
# ===========================================================================

_GO_CACHE: Dict[str, dict] = {}


def query_yeast_go_annotations(
    gene: str,
) -> dict:
    """
    SPECIES: Yeast (Saccharomyces cerevisiae).
    Query the Saccharomyces Genome Database (SGD) for Gene Ontology (GO)
    annotations of a yeast gene. Returns molecular function, biological
    process, and cellular component annotations with evidence codes and
    references.

    Useful for understanding what a gene does, where its product localizes,
    and what biological processes it participates in. When querying a gene
    pair, call this tool for each gene separately to compare their functional
    profiles.

    Parameters
    ----------
    gene : str
        Yeast gene standard name or systematic ORF name (e.g. "SGS1",
        "YMR190C").

    Returns
    -------
    dict
        {
            "success": bool,
            "gene": str,
            "molecular_function": [{"term": str, "go_id": str, "evidence": str, "qualifier": str, "reference": str}],
            "biological_process": [{"term": str, "go_id": str, "evidence": str, "qualifier": str, "reference": str}],
            "cellular_component": [{"term": str, "go_id": str, "evidence": str, "qualifier": str, "reference": str}],
            "num_annotations": int,
            "summary": str,
            "error": str or None,
        }
    """
    TOOL = "SGD_GO"
    gene = gene.strip().upper()
    _log(TOOL, f"Querying GO annotations: {gene}")

    if gene in _GO_CACHE:
        _log(TOOL, "Cache hit")
        return _GO_CACHE[gene]

    result = {
        "success": False, "gene": gene,
        "molecular_function": [], "biological_process": [],
        "cellular_component": [],
        "num_annotations": 0, "summary": "", "error": None,
    }

    url = f"{SGD_API_BASE}/locus/{gene}/go_details"
    resp = _safe_request(url, timeout=SGD_API_TIMEOUT, tool=TOOL)

    if resp is None:
        result["error"] = f"SGD API unreachable for {gene}"
        _log(TOOL, result["error"], "ERROR")
        return result

    try:
        data = resp.json()
    except (json.JSONDecodeError, ValueError) as e:
        result["error"] = f"Failed to parse SGD GO response: {e}"
        _log(TOOL, result["error"], "ERROR")
        return result

    if not data:
        result["success"] = True
        result["summary"] = f"No GO annotations found for {gene} in SGD."
        _log(TOOL, result["summary"])
        _GO_CACHE[gene] = result
        return result

    # Parse annotations by GO aspect
    aspect_map = {
        "molecular function": "molecular_function",
        "biological process": "biological_process",
        "cellular component": "cellular_component",
    }

    for rec in data:
        go_obj = rec.get("go", {})
        aspect = go_obj.get("go_aspect", "")
        field = aspect_map.get(aspect)
        if not field:
            continue

        term = go_obj.get("display_name", "")
        go_id = go_obj.get("go_id", "")
        evidence = rec.get("experiment", {}).get("display_name", "")
        qualifier = rec.get("qualifier", "")
        ref_obj = rec.get("reference", {})
        reference = ref_obj.get("display_name", "") if ref_obj else ""
        annotation_type = rec.get("annotation_type", "")

        result[field].append({
            "term": term, "go_id": go_id, "evidence": evidence,
            "qualifier": qualifier, "reference": reference,
            "annotation_type": annotation_type,
        })

    result["success"] = True
    n_mf = len(result["molecular_function"])
    n_bp = len(result["biological_process"])
    n_cc = len(result["cellular_component"])
    result["num_annotations"] = n_mf + n_bp + n_cc

    # Build concise summary — deduplicate terms (same term can have multiple evidence codes)
    def _unique_terms(annotations):
        seen = set()
        terms = []
        for a in annotations:
            if a["term"] not in seen:
                seen.add(a["term"])
                terms.append(a["term"])
        return terms

    mf_terms = _unique_terms(result["molecular_function"])
    bp_terms = _unique_terms(result["biological_process"])
    cc_terms = _unique_terms(result["cellular_component"])

    parts = [f"{gene} GO annotations ({result['num_annotations']} total):"]
    if mf_terms:
        parts.append(f"Molecular Function ({len(mf_terms)}): {', '.join(mf_terms[:8])}"
                      + (f" (+{len(mf_terms)-8} more)" if len(mf_terms) > 8 else ""))
    if bp_terms:
        parts.append(f"Biological Process ({len(bp_terms)}): {', '.join(bp_terms[:8])}"
                      + (f" (+{len(bp_terms)-8} more)" if len(bp_terms) > 8 else ""))
    if cc_terms:
        parts.append(f"Cellular Component ({len(cc_terms)}): {', '.join(cc_terms[:5])}"
                      + (f" (+{len(cc_terms)-5} more)" if len(cc_terms) > 5 else ""))
    result["summary"] = " | ".join(parts)

    _log(TOOL, f"{gene}: {n_mf} MF, {n_bp} BP, {n_cc} CC annotations")
    _GO_CACHE[gene] = result
    return result


# ===========================================================================
# Backward-compatible alias (old name → new name)
# ===========================================================================
query_sgd_genetic_interactions = query_sgd_interactions
