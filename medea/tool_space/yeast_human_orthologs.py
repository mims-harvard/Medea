"""
Yeast-Human Gene Ortholog Mapping Tools

Provides tools to map genes between S. cerevisiae (budding yeast) and Homo sapiens
using two complementary data sources:

1. SGD (Saccharomyces Genome Database) - Direct yeast→human ortholog mappings
   via Alliance/DIOPT data, plus experimentally validated functional complementation.
   API: https://www.yeastgenome.org/backend/locus/{sgd_id}/homolog_details

2. PomBase - Triangulated ortholog mappings via S. pombe as a bridge species.
   Bulk TSV files: https://pombase.org/data/orthologs/
   Path: cerevisiae → pombe → human (validates & fills gaps in direct SGD mappings)

Tools:
    - YeastHumanOrthologMapper: Given yeast gene(s), find human orthologs
    - HumanYeastOrthologMapper: Given human gene(s), find yeast orthologs
    - YeastHumanComplementFinder: Find experimentally validated functional complementation pairs
    - YeastGeneInfoRetriever: Get comprehensive yeast gene info from SGD
"""

import os
import csv
import json
import random
import time
import logging
import tempfile
import requests
import urllib.parse
from io import StringIO
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Any, Set
from collections import defaultdict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Base cache directory for all yeast-human tool data
MEDEA_CACHE_BASE = Path.home() / ".medea" / "cache"

# SGD API
SGD_API_BASE = "https://www.yeastgenome.org/backend"
SGD_FEATURES_URL = "https://downloads.yeastgenome.org/curation/chromosomal_feature/SGD_features.tab"
SGD_REQUEST_DELAY = 0.15      # seconds between API requests (politeness)
SGD_REQUEST_TIMEOUT = 15      # seconds per request
SGD_MAX_RETRIES = 5
SGD_API_CHECK_TIMEOUT = 5     # seconds for quick API availability check

# PomBase bulk ortholog TSV URLs
POMBASE_CEREVISIAE_URL = "https://pombase.org/data/orthologs/pombe-cerevisiae-orthologs.tsv"
POMBASE_HUMAN_URL = "https://pombase.org/data/orthologs/pombe-human-orthologs.tsv"

# MyGene.info (human gene validation / HGNC resolution)
MYGENE_API_TIMEOUT = 10

# Cache TTL (days)
CACHE_TTL_DAYS = 30           # how long to keep cached data before re-downloading
DOWNLOAD_TIMEOUT = 60         # seconds for bulk file downloads


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class OrthologMapping:
    """A single ortholog mapping between two genes."""
    source_gene: str            # Source gene symbol
    source_systematic: str      # Source systematic name (e.g., YER095W)
    source_organism: str        # e.g., "Saccharomyces cerevisiae"
    target_gene: str            # Target gene symbol
    target_id: str              # Target gene ID (e.g., HGNC ID, SGD ID)
    target_organism: str        # e.g., "Homo sapiens"
    data_source: str            # "SGD", "PomBase_direct", "PomBase_triangulated"
    prediction_methods: List[str] = field(default_factory=list)
    num_methods: int = 0        # Number of prediction methods supporting this
    is_best_score: bool = False # Whether this is the best-scoring ortholog
    confidence: str = "unknown" # "high", "medium", "low"
    pmid_references: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ComplementationResult:
    """An experimentally validated functional complementation pair."""
    yeast_gene: str
    yeast_systematic: str
    human_gene: str
    human_gene_id: str
    direction: str              # "yeast_complements_human" or "human_complements_yeast"
    strain_background: str
    details: str
    reference_pmid: str
    source: str                 # "SGD" or "P-POD"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class YeastGeneInfo:
    """Comprehensive information about a yeast gene."""
    gene_name: str
    systematic_name: str
    sgd_id: str
    description: str
    gene_type: str              # e.g., "ORF", "tRNA", etc.
    qualifier: str              # e.g., "Verified", "Uncharacterized"
    go_annotations: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    url: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# SGD Gene Features Cache (Bulk TSV Download — replaces per-gene API calls)
# =============================================================================

class SGDFeatureCache:
    """
    Downloads and caches the SGD_features.tab bulk file for fast local gene lookups.
    Eliminates the need for per-gene SGD API calls, avoiding rate limiting / IP bans.
    File is ~3 MB, refreshed every 30 days.
    """
    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = MEDEA_CACHE_BASE / "sgd"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._genes: Dict[str, Dict] = {}
        self._loaded = False
        self._logger = logging.getLogger("SGD_FEATURE_CACHE")

    def _log(self, msg, level="INFO"):
        print(f"[SGD_FEATURE_CACHE] {level}: {msg}", flush=True)

    def _download(self, max_retries: int = 3) -> str:
        local_path = self.cache_dir / "SGD_features.tab"
        if local_path.exists():
            age_days = (time.time() - local_path.stat().st_mtime) / 86400
            if age_days < CACHE_TTL_DAYS:
                self._log(f"Using cached SGD_features.tab ({age_days:.0f} days old)")
                return str(local_path)

        self._log("Downloading SGD_features.tab (~3 MB)...")
        for attempt in range(max_retries):
            try:
                resp = requests.get(SGD_FEATURES_URL, timeout=DOWNLOAD_TIMEOUT)
                resp.raise_for_status()
                # Atomic write: temp file + rename (safe for concurrent processes)
                fd, tmp = tempfile.mkstemp(dir=str(local_path.parent), suffix=".tmp")
                try:
                    with os.fdopen(fd, "wb") as f:
                        f.write(resp.content)
                    os.replace(tmp, str(local_path))
                except BaseException:
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass
                    raise
                self._log(f"Downloaded SGD_features.tab ({len(resp.content)} bytes)")
                return str(local_path)
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    self._log(f"Download failed, retrying in {wait}s: {e}", "WARNING")
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"Failed to download SGD_features.tab after {max_retries} attempts: {e}")

    def _load(self):
        if self._loaded:
            return
        path = self._download()
        with open(path, 'r') as f:
            for line in f:
                parts = line.rstrip('\n').split('\t')
                if len(parts) < 16:
                    continue
                sgd_id = parts[0]
                feature_type = parts[1]
                qualifier = parts[2] if len(parts) > 2 else ""
                systematic_name = parts[3] if len(parts) > 3 else ""
                gene_name = parts[4] if len(parts) > 4 else ""
                aliases = parts[5] if len(parts) > 5 else ""
                description = parts[15] if len(parts) > 15 else ""

                if feature_type not in ("ORF", "tRNA", "rRNA", "snoRNA", "snRNA", "ncRNA"):
                    continue

                key_name = (gene_name or systematic_name).upper()
                key_sys = systematic_name.upper()

                entry = {
                    "sgd_id": sgd_id,
                    "gene_name": gene_name or systematic_name,
                    "systematic_name": systematic_name,
                    "description": description,
                    "gene_type": feature_type,
                    "qualifier": qualifier,
                    "aliases": [a.strip() for a in aliases.split("|")] if aliases else [],
                    "go_annotations": [],
                    "url": f"https://www.yeastgenome.org/locus/{sgd_id}",
                }
                if key_name:
                    self._genes[key_name] = entry
                if key_sys and key_sys != key_name:
                    self._genes[key_sys] = entry

        self._log(f"Loaded {len(self._genes)} gene entries from SGD_features.tab")
        self._loaded = True

    def get_gene_info(self, gene_name: str) -> Optional[Dict]:
        self._load()
        return self._genes.get(gene_name.strip().upper())


_sgd_feature_cache: Optional[SGDFeatureCache] = None

def _get_sgd_feature_cache() -> SGDFeatureCache:
    global _sgd_feature_cache
    if _sgd_feature_cache is None:
        _sgd_feature_cache = SGDFeatureCache()
    return _sgd_feature_cache


# =============================================================================
# PomBase Ortholog Cache (Bulk TSV Download + In-Memory Lookup)
# =============================================================================

class PomBaseOrthologCache:
    """
    Downloads and caches PomBase bulk ortholog TSV files for fast lookups.
    
    Provides two mapping tables:
    - pombe ↔ cerevisiae
    - pombe ↔ human
    
    Combined, these enable triangulated cerevisiae→pombe→human mapping.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = MEDEA_CACHE_BASE / "pombase"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory lookup tables
        self._pombe_to_cerevisiae: Dict[str, List[Dict]] = {}  # pombe_id -> [{cerevisiae_systematic, cerevisiae_name, ...}]
        self._cerevisiae_to_pombe: Dict[str, List[Dict]] = {}  # cerevisiae_systematic -> [{pombe_id, pombe_name, ...}]
        self._pombe_to_human: Dict[str, List[Dict]] = {}       # pombe_id -> [{human_gene, human_id, ...}]
        self._human_to_pombe: Dict[str, List[Dict]] = {}       # human_gene_upper -> [{pombe_id, pombe_name, ...}]
        
        # Gene name indexes for flexible lookups
        self._cerevisiae_name_to_systematic: Dict[str, str] = {}  # RAD51 -> YER095W
        
        self._loaded = False
    
    def _download_file(self, url: str, filename: str, max_retries: int = 3) -> str:
        """Download a file with retry logic, return local path."""
        local_path = self.cache_dir / filename
        
        # Check if cached file exists and is less than 30 days old
        if local_path.exists():
            age_days = (time.time() - local_path.stat().st_mtime) / 86400
            if age_days < CACHE_TTL_DAYS:
                self._log(f"Using cached {filename} ({age_days:.0f} days old)")
                return str(local_path)
        
        self._log(f"Downloading {filename} from PomBase...")
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=DOWNLOAD_TIMEOUT)
                response.raise_for_status()
                # Atomic write: temp file + rename (safe for concurrent processes)
                fd, tmp = tempfile.mkstemp(dir=str(local_path.parent), suffix=".tmp")
                try:
                    with os.fdopen(fd, "wb") as f:
                        f.write(response.text.encode("utf-8"))
                    os.replace(tmp, str(local_path))
                except BaseException:
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass
                    raise
                self._log(f"Downloaded {filename} ({len(response.text)} bytes)")
                return str(local_path)
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    self._log(f"Download failed, retrying in {wait}s: {e}", "WARNING")
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"Failed to download {url} after {max_retries} attempts: {e}")
        
        return str(local_path)
    
    def _parse_tsv(self, filepath: str) -> List[List[str]]:
        """Parse a PomBase TSV file, skipping comment/header lines."""
        rows = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                rows.append(line.split('\t'))
        return rows
    
    def load(self) -> None:
        """Download (if needed) and parse PomBase ortholog files into memory."""
        if self._loaded:
            return
        
        try:
            # Download files
            cer_path = self._download_file(POMBASE_CEREVISIAE_URL, "pombe-cerevisiae-orthologs.tsv")
            human_path = self._download_file(POMBASE_HUMAN_URL, "pombe-human-orthologs.tsv")
            
            # Parse pombe-cerevisiae
            # Format: 2-column TSV: pombe_systematic_id \t cerevisiae_systematic_name
            # e.g.: SPAC1782.04\tYLR204W
            cer_rows = self._parse_tsv(cer_path)
            for row in cer_rows:
                if len(row) < 2:
                    continue
                pombe_id = row[0].strip()
                cer_systematic = row[1].strip()
                
                if not pombe_id or not cer_systematic:
                    continue
                
                entry = {
                    "pombe_id": pombe_id,
                    "pombe_name": "",
                    "cerevisiae_systematic": cer_systematic,
                    "cerevisiae_name": "",  # Not in bulk file; resolved later via SGD if needed
                    "pmids": []
                }
                
                self._pombe_to_cerevisiae.setdefault(pombe_id, []).append(entry)
                self._cerevisiae_to_pombe.setdefault(cer_systematic.upper(), []).append(entry)
            
            # Parse pombe-human
            # Format: 2-column TSV: pombe_systematic_id \t human_hgnc_id
            # e.g.: SPAC1039.07c\tHGNC:14412
            human_rows = self._parse_tsv(human_path)
            for row in human_rows:
                if len(row) < 2:
                    continue
                pombe_id = row[0].strip()
                human_id = row[1].strip()  # e.g., "HGNC:14412"
                
                if not pombe_id or not human_id:
                    continue
                
                # Extract gene name from HGNC ID if possible (store the ID for now)
                entry = {
                    "pombe_id": pombe_id,
                    "pombe_name": "",
                    "human_gene": human_id,  # Will be HGNC ID like "HGNC:14412"
                    "human_id": human_id,
                    "pmids": []
                }
                
                self._pombe_to_human.setdefault(pombe_id, []).append(entry)
                # Index by HGNC ID (uppercase) for reverse lookups
                self._human_to_pombe.setdefault(human_id.upper(), []).append(entry)
            
            self._loaded = True
            self._log(
                f"Loaded PomBase orthologs: "
                f"{len(self._cerevisiae_to_pombe)} cerevisiae genes, "
                f"{len(self._human_to_pombe)} human genes"
            )
            
        except Exception as e:
            self._log(f"Failed to load PomBase data: {e}", "ERROR")
            raise
    
    def get_cerevisiae_systematic(self, gene_name: str) -> Optional[str]:
        """Resolve a cerevisiae gene name to its systematic name."""
        upper = gene_name.upper()
        # Already a systematic name?
        if upper in self._cerevisiae_to_pombe:
            return upper
        # Standard name?
        return self._cerevisiae_name_to_systematic.get(upper)
    
    def find_human_orthologs_via_pombe(self, cerevisiae_gene: str) -> List[OrthologMapping]:
        """
        Triangulated lookup: cerevisiae → pombe → human.
        Returns list of OrthologMapping with data_source="PomBase_triangulated".
        """
        self.load()
        
        # Resolve to systematic name
        systematic = self.get_cerevisiae_systematic(cerevisiae_gene)
        if not systematic:
            systematic = cerevisiae_gene.upper()
        
        # Step 1: cerevisiae → pombe
        pombe_entries = self._cerevisiae_to_pombe.get(systematic, [])
        if not pombe_entries:
            return []
        
        results = []
        seen_human = set()
        
        for pombe_entry in pombe_entries:
            pombe_id = pombe_entry["pombe_id"]
            
            # Step 2: pombe → human
            human_entries = self._pombe_to_human.get(pombe_id, [])
            for human_entry in human_entries:
                human_gene = human_entry["human_gene"]
                if human_gene.upper() in seen_human:
                    continue
                seen_human.add(human_gene.upper())
                
                # Combine PMIDs from both links
                combined_pmids = list(set(
                    pombe_entry.get("pmids", []) + human_entry.get("pmids", [])
                ))
                
                results.append(OrthologMapping(
                    source_gene=pombe_entry.get("cerevisiae_name", cerevisiae_gene),
                    source_systematic=systematic,
                    source_organism="Saccharomyces cerevisiae",
                    target_gene=human_gene,
                    target_id=human_entry.get("human_id", ""),
                    target_organism="Homo sapiens",
                    data_source="PomBase_triangulated",
                    prediction_methods=["PomBase_curated_orthology"],
                    num_methods=1,
                    confidence="medium",  # Triangulated = medium confidence by default
                    pmid_references=combined_pmids,
                    is_best_score=False
                ))
        
        return results
    
    def find_yeast_orthologs_via_pombe(self, human_gene: str) -> List[OrthologMapping]:
        """
        Reverse triangulated lookup: human → pombe → cerevisiae.
        Accepts both gene symbols (e.g., "BRCA1") and HGNC IDs (e.g., "HGNC:1100").
        """
        self.load()
        
        # Step 1: human → pombe
        # Try direct lookup (works for HGNC IDs like "HGNC:1100")
        pombe_entries = self._human_to_pombe.get(human_gene.upper(), [])
        
        # If not found and looks like a gene symbol, try HGNC ID format
        if not pombe_entries and not human_gene.upper().startswith("HGNC:"):
            # Try all HGNC entries — this is a linear scan but the data is small
            # We could optimize with a reverse symbol→HGNC index, but PomBase
            # doesn't include symbols in its bulk file, so we skip for now
            pass
        if not pombe_entries:
            return []
        
        results = []
        seen_yeast = set()
        
        for pombe_entry in pombe_entries:
            pombe_id = pombe_entry["pombe_id"]
            
            # Step 2: pombe → cerevisiae
            cer_entries = self._pombe_to_cerevisiae.get(pombe_id, [])
            for cer_entry in cer_entries:
                cer_systematic = cer_entry["cerevisiae_systematic"]
                if cer_systematic.upper() in seen_yeast:
                    continue
                seen_yeast.add(cer_systematic.upper())
                
                combined_pmids = list(set(
                    pombe_entry.get("pmids", []) + cer_entry.get("pmids", [])
                ))
                
                results.append(OrthologMapping(
                    source_gene=human_gene,
                    source_systematic="",
                    source_organism="Homo sapiens",
                    target_gene=cer_entry.get("cerevisiae_name", cer_systematic),
                    target_id=cer_systematic,
                    target_organism="Saccharomyces cerevisiae",
                    data_source="PomBase_triangulated",
                    prediction_methods=["PomBase_curated_orthology"],
                    num_methods=1,
                    confidence="medium",
                    pmid_references=combined_pmids,
                    is_best_score=False
                ))
        
        return results
    
    @staticmethod
    def _log(message: str, level: str = "INFO"):
        prefix = "[POMBASE_ORTHOLOGS]"
        print(f"{prefix} {level}: {message}", flush=True)


# =============================================================================
# SGD API Client
# =============================================================================

class SGDClient:
    """
    Client for the SGD (Saccharomyces Genome Database) REST API.
    
    Endpoints used:
    - /backend/locus/{id}                    → Basic gene info
    - /backend/locus/{id}/homolog_details    → Non-fungal homologs (Alliance/DIOPT)
    - /backend/locus/{id}/complement_details → Functional complementation data
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "Medea/1.0 (yeast-human-ortholog-tools)"
        })
        self._sgd_id_cache: Dict[str, str] = {}  # gene_name -> SGD ID
        self._request_delay = SGD_REQUEST_DELAY
        self._max_retries = SGD_MAX_RETRIES
    
    def _request(self, url: str, params: Optional[Dict] = None) -> Optional[Any]:
        """Make a rate-limited request with retries."""
        for attempt in range(self._max_retries):
            try:
                time.sleep(self._request_delay + random.uniform(0, 0.5))
                response = self.session.get(url, params=params, timeout=SGD_REQUEST_TIMEOUT)
                
                if response.status_code == 404:
                    self._log(f"Resource not found: {url}", "WARNING")
                    return None
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code in (429, 403):
                    wait = (2 ** attempt) * 3 + random.uniform(0, 2)
                    self._log(f"Rate limited ({e.response.status_code}), waiting {wait:.1f}s... (attempt {attempt+1}/{self._max_retries})", "WARNING")
                    time.sleep(wait)
                elif attempt == self._max_retries - 1:
                    self._log(f"Request failed after {self._max_retries} attempts: {e}", "ERROR")
                    return None
                else:
                    time.sleep(2 ** attempt + random.uniform(0, 1))
            except requests.RequestException as e:
                if attempt == self._max_retries - 1:
                    self._log(f"Request failed: {e}", "ERROR")
                    return None
                time.sleep(2 ** attempt + random.uniform(0, 1))
        
        return None
    
    def resolve_gene_to_sgd_id(self, gene_name: str) -> Optional[str]:
        """
        Resolve a yeast gene name (standard or systematic) to its SGD ID.
        Uses the SGD search API to find the locus.
        """
        if gene_name in self._sgd_id_cache:
            return self._sgd_id_cache[gene_name]
        
        # Try direct locus lookup first (works with standard names like RAD51)
        url = f"{SGD_API_BASE}/locus/{urllib.parse.quote(gene_name)}"
        data = self._request(url)
        
        if data and isinstance(data, dict):
            sgd_id = data.get("sgdid")
            if sgd_id:
                self._sgd_id_cache[gene_name] = sgd_id
                self._log(f"Resolved {gene_name} → {sgd_id}")
                return sgd_id
        
        self._log(f"Could not resolve gene '{gene_name}' to SGD ID", "WARNING")
        return None
    
    def get_gene_info(self, gene_name: str) -> Optional[YeastGeneInfo]:
        """Get comprehensive gene information from SGD."""
        url = f"{SGD_API_BASE}/locus/{urllib.parse.quote(gene_name)}"
        data = self._request(url)
        
        if not data or not isinstance(data, dict):
            return None
        
        return YeastGeneInfo(
            gene_name=data.get("display_name", gene_name),
            systematic_name=data.get("format_name", ""),
            sgd_id=data.get("sgdid", ""),
            description=data.get("headline", "") or data.get("description", ""),
            gene_type=data.get("locus_type", ""),
            qualifier=data.get("qualifier", ""),
            aliases=[a.get("display_name", "") for a in data.get("aliases", [])],
            url=f"https://www.yeastgenome.org/locus/{data.get('sgdid', gene_name)}"
        )
    
    def get_human_orthologs(self, gene_name: str) -> List[OrthologMapping]:
        """
        Get human orthologs for a yeast gene from SGD (Alliance/DIOPT data).
        """
        # First resolve to get SGD ID and gene info
        url = f"{SGD_API_BASE}/locus/{urllib.parse.quote(gene_name)}"
        locus_data = self._request(url)
        
        if not locus_data or not isinstance(locus_data, dict):
            self._log(f"Gene '{gene_name}' not found in SGD", "WARNING")
            return []
        
        sgd_id = locus_data.get("sgdid", "")
        display_name = locus_data.get("display_name", gene_name)
        systematic_name = locus_data.get("format_name", "")
        
        # Fetch homolog details
        homolog_url = f"{SGD_API_BASE}/locus/{sgd_id}/homolog_details"
        homolog_data = self._request(homolog_url)
        
        if not homolog_data or not isinstance(homolog_data, list):
            self._log(f"No homolog data returned for {gene_name}")
            return []
        
        results = []
        for entry in homolog_data:
            # Filter for human orthologs only
            # SGD API returns species as a plain string (e.g., "Homo sapiens")
            species_raw = entry.get("species", "")
            if isinstance(species_raw, dict):
                species_name = species_raw.get("display_name", "")
            else:
                species_name = str(species_raw)
            
            if "sapiens" not in species_name.lower() and "human" not in species_name.lower():
                continue
            
            # SGD API returns flat fields: gene_name, gene_id, source, link_url
            target_gene = entry.get("gene_name", "")
            target_id = entry.get("gene_id", "")
            link_url = entry.get("link_url", "")
            source = entry.get("source", "")
            description = entry.get("description", "")
            
            # The Alliance/DIOPT source doesn't include per-entry method counts
            # in the SGD API, so we mark it as Alliance-sourced
            method_names = [source] if source else ["Alliance"]
            num_methods = 1  # SGD API doesn't expose per-entry method count
            
            results.append(OrthologMapping(
                source_gene=display_name,
                source_systematic=systematic_name,
                source_organism="Saccharomyces cerevisiae",
                target_gene=target_gene,
                target_id=target_id,
                target_organism="Homo sapiens",
                data_source="SGD",
                prediction_methods=method_names,
                num_methods=num_methods,
                is_best_score=True,  # SGD only returns curated best matches
                confidence="high",   # SGD Alliance orthologs are curated
                pmid_references=[]
            ))
        
        if results:
            self._log(f"Found {len(results)} human ortholog(s) for {gene_name} via SGD")
        else:
            self._log(f"No human orthologs found for {gene_name} in SGD")
        
        return results
    
    def get_functional_complementation(self, gene_name: str) -> List[ComplementationResult]:
        """
        Get experimentally validated functional complementation data.
        These are cases where a yeast gene can replace a human gene (or vice versa).
        """
        url = f"{SGD_API_BASE}/locus/{urllib.parse.quote(gene_name)}"
        locus_data = self._request(url)
        
        if not locus_data or not isinstance(locus_data, dict):
            return []
        
        sgd_id = locus_data.get("sgdid", "")
        display_name = locus_data.get("display_name", gene_name)
        systematic_name = locus_data.get("format_name", "")
        
        complement_url = f"{SGD_API_BASE}/locus/{sgd_id}/complement_details"
        complement_data = self._request(complement_url)
        
        if not complement_data or not isinstance(complement_data, list):
            return []
        
        results = []
        for entry in complement_data:
            # Filter for human complementation
            # SGD complement API returns species as a plain string
            species_raw = entry.get("species", "")
            if isinstance(species_raw, dict):
                species_name = species_raw.get("display_name", "")
            else:
                species_name = str(species_raw)
            
            if "sapiens" not in species_name.lower() and "human" not in species_name.lower():
                continue
            
            # Determine direction
            # SGD uses phrases like "other complements yeast" or "yeast complements other"
            direction_raw = entry.get("direction", "")
            direction_str = str(direction_raw).lower()
            if "other complements yeast" in direction_str:
                direction = "human_complements_yeast"
            elif "yeast complements other" in direction_str:
                direction = "yeast_complements_human"
            else:
                direction = str(direction_raw)
            
            # Extract references (list of dicts with pubmed_id)
            refs = entry.get("references", [])
            pmids = []
            if isinstance(refs, list):
                for ref in refs:
                    if isinstance(ref, dict):
                        pid = ref.get("pubmed_id", "")
                        if pid:
                            pmids.append(str(pid))
            # Fallback to single "reference" field
            if not pmids:
                ref = entry.get("reference", {})
                if isinstance(ref, dict):
                    pid = ref.get("pubmed_id", "") or ref.get("pmid", "")
                    if pid:
                        pmids.append(str(pid))
            
            # Human gene info: SGD complement API uses flat fields
            human_gene_name = entry.get("gene_name", "")
            human_gene_id = entry.get("dbxref_id", "") or entry.get("gene_id", "")
            
            # Source info
            source_raw = entry.get("source", "SGD")
            if isinstance(source_raw, dict):
                source_name = source_raw.get("display_name", "SGD")
            else:
                source_name = str(source_raw) if source_raw else "SGD"
            
            results.append(ComplementationResult(
                yeast_gene=display_name,
                yeast_systematic=systematic_name,
                human_gene=human_gene_name,
                human_gene_id=human_gene_id,
                direction=direction,
                strain_background=entry.get("strain_background", ""),
                details=entry.get("curator_comment", "") or entry.get("details", "") or entry.get("note", ""),
                reference_pmid=",".join(pmids) if pmids else "",
                source=source_name
            ))
        
        if results:
            self._log(f"Found {len(results)} complementation record(s) for {gene_name}")
        
        return results
    
    @staticmethod
    def _score_confidence(num_methods: int, is_best: bool) -> str:
        """Score confidence based on number of prediction methods and best-score flag."""
        if num_methods >= 8 or (num_methods >= 5 and is_best):
            return "high"
        elif num_methods >= 3:
            return "medium"
        elif num_methods >= 1:
            return "low"
        return "unknown"
    
    @staticmethod
    def _log(message: str, level: str = "INFO"):
        prefix = "[SGD_ORTHOLOGS]"
        print(f"{prefix} {level}: {message}", flush=True)


# =============================================================================
# Alliance Ortholog Cache (Local Bulk TSV — Primary Data Source)
# =============================================================================

class AllianceOrthologCache:
    """
    Loads the Alliance ORTHOLOGY-ALLIANCE_COMBINED.tsv(.gz) bulk file for
    fast local yeast↔human ortholog lookups.  This file contains multi-algorithm
    ortholog predictions curated by the Alliance of Genome Resources and is the
    most comprehensive single source for cross-species mapping.

    Expected location: ~/.medea/cache/alliance/ORTHOLOGY-ALLIANCE_COMBINED.tsv.gz
    (auto-downloaded by the existing SGD ortholog pipeline).
    """

    _YEAST_TAXON = "NCBITaxon:559292"
    _HUMAN_TAXON = "NCBITaxon:9606"

    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = MEDEA_CACHE_BASE / "alliance"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._yeast_to_human: Dict[str, List[Dict]] = {}   # YEAST_SYMBOL_UPPER -> [records]
        self._human_to_yeast: Dict[str, List[Dict]] = {}   # HUMAN_SYMBOL_UPPER -> [records]
        self._loaded = False

    def load(self) -> bool:
        """Parse the Alliance TSV into in-memory lookup tables. Returns True on success."""
        if self._loaded:
            return True

        import gzip as _gzip

        candidates = [
            self.cache_dir / "ORTHOLOGY-ALLIANCE_COMBINED.tsv.gz",
            self.cache_dir / "ORTHOLOGY-ALLIANCE_COMBINED.tsv",
        ]
        filepath = None
        for c in candidates:
            if c.exists():
                filepath = c
                break

        if filepath is None:
            self._log("Alliance orthology file not found — skipping Alliance source", "WARNING")
            return False

        open_fn = _gzip.open if str(filepath).endswith(".gz") else open
        try:
            with open_fn(filepath, "rt", encoding="utf-8") as fh:
                header_seen = False
                for line in fh:
                    if line.startswith("#"):
                        continue
                    if not header_seen:
                        header_seen = True
                        continue  # skip column header row

                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 12:
                        continue

                    g1_sym, g1_tax = parts[1], parts[2]
                    g2_sym, g2_tax = parts[5], parts[6]
                    algorithms = parts[8] if len(parts) > 8 else ""
                    n_match = int(parts[9]) if len(parts) > 9 and parts[9].isdigit() else 0
                    n_total = int(parts[10]) if len(parts) > 10 and parts[10].isdigit() else 0
                    is_best = parts[11].strip().lower() == "yes" if len(parts) > 11 else False

                    yeast_sym = human_sym = None
                    if self._YEAST_TAXON in g1_tax and self._HUMAN_TAXON in g2_tax:
                        yeast_sym, human_sym = g1_sym, g2_sym
                    elif self._YEAST_TAXON in g2_tax and self._HUMAN_TAXON in g1_tax:
                        yeast_sym, human_sym = g2_sym, g1_sym

                    if yeast_sym and human_sym:
                        record = {
                            "yeast_gene": yeast_sym,
                            "human_gene": human_sym,
                            "algorithms": algorithms,
                            "num_methods": n_match,
                            "num_total": n_total,
                            "is_best_score": is_best,
                        }
                        self._yeast_to_human.setdefault(yeast_sym.upper(), []).append(record)
                        self._human_to_yeast.setdefault(human_sym.upper(), []).append(record)

            self._loaded = True
            self._log(
                f"Loaded Alliance orthologs: "
                f"{len(self._yeast_to_human)} yeast genes, "
                f"{len(self._human_to_yeast)} human genes"
            )
            return True
        except Exception as e:
            self._log(f"Failed to load Alliance file: {e}", "ERROR")
            return False

    def find_human_orthologs(self, yeast_gene: str) -> List[OrthologMapping]:
        """Return human orthologs for *yeast_gene* from Alliance data."""
        if not self.load():
            return []

        sgd_cache = _get_sgd_feature_cache()
        gene_info = sgd_cache.get_gene_info(yeast_gene)
        systematic = gene_info["systematic_name"] if gene_info else ""

        records = self._yeast_to_human.get(yeast_gene.upper(), [])
        if not records and systematic:
            records = self._yeast_to_human.get(systematic.upper(), [])

        results = []
        seen = set()
        for rec in records:
            hg = rec["human_gene"].upper()
            if hg in seen:
                continue
            seen.add(hg)

            n_methods = rec["num_methods"]
            is_best = rec["is_best_score"]
            if n_methods >= 8 or (n_methods >= 5 and is_best):
                conf = "high"
            elif n_methods >= 3:
                conf = "medium"
            else:
                conf = "low"

            results.append(OrthologMapping(
                source_gene=yeast_gene,
                source_systematic=systematic,
                source_organism="Saccharomyces cerevisiae",
                target_gene=rec["human_gene"],
                target_id="",
                target_organism="Homo sapiens",
                data_source="Alliance",
                prediction_methods=rec["algorithms"].split("|") if rec["algorithms"] else [],
                num_methods=n_methods,
                is_best_score=is_best,
                confidence=conf,
                pmid_references=[],
            ))

        if results:
            self._log(f"Found {len(results)} human ortholog(s) for {yeast_gene} via Alliance")
        return results

    def find_yeast_orthologs(self, human_gene: str) -> List[OrthologMapping]:
        """Return yeast orthologs for *human_gene* from Alliance data."""
        if not self.load():
            return []

        records = self._human_to_yeast.get(human_gene.upper(), [])
        results = []
        seen = set()
        for rec in records:
            yg = rec["yeast_gene"].upper()
            if yg in seen:
                continue
            seen.add(yg)

            n_methods = rec["num_methods"]
            is_best = rec["is_best_score"]
            if n_methods >= 8 or (n_methods >= 5 and is_best):
                conf = "high"
            elif n_methods >= 3:
                conf = "medium"
            else:
                conf = "low"

            results.append(OrthologMapping(
                source_gene=human_gene,
                source_systematic="",
                source_organism="Homo sapiens",
                target_gene=rec["yeast_gene"],
                target_id="",
                target_organism="Saccharomyces cerevisiae",
                data_source="Alliance",
                prediction_methods=rec["algorithms"].split("|") if rec["algorithms"] else [],
                num_methods=n_methods,
                is_best_score=is_best,
                confidence=conf,
                pmid_references=[],
            ))
        return results

    @staticmethod
    def _log(message: str, level: str = "INFO"):
        print(f"[ALLIANCE_ORTHOLOGS] {level}: {message}", flush=True)


# =============================================================================
# Integrated Ortholog Mapper (Combines Alliance + SGD + PomBase)
# =============================================================================

class IntegratedOrthologMapper:
    """
    Combines Alliance local orthologs (primary), SGD API orthologs,
    and PomBase triangulated orthologs for comprehensive yeast↔human mapping.
    """
    
    def __init__(self, pombase_cache_dir: Optional[str] = None):
        self.alliance = AllianceOrthologCache()
        self.sgd = SGDClient()
        self.pombase = PomBaseOrthologCache(cache_dir=pombase_cache_dir)
        self._hgnc_cache: Dict[str, str] = {}  # gene_symbol -> HGNC ID
    
    def _resolve_human_gene_to_hgnc(self, gene_symbol: str) -> Optional[str]:
        """Resolve a human gene symbol to its HGNC ID using MyGene.info."""
        if gene_symbol in self._hgnc_cache:
            return self._hgnc_cache[gene_symbol]
        
        if gene_symbol.upper().startswith("HGNC:"):
            return gene_symbol.upper()
        
        try:
            encoded = urllib.parse.quote(gene_symbol)
            url = f"https://mygene.info/v3/query?q={encoded}&fields=HGNC&species=human"
            response = requests.get(url, timeout=MYGENE_API_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            hits = data.get("hits", [])
            if hits:
                hgnc_raw = hits[0].get("HGNC", "")
                if hgnc_raw:
                    hgnc_id = f"HGNC:{hgnc_raw}" if not str(hgnc_raw).startswith("HGNC:") else str(hgnc_raw)
                    self._hgnc_cache[gene_symbol] = hgnc_id
                    self._log(f"Resolved {gene_symbol} → {hgnc_id}")
                    return hgnc_id
        except Exception as e:
            self._log(f"HGNC resolution failed for {gene_symbol}: {e}", "WARNING")
        
        return None
    
    def _sgd_api_available(self) -> bool:
        """Quick check if SGD API is reachable (cached for 10 minutes)."""
        now = time.time()
        if hasattr(self, '_sgd_api_ok') and now - self._sgd_api_check_time < 600:
            return self._sgd_api_ok
        try:
            resp = self.sgd.session.get(
                f"{SGD_API_BASE}/locus/S000003097",
                timeout=SGD_API_CHECK_TIMEOUT
            )
            ok = resp.status_code == 200
        except Exception:
            ok = False
        self._sgd_api_ok = ok
        self._sgd_api_check_time = now
        if not ok:
            self._log("SGD API unavailable (403/timeout). Using PomBase-only mode.", "WARNING")
        return ok

    def yeast_to_human(self, yeast_gene: str, include_pombase: bool = True) -> Dict[str, Any]:
        """
        Find all human orthologs for a yeast gene.
        Priority: Alliance local cache (fast, comprehensive) → SGD API → PomBase triangulation.
        
        Args:
            yeast_gene: Yeast gene name (standard or systematic, e.g. "RAD51" or "YER095W")
            include_pombase: Whether to include PomBase triangulated results (default: True)
        
        Returns:
            Dict with keys: yeast_gene, yeast_info, orthologs, complementation, summary
        """
        self._log(f"Looking up human orthologs for yeast gene: {yeast_gene}")
        
        use_sgd_api = self._sgd_api_available()

        # 1. Get gene info — prefer bulk cache, fallback to API
        gene_info = None
        try:
            bulk_cache = _get_sgd_feature_cache()
            bulk_info = bulk_cache.get_gene_info(yeast_gene)
            if bulk_info:
                gene_info = type('GeneInfo', (), {
                    'to_dict': lambda self=bulk_info: bulk_info,
                    'url': bulk_info.get('url', ''),
                    **{k: bulk_info[k] for k in bulk_info}
                })()
        except Exception:
            pass
        if gene_info is None and use_sgd_api:
            gene_info = self.sgd.get_gene_info(yeast_gene)
        
        # 2. Alliance local cache (primary — fast, no network, most comprehensive)
        alliance_orthologs = self.alliance.find_human_orthologs(yeast_gene)
        
        # 3. SGD API orthologs (supplement — skip if API unavailable)
        sgd_orthologs = []
        if use_sgd_api:
            sgd_orthologs = self.sgd.get_human_orthologs(yeast_gene)
        
        # 4. PomBase triangulated orthologs (fallback)
        pombase_orthologs = []
        if include_pombase:
            try:
                pombase_orthologs = self.pombase.find_human_orthologs_via_pombe(yeast_gene)
            except Exception as e:
                self._log(f"PomBase lookup failed (non-fatal): {e}", "WARNING")
        
        # 5. Functional complementation (skip if API unavailable)
        complementation = []
        if use_sgd_api:
            complementation = self.sgd.get_functional_complementation(yeast_gene)
        
        # 6. Merge: Alliance first, then SGD + PomBase to supplement
        merged = self._merge_orthologs(
            alliance_orthologs + sgd_orthologs, pombase_orthologs, complementation
        )
        
        # 7. Generate summary
        summary = self._generate_summary(yeast_gene, merged, complementation)
        if not use_sgd_api and not alliance_orthologs:
            summary += " (SGD API unavailable — results from PomBase only)"
        
        return {
            "success": True,
            "yeast_gene": yeast_gene,
            "yeast_info": gene_info.to_dict() if gene_info else None,
            "orthologs": [o.to_dict() for o in merged],
            "num_orthologs": len(merged),
            "complementation": [c.to_dict() for c in complementation],
            "num_complementation": len(complementation),
            "data_sources_used": self._list_sources(merged, complementation),
            "summary": summary
        }
    
    def human_to_yeast(self, human_gene: str, include_pombase: bool = True) -> Dict[str, Any]:
        """
        Find all yeast orthologs for a human gene.
        Priority: Alliance local cache → PomBase triangulation → SGD API validation.
        
        Args:
            human_gene: Human gene symbol (e.g. "TP53", "BRCA1")
            include_pombase: Whether to include PomBase triangulated results (default: True)
        
        Returns:
            Dict with keys: human_gene, orthologs, summary
        """
        self._log(f"Looking up yeast orthologs for human gene: {human_gene}")

        # 1. Alliance local cache (primary — fast, no network)
        alliance_orthologs = self.alliance.find_yeast_orthologs(human_gene)

        # 2. PomBase reverse lookup as supplement
        pombase_orthologs = []
        if include_pombase:
            try:
                hgnc_id = self._resolve_human_gene_to_hgnc(human_gene)
                if hgnc_id:
                    pombase_orthologs = self.pombase.find_yeast_orthologs_via_pombe(hgnc_id)
                if not pombase_orthologs:
                    pombase_orthologs = self.pombase.find_yeast_orthologs_via_pombe(human_gene)
            except Exception as e:
                self._log(f"PomBase reverse lookup failed (non-fatal): {e}", "WARNING")

        # 3. Merge and deduplicate (Alliance first)
        all_orthologs = alliance_orthologs + pombase_orthologs
        seen = set()
        deduped = []
        for o in all_orthologs:
            key = (o.target_gene.upper() if o.target_gene else o.target_id.upper())
            if key not in seen:
                seen.add(key)
                deduped.append(o)
        
        summary = self._generate_reverse_summary(human_gene, deduped)
        
        return {
            "success": True,
            "human_gene": human_gene,
            "orthologs": [o.to_dict() for o in deduped],
            "num_orthologs": len(deduped),
            "data_sources_used": list(set(o.data_source for o in deduped)),
            "summary": summary
        }
    
    def batch_yeast_to_human(self, yeast_genes: List[str], include_pombase: bool = True) -> Dict[str, Any]:
        """
        Batch lookup: find human orthologs for multiple yeast genes.
        
        Args:
            yeast_genes: List of yeast gene names
            include_pombase: Whether to include PomBase triangulated results
        
        Returns:
            Dict with per-gene results and overall summary
        """
        self._log(f"Batch lookup: {len(yeast_genes)} yeast genes → human")
        
        results = {}
        total_orthologs = 0
        genes_with_orthologs = 0
        
        for gene in yeast_genes:
            try:
                result = self.yeast_to_human(gene, include_pombase=include_pombase)
                results[gene] = result
                if result.get("num_orthologs", 0) > 0:
                    genes_with_orthologs += 1
                    total_orthologs += result["num_orthologs"]
            except Exception as e:
                results[gene] = {"success": False, "error": str(e)}
        
        return {
            "success": True,
            "total_genes_queried": len(yeast_genes),
            "genes_with_human_orthologs": genes_with_orthologs,
            "total_ortholog_mappings": total_orthologs,
            "results": results,
            "summary": (
                f"Mapped {genes_with_orthologs}/{len(yeast_genes)} yeast genes to "
                f"{total_orthologs} human orthologs."
            )
        }
    
    def _merge_orthologs(
        self,
        sgd_orthologs: List[OrthologMapping],
        pombase_orthologs: List[OrthologMapping],
        complementation: List[ComplementationResult]
    ) -> List[OrthologMapping]:
        """Merge orthologs from different sources, upgrading confidence for cross-validated ones."""
        by_gene: Dict[str, OrthologMapping] = {}
        # Also index by HGNC ID for cross-referencing PomBase (which uses HGNC IDs)
        hgnc_to_key: Dict[str, str] = {}
        
        # Add SGD orthologs first (higher priority)
        for orth in sgd_orthologs:
            key = orth.target_gene.upper()
            by_gene[key] = orth
            # Map HGNC ID → gene key for PomBase cross-reference
            if orth.target_id and orth.target_id.upper().startswith("HGNC:"):
                hgnc_to_key[orth.target_id.upper()] = key
        
        # Merge PomBase orthologs (which use HGNC IDs as identifiers)
        for orth in pombase_orthologs:
            pombase_id = orth.target_id.upper() if orth.target_id else ""
            pombase_gene = orth.target_gene.upper() if orth.target_gene else ""
            
            # Try to match by HGNC ID first, then by gene name
            matched_key = None
            if pombase_id in hgnc_to_key:
                matched_key = hgnc_to_key[pombase_id]
            elif pombase_gene in by_gene:
                matched_key = pombase_gene
            
            if matched_key:
                # Cross-validated! Upgrade confidence
                existing = by_gene[matched_key]
                existing.confidence = self._upgrade_confidence(existing.confidence)
                existing.data_source = "SGD+PomBase_triangulated"
                existing.pmid_references = list(set(
                    existing.pmid_references + orth.pmid_references
                ))
            else:
                # New ortholog from PomBase only
                key = pombase_id or pombase_gene
                if key:
                    by_gene[key] = orth
        
        # Mark complementation-validated orthologs
        complement_genes = {c.human_gene.upper() for c in complementation if c.human_gene}
        complement_hgnc = {c.human_gene_id.upper() for c in complementation if c.human_gene_id}
        for key, orth in by_gene.items():
            if key in complement_genes or orth.target_id.upper() in complement_hgnc:
                if orth.confidence != "high":
                    orth.confidence = self._upgrade_confidence(orth.confidence)
                orth.prediction_methods = list(set(
                    orth.prediction_methods + ["functional_complementation"]
                ))
        
        # Sort: high confidence first, then by number of methods
        result = sorted(
            by_gene.values(),
            key=lambda o: (
                {"high": 3, "medium": 2, "low": 1, "unknown": 0}.get(o.confidence, 0),
                o.num_methods
            ),
            reverse=True
        )
        
        return result
    
    @staticmethod
    def _upgrade_confidence(current: str) -> str:
        upgrades = {"unknown": "low", "low": "medium", "medium": "high", "high": "high"}
        return upgrades.get(current, "medium")
    
    @staticmethod
    def _list_sources(orthologs: List[OrthologMapping], complementation: List[ComplementationResult]) -> List[str]:
        sources = set()
        for o in orthologs:
            sources.add(o.data_source)
        if complementation:
            sources.add("SGD_complementation")
        return sorted(sources)
    
    @staticmethod
    def _generate_summary(yeast_gene: str, orthologs: List[OrthologMapping], complementation: List[ComplementationResult]) -> str:
        if not orthologs:
            return f"No human orthologs found for yeast gene {yeast_gene}."
        
        high_conf = [o for o in orthologs if o.confidence == "high"]
        med_conf = [o for o in orthologs if o.confidence == "medium"]
        
        parts = [f"Found {len(orthologs)} human ortholog(s) for yeast gene {yeast_gene}."]
        
        if high_conf:
            genes = ", ".join(o.target_gene for o in high_conf[:5])
            parts.append(f"High confidence: {genes}.")
        
        if med_conf:
            genes = ", ".join(o.target_gene for o in med_conf[:5])
            parts.append(f"Medium confidence: {genes}.")
        
        # Note cross-validated results
        cross_validated = [o for o in orthologs if "+" in o.data_source]
        if cross_validated:
            parts.append(
                f"{len(cross_validated)} ortholog(s) cross-validated by both SGD and PomBase triangulation."
            )
        
        if complementation:
            parts.append(
                f"{len(complementation)} experimentally validated functional complementation record(s) found."
            )
        
        return " ".join(parts)
    
    @staticmethod
    def _generate_reverse_summary(human_gene: str, orthologs: List[OrthologMapping]) -> str:
        if not orthologs:
            return f"No yeast orthologs found for human gene {human_gene}."
        
        yeast_genes = [o.target_gene or o.target_id for o in orthologs]
        return (
            f"Found {len(orthologs)} yeast ortholog(s) for human gene {human_gene}: "
            f"{', '.join(yeast_genes[:10])}."
        )
    
    @staticmethod
    def _log(message: str, level: str = "INFO"):
        prefix = "[ORTHOLOG_MAPPER]"
        print(f"{prefix} {level}: {message}", flush=True)


# =============================================================================
# Convenience Functions (Agent-Friendly API)
# =============================================================================

# Module-level mapper instance (lazy init)
_mapper: Optional[IntegratedOrthologMapper] = None


def _get_mapper() -> IntegratedOrthologMapper:
    """Get or create the shared mapper instance."""
    global _mapper
    if _mapper is None:
        _mapper = IntegratedOrthologMapper()
    return _mapper


_ORTHOLOG_CACHE_DIR = MEDEA_CACHE_BASE / "sgd_orthologs"

def find_human_orthologs_for_yeast_gene(
    yeast_gene: str,
    include_pombase: bool = True
) -> Dict[str, Any]:
    """
    Find human orthologs for a yeast gene using SGD + PomBase data.
    Results are cached to disk (30 days) to avoid repeated API calls.
    
    Args:
        yeast_gene: Yeast gene name (standard or systematic, e.g. "RAD51" or "YER095W")
        include_pombase: Whether to include PomBase triangulated results (default: True)
    
    Returns:
        Dict with keys: success, yeast_gene, yeast_info, orthologs, complementation, summary
    """
    _ORTHOLOG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = f"{yeast_gene.strip().upper()}_pb{int(include_pombase)}"
    cache_file = _ORTHOLOG_CACHE_DIR / f"{cache_key}.json"

    if cache_file.exists():
        age_days = (time.time() - cache_file.stat().st_mtime) / 86400
        if age_days < CACHE_TTL_DAYS:
            try:
                with open(cache_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

    mapper = _get_mapper()
    try:
        result = mapper.yeast_to_human(yeast_gene, include_pombase=include_pombase)
    except Exception as e:
        return {"success": False, "error": str(e), "yeast_gene": yeast_gene}

    if result.get("success"):
        try:
            fd, tmp = tempfile.mkstemp(
                dir=str(Path(cache_file).parent), suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                json.dump(result, f, default=str)
            os.replace(tmp, cache_file)
        except OSError:
            try:
                os.unlink(tmp)
            except (OSError, UnboundLocalError):
                pass

    return result


def find_yeast_orthologs_for_human_gene(
    human_gene: str,
    include_pombase: bool = True
) -> Dict[str, Any]:
    """
    Find yeast (S. cerevisiae) orthologs for a human gene.
    
    Args:
        human_gene: Human gene symbol (e.g. "TP53", "BRCA1")
        include_pombase: Whether to include PomBase triangulated results (default: True)
    
    Returns:
        Dict with keys: success, human_gene, orthologs, summary
    
    Example:
        >>> result = find_yeast_orthologs_for_human_gene("BRCA1")
        >>> for orth in result["orthologs"]:
        ...     print(f"{orth['target_gene']} ({orth['confidence']})")
    """
    mapper = _get_mapper()
    try:
        return mapper.human_to_yeast(human_gene, include_pombase=include_pombase)
    except Exception as e:
        return {"success": False, "error": str(e), "human_gene": human_gene}


_COMPLEMENT_CACHE_DIR = MEDEA_CACHE_BASE / "sgd_complement"


def find_yeast_human_complementation(yeast_gene: str) -> Dict[str, Any]:
    """
    Find experimentally validated functional complementation pairs for a yeast gene.
    These represent cases where a yeast gene and human gene can functionally replace
    each other, providing the strongest evidence for functional conservation.
    Results are cached to disk (30 days) to avoid repeated API calls.

    Args:
        yeast_gene: Yeast gene name

    Returns:
        Dict with keys: success, yeast_gene, complementation_pairs, summary
    """
    _COMPLEMENT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = yeast_gene.strip().upper()
    cache_file = _COMPLEMENT_CACHE_DIR / f"{cache_key}.json"

    if cache_file.exists():
        age_days = (time.time() - cache_file.stat().st_mtime) / 86400
        if age_days < CACHE_TTL_DAYS:
            try:
                with open(cache_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

    mapper = _get_mapper()
    try:
        results = mapper.sgd.get_functional_complementation(yeast_gene)

        if results:
            summary = (
                f"Found {len(results)} functional complementation record(s) for {yeast_gene}. "
                + " ".join(
                    f"{r.human_gene} ({r.direction.replace('_', ' ')})"
                    for r in results[:5]
                )
            )
        else:
            summary = f"No functional complementation data found for {yeast_gene} in SGD."

        result = {
            "success": True,
            "yeast_gene": yeast_gene,
            "complementation_pairs": [r.to_dict() for r in results],
            "num_pairs": len(results),
            "summary": summary
        }

        try:
            fd, tmp = tempfile.mkstemp(
                dir=str(Path(cache_file).parent), suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                json.dump(result, f, default=str)
            os.replace(tmp, cache_file)
        except OSError:
            try:
                os.unlink(tmp)
            except (OSError, UnboundLocalError):
                pass

        return result
    except Exception as e:
        return {"success": False, "error": str(e), "yeast_gene": yeast_gene}


def get_yeast_gene_info(yeast_gene: str) -> Dict[str, Any]:
    """
    Get comprehensive information about a yeast gene from SGD.
    Tries the bulk SGD_features.tab cache first (fast, no rate limiting),
    then falls back to the SGD API if the gene isn't found locally.
    
    Args:
        yeast_gene: Yeast gene name (standard or systematic)
    
    Returns:
        Dict with keys: success, gene_info, url
    """
    # Try bulk cache first (no API call needed)
    try:
        cache = _get_sgd_feature_cache()
        info = cache.get_gene_info(yeast_gene)
        if info:
            return {
                "success": True,
                "gene_info": info,
                "url": info["url"]
            }
    except Exception:
        pass

    # Fallback to SGD API for genes not in bulk file
    mapper = _get_mapper()
    try:
        api_info = mapper.sgd.get_gene_info(yeast_gene)
        if api_info:
            return {
                "success": True,
                "gene_info": api_info.to_dict(),
                "url": api_info.url
            }
    except Exception as e:
        return {"success": False, "error": str(e), "yeast_gene": yeast_gene}

    return {
        "success": False,
        "error": f"Gene '{yeast_gene}' not found in SGD (bulk cache + API)",
        "yeast_gene": yeast_gene
    }


def batch_yeast_to_human_mapping(
    yeast_genes: List[str],
    include_pombase: bool = True
) -> Dict[str, Any]:
    """
    Batch mapping: find human orthologs for multiple yeast genes.
    
    Args:
        yeast_genes: List of yeast gene names
        include_pombase: Whether to include PomBase data
    
    Returns:
        Dict with per-gene results and overall summary statistics
    """
    mapper = _get_mapper()
    try:
        return mapper.batch_yeast_to_human(yeast_genes, include_pombase=include_pombase)
    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# CLI Entry Point for Testing
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("Yeast-Human Ortholog Mapping Tool - Test Suite")
    print("=" * 60)
    
    test_gene = sys.argv[1] if len(sys.argv) > 1 else "RAD51"
    
    print(f"\n--- Test 1: Yeast Gene Info for {test_gene} ---")
    info = get_yeast_gene_info(test_gene)
    print(json.dumps(info, indent=2, default=str))
    
    print(f"\n--- Test 2: Human Orthologs for {test_gene} ---")
    orthologs = find_human_orthologs_for_yeast_gene(test_gene)
    print(json.dumps(orthologs, indent=2, default=str))
    
    print(f"\n--- Test 3: Functional Complementation for {test_gene} ---")
    comp = find_yeast_human_complementation(test_gene)
    print(json.dumps(comp, indent=2, default=str))
    
    print(f"\n--- Test 4: Reverse Lookup (human→yeast) for BRCA1 ---")
    reverse = find_yeast_orthologs_for_human_gene("BRCA1")
    print(json.dumps(reverse, indent=2, default=str))
    
    print(f"\n--- Test 5: Batch Mapping ---")
    batch = batch_yeast_to_human_mapping(["RAD51", "CDC28", "TUB1"])
    print(f"Summary: {batch.get('summary', 'N/A')}")
