import numpy as np
import os
import h5py
from typing import Dict, Tuple, Optional, Union, List

from .env_utils import get_medeadb_path as _get_medeadb_path


class GeneCorrelationLookup:
    """
    Efficient lookup tool for gene-gene correlations and p-values from preprocessed data.
    
    This class provides fast access to Pearson correlation coefficients between 
    gene effect signatures derived from CERES scores across pan-cancer cell lines.
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize the lookup tool by loading the preprocessed gene correlation data.
        
        Parameters:
        -----------
        data_dir : str
            Path to the directory containing the preprocessed gene correlation data
            (should contain gene_names.txt and correlation matrix files)
        """
        self.data_dir = data_dir
        
        # Load data based on available file format (dense npy or sparse h5)
        self._load_gene_index()
        self._load_correlation_data()
        
    def _load_gene_index(self):
        """Load gene names and create mapping for fast lookups"""
        try:
            # Try loading from numpy array first (faster)
            gene_idx_path = os.path.join(self.data_dir, "gene_idx_array.npy")
            if os.path.exists(gene_idx_path):
                self.gene_names = np.load(gene_idx_path, allow_pickle=True, mmap_mode='r')
            else:
                # Fall back to text file
                gene_names_path = os.path.join(self.data_dir, "gene_names.txt")
                with open(gene_names_path, 'r') as f:
                    self.gene_names = np.array([line.strip() for line in f])
            
            # Create fast lookup dictionary
            self.gene_to_idx = {gene: idx for idx, gene in enumerate(self.gene_names)}
            self.num_genes = len(self.gene_names)
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Gene names file not found in {self.data_dir}")
    
    def _load_correlation_data(self):
        """Load correlation and p-value matrices based on available format"""
        # Check for dense matrix format first
        corr_matrix_path = os.path.join(self.data_dir, "corr_matrix.npy")
        p_val_matrix_path = os.path.join(self.data_dir, "p_val_matrix.npy")
        
        if os.path.exists(corr_matrix_path) and os.path.exists(p_val_matrix_path):
            # Dense matrix format
            self.corr_matrix = np.load(corr_matrix_path, mmap_mode='r')
            self.p_val_matrix = np.load(p_val_matrix_path, mmap_mode='r')
            self.format = "dense"
            
            # Check for adjusted p-values (optional)
            p_adj_matrix_path = os.path.join(self.data_dir, "p_adj_matrix.npy")
            if os.path.exists(p_adj_matrix_path):
                self.p_adj_matrix = np.load(p_adj_matrix_path, mmap_mode='r')
                self.has_adj_p = True
            else:
                self.has_adj_p = False
                
        else:
            # Try sparse HDF5 format
            h5_path = os.path.join(self.data_dir, "gene_correlations.h5")
            if not os.path.exists(h5_path):
                raise FileNotFoundError(f"No correlation data found in {self.data_dir}")
            
            # Load sparse matrices
            self.h5_file = h5py.File(h5_path, 'r')
            self.format = "sparse"
            
            # Check for adjusted p-values
            self.has_adj_p = 'p_adj' in self.h5_file
    
    def get_correlation(self, gene_a: str, gene_b: str) -> Dict[str, float]:
        """
        Get correlation coefficient and p-value between two genes.
        
        Parameters:
        -----------
        gene_a : str
            First gene symbol
        gene_b : str
            Second gene symbol
            
        Returns:
        --------
        dict
            Dictionary with 'correlation' and 'p_value' keys. If available,
            also includes 'adjusted_p_value'.
        
        Raises:
        -------
        KeyError
            If either gene is not found in the dataset
        """
        # Check if genes exist in our dataset
        if gene_a not in self.gene_to_idx:
            raise KeyError(f"Gene '{gene_a}' not avaliable in the gene correlation matrix")
        if gene_b not in self.gene_to_idx:
            raise KeyError(f"Gene '{gene_b}' not avaliable in the gene correlation matrix")
        
        # Get indices for the genes
        idx_a = self.gene_to_idx[gene_a]
        idx_b = self.gene_to_idx[gene_b]
        
        # Get values based on storage format
        if self.format == "dense":
            correlation = float(self.corr_matrix[idx_a, idx_b])
            p_value = float(self.p_val_matrix[idx_a, idx_b])
            
            result = {
                "correlation": correlation,
                "p_value": p_value
            }
            
            if self.has_adj_p:
                result["adjusted_p_value"] = float(self.p_adj_matrix[idx_a, idx_b])
                
        else:  # sparse format
            # Access sparse data from HDF5 file
            corr_data = self.h5_file['corr']
            p_val_data = self.h5_file['p_val']
            
            # We need to reconstruct the CSR matrix access pattern
            def get_csr_value(group, row, col):
                indptr = group['indptr'][:]
                indices = group['indices'][:]
                data = group['data'][:]
                
                # CSR lookup: check elements between indptr[row] and indptr[row+1]
                for i in range(indptr[row], indptr[row+1]):
                    if indices[i] == col:
                        return float(data[i])
                return 0.0  # If not found, assume zero (sparse matrix default)
            
            correlation = get_csr_value(corr_data, idx_a, idx_b)
            p_value = get_csr_value(p_val_data, idx_a, idx_b)
            
            result = {
                "correlation": correlation,
                "p_value": p_value
            }
            
            if self.has_adj_p:
                p_adj_data = self.h5_file['p_adj']
                result["adjusted_p_value"] = get_csr_value(p_adj_data, idx_a, idx_b)
        
        return result
    
    def get_cell_viability_effect(self, gene_a: str, gene_b: str) -> Dict[str, Union[float, str]]:
        """
        Get the interpreted cell viability effect based on correlation between two genes.
        
        Parameters:
        -----------
        gene_a : str
            First gene symbol
        gene_b : str
            Second gene symbol
            
        Returns:
        --------
        dict
            Dictionary with correlation statistics and interpretation of the relationship
            between the two genes in terms of cell viability effects.
        """
        corr_data = self.get_correlation(gene_a, gene_b)
        correlation = corr_data["correlation"]
        p_value = corr_data["p_value"]
        
        # Determine interaction interpretation
        if p_value > 0.05:
            interaction = "No statistically significant relationship"
        else:
            if correlation > 0.7:
                interaction = "Strong similar effect on cell viability"
            elif correlation > 0.5:
                interaction = "Moderate similar effect on cell viability"
            elif correlation > 0.3:
                interaction = "Weak similar effect on cell viability"
            elif correlation > -0.3:
                interaction = "Little to no relationship in cell viability effect"
            elif correlation > -0.5:
                interaction = "Weak opposing effect on cell viability"
            elif correlation > -0.7:
                interaction = "Moderate opposing effect on cell viability"
            else:
                interaction = "Strong opposing effect on cell viability"
        
        result = {
            "correlation": correlation,
            "p_value": p_value,
            "interaction": interaction
        }
        
        if "adjusted_p_value" in corr_data:
            result["adjusted_p_value"] = corr_data["adjusted_p_value"]
        
        return result
    
    def find_similar_genes(self, gene: str, top_n: int = 10, 
                          min_correlation: float = 0.5,
                          max_p_value: float = 0.05) -> List[Dict[str, Union[str, float]]]:
        """
        Find genes with similar effects on cell viability as the query gene.
        
        Parameters:
        -----------
        gene : str
            Query gene symbol
        top_n : int
            Number of top similar genes to return
        min_correlation : float
            Minimum correlation coefficient to consider
        max_p_value : float
            Maximum p-value to consider statistically significant
            
        Returns:
        --------
        list
            List of dictionaries with gene names and correlation statistics
        """
        if gene not in self.gene_to_idx:
            raise KeyError(f"Gene '{gene}' not found in dataset")
        
        idx = self.gene_to_idx[gene]
        similar_genes = []
        
        # Process based on storage format
        if self.format == "dense":
            # Get all correlations for this gene
            correlations = self.corr_matrix[idx, :]
            p_values = self.p_val_matrix[idx, :]
            
            # Create array of gene indices
            gene_indices = np.arange(self.num_genes)
            
            # Filter out the query gene itself, apply correlation and p-value filters
            mask = (gene_indices != idx) & (correlations >= min_correlation) & (p_values <= max_p_value)
            filtered_indices = gene_indices[mask]
            filtered_correlations = correlations[mask]
            filtered_p_values = p_values[mask]
            
            # Sort by correlation (descending)
            sorted_indices = np.argsort(filtered_correlations)[::-1]
            
            # Take top N
            top_indices = sorted_indices[:top_n]
            
            # Create result list
            for i in top_indices:
                gene_idx = filtered_indices[i]
                similar_genes.append({
                    "gene": self.gene_names[gene_idx],
                    "correlation": float(filtered_correlations[i]),
                    "p_value": float(filtered_p_values[i])
                })
        
        else:  # sparse format
            # For sparse format, we need to retrieve the entire row
            corr_data = self.h5_file['corr']
            p_val_data = self.h5_file['p_val']
            
            # Get row data for the query gene
            def get_csr_row(group, row):
                indptr = group['indptr'][:]
                indices = group['indices'][:]
                data = group['data'][:]
                
                row_indices = indices[indptr[row]:indptr[row+1]]
                row_data = data[indptr[row]:indptr[row+1]]
                
                return row_indices, row_data
            
            corr_indices, corr_values = get_csr_row(corr_data, idx)
            p_val_indices, p_val_values = get_csr_row(p_val_data, idx)
            
            # Create a dictionary for p-values
            p_val_dict = {int(i): float(v) for i, v in zip(p_val_indices, p_val_values)}
            
            # Filter and sort correlations
            candidates = []
            for i, v in zip(corr_indices, corr_values):
                i = int(i)
                if i == idx:  # Skip self
                    continue
                    
                corr = float(v)
                p_val = p_val_dict.get(i, 1.0)  # Default to 1.0 if not found
                
                if corr >= min_correlation and p_val <= max_p_value:
                    candidates.append((i, corr, p_val))
            
            # Sort by correlation (descending)
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Take top N
            for i, corr, p_val in candidates[:top_n]:
                similar_genes.append({
                    "gene": self.gene_names[i],
                    "correlation": corr,
                    "p_value": p_val
                })
        
        return similar_genes
    
    def __del__(self):
        """Clean up HDF5 file handle if using sparse format"""
        if hasattr(self, 'format') and self.format == "sparse" and hasattr(self, 'h5_file'):
            self.h5_file.close()


def compute_depmap24q2_gene_correlations(gene_a, gene_b, data_dir=None):
    """
    Robust cell viability correlation analysis between two genes using DepMap 24Q2 CERES data.
    
    This function computes Pearson correlation coefficients between gene knockout effects
    across 1,320 cancer cell lines, providing evidence-based insights into genetic 
    dependencies and cell viability relationships.
    
    Parameters:
    -----------
    gene_a : str
        First gene symbol for correlation analysis
    gene_b : str  
        Second gene symbol for correlation analysis
    data_dir : str, optional
        Path to DepMap 24Q2 preprocessed correlation data. If None, uses MEDEADB_PATH environment variable.
        
    Returns:
    --------
    tuple
        (correlation_coefficient, p_value, adjusted_p_value)
        Returns (None, None, None) if analysis fails
    """
    # Set default data_dir if not provided
    if data_dir is None:
        data_dir = os.path.join(_get_medeadb_path(), "depmap_24q2")
    
    def _log_insight(message, level="ANALYSIS"):
        """Structured logging for cell viability insights"""
        print(f"[DEPMAP] {level}: {message}", flush=True)
    
    def _validate_gene_symbols(gene_a, gene_b):
        """Validate and standardize gene symbols"""
        # Convert to uppercase for consistency
        gene_a_std = gene_a.upper().strip()
        gene_b_std = gene_b.upper().strip()
        
        # Basic validation
        if not gene_a_std or not gene_b_std:
            raise ValueError("Gene symbols cannot be empty")
        if gene_a_std == gene_b_std:
            _log_insight(f"Warning: Analyzing self-correlation for gene {gene_a_std}", "WARNING")
            
        return gene_a_std, gene_b_std
    
    def _interpret_correlation_strength(correlation):
        """Provide evidence-based interpretation of correlation strength"""
        abs_corr = abs(correlation)
        if abs_corr >= 0.8:
            return "very strong"
        elif abs_corr >= 0.6:
            return "strong" 
        elif abs_corr >= 0.4:
            return "moderate"
        elif abs_corr >= 0.2:
            return "weak"
        else:
            return "negligible"
    
    def _interpret_statistical_significance(p_value, adj_p_value=None):
        """Interpret statistical significance with multiple testing context"""
        if adj_p_value is not None and adj_p_value <= 0.001:
            return "highly significant (survives multiple testing correction)"
        elif adj_p_value is not None and adj_p_value <= 0.05:
            return "significant (survives multiple testing correction)"
        elif p_value <= 0.001:
            return "highly significant (uncorrected)"
        elif p_value <= 0.05:
            return "significant (uncorrected)"
        elif p_value <= 0.1:
            return "marginally significant"
        else:
            return "not statistically significant"
    
    def _generate_biological_interpretation(correlation, p_value, gene_a, gene_b, adj_p_value=None):
        """Generate evidence-based biological interpretation"""
        strength = _interpret_correlation_strength(correlation)
        significance = _interpret_statistical_significance(p_value, adj_p_value)
        
        if p_value > 0.05:
            return f"No reliable evidence for correlated cell viability effects between {gene_a} and {gene_b}"
        
        direction = "similar" if correlation > 0 else "opposing"
        
        # Detailed biological context
        if abs(correlation) >= 0.6 and p_value <= 0.001:
            confidence = "high confidence"
            evidence = "strong empirical evidence"
        elif abs(correlation) >= 0.4 and p_value <= 0.05:
            confidence = "moderate confidence" 
            evidence = "substantial evidence"
        else:
            confidence = "low confidence"
            evidence = "limited evidence"
            
        interpretation = (f"{confidence.capitalize()} of {direction} cell viability effects "
                         f"({strength} correlation, {significance}). "
                         f"This suggests {evidence} for ")
        
        if correlation > 0:
            interpretation += f"co-dependency or shared pathway involvement between {gene_a} and {gene_b}"
        else:
            interpretation += f"compensatory or antagonistic relationship between {gene_a} and {gene_b}"
            
        return interpretation
    
    try:
        # Input validation and standardization
        _log_insight(f"Initiating cell viability correlation analysis: {gene_a} ↔ {gene_b}")
        gene_a_std, gene_b_std = _validate_gene_symbols(gene_a, gene_b)
        
        # Data source validation
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"DepMap data directory not accessible: {data_dir}")
        
        _log_insight(f"Gene symbols standardized: {gene_a_std}, {gene_b_std}")
        _log_insight(f"Loading DepMap 24Q2 correlation matrix from: {data_dir}")
        
        # Initialize correlation lookup with error handling
        try:
            lookup = GeneCorrelationLookup(data_dir)
            _log_insight(f"Successfully loaded correlation data for {lookup.num_genes:,} genes")
        except Exception as e:
            _log_insight(f"Failed to initialize DepMap correlation data: {str(e)}", "ERROR")
            raise
        
        # Gene availability validation
        missing_genes = []
        if gene_a_std not in lookup.gene_to_idx:
            missing_genes.append(gene_a_std)
        if gene_b_std not in lookup.gene_to_idx:
            missing_genes.append(gene_b_std)
            
        if missing_genes:
            _log_insight(f"Genes not found in DepMap dataset: {', '.join(missing_genes)}", "ERROR")
            _log_insight(f"Available genes: {lookup.num_genes:,} total in DepMap 24Q2", "INFO")
            raise KeyError(f"Gene(s) not available in DepMap correlation matrix: {', '.join(missing_genes)}")
        
        _log_insight(f"Gene pair validation successful - both genes found in dataset")
        
        # Perform correlation analysis
        _log_insight(f"Computing CERES-based correlation across 1,320 cancer cell lines")
        result = lookup.get_cell_viability_effect(gene_a_std, gene_b_std)
        
        correlation = result['correlation']
        p_value = result['p_value'] 
        adj_p_value = result.get('adjusted_p_value')
        
        # Data quality validation
        if not (-1 <= correlation <= 1):
            _log_insight(f"Warning: Correlation value outside expected range: {correlation}", "WARNING")
        if p_value < 0 or p_value > 1:
            _log_insight(f"Warning: P-value outside expected range: {p_value}", "WARNING")
            
        # Statistical analysis summary
        _log_insight(f"Correlation coefficient: {correlation:.4f}")
        _log_insight(f"Statistical significance: p = {p_value:.2e}")
        if adj_p_value is not None:
            _log_insight(f"Multiple testing correction: adj_p = {adj_p_value:.2e}")
        
        # Evidence-based interpretation
        biological_interpretation = _generate_biological_interpretation(
            correlation, p_value, gene_a_std, gene_b_std, adj_p_value
        )
        _log_insight(f"Biological interpretation: {biological_interpretation}")
        
        # Clinical relevance context
        if abs(correlation) >= 0.5 and p_value <= 0.01:
            _log_insight(f"Clinical relevance: Strong correlation suggests potential for combination "
                        f"therapy or synthetic lethality screening", "INSIGHT")
        elif abs(correlation) >= 0.3 and p_value <= 0.05:
            _log_insight(f"Research priority: Moderate correlation warrants further investigation "
                        f"in relevant cancer contexts", "INSIGHT")
        
        # Analysis summary
        _log_insight(f"Analysis completed successfully - correlation: {correlation:.4f}, "
                    f"significance: {_interpret_statistical_significance(p_value, adj_p_value)}")
        
        return correlation, p_value, adj_p_value
        
    except KeyError as e:
        _log_insight(f"Gene availability error: {str(e)}", "ERROR")
        _log_insight(f"Recommendation: Verify gene symbols or check DepMap gene coverage", "INFO")
        return None, None, None
        
    except FileNotFoundError as e:
        _log_insight(f"Data access error: {str(e)}", "ERROR")
        _log_insight(f"Recommendation: Verify DepMap data directory path and permissions", "INFO")
        return None, None, None
        
    except ValueError as e:
        _log_insight(f"Input validation error: {str(e)}", "ERROR")
        _log_insight(f"Recommendation: Check gene symbol format and validity", "INFO")
        return None, None, None
        
    except Exception as e:
        _log_insight(f"Unexpected analysis error: {str(e)}", "ERROR")
        _log_insight(f"System context: DepMap correlation analysis framework", "DEBUG")
        return None, None, None


# Simple example usage
def main():
    """Example usage of the GeneCorrelationLookup class"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Look up gene correlation and cell viability effects")
    parser.add_argument("--data-dir", "-d", required=True, 
                      help="Directory containing preprocessed gene correlation data")
    parser.add_argument("--gene-a", "-a", required=True, 
                      help="First gene symbol")
    parser.add_argument("--gene-b", "-b", 
                      help="Second gene symbol (optional, if not provided, will show similar genes to gene-a)")
    parser.add_argument("--top-n", "-n", type=int, default=10,
                      help="Number of similar genes to show (when gene-b is not provided)")
    
    args = parser.parse_args()
    
    try:
        # Initialize lookup tool
        lookup = GeneCorrelationLookup(args.data_dir)
        
        # Look up correlation between two genes
        result = lookup.get_cell_viability_effect(args.gene_a, args.gene_b)
        
        print(f"\nCell Viability Effect Analysis: {args.gene_a} vs {args.gene_b}")
        print("-" * 60)
        print(f"Correlation:   {result['correlation']:.4f}")
        print(f"P-value:       {result['p_value']:.4e}")
        if "adjusted_p_value" in result:
            print(f"Adjusted P:    {result['adjusted_p_value']:.4e}")
        # print(f"Interpretation: {result['interaction']}")
        return result['correlation'], result['p_value'], result['adjusted_p_value']
    
    except Exception as e:
        print(f"Warning: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    main()