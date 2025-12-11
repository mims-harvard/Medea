#!/usr/bin/env python3
"""
Script to extract all user queries from experiment log files.
Extracts text between "[User Query]:" and "Agent [agent_name] receives the following TaskPackage:"
Separates user questions from experiment instructions and saves to CSV.
Also extracts gene pairs, cell line, and interaction data from evaluation CSV files.

Usage:
    python extract_user_queries_clean.py [log_directory] [output_csv]
    python extract_user_queries_clean.py --file log_file.out [output_csv]
    
Examples:
    python extract_user_queries_clean.py
    python extract_user_queries_clean.py results
    python extract_user_queries_clean.py results/raw_log
    python extract_user_queries_clean.py results queries.csv
    python extract_user_queries_clean.py --file results/raw_log/medea-MCF7-43.out
    python extract_user_queries_clean.py --file results/raw_log/medea-MCF7-43.out single_file_queries.csv
"""

import os
import re
import argparse
import csv
import pandas as pd
from typing import List, Dict, Tuple, Optional

def separate_question_and_instructions(query_text: str) -> Tuple[str, str]:
    """
    Separate the user question from experiment instructions.
    
    Args:
        query_text: The full query text
        
    Returns:
        Tuple of (user_question, experiment_instructions)
    """
    # Look for common patterns that indicate the start of experiment instructions
    instruction_markers = [
        "Use DepMap data",
        "Next, perform",
        "Synthesize the",
        "evaluate whether",
        "find literature support"
    ]
    
    user_question = query_text.strip()
    experiment_instructions = ""
    
    # Find the first occurrence of any instruction marker
    first_instruction_pos = -1
    for marker in instruction_markers:
        pos = query_text.find(marker)
        if pos != -1 and (first_instruction_pos == -1 or pos < first_instruction_pos):
            first_instruction_pos = pos
    
    if first_instruction_pos != -1:
        user_question = query_text[:first_instruction_pos].strip()
        experiment_instructions = query_text[first_instruction_pos:].strip()
    
    return user_question, experiment_instructions

def extract_file_info_from_filename(filename: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extract dataset type, cell line, and experiment number from filename.
    
    Args:
        filename: Log file filename (e.g., "medea-MCF7-42.out")
        
    Returns:
        Tuple of (dataset_type, cell_line, experiment_number)
    """
    # Remove .out extension
    base_name = filename.replace('.out', '')
    
    # Split by hyphens
    parts = base_name.split('-')
    
    if len(parts) >= 3:
        # Format: model-cellline-number (e.g., medea-MCF7-42)
        model = parts[0]
        cell_line = parts[1]
        experiment_num = parts[2]
        return model, cell_line, experiment_num
    elif len(parts) == 2:
        # Format: model-cellline (e.g., medea-MCF7)
        model = parts[0]
        cell_line = parts[1]
        return model, cell_line, None
    else:
        return None, None, None

def find_evaluation_file(cell_line: str, experiment_num: str) -> Optional[str]:
    """
    Find the corresponding evaluation file for a given cell line and experiment number.
    
    Args:
        cell_line: Cell line name (e.g., "MCF7")
        experiment_num: Experiment number (e.g., "42")
        
    Returns:
        Path to evaluation file if found, None otherwise
    """
    evaluation_dir = "evaluation"
    
    # Check different subdirectories for evaluation files
    subdirs = ["sl", "targetID", "immune_response"]
    
    for subdir in subdirs:
        subdir_path = os.path.join(evaluation_dir, subdir)
        if os.path.exists(subdir_path):
            # Look for files matching the pattern
            for filename in os.listdir(subdir_path):
                if filename.endswith('.csv'):
                    # Check if filename contains cell line and experiment number
                    if cell_line.lower() in filename.lower() and experiment_num in filename:
                        return os.path.join(subdir_path, filename)
    
    return None

def extract_gene_pairs_from_evaluation_file(evaluation_file: str) -> Tuple[List[Dict[str, str]], str]:
    """
    Extract gene pairs, interaction data, and cell line from evaluation file.
    Returns (list of dicts, cell_line_value)
    """
    try:
        df = pd.read_csv(evaluation_file)
        gene_cols = []
        interaction_col = None
        cell_line_col = None
        # Find gene columns and cell line column
        for col in df.columns:
            col_lower = col.lower()
            if 'gene' in col_lower and ('a' in col_lower or '1' in col_lower):
                gene_cols.append(col)
            elif 'gene' in col_lower and ('b' in col_lower or '2' in col_lower):
                gene_cols.append(col)
            elif 'interaction' in col_lower or 'label' in col_lower or 'y' in col_lower:
                interaction_col = col
            elif 'cell' in col_lower and 'line' in col_lower:
                cell_line_col = col
        # Determine cell line value
        cell_line_value = None
        if cell_line_col and cell_line_col in df.columns:
            # Use the first non-empty value
            cell_line_value = str(df[cell_line_col].dropna().iloc[0]) if not df[cell_line_col].dropna().empty else ''
        if not cell_line_value:
            # Fallback: try to extract from filename
            import re
            m = re.search(r'([A-Za-z0-9]+)[-_]', os.path.basename(evaluation_file))
            if m:
                cell_line_value = m.group(1)
            else:
                cell_line_value = ''
        # Extract gene pairs and interaction
        results = []
        if len(gene_cols) >= 2 and interaction_col:
            for _, row in df.iterrows():
                gene_a = str(row[gene_cols[0]]) if pd.notna(row[gene_cols[0]]) else ""
                gene_b = str(row[gene_cols[1]]) if pd.notna(row[gene_cols[1]]) else ""
                interaction = str(row[interaction_col]) if pd.notna(row[interaction_col]) else ""
                results.append({
                    'gene_a': gene_a,
                    'gene_b': gene_b,
                    'interaction': interaction
                })
        return results, cell_line_value
    except Exception as e:
        print(f"Error reading evaluation file {evaluation_file}: {e}")
        return [], ''

def match_evaluation_to_queries(evaluation_data: List[Dict[str, str]], queries: List[Dict[str, str]], cell_line_value: str) -> List[Dict[str, str]]:
    """
    For each gene pair in the evaluation file, find a matching query (where both gene names appear in the query text).
    If no match, leave user_question, experiment_instruction, and full_query empty.
    Returns a list of dicts, one per evaluation row, with cell_line included.
    """
    matched_rows = []
    for eval_row in evaluation_data:
        gene_a = eval_row['gene_a'].lower()
        gene_b = eval_row['gene_b'].lower()
        found = False
        for query in queries:
            query_text = query['query'].lower()
            if gene_a in query_text and gene_b in query_text:
                matched_rows.append({
                    'gene_a': eval_row['gene_a'],
                    'gene_b': eval_row['gene_b'],
                    'cell_line': cell_line_value,
                    'interaction': eval_row.get('interaction', ''),
                    'user_question': query.get('user_question', ''),
                    'experiment_instruction': query.get('experiment_instruction', ''),
                    'full_query': query['query']
                })
                found = True
                break
        if not found:
            matched_rows.append({
                'gene_a': eval_row['gene_a'],
                'gene_b': eval_row['gene_b'],
                'cell_line': cell_line_value,
                'interaction': eval_row.get('interaction', ''),
                'user_question': '',
                'experiment_instruction': '',
                'full_query': ''
            })
    return matched_rows

def extract_user_queries_from_file(file_path: str) -> List[Dict[str, str]]:
    """
    Extract user queries from a single log file.
    """
    queries = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return queries
    
    # Find all positions of "[User Query]:"
    query_markers = []
    start_pos = 0
    while True:
        pos = content.find('[User Query]:', start_pos)
        if pos == -1:
            break
        query_markers.append(pos)
        start_pos = pos + 1
    
    print(f"Found {len(query_markers)} user query markers")
    
    for i, marker_pos in enumerate(query_markers):
        query_start = content.find(':', marker_pos) + 1
        if query_start == 0:
            continue
        
        next_agent_pos = content.find('Agent', query_start)
        if next_agent_pos == -1:
            query_text = content[query_start:].strip()
        else:
            agent_line_end = content.find('\n', next_agent_pos)
            if agent_line_end == -1:
                agent_line_end = len(content)
            agent_line = content[next_agent_pos:agent_line_end]
            if 'receives the following' in agent_line and 'TaskPackage' in agent_line:
                query_text = content[query_start:next_agent_pos].strip()
            else:
                next_agent_pos = content.find('Agent', agent_line_end)
                if next_agent_pos == -1:
                    query_text = content[query_start:].strip()
                else:
                    query_text = content[query_start:next_agent_pos].strip()
        
        query_text = query_text.strip()
        if query_text:
            user_question, experiment_instructions = separate_question_and_instructions(query_text)
            
            queries.append({
                'file': os.path.basename(file_path),
                'file_path': file_path,
                'query_number': i + 1,
                'query': query_text,
                'user_question': user_question,
                'experiment_instruction': experiment_instructions,
                'gene_a': '',
                'gene_b': '',
                'interaction': ''
            })
    
    return queries

def extract_queries_from_single_file(file_path: str) -> List[Dict[str, str]]:
    """
    Extract user queries from a single specified log file.
    For each gene pair in the evaluation file, find a matching query.
    """
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return []
    if not file_path.endswith('.out'):
        print(f"Warning: File '{file_path}' does not have .out extension.")
    print(f"Processing single file: {file_path}")
    queries = extract_user_queries_from_file(file_path)
    print(f"  - Extracted {len(queries)} queries")
    # Try to find corresponding evaluation file
    filename = os.path.basename(file_path)
    model, cell_line, experiment_num = extract_file_info_from_filename(filename)
    if cell_line and experiment_num:
        evaluation_file = find_evaluation_file(cell_line, experiment_num)
        if evaluation_file:
            print(f"  - Found evaluation file: {evaluation_file}")
            evaluation_data, cell_line_value = extract_gene_pairs_from_evaluation_file(evaluation_file)
            print(f"  - Extracted {len(evaluation_data)} gene pairs from evaluation file")
            # Match evaluation to queries (one row per evaluation entry)
            return match_evaluation_to_queries(evaluation_data, queries, cell_line_value)
        else:
            print(f"  - No evaluation file found for {cell_line}-{experiment_num}")
    else:
        print(f"  - Could not extract cell line and experiment number from filename: {filename}")
    return []

def extract_all_queries_from_directory(directory_path: str) -> List[Dict[str, str]]:
    """
    Extract user queries from all .out files in the specified directory and subdirectories.
    
    Args:
        directory_path: Path to the directory containing log files
        
    Returns:
        List of dictionaries containing all query information
    """
    all_queries = []
    
    # Check if directory exists
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist.")
        return all_queries
    
    # Find all .out files in the directory and subdirectories
    log_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.out'):
                log_files.append(os.path.join(root, file))
    
    if not log_files:
        print(f"No .out files found in directory '{directory_path}'")
        return all_queries
    
    print(f"Found {len(log_files)} log files to process...")
    
    for file_path in log_files:
        print(f"Processing: {file_path}")
        queries = extract_queries_from_single_file(file_path)
        all_queries.extend(queries)
        print(f"  - Extracted {len(queries)} queries")
    
    return all_queries

def save_queries_to_csv(queries: List[Dict[str, str]], output_file: str):
    """
    Save extracted queries to a CSV file with separated user questions and experiment instructions.
    - No quotes in column names
    - No quotes in the first 4 columns (gene_a, gene_b, cell_line, interaction)
    - The 3rd column is cell_line
    - Only quote fields if needed (for user_question, experiment_instruction, full_query)
    """
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            # Write header manually, no quotes
            f.write('gene_a,gene_b,cell_line,interaction,user_question,experiment_instruction,full_query\n')
            for row in queries:
                def strip_quotes(s):
                    if isinstance(s, str):
                        s = s.strip()
                        if s.startswith('"') and s.endswith('"'):
                            s = s[1:-1]
                        return s.replace('"', '').strip()
                    return s
                # Compose row: gene_a, gene_b, cell_line, interaction (no quotes), rest quoted if needed
                gene_a = strip_quotes(row.get('gene_a', ''))
                gene_b = strip_quotes(row.get('gene_b', ''))
                # Try to get cell_line from row, fallback to empty string
                cell_line = strip_quotes(row.get('cell_line', ''))
                interaction = strip_quotes(row.get('interaction', ''))
                # Quote user_question, experiment_instruction, full_query if they contain commas or newlines
                def quote_if_needed(s):
                    if not isinstance(s, str):
                        s = ''
                    if ',' in s or '\n' in s or '"' in s:
                        s = '"' + s.replace('"', '""') + '"'
                    return s
                user_question = quote_if_needed(row.get('user_question', ''))
                experiment_instruction = quote_if_needed(row.get('experiment_instruction', ''))
                full_query = quote_if_needed(row.get('full_query', ''))
                # Write row
                f.write(f'{gene_a},{gene_b},{cell_line},{interaction},{user_question},{experiment_instruction},{full_query}\n')
        print(f"Saved {len(queries)} queries to {output_file}")
    except Exception as e:
        print(f"Error saving to {output_file}: {e}")

def print_summary(queries: List[Dict[str, str]]):
    """
    Print a summary of extracted queries.
    
    Args:
        queries: List of query dictionaries
    """
    print(f"\n=== SUMMARY ===")
    print(f"Total evaluation rows: {len(queries)}")
    
    if not queries:
        print("No queries found.")
        return
    
    # Count queries with matched user questions
    matched_count = sum(1 for q in queries if q.get('user_question'))
    print(f"Rows with matched user queries: {matched_count}/{len(queries)}")
    
    # Show some example rows
    print(f"\nExample rows:")
    for i, row in enumerate(queries[:3]):
        print(f"\n{i+1}. Gene pair: {row.get('gene_a', 'N/A')} + {row.get('gene_b', 'N/A')}")
        print(f"   Interaction: {row.get('interaction', 'N/A')}")
        uq = row.get('user_question', '')
        print(f"   User question: {uq[:120]}{'...' if len(uq) > 120 else ''}")
        print(f"   Matched: {'Yes' if uq else 'No'}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract user queries from experiment log files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Use default 'results' directory
  %(prog)s results                   # Specify log directory
  %(prog)s results/raw_log           # Use specific subdirectory
  %(prog)s results queries.csv       # Specify output CSV file
  %(prog)s --file log_file.out       # Process single file
  %(prog)s --file log_file.out output.csv  # Process single file with custom output
        """
    )
    
    # Add file argument
    parser.add_argument(
        '--file',
        '-f',
        metavar='LOG_FILE',
        help='Process a single log file instead of a directory'
    )
    
    # Add positional arguments
    parser.add_argument(
        'log_directory',
        nargs='?',
        default='results',
        help='Directory containing log files (default: results)'
    )
    
    parser.add_argument(
        'output_csv',
        nargs='?',
        default='extracted_user_queries.csv',
        help='Output CSV file (default: extracted_user_queries.csv)'
    )
    
    return parser.parse_args()

def main():
    """Main function to extract user queries from log files."""
    args = parse_arguments()
    
    # Determine if we're processing a single file or directory
    if args.file:
        # Process single file
        print(f"Extracting user queries from single file: {args.file}")
        print(f"Output CSV: {args.output_csv}")
        print("-" * 50)
        
        queries = extract_queries_from_single_file(args.file)
    else:
        # Process directory
        print(f"Extracting user queries from directory: {args.log_directory}")
        print(f"Output CSV: {args.output_csv}")
        print("-" * 50)
        
        queries = extract_all_queries_from_directory(args.log_directory)
    
    if not queries:
        print("No queries found in any log files.")
        return
    
    # Print summary
    print_summary(queries)
    
    # Save to CSV file
    save_queries_to_csv(queries, args.output_csv)
    
    print(f"\nExtraction complete! Found {len(queries)} user queries.")
    print(f"Results saved to: {args.output_csv}")

if __name__ == "__main__":
    main() 