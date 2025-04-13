# kg_platform/scripts/kg_builder.py

import argparse
from pathlib import Path
import sys
import os
from rdflib import Graph, Dataset, URIRef, Namespace
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.kg_functions import (
    load_tbox_from_owl,
    extract_datasource_from_mapping,
    execute_r2rml_mapping,
    tbox_to_graph,
    load_mappings_directory
)

def export_nquads(dataset, output_path):
    """Export a dataset as N-Quads format"""
    with open(output_path, 'w') as f:
        f.write(dataset.serialize(format='nquads'))
    return True

def build_knowledge_graph(mappings_dir, data_dir, output_file=None, output_dir=None, tbox_file=None, base_uri=None, single_output=False, include_tbox=False, separate_files=False):
    """
    Build a knowledge graph from R2RML mappings and CSV data files.
    
    Args:
        mappings_dir (str): Directory containing R2RML mapping files
        data_dir (str): Directory containing CSV data files
        output_file (str, optional): Path to save the output graph
        output_dir (str, optional): Directory to save separate output files (when separate_files=True)
        tbox_file (str, optional): Path to OWL file containing TBox
        base_uri (str, optional): Base URI for the generated graph
        single_output (bool): If True, create a dataset with named graphs
        include_tbox (bool): If True, include TBox in a separate named graph
        separate_files (bool): If True, create separate .nq files for each mapping
        
    Returns:
        Graph or Dataset: The resulting knowledge graph if separate_files=False,
                         otherwise a dictionary of file paths created
    """
    from core.kg_functions import load_tbox_from_owl, load_mappings_directory, extract_datasource_from_mapping, execute_r2rml_mapping
    from rdflib import Graph, Dataset, URIRef, Namespace
    import os
    import logging
    
    logger = logging.getLogger('kg_builder')
    
    # Create output directory if it doesn't exist and separate_files is True
    if separate_files and output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Set default base URI if not provided
    if base_uri is None:
        base_uri = "https://kg-explorer-mvp/KnowledgeGraph.owl#"
    
    # Load TBox if specified
    tbox = None
    if tbox_file and os.path.exists(tbox_file):
        logger.info(f"Loading TBox from {tbox_file}")
        tbox = load_tbox_from_owl(tbox_file)
    
    # Create output graph or dataset
    if single_output:
        kg = Dataset()
    else:
        kg = Graph()
    
    # Load and process mappings
    result = load_mappings_directory(
        mappings_dir,
        data_dir,
        kg=kg,
        base_uri=base_uri,
        single_output=single_output,
        include_tbox=include_tbox,
        tbox=tbox
    )
    
    # Save output
    if output_file and not separate_files:
        format = 'nquads' if single_output else 'turtle'
        logger.info(f"Saving knowledge graph to {output_file}")
        kg.serialize(destination=output_file, format=format)
    
    return result

def main():
    """
    Main function to parse arguments and build a knowledge graph.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Build a knowledge graph from R2RML mappings and CSV data.")
    parser.add_argument("--mappings-dir", required=True, help="Directory containing R2RML mapping files")
    parser.add_argument("--data-dir", required=True, help="Directory containing CSV data files")
    parser.add_argument("--output", help="Path to save the knowledge graph")
    parser.add_argument("--output-dir", help="Directory to save separate output files (when using --separate-files)")
    parser.add_argument("--tbox-file", help="Path to OWL file with TBox definitions")
    parser.add_argument("--base-uri", default="https://kg-explorer-mvp/KnowledgeGraph.owl#", help="Base URI for generated resources")
    parser.add_argument("--single-output", action="store_true", help="Create a single output file with named graphs")
    parser.add_argument("--include-tbox", action="store_true", help="Include TBox information in the graph")
    parser.add_argument("--separate-files", action="store_true", help="Create separate .nq files for each mapping")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Configure logging level
    import logging
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    
    # Check for required arguments based on mode
    if not args.separate_files and not args.output:
        parser.error("--output is required unless --separate-files is specified")
    
    # Build knowledge graph
    build_knowledge_graph(
        args.mappings_dir,
        args.data_dir,
        output_file=args.output,
        output_dir=args.output_dir,
        tbox_file=args.tbox_file,
        base_uri=args.base_uri,
        single_output=args.single_output,
        include_tbox=args.include_tbox,
        separate_files=args.separate_files
    )
    
    print("Knowledge graph building complete!")

if __name__ == "__main__":
    main()