# Knowledge Graph Explorer

A comprehensive toolkit for building, querying, and visualizing knowledge graphs from heterogeneous data sources using RDF and OWL standards.

## Overview

Knowledge Graph Explorer is a Python-based platform that enables you to:

1. Build knowledge graphs from various data sources using R2RML mappings
2. Query and analyze graph data using SPARQL
3. Visualize knowledge graphs with interactive web interfaces
4. Explore semantic relationships between entities

## Features

- **R2RML Mapping Engine**: Transform tabular data (CSV) into RDF knowledge graphs
- **OWL TBox Integration**: Import and use ontology schemas from OWL files 
- **Flexible Visualization**: Interactive graph visualizations using Cytoscape.js or D3.js
- **SPARQL Querying**: Execute SPARQL queries against your knowledge graphs
- **Named Graph Support**: Organize data in separate named graphs
- **Aggregation Options**: Simplify complex graphs by aggregating nodes by class

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/kg-explorer.git
cd kg-explorer

# Install required dependencies
pip install -r requirements.txt
```

## Usage

### Building a Knowledge Graph

```bash
python scripts/kg_builder.py \
  --mappings-dir examples/mappings \
  --data-dir examples/data \
  --output kg_output.nq \
  --tbox-file examples/ontology/world_model.owl \
  --base-uri "https://kg-explorer-mvp/KnowledgeGraph.owl#" \
  --single-output \
  --include-tbox
```

### Visualizing a Knowledge Graph

```bash
python scripts/visualize_kg.py \
  --input kg_output.nq \
  --output outputs/kg_visualization.html \
  --vis-type cytoscape \
  --aggregate-by-class
```

### API Server

Start the API server to interact with your knowledge graphs through a web interface:

```bash
python api/main.py
```

Then open a web browser and navigate to http://localhost:8000

## Project Structure

- `core/`: Core knowledge graph functionality
  - `kg_functions.py`: Primary functions for graph manipulation
- `scripts/`: Executable scripts
  - `kg_builder.py`: Build knowledge graphs from data sources
  - `visualize_kg.py`: Create visualizations from RDF graphs
- `api/`: REST API for interacting with knowledge graphs
- `utils/`: Utility functions
  - `visualization.py`: Graph visualization utilities
- `examples/`: Example files
  - `mappings/`: R2RML mapping files
  - `data/`: Sample data files
  - `ontology/`: OWL ontology files
- `outputs/`: Generated output files

## Examples

The repository includes several examples of R2RML mappings for building knowledge graphs from various data sources:

- Building Connected data (bidPackage, threads, etc.)
- Sample ontology in OWL format

## Advanced Usage

### Focusing on Specific Entities

```bash
python scripts/visualize_kg.py \
  --input kg_output.nq \
  --entity "https://kg-explorer-mvp/KnowledgeGraph.owl#Individual/BidPackage/BuildingConnected/123"
```

### Filtering by Entity Types

```bash
python scripts/visualize_kg.py \
  --input kg_output.nq \
  --filter-types "https://kg-explorer-mvp/KnowledgeGraph.owl#Class/BidPackage" "https://kg-explorer-mvp/KnowledgeGraph.owl#Class/Person"
```

## Requirements

- Python 3.8+
- RDFLib
- OWLReady2
- Pandas
- FastAPI (for API server)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.