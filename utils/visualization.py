from typing import Dict, List, Optional, Any
from rdflib import Graph, URIRef
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.kg_functions import execute_sparql_query

def create_d3_graph_data(graph: Graph) -> Dict[str, Any]:
    """
    Create data for D3.js force-directed graph visualization
    
    Args:
        graph: The RDF graph to visualize
        
    Returns:
        Dictionary with nodes and links in D3 format
    """
    from rdflib.namespace import RDF
    
    # Initialize data structure
    d3_data = {
        "nodes": [],
        "links": []
    }
    
    # Track nodes to avoid duplicates
    nodes_map = {}
    
    # Get entity types from graph
    query = """
    SELECT DISTINCT ?type (COUNT(?s) as ?count)
    WHERE {
        ?s a ?type .
    }
    GROUP BY ?type
    ORDER BY DESC(?count)
    """
    
    entity_types = execute_sparql_query(graph, query)
    
    # Create a color map for types
    type_colors = {}
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]
    
    for i, type_info in enumerate(entity_types):
        type_uri = type_info["type"]
        type_name = type_uri.split('#')[-1] if '#' in type_uri else type_uri.split('/')[-1]
        type_colors[type_uri] = colors[i % len(colors)]
    
    # Find all subjects with their types
    query = """
    SELECT ?s ?label ?type
    WHERE {
        ?s a ?type .
        OPTIONAL { ?s rdfs:label ?label }
    }
    """
    
    subjects = execute_sparql_query(graph, query)
    
    # Add nodes
    for subject in subjects:
        uri = subject["s"]
        
        # Skip if already processed
        if uri in nodes_map:
            continue
        
        # Get label
        label = subject.get("label")
        if not label:
            label = uri.split('#')[-1] if '#' in uri else uri.split('/')[-1]
        
        # Get type
        type_uri = subject.get("type")
        type_name = type_uri.split('#')[-1] if '#' in type_uri else type_uri.split('/')[-1]
        
        # Create node
        node = {
            "id": uri,
            "label": label,
            "type": type_name,
            "color": type_colors.get(type_uri, "#999999")
        }
        
        # Add to result
        d3_data["nodes"].append(node)
        nodes_map[uri] = len(d3_data["nodes"]) - 1
    
    # Add links
    for s, p, o in graph:
        if not isinstance(o, URIRef) or p == RDF.type:
            continue
        
        s_str = str(s)
        o_str = str(o)
        
        # Skip if either node was not included
        if s_str not in nodes_map or o_str not in nodes_map:
            continue
        
        # Get predicate label
        p_str = str(p)
        p_label = p_str.split('#')[-1] if '#' in p_str else p_str.split('/')[-1]
        
        # Create link
        link = {
            "source": s_str,
            "target": o_str,
            "label": p_label
        }
        
        d3_data["links"].append(link)
    
    return d3_data

def create_cytoscape_graph_data(graph: Graph, aggregate_by_class=False, aggregate_edges=False) -> Dict[str, Any]:
    """
    Create a data structure suitable for Cytoscape visualization from an RDF graph.
    
    Args:
        graph (Graph): The RDF graph to visualize
        aggregate_by_class (bool): Whether to aggregate nodes by class
        aggregate_edges (bool): Whether to aggregate edges between classes
        
    Returns:
        Dict[str, Any]: Cytoscape-compatible data structure
    """
    from rdflib.namespace import RDF
    
    # Initialize result structure
    nodes = []
    edges = []
    class_nodes = {}
    
    # Define color scheme for different node types
    type_colors = {
        "default": "#666666",  # Default gray
        # Add specific type colors here
    }
    
    # Handle class aggregation
    if aggregate_by_class:
        # First pass: collect all nodes by type
        for s, p, o in graph.triples((None, RDF.type, None)):
            if isinstance(o, URIRef):
                # Get the class URI
                class_uri = str(o)
                
                # Get class label
                class_label = class_uri.split('/')[-1]
                if '#' in class_label:
                    class_label = class_label.split('#')[-1]
                
                # Skip if already processed this class
                if class_uri in class_nodes:
                    # Fix: Access count inside the data dictionary
                    class_nodes[class_uri]["data"]["count"] += 1
                else:
                    # Get color for this class
                    class_color = type_colors.get(class_uri, type_colors["default"])
                    
                    # Create class node
                    class_nodes[class_uri] = {
                        "data": {
                            "id": class_uri,
                            "label": class_label,
                            "color": class_color,
                            "count": 1,
                            "type": "class"
                        }
                    }
        
        # Add class nodes to the result
        nodes.extend(list(class_nodes.values()))
        
        # Second pass: handle edges between classes
        # ... implement edge aggregation logic ...
    
    # Regular (non-aggregated) graph processing
    else:
        # ... implement regular node and edge processing ...
    
    return {"nodes": nodes, "edges": edges}

def filter_graph_by_type(graph: Graph, types: List[str]) -> Graph:
    """
    Filter the graph to only include entities of specified types
    
    Args:
        graph: The RDF graph to filter
        types: List of type URIs to include
        
    Returns:
        Filtered RDF graph
    """
    from rdflib.namespace import RDF
    
    result_graph = Graph()
    
    # Convert types to URIRefs if they are strings
    type_uris = [URIRef(t) if not isinstance(t, URIRef) else t for t in types]
    
    # Find all entities of the specified types
    entities = set()
    for type_uri in type_uris:
        for entity in graph.subjects(RDF.type, type_uri):
            entities.add(entity)
    
    # Add all triples that have the entities as subject or object
    for entity in entities:
        # Add triples where entity is subject
        for p, o in graph.predicate_objects(entity):
            result_graph.add((entity, p, o))
        
        # Add triples where entity is object
        for s, p in graph.subject_predicates(entity):
            if s in entities:  # Only include if subject is also of the specified types
                result_graph.add((s, p, entity))
    
    return result_graph

def extract_subgraph(graph: Graph, center_node: str, depth: int = 1) -> Graph:
    """
    Extract a subgraph around a center node up to a specified depth
    
    Args:
        graph: The RDF graph
        center_node: URI of the center node
        depth: Number of hops from center node
        
    Returns:
        Subgraph centered around the specified node
    """
    center = URIRef(center_node)
    result_graph = Graph()
    
    # Set of nodes to explore at current depth
    current_nodes = {center}
    
    # Explore up to the specified depth
    for _ in range(depth):
        next_nodes = set()
        
        for node in current_nodes:
            # Add triples where node is subject
            for p, o in graph.predicate_objects(node):
                result_graph.add((node, p, o))
                if isinstance(o, URIRef):
                    next_nodes.add(o)
            
            # Add triples where node is object
            for s, p in graph.subject_predicates(node):
                result_graph.add((s, p, node))
                next_nodes.add(s)
        
        # Update current nodes for next iteration
        current_nodes = next_nodes
    
    return result_graph

def get_graph_degree_distribution(graph: Graph) -> Dict[str, Dict[int, int]]:
    """
    Calculate the degree distribution of nodes in the graph
    
    Args:
        graph: The RDF graph
        
    Returns:
        Dictionary with in-degree and out-degree distributions
    """
    # Initialize degree counters
    in_degrees = {}
    out_degrees = {}
    
    # Count out-degrees (triples where node is subject)
    for s in set(graph.subjects()):
        out_degree = len(list(graph.predicate_objects(s)))
        if out_degree in out_degrees:
            out_degrees[out_degree] += 1
        else:
            out_degrees[out_degree] = 1
    
    # Count in-degrees (triples where node is object)
    for o in set([o for o in graph.objects() if isinstance(o, URIRef)]):
        in_degree = len(list(graph.subject_predicates(o)))
        if in_degree in in_degrees:
            in_degrees[in_degree] += 1
        else:
            in_degrees[in_degree] = 1
    
    return {
        "in_degree": in_degrees,
        "out_degree": out_degrees
    }

def calculate_centrality(graph: Graph, centrality_type: str = "degree") -> Dict[str, float]:
    """
    Calculate centrality measures for nodes in the graph
    
    Args:
        graph: The RDF graph
        centrality_type: Type of centrality to calculate (degree, closeness, betweenness)
        
    Returns:
        Dictionary mapping node URIs to centrality values
    """
    # Convert RDF graph to NetworkX for centrality calculation
    import networkx as nx
    
    G = nx.DiGraph()
    
    # Add nodes and edges
    for s, p, o in graph:
        if isinstance(o, URIRef):  # Only include URI objects, not literals
            G.add_edge(str(s), str(o), label=str(p))
    
    if centrality_type == "degree":
        # Calculate degree centrality
        centrality = nx.degree_centrality(G)
    elif centrality_type == "closeness":
        # Calculate closeness centrality
        centrality = nx.closeness_centrality(G)
    elif centrality_type == "betweenness":
        # Calculate betweenness centrality
        centrality = nx.betweenness_centrality(G)
    else:
        raise ValueError(f"Unknown centrality type: {centrality_type}")
    
    return centrality
