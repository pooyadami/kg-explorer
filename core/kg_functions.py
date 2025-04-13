# Organize imports by standard library, third-party, and local
import os
import re
import time
import traceback
import logging

# Third-party imports
import owlready2
import pandas as pd
import rdflib
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('kg_functions')

def load_tbox_from_owl(owl_file_path):
    """
    Load the TBox (terminological box) from an OWL file.
    
    Args:
        owl_file_path (str): Path to the OWL file
        
    Returns:
        dict: A dictionary containing TBox components organized by type
    """
    # Load the ontology
    onto = owlready2.get_ontology(owl_file_path).load()
    
    # Extract TBox components
    tbox = {
        "classes": list(onto.classes()),
        "object_properties": list(onto.object_properties()),
        "data_properties": list(onto.data_properties()),
        "annotation_properties": list(onto.annotation_properties()),
        "subclass_relations": []
    }
    
    # Extract subclass relationships
    for cls in onto.classes():
        for parent in cls.is_a:
            if isinstance(parent, owlready2.entity.ThingClass):
                tbox["subclass_relations"].append((cls, parent))
    
    return tbox

def load_mappings_from_ttl(ttl_file_path):
    """
    Load RDF mappings from a TTL file.
    
    Args:
        ttl_file_path (str): Path to the TTL file with mappings
        
    Returns:
        rdflib.Graph: A graph containing the loaded mappings
    """
    # Create a new RDF graph
    g = Graph()
    
    # Parse the TTL file
    g.parse(ttl_file_path, format="turtle")
    
    return g

def apply_mappings_to_kg(kg, mappings_graph):
    """
    Apply mappings from a mappings graph to a knowledge graph.
    
    Args:
        kg (rdflib.Graph): The target knowledge graph to update
        mappings_graph (rdflib.Graph): Graph containing the mappings to apply
        
    Returns:
        rdflib.Graph: The updated knowledge graph
    """
    # Simply add all triples from the mappings graph to the knowledge graph
    # More complex mapping logic can be implemented based on specific requirements
    for s, p, o in mappings_graph:
        kg.add((s, p, o))
    
    return kg

def load_mappings_directory(mappings_dir, data_dir, kg=None, base_uri=None, single_output=False, include_tbox=False, tbox=None):
    """
    Process all mapping files in a directory and build a knowledge graph.
    
    Args:
        mappings_dir (str): Directory containing R2RML mapping files
        data_dir (str): Directory containing CSV data files
        kg (Graph or Dataset, optional): Existing graph or dataset to add triples to
        base_uri (str, optional): Base URI for generated resources
        single_output (bool): If True, create a dataset with named graphs
        include_tbox (bool): If True, include TBox in a separate named graph
        tbox (dict, optional): TBox data to include
        
    Returns:
        Graph or Dataset: The resulting knowledge graph
    """
    from rdflib import Graph, Dataset, URIRef
    import os
    import time
    
    # Initialize statistics
    start_time = time.time()
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    # Create graph if none provided
    if kg is None:
        if single_output:
            kg = Dataset()
        else:
            kg = Graph()
    
    # Get initial triple count
    initial_triple_count = len(kg)
    
    # Find TTL mapping files
    ttl_files = [f for f in os.listdir(mappings_dir) if f.endswith('.ttl')]
    logger.info(f"Found {len(ttl_files)} mapping files")
    
    # Process each mapping file
    for ttl_file in ttl_files:
        file_path = os.path.join(mappings_dir, ttl_file)
        logger.info(f"Processing mapping file: {ttl_file}")
        
        # Extract datasource information
        datasource = extract_datasource_from_mapping(file_path)
        if not datasource:
            logger.warning(f"Skipping: Could not determine datasource for {ttl_file}")
            skipped_count += 1
            continue
        
        # Look for the corresponding CSV file
        csv_file = os.path.join(data_dir, f"{datasource}.csv")
        if not os.path.exists(csv_file):
            logger.warning(f"Skipping: CSV file not found: {csv_file}")
            skipped_count += 1
            continue
        
        logger.info(f"Found data source: {csv_file}")
        
        # Create graph name if using named graphs
        graph_uri = None
        if single_output and base_uri:
            # Remove extension and convert to URI
            graph_name = ttl_file.replace('.ttl', '')
            # Full graph URI
            graph_uri = URIRef(f"{base_uri}{graph_name}")
        
        # Execute the R2RML mapping
        try:
            # Create temporary graph for the mapping result
            temp_graph = Graph()
            temp_graph = execute_r2rml_mapping(file_path, csv_file, temp_graph)
            
            if len(temp_graph) == 0:
                logger.warning(f"No triples generated from {ttl_file}")
                continue
                
            logger.info(f"Generated {len(temp_graph)} triples from mapping")
            
            # Add triples to the knowledge graph
            if single_output and isinstance(kg, Dataset):
                # Add to a named graph
                g = kg.graph(graph_uri)
                for s, p, o in temp_graph:
                    g.add((s, p, o))
                
                # Add TBox to a separate graph if requested
                if include_tbox and tbox and base_uri:
                    tbox_graph_uri = URIRef(f"{base_uri}TBox")
                    g_tbox = kg.graph(tbox_graph_uri)
                    tbox_to_graph(tbox, g_tbox)
                    logger.info(f"Added TBox to graph {tbox_graph_uri}")
            else:
                # Add directly to the main graph
                for s, p, o in temp_graph:
                    kg.add((s, p, o))
            
            processed_count += 1
        except Exception as e:
            logger.error(f"Error applying mapping: {str(e)}")
            logger.error(traceback.format_exc())
            error_count += 1
    
    # Calculate statistics
    duration = time.time() - start_time
    total_triples = len(kg) - initial_triple_count
    
    logger.info(f"Mapping directory processing completed in {duration:.2f} seconds")
    logger.info(f"Processed {processed_count} mapping files")
    logger.info(f"Skipped {skipped_count} mapping files")
    logger.info(f"Encountered {error_count} errors")
    logger.info(f"Added {total_triples} triples to the knowledge graph")
    
    return kg

def extract_datasource_from_mapping(file_path):
    """
    Extract datasource name from a mapping file.
    
    Args:
        file_path (str): Path to the mapping file
        
    Returns:
        str: Extracted datasource name or None if not found
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Try to extract from comments first
        datasource = _extract_datasource_from_comments(content)
        if datasource:
            return datasource
            
        # Fall back to filename extraction
        return _extract_datasource_from_filename(file_path)
    except Exception as e:
        logger.error(f"Error extracting datasource from {file_path}: {str(e)}")
        return None

def _extract_datasource_from_comments(content: str) -> str:
    """Extract datasource from file comments."""
    datasource_match = re.search(r'#\s*datasource:\s*#\s*(\w+)', content)
    if datasource_match:
        datasource = datasource_match.group(1)
        logger.debug(f"Found datasource in comments: {datasource}")
        return datasource
    return None

def _extract_datasource_from_filename(file_path: str) -> str:
    """Extract datasource from filename."""
    base_name = os.path.basename(file_path)
    logger.debug(f"No datasource in comments, using filename: {base_name}")
    
    prefix = "building_connected-"
    if base_name.startswith(prefix):
        datasource = base_name[len(prefix):-4].replace('-', '_')
        logger.debug(f"Extracted datasource from filename: {datasource}")
        return datasource
    return None

def execute_r2rml_mapping(mapping_file, csv_file, kg=None):
    """
    Execute an R2RML mapping on a CSV file and add the results to a knowledge graph.
    
    Args:
        mapping_file (str): Path to the R2RML mapping file
        csv_file (str): Path to the CSV data file
        kg (rdflib.Graph, optional): Existing knowledge graph to update
    
    Returns:
        rdflib.Graph: The updated knowledge graph
    """
    start_time = time.time()
    logger.info(f"Starting R2RML mapping execution: {mapping_file} with {csv_file}")
    
    if kg is None:
        kg = Graph()
        logger.debug("Created new RDF graph")
    else:
        logger.debug(f"Using existing graph with {len(kg)} triples")
    
    initial_triple_count = len(kg)
    
    try:
        # Load CSV data into a pandas DataFrame
        logger.debug(f"Loading CSV data from {csv_file}")
        df = pd.read_csv(csv_file)
        logger.info(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        logger.debug(f"CSV columns: {list(df.columns)}")
        
        # Load the R2RML mapping
        logger.debug(f"Loading R2RML mapping from {mapping_file}")
        mapping_graph = Graph()
        mapping_graph.parse(mapping_file, format="turtle")
        logger.debug(f"Loaded mapping graph with {len(mapping_graph)} triples")
        
        # Define common namespaces
        r2rml = Namespace("http://www.w3.org/ns/r2rml#")
        
        # Keep track of missing columns
        missing_columns = set()
        
        # Find all logical table mappings
        triples_maps = list(mapping_graph.subjects(RDF.type, r2rml.TriplesMap))
        if not triples_maps:
            # Use alternative approach if no explicit TriplesMap types are found
            for s, p, o in mapping_graph.triples((None, r2rml.logicalTable, None)):
                triples_maps.append(s)
            for s, p, o in mapping_graph.triples((None, r2rml.subjectMap, None)):
                triples_maps.append(s)
            triples_maps = list(set(triples_maps))
            
        logger.info(f"Found {len(triples_maps)} triples maps in mapping file")
        
        row_count = 0
        for subj in triples_maps:
            logger.debug(f"Processing triples map: {subj}")
            
            # Get subject map
            subject_map = mapping_graph.value(subj, r2rml.subjectMap)
            if subject_map is None:
                logger.error(f"No subject map found for {subj}")
                continue
                
            subject_template = mapping_graph.value(subject_map, r2rml.template)
            if subject_template is None:
                logger.error(f"No template found in subject map {subject_map}")
                continue
                
            subject_template = str(subject_template)
            
            # Check subject template for columns that don't exist
            for column_match in re.finditer(r'\{([^}]+)\}', subject_template):
                column = column_match.group(1)
                if column not in df.columns:
                    missing_columns.add(column)
            
            # IMPORTANT FIX: Properly extract class information
            # First check with rr:class (standard R2RML)
            classes = list(mapping_graph.objects(subject_map, r2rml.class_))
            
            # If no classes found, try :class (some R2RML implementations use this)
            if not classes:
                alt_class = Namespace("http://www.w3.org/ns/r2rml#").term("class")
                classes = list(mapping_graph.objects(subject_map, alt_class))
            
            # If still no classes, check if there's a custom namespace used
            if not classes:
                for p, o in mapping_graph.predicate_objects(subject_map):
                    if p.n3().endswith('class>'):
                        classes.append(o)
            
            # Debug info
            if classes:
                logger.debug(f"Found class declarations: {classes}")
            else:
                logger.warning(f"No class declarations found for subject map {subject_map}")
            
            # Process each row in the CSV
            for index, row in df.iterrows():
                row_count += 1
                
                # Create subject URI from template
                subject_uri = subject_template
                missing_placeholders = False
                
                # Find all placeholders and check if they can be resolved
                for column_match in re.finditer(r'\{([^}]+)\}', subject_template):
                    column = column_match.group(1)
                    if column not in df.columns:
                        missing_columns.add(column)
                        missing_placeholders = True
                        break
                    if pd.isna(row[column]):
                        missing_placeholders = True
                        break
                
                # Skip this row entirely if subject URI can't be created
                if missing_placeholders:
                    logger.debug(f"Row {index}: Skipping row due to unresolvable placeholder in subject template")
                    continue
                
                # Replace placeholders in subject template
                for col in df.columns:
                    placeholder = "{" + col + "}"
                    if placeholder in subject_uri and not pd.isna(row[col]):
                        subject_uri = subject_uri.replace(placeholder, str(row[col]))
                
                # Verify all placeholders were replaced
                if re.search(r'\{([^}]+)\}', subject_uri):
                    logger.warning(f"Row {index}: Not all placeholders replaced in subject URI: {subject_uri}, skipping row")
                    continue
                
                subject = URIRef(subject_uri)
                
                # IMPORTANT FIX: Add class assertions
                for class_uri in classes:
                    kg.add((subject, RDF.type, class_uri))
                    logger.debug(f"Added type assertion: {subject} rdf:type {class_uri}")
                
                # Process predicate-object maps
                po_maps = list(mapping_graph.objects(subj, r2rml.predicateObjectMap))
                
                for po_map in po_maps:
                    predicate = mapping_graph.value(po_map, r2rml.predicate)
                    if predicate is None:
                        continue
                        
                    obj_map = mapping_graph.value(po_map, r2rml.objectMap)
                    if obj_map is None:
                        continue
                    
                    # Check if there's a column reference
                    column = mapping_graph.value(obj_map, r2rml.column)
                    if column:
                        column = str(column)
                        
                        # Check if column exists in CSV
                        if column not in df.columns:
                            missing_columns.add(column)
                            continue
                            
                        if pd.isna(row[column]):
                            continue
                            
                        datatype = mapping_graph.value(obj_map, r2rml.datatype)
                        if datatype:
                            # Convert to appropriate datatype
                            try:
                                literal = Literal(row[column], datatype=datatype)
                                kg.add((subject, predicate, literal))
                            except Exception as e:
                                logger.error(f"Error creating typed literal for {column}: {str(e)}")
                        else:
                            literal = Literal(row[column])
                            kg.add((subject, predicate, literal))
                    
                    # Check if there's a template
                    template = mapping_graph.value(obj_map, r2rml.template)
                    if template:
                        template_str = str(template)
                        original_template = template_str
                        
                        # Check template for columns that don't exist
                        missing_placeholders = False
                        placeholders_found = []
                        
                        # Find all placeholders in the template
                        for column_match in re.finditer(r'\{([^}]+)\}', template_str):
                            column = column_match.group(1)
                            placeholders_found.append(column)
                            
                            if column not in df.columns:
                                missing_columns.add(column)
                                missing_placeholders = True
                                break
                            
                            # If column exists but value is null/NA for this row, also skip
                            if pd.isna(row[column]):
                                missing_placeholders = True
                                break
                        
                        # Skip this triple if any placeholder can't be resolved
                        if missing_placeholders:
                            logger.debug(f"Row {index}: Skipping template with unresolvable placeholder: {template_str}")
                            continue
                        
                        # Replace all placeholders with actual values
                        object_uri = template_str
                        for col in placeholders_found:
                            placeholder = "{" + col + "}"
                            object_uri = object_uri.replace(placeholder, str(row[col]))
                        
                        # Verify all placeholders were replaced
                        if re.search(r'\{([^}]+)\}', object_uri):
                            logger.warning(f"Row {index}: Not all placeholders replaced in: {object_uri}, skipping triple")
                            continue
                            
                        kg.add((subject, predicate, URIRef(object_uri)))
        
        # Log missing columns if any were found
        if missing_columns:
            logger.warning(f"The following columns referenced in the mapping were not found in the CSV: {', '.join(missing_columns)}")
        
        # Calculate statistics
        duration = time.time() - start_time
        new_triples = len(kg) - initial_triple_count
        
        logger.info(f"R2RML mapping completed in {duration:.2f} seconds")
        logger.info(f"Processed {row_count} rows")
        logger.info(f"Added {new_triples} new triples to the graph")
        
        return kg
        
    except Exception as e:
        logger.error(f"Error executing R2RML mapping: {str(e)}")
        logger.error(traceback.format_exc())
        return kg

def load_ontology(owl_file_path, kg=None):
    """Adapter function for load_tbox_from_owl"""
    if kg is None:
        kg = Graph()
    
    # Load the TBox
    tbox = load_tbox_from_owl(owl_file_path)
    
    # You could add the classes, properties to the graph if needed
    return kg

def load_r2rml_mapping(r2rml_file_path):
    """Adapter function for load_mappings_from_ttl"""
    return load_mappings_from_ttl(r2rml_file_path)

def build_kg_from_csv_and_r2rml(csv_file_path, r2rml_file_path, kg=None):
    """Adapter function for execute_r2rml_mapping"""
    return execute_r2rml_mapping(r2rml_file_path, csv_file_path, kg)

def export_graph(graph, output_path, format='turtle'):
    """Export an RDF graph to a file"""
    graph.serialize(destination=output_path, format=format)
    return True

def tbox_to_graph(tbox, kg=None):
    """
    Convert a TBox dictionary to an RDF graph.
    
    Args:
        tbox (dict): The TBox dictionary from load_tbox_from_owl
        kg (rdflib.Graph, optional): Existing graph to add TBox to
        
    Returns:
        rdflib.Graph: Graph with TBox data
    """
    if kg is None:
        kg = Graph()
    
    # Add namespace bindings
    owl = Namespace("http://www.w3.org/2002/07/owl#")
    rdfs = Namespace("http://www.w3.org/2000/01/rdf-schema#")
    kg.bind('owl', owl)
    kg.bind('rdfs', rdfs)
    
    # Add classes
    for cls in tbox['classes']:
        cls_uri = URIRef(cls.iri)
        kg.add((cls_uri, RDF.type, owl.Class))
        if hasattr(cls, 'label') and cls.label:
            kg.add((cls_uri, rdfs.label, Literal(cls.label)))
    
    # Add object properties
    for prop in tbox['object_properties']:
        prop_uri = URIRef(prop.iri)
        kg.add((prop_uri, RDF.type, owl.ObjectProperty))
        if hasattr(prop, 'label') and prop.label:
            kg.add((prop_uri, rdfs.label, Literal(prop.label)))
    
    # Add data properties
    for prop in tbox['data_properties']:
        prop_uri = URIRef(prop.iri)
        kg.add((prop_uri, RDF.type, owl.DatatypeProperty))
        if hasattr(prop, 'label') and prop.label:
            kg.add((prop_uri, rdfs.label, Literal(prop.label)))
    
    # Add subclass relationships
    for cls, parent in tbox['subclass_relations']:
        cls_uri = URIRef(cls.iri)
        parent_uri = URIRef(parent.iri)
        kg.add((cls_uri, rdfs.subClassOf, parent_uri))
    
    return kg

def execute_sparql_query(graph, query_string):
    """
    Execute a SPARQL query against an RDF graph.
    
    Args:
        graph (rdflib.Graph): The RDF graph to query
        query_string (str): The SPARQL query string
    
    Returns:
        list: List of query results as dictionaries
    """
    results = []
    qres = graph.query(query_string)
    
    # Convert results to dictionaries
    for row in qres:
        result_dict = {}
        for var in qres.vars:
            value = row.get(var)
            if value is not None:
                result_dict[var] = value
        results.append(result_dict)
    
    return results
