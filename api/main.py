# kg_platform/api/main.py

from fastapi import FastAPI, HTTPException, Query, Body, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import os
import json
from typing import List, Optional, Dict, Any
from rdflib import Graph, URIRef
import sys
from rdflib.namespace import RDF, RDFS, OWL
import openai
from dotenv import load_dotenv

# Import KG functions
sys.path.append(str(Path(__file__).parent.parent))
from core.kg_functions import execute_sparql_query
from utils.visualization import create_cytoscape_graph_data, create_d3_graph_data, filter_graph_by_type, extract_subgraph
from scripts.visualize_kg import create_html_visualization, load_graph

app = FastAPI(
    title="Knowledge Graph Access API",
    description="API for accessing and visualizing pre-created knowledge graphs",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
KG_DIRECTORY = os.path.join(Path(__file__).parent.parent, "outputs")

# In-memory store for loaded graphs
kg_store = {}

@app.get("/", response_class=HTMLResponse)
async def root(
    limit_nodes: int = Query(0, description="Limit the number of nodes (0 for no limit)"),
    aggregate_by_class: bool = Query(False, description="Aggregate nodes by class"),
    aggregate_edges: bool = Query(False, description="Aggregate edges by type")
):
    """Root endpoint that directly shows the visualization"""
    try:
        # Get the first available graph file
        graph_files = []
        if os.path.exists(KG_DIRECTORY):
            for file in os.listdir(KG_DIRECTORY):
                if file.endswith(('.nq', '.ttl', '.nt', '.trig')):
                    graph_files.append(file)
        
        if not graph_files:
            return """
            <html>
                <body>
                    <h1>Knowledge Graph Visualization</h1>
                    <p>No graph files found in the outputs directory. Please create some knowledge graphs first.</p>
                </body>
            </html>
            """
        
        # Prefer combined_kg.nq if it exists
        graph_id = "combined_kg.nq" if "combined_kg.nq" in graph_files else graph_files[0]
        
        # Get visualization data
        graph = load_graph(os.path.join(KG_DIRECTORY, graph_id))
        
        # Apply node limit if specified
        if limit_nodes > 0:
            from scripts.visualize_kg import limit_graph_size
            graph = limit_graph_size(graph, limit_nodes)
        
        # Generate visualization data
        vis_data = create_cytoscape_graph_data(
            graph, 
            aggregate_by_class=aggregate_by_class,
            aggregate_edges=aggregate_edges
        )
        
        # Create HTML
        html_content = create_html_visualization(vis_data, "cytoscape")
        return html_content
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"""
        <html>
            <body>
                <h1>Error loading visualization</h1>
                <p>{str(e)}</p>
                <pre>{error_details}</pre>
                <p>Try accessing <a href="/graphs">the list of available graphs</a> instead.</p>
            </body>
        </html>
        """

@app.get("/graphs")
async def list_graphs():
    """List all available knowledge graph files"""
    try:
        graph_files = []
        if os.path.exists(KG_DIRECTORY):
            for file in os.listdir(KG_DIRECTORY):
                if file.endswith(('.nq', '.ttl', '.nt', '.trig')):
                    size = os.path.getsize(os.path.join(KG_DIRECTORY, file))
                    graph_files.append({
                        "filename": file,
                        "size": size,
                        "path": os.path.join(KG_DIRECTORY, file)
                    })
        return graph_files
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing graphs: {str(e)}")

@app.get("/graphs/{graph_id}")
async def get_graph_info(graph_id: str):
    """Get information about a specific graph"""
    try:
        # Load the graph if not already loaded
        if graph_id not in kg_store:
            await load_graph_to_store(graph_id)
            
        graph = kg_store[graph_id]
        
        # Count triples
        triple_count = len(graph)
        
        # Count entity types
        types_query = """
        SELECT ?type (COUNT(?s) as ?count)
        WHERE {
            ?s a ?type .
        }
        GROUP BY ?type
        ORDER BY DESC(?count)
        """
        types = execute_sparql_query(graph, types_query)
        
        # Count predicates
        predicates_query = """
        SELECT ?p (COUNT(*) as ?count)
        WHERE {
            ?s ?p ?o .
        }
        GROUP BY ?p
        ORDER BY DESC(?count)
        """
        predicates = execute_sparql_query(graph, predicates_query)
        
        return {
            "id": graph_id,
            "triple_count": triple_count,
            "entity_types": types,
            "predicates": predicates
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting graph info: {str(e)}")

@app.post("/graphs/{graph_id}/query")
async def query_graph(graph_id: str, query: str = Body(..., embed=True)):
    """Execute a SPARQL query against a graph"""
    try:
        # Load the graph if not already loaded
        if graph_id not in kg_store:
            await load_graph_to_store(graph_id)
            
        graph = kg_store[graph_id]
        
        # Execute the query
        results = execute_sparql_query(graph, query)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing query: {str(e)}")

@app.get("/graphs/{graph_id}/visualization")
async def get_visualization(
    graph_id: str, 
    vis_type: str = Query("cytoscape", enum=["cytoscape", "d3"]),
    aggregate_by_class: bool = Query(False),
    aggregate_edges: bool = Query(False),
    limit_nodes: int = Query(0),
    entity_uri: Optional[str] = Query(None),
    entity_types: Optional[List[str]] = Query(None)
):
    """Get visualization data for a graph"""
    try:
        # Load the graph if not already loaded
        if graph_id not in kg_store:
            await load_graph_to_store(graph_id)
            
        graph = kg_store[graph_id]
        
        # Apply filters if specified
        if entity_uri:
            graph = extract_subgraph(graph, entity_uri, depth=1)
            
        if entity_types:
            graph = filter_graph_by_type(graph, entity_types)
            
        # Apply node limit if specified
        if limit_nodes > 0:
            # Import the function here to avoid circular imports
            from scripts.visualize_kg import limit_graph_size
            graph = limit_graph_size(graph, limit_nodes)
        
        # Generate visualization data
        if vis_type == "d3":
            vis_data = create_d3_graph_data(graph)
        else:
            vis_data = create_cytoscape_graph_data(
                graph, 
                aggregate_by_class=aggregate_by_class,
                aggregate_edges=aggregate_edges
            )
            
        return vis_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating visualization: {str(e)}")

@app.get("/graphs/{graph_id}/visualization-html", response_class=HTMLResponse)
async def get_visualization_html(
    graph_id: str, 
    vis_type: str = Query("cytoscape", enum=["cytoscape", "d3"]),
    aggregate_by_class: bool = Query(False),
    limit_nodes: int = Query(0),
    entity_uri: Optional[str] = Query(None),
    entity_types: Optional[List[str]] = Query(None)
):
    """Get HTML visualization for a graph"""
    try:
        # First get the visualization data
        vis_data = await get_visualization(
            graph_id, 
            vis_type, 
            aggregate_by_class, 
            limit_nodes, 
            entity_uri, 
            entity_types
        )
        
        # Create HTML
        html_content = create_html_visualization(vis_data, vis_type)
        return html_content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating HTML visualization: {str(e)}")

@app.get("/graphs/{graph_id}/entity/{entity_uri}")
async def get_entity_info(graph_id: str, entity_uri: str):
    """Get information about a specific entity"""
    try:
        # Load the graph if not already loaded
        if graph_id not in kg_store:
            await load_graph_to_store(graph_id)
            
        graph = kg_store[graph_id]
        
        # Create URI object
        entity = URIRef(entity_uri)
        
        # Get entity types
        types = [str(o) for o in graph.objects(entity, URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"))]
        
        # Get outgoing relationships
        outgoing = []
        for p, o in graph.predicate_objects(entity):
            if isinstance(o, URIRef):
                outgoing.append({
                    "predicate": str(p),
                    "object": str(o)
                })
            else:
                outgoing.append({
                    "predicate": str(p),
                    "value": str(o)
                })
        
        # Get incoming relationships
        incoming = []
        for s, p in graph.subject_predicates(entity):
            incoming.append({
                "subject": str(s),
                "predicate": str(p)
            })
        
        return {
            "uri": entity_uri,
            "types": types,
            "outgoing_relationships": outgoing,
            "incoming_relationships": incoming
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting entity info: {str(e)}")

@app.get("/graphs/{graph_id}/entity-network/{entity_uri}")
async def get_entity_network(
    graph_id: str, 
    entity_uri: str,
    depth: int = Query(1, ge=1, le=3)
):
    """Get a network around a specific entity"""
    try:
        # Load the graph if not already loaded
        if graph_id not in kg_store:
            await load_graph_to_store(graph_id)
            
        graph = kg_store[graph_id]
        
        # Extract subgraph
        subgraph = extract_subgraph(graph, entity_uri, depth=depth)
        
        # Generate visualization data
        vis_data = create_cytoscape_graph_data(subgraph)
        
        return vis_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting entity network: {str(e)}")

@app.get("/graphs/{graph_id}/class/{class_uri}")
async def get_class_visualization(
    graph_id: str,
    class_uri: str,
    limit_nodes: int = Query(100, ge=0),
    vis_type: str = Query("cytoscape", enum=["cytoscape", "d3"])
):
    """Show detailed visualization of a specific class"""
    try:
        # Load the graph if not already loaded
        if graph_id not in kg_store:
            await load_graph_to_store(graph_id)
            
        graph = kg_store[graph_id]
        
        # Filter to only include entities of the specified class
        filtered_graph = filter_graph_by_type(graph, [class_uri])
        
        # Apply node limit if needed
        if limit_nodes > 0 and len(filtered_graph) > limit_nodes * 3:  # Rough estimate
            from scripts.visualize_kg import limit_graph_size
            filtered_graph = limit_graph_size(filtered_graph, limit_nodes)
        
        # Generate visualization data
        if vis_type == "d3":
            vis_data = create_d3_graph_data(filtered_graph)
        else:
            vis_data = create_cytoscape_graph_data(filtered_graph, aggregate_by_class=False)
            
        # Create HTML
        html_content = create_html_visualization(vis_data, vis_type)
        
        # Add navigation back to class view
        back_link = f"""
        <div style="position: fixed; top: 15px; right: 15px; z-index: 1000; background: rgba(255,255,255,0.8); padding: 10px; border-radius: 5px;">
            <a href="/graphs/{graph_id}/visualization-html?aggregate_by_class=true" target="_blank">Back to Class Overview</a>
        </div>
        """
        html_content = html_content.replace('</body>', f'{back_link}</body>')
        
        return HTMLResponse(content=html_content)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating class visualization: {str(e)}")

@app.post("/graphs/{graph_id}/chat")
async def chat_with_kg(
    graph_id: str,
    query: Dict[str, str] = Body(...)
):
    """
    Chat with the knowledge graph using natural language
    
    Args:
        graph_id: ID of the graph to query
        query: JSON object with a "question" field containing the natural language query
        
    Returns:
        JSON response with answer and supporting data
    """
    try:
        # Load the graph if not already loaded
        if graph_id not in kg_store:
            await load_graph_to_store(graph_id)
            
        graph = kg_store[graph_id]
        
        # Get the question from the request body
        question = query.get("question", "")
        if not question:
            raise HTTPException(status_code=400, detail="Question is required")
        
        # Extract schema information from the graph
        schema_info = extract_schema_information(graph)
        
        # Generate SPARQL query from natural language using schema information
        sparql_query = natural_language_to_sparql(question, schema_info)
        
        # Execute the generated query
        try:
            results = execute_sparql_query(graph, sparql_query)
            
            # Format the answer
            answer = format_query_results(results)
            
            return {
                "question": question,
                "sparql_query": sparql_query,
                "answer": answer,
                "raw_results": results
            }
        except Exception as query_error:
            # If the generated query fails, try a different approach or inform the user
            return {
                "question": question,
                "error": f"Failed to execute generated query: {str(query_error)}",
                "sparql_query": sparql_query,
                "suggestion": "Your question might need to be more specific or use different terminology."
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

def extract_schema_information(graph):
    """
    Extract schema information from the knowledge graph
    
    Args:
        graph: RDFLib Graph object
        
    Returns:
        Dictionary containing classes, data properties, and object properties
    """
    schema_info = {
        "classes": [],
        "data_properties": {},
        "object_properties": []
    }
    
    # Find all classes in the graph
    class_query = """
    SELECT DISTINCT ?class ?label ?comment
    WHERE {
      {?class a owl:Class} UNION {?instance a ?class}
      OPTIONAL {?class rdfs:label ?label}
      OPTIONAL {?class rdfs:comment ?comment}
    }
    ORDER BY ?class
    """
    
    class_results = execute_sparql_query(graph, class_query)
    
    # Process class results
    for result in class_results:
        class_uri = str(result.get('class', ''))
        if not class_uri:
            continue
            
        class_name = class_uri.split('#')[-1] if '#' in class_uri else class_uri.split('/')[-1]
        
        class_info = {
            "uri": class_uri,
            "name": class_name,
            "label": str(result.get('label', class_name)),
            "comment": str(result.get('comment', '')),
            "instance_count": count_instances(graph, class_uri)
        }
        
        schema_info["classes"].append(class_info)
        
        # Initialize data properties dictionary for this class
        schema_info["data_properties"][class_uri] = []
    
    # Find object properties
    object_property_query = """
    SELECT DISTINCT ?prop ?label ?domain ?range ?comment
    WHERE {
      {?prop a owl:ObjectProperty} 
      UNION 
      {?s ?prop ?o . 
       FILTER(isIRI(?o) && ?prop != rdf:type)}
      
      OPTIONAL {?prop rdfs:label ?label}
      OPTIONAL {?prop rdfs:domain ?domain}
      OPTIONAL {?prop rdfs:range ?range}
      OPTIONAL {?prop rdfs:comment ?comment}
    }
    ORDER BY ?prop
    """
    
    op_results = execute_sparql_query(graph, object_property_query)
    
    # Process object property results
    for result in op_results:
        prop_uri = str(result.get('prop', ''))
        if not prop_uri or prop_uri in [RDF.type, RDFS.label, RDFS.comment]:
            continue
            
        prop_name = prop_uri.split('#')[-1] if '#' in prop_uri else prop_uri.split('/')[-1]
        domain = str(result.get('domain', ''))
        range_val = str(result.get('range', ''))
        
        prop_info = {
            "uri": prop_uri,
            "name": prop_name,
            "label": str(result.get('label', prop_name)),
            "domain": domain,
            "range": range_val,
            "comment": str(result.get('comment', ''))
        }
        
        schema_info["object_properties"].append(prop_info)
    
    # Find data properties
    data_property_query = """
    SELECT DISTINCT ?prop ?label ?domain ?range ?comment
    WHERE {
      {?prop a owl:DatatypeProperty}
      UNION
      {?s ?prop ?o . 
       FILTER(isLiteral(?o) && ?prop != rdfs:label && ?prop != rdfs:comment)}
      
      OPTIONAL {?prop rdfs:label ?label}
      OPTIONAL {?prop rdfs:domain ?domain}
      OPTIONAL {?prop rdfs:range ?range}
      OPTIONAL {?prop rdfs:comment ?comment}
    }
    ORDER BY ?prop
    """
    
    dp_results = execute_sparql_query(graph, data_property_query)
    
    # Process data property results
    for result in dp_results:
        prop_uri = str(result.get('prop', ''))
        if not prop_uri or prop_uri in [RDFS.label, RDFS.comment]:
            continue
            
        prop_name = prop_uri.split('#')[-1] if '#' in prop_uri else prop_uri.split('/')[-1]
        domain = str(result.get('domain', ''))
        
        prop_info = {
            "uri": prop_uri,
            "name": prop_name,
            "label": str(result.get('label', prop_name)),
            "domain": domain,
            "range": str(result.get('range', '')),
            "comment": str(result.get('comment', ''))
        }
        
        # If we have a domain, add this property to that class's data properties
        if domain and domain in schema_info["data_properties"]:
            schema_info["data_properties"][domain].append(prop_info)
        else:
            # If no domain is specified, we need to infer it by looking at usage
            usage_query = f"""
            SELECT DISTINCT ?class
            WHERE {{
              ?instance a ?class .
              ?instance <{prop_uri}> ?value .
            }}
            LIMIT 10
            """
            
            try:
                usage_results = execute_sparql_query(graph, usage_query)
                for usage in usage_results:
                    class_uri = str(usage.get('class', ''))
                    if class_uri and class_uri in schema_info["data_properties"]:
                        # Add this property to the inferred class
                        schema_info["data_properties"][class_uri].append(prop_info)
            except:
                # If the query fails, add to a generic list
                if "unclassified" not in schema_info["data_properties"]:
                    schema_info["data_properties"]["unclassified"] = []
                schema_info["data_properties"]["unclassified"].append(prop_info)
    
    # Format schema for prompt
    formatted_schema = format_schema_for_prompt(schema_info)
    return formatted_schema

def count_instances(graph, class_uri):
    """Count the number of instances of a given class"""
    query = f"""
    SELECT (COUNT(DISTINCT ?instance) as ?count)
    WHERE {{
      ?instance a <{class_uri}> .
    }}
    """
    
    try:
        results = execute_sparql_query(graph, query)
        if results and results[0].get('count'):
            return int(results[0]['count'])
        return 0
    except:
        return 0

def format_schema_for_prompt(schema_info):
    """Format the schema information for inclusion in the LLM prompt"""
    formatted = "KNOWLEDGE GRAPH SCHEMA:\n\n"
    
    # Add classes
    formatted += "CLASSES:\n"
    for cls in schema_info["classes"]:
        formatted += f"- {cls['name']} ({cls['instance_count']} instances)"
        if cls['comment'] and len(cls['comment']) > 0 and cls['comment'] != 'None':
            # Truncate long comments
            comment = cls['comment']
            if len(comment) > 100:
                comment = comment[:97] + "..."
            formatted += f": {comment}"
        formatted += "\n"
    
    formatted += "\nOBJECT PROPERTIES (relationships between entities):\n"
    for prop in schema_info["object_properties"]:
        domain_name = prop['domain'].split('#')[-1] if '#' in prop['domain'] else prop['domain'].split('/')[-1]
        range_name = prop['range'].split('#')[-1] if '#' in prop['range'] else prop['range'].split('/')[-1]
        
        if domain_name and range_name and domain_name != 'None' and range_name != 'None':
            formatted += f"- {prop['name']}: connects {domain_name} to {range_name}\n"
        else:
            formatted += f"- {prop['name']}\n"
    
    formatted += "\nDATA PROPERTIES (attributes of entities):\n"
    for class_uri, properties in schema_info["data_properties"].items():
        if not properties:
            continue
            
        class_name = class_uri.split('#')[-1] if '#' in class_uri else class_uri.split('/')[-1]
        if class_name == "unclassified":
            formatted += "\nUnclassified properties:\n"
        else:
            formatted += f"\nProperties of {class_name}:\n"
            
        for prop in properties:
            formatted += f"- {prop['name']}"
            if prop['range'] and prop['range'] != 'None':
                range_name = prop['range'].split('#')[-1] if '#' in prop['range'] else prop['range'].split('/')[-1]
                formatted += f" (type: {range_name})"
            formatted += "\n"
    
    return formatted

def natural_language_to_sparql(question, schema_info):
    """
    Convert natural language question to SPARQL query using an LLM
    
    Args:
        question: The natural language question
        schema_info: Formatted schema information from the knowledge graph
        
    Returns:
        SPARQL query string
    """
    prompt = f"""
You are a SPARQL query generator for a knowledge graph. Convert the following question into a valid SPARQL query.

{schema_info}

Based on the schema above, please generate a valid SPARQL query that answers this question:
"{question}"

Remember:
1. Use the exact class and property names as shown in the schema
2. Make sure to use proper prefixes or full URIs
3. Include appropriate FILTER, OPTIONAL, or GROUP BY clauses as needed
4. Limit results to a reasonable number (e.g., LIMIT 25) unless counting or aggregating

Return ONLY the SPARQL query without any explanation or markdown formatting. The query should be executable as-is.
"""
    
    try:
        # Using OpenAI API to generate SPARQL query
        response = openai.ChatCompletion.create(
            model="gpt-4",  # or another appropriate model
            messages=[
                {"role": "system", "content": "You are a SPARQL query generator that converts natural language to precise SPARQL queries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,  # Lower temperature for more precise outputs
            max_tokens=700
        )
        
        # Extract the generated SPARQL query
        sparql_query = response.choices[0].message.content.strip()
        
        # Remove markdown code blocks if present
        if sparql_query.startswith("```") and sparql_query.endswith("```"):
            sparql_query = sparql_query[3:-3].strip()
        if sparql_query.startswith("```sparql"):
            sparql_query = sparql_query[9:].strip()
            if sparql_query.endswith("```"):
                sparql_query = sparql_query[:-3].strip()
        
        return sparql_query
    
    except Exception as e:
        print(f"Error generating SPARQL: {str(e)}")
        # Fallback to a simple query if generation fails
        return f"""
SELECT ?s ?p ?o 
WHERE {{ 
    ?s ?p ?o . 
    FILTER(CONTAINS(LCASE(STR(?o)), LCASE("{question}"))) 
}}
LIMIT 20
"""

# ... existing format_query_results function ...

async def load_graph_to_store(graph_id: str):
    """Helper function to load a graph into memory"""
    try:
        # Find the file path
        file_path = None
        if os.path.exists(os.path.join(KG_DIRECTORY, graph_id)):
            file_path = os.path.join(KG_DIRECTORY, graph_id)
        else:
            for file in os.listdir(KG_DIRECTORY):
                if file.startswith(graph_id):
                    file_path = os.path.join(KG_DIRECTORY, file)
                    break
                    
        if not file_path:
            raise HTTPException(status_code=404, detail=f"Graph '{graph_id}' not found")
        
        # Load the graph
        kg_store[graph_id] = load_graph(file_path)
        return True
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading graph: {str(e)}")

@app.get("/test-graph-visualization/{graph_id}")
async def test_graph_visualization(graph_id: str):
    """Test endpoint to debug graph visualization issues"""
    try:
        # Load the graph if not already loaded
        if graph_id not in kg_store:
            await load_graph_to_store(graph_id)
            
        graph = kg_store[graph_id]
        
        # Get basic graph stats
        triple_count = len(graph)
        
        # Count entities (subjects that have a type)
        entity_query = """
        SELECT (COUNT(DISTINCT ?s) as ?count)
        WHERE {
            ?s a ?type .
        }
        """
        entity_results = execute_sparql_query(graph, entity_query)
        entity_count = int(entity_results[0].get('count', 0)) if entity_results else 0
        
        # Sample a few entities
        sample_query = """
        SELECT ?s ?type
        WHERE {
            ?s a ?type .
        }
        LIMIT 5
        """
        sample_results = execute_sparql_query(graph, sample_query)
        
        # Create test visualization data
        from utils.visualization import create_cytoscape_graph_data
        
        # Create small test graph with just a few entities
        test_graph = Graph()
        for result in sample_results:
            s = result.get('s')
            type_uri = result.get('type')
            if s and type_uri:
                test_graph.add((s, RDF.type, type_uri))
                # Get a few properties for each entity
                for p, o in graph.predicate_objects(s):
                    test_graph.add((s, p, o))
                    if len(test_graph) > 50:  # Limit to 50 triples
                        break
        
        # Create visualization data
        viz_data = create_cytoscape_graph_data(test_graph, aggregate_by_class=False)
        
        return {
            "graph_id": graph_id,
            "triple_count": triple_count,
            "entity_count": entity_count,
            "test_graph_size": len(test_graph),
            "visualization_data": viz_data,
            "sample_entities": sample_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error testing graph: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)