import argparse
from pathlib import Path
import sys
import json
from rdflib import Graph, Dataset, URIRef
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.visualization import create_d3_graph_data, create_cytoscape_graph_data, filter_graph_by_type, extract_subgraph, get_graph_degree_distribution

def load_graph(file_path):
    """
    Load an RDF graph from file
    
    Args:
        file_path: Path to the N-Quads or other RDF file
        
    Returns:
        Loaded RDF graph
    """
    print(f"Loading graph from {file_path}...")
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    # Different loading approach based on file type
    if file_ext == '.nq':
        # For N-Quads, load into a dataset then merge graphs
        dataset = Dataset()
        dataset.parse(file_path, format='nquads')
        
        # Merge all named graphs into one for visualization
        graph = Graph()
        for g in dataset.graphs():
            for s, p, o in g:
                graph.add((s, p, o))
                
        print(f"Loaded {len(graph)} triples from dataset with {len(list(dataset.graphs()))} graphs")
        return graph
    else:
        # For other formats like Turtle, N3, etc.
        graph = Graph()
        graph.parse(file_path)
        print(f"Loaded {len(graph)} triples")
        return graph

def create_html_visualization(graph_data, vis_type="d3", includes_form=False, current_params=None):
    """
    Create an HTML file with the visualization
    
    Args:
        graph_data: Prepared data for visualization
        vis_type: Type of visualization ('d3' or 'cytoscape')
        includes_form: Whether to include a form for adjusting parameters
        current_params: Current parameter values for the form
        
    Returns:
        HTML content as string
    """
    # Set default current_params if not provided
    if current_params is None:
        current_params = {
            "limit_nodes": 0,
            "entity_uri": None,
            "entity_types": None,
            "aggregate_by_class": False,
            "aggregate_edges": False
        }
    
    # Debug the graph_data
    print(f"Graph data type: {type(graph_data)}")
    print(f"Graph data has nodes: {'nodes' in graph_data}")
    print(f"Graph data has edges: {'edges' in graph_data}")
    if 'nodes' in graph_data:
        print(f"Number of nodes: {len(graph_data['nodes'])}")
    if 'edges' in graph_data:
        print(f"Number of edges: {len(graph_data['edges'])}")
    
    # Make sure graph_data is properly formatted
    if not isinstance(graph_data, dict):
        print(f"WARNING: graph_data is not a dict! Type: {type(graph_data)}")
        graph_data = {'nodes': [], 'edges': []}
    
    if 'nodes' not in graph_data or 'edges' not in graph_data:
        print("WARNING: graph_data missing nodes or edges keys!")
        if 'nodes' not in graph_data:
            graph_data['nodes'] = []
        if 'edges' not in graph_data:
            graph_data['edges'] = []
    
    if not graph_data.get('nodes') or len(graph_data.get('nodes', [])) == 0:
        print("WARNING: No nodes in graph_data! Adding sample data for testing.")
        # Add sample data for testing
        graph_data = {
            'nodes': [
                {'data': {'id': 'sample1', 'label': 'Sample Node 1', 'type': 'TestClass', 'color': '#4287f5', 'size': 30}},
                {'data': {'id': 'sample2', 'label': 'Sample Node 2', 'type': 'TestClass', 'color': '#4287f5', 'size': 30}},
            ],
            'edges': [
                {'data': {'id': 'e1', 'source': 'sample1', 'target': 'sample2', 'label': 'testRelation', 'width': 2}}
            ]
        }
        print(f"Added sample data with {len(graph_data['nodes'])} nodes and {len(graph_data['edges'])} edges")
    
    # Generate control form HTML if requested
    control_form_html = ""
    if includes_form:
        control_form_html = f"""
        <div id="control-panel" style="position: absolute; top: 10px; right: 10px; background: white; padding: 15px; border: 1px solid #ccc; border-radius: 5px; z-index: 10; max-width: 300px; box-shadow: 0 0 10px rgba(0,0,0,0.1);">
            <h3>Visualization Controls</h3>
            <form action="/" method="get">
                <div style="margin-bottom: 10px;">
                    <label for="limit_nodes">Limit nodes:</label>
                    <input type="number" id="limit_nodes" name="limit_nodes" value="{current_params.get('limit_nodes', 0)}" min="0" style="width: 80px;">
                    <span style="color: #666; font-size: 0.8em;">(0 = no limit)</span>
                </div>
                
                <div style="margin-bottom: 10px;">
                    <label for="entity_uri">Focus entity URI:</label>
                    <input type="text" id="entity_uri" name="entity_uri" value="{current_params.get('entity_uri', '')}" style="width: 100%;">
                </div>
                
                <div style="margin-bottom: 10px;">
                    <label>Aggregation options:</label><br>
                    <label>
                        <input type="checkbox" name="aggregate_by_class" {' checked' if current_params.get('aggregate_by_class') else ''}>
                        Aggregate by class
                    </label><br>
                    <label>
                        <input type="checkbox" name="aggregate_edges" {' checked' if current_params.get('aggregate_edges') else ''}>
                        Aggregate edges
                    </label>
                </div>
                
                <div style="margin-bottom: 10px;">
                    <input type="submit" value="Apply Changes" style="padding: 5px 10px; background: #4CAF50; color: white; border: none; border-radius: 3px; cursor: pointer;">
                    <button type="button" onclick="document.getElementById('control-panel').style.display='none'" style="padding: 5px 10px; background: #f44336; color: white; border: none; border-radius: 3px; cursor: pointer; margin-left: 5px;">Hide</button>
                </div>
            </form>
        </div>
        """
    
    if vis_type == "d3":
        # D3.js force-directed graph visualization
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Knowledge Graph Visualization</title>
            <script src="https://d3js.org/d3.v7.min.js"></script>
            <style>
                body { margin: 0; font-family: Arial, sans-serif; }
                .links line { stroke: #999; stroke-opacity: 0.6; }
                .nodes circle { stroke: #fff; stroke-width: 1.5px; }
                .node-labels { font-size: 10px; }
                .link-labels { font-size: 8px; fill: #666; }
                #graph { width: 100%; height: 100vh; }
                #controls {
                    position: absolute;
                    top: 10px;
                    left: 10px;
                    background: rgba(255,255,255,0.8);
                    padding: 10px;
                    border-radius: 5px;
                }
            </style>
        </head>
        <body>
            <div id="controls">
                <h3>Knowledge Graph Explorer</h3>
                <p>Nodes: <span id="node-count">0</span> | Links: <span id="link-count">0</span></p>
                <label>
                    <input type="checkbox" id="show-labels" checked>
                    Show Labels
                </label>
            </div>
            <div id="graph"></div>
            
            <div id="node-info-modal" style="display: none; position: fixed; z-index: 20; left: 0; top: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.4);">
                <div style="background-color: white; margin: 10% auto; padding: 20px; width: 60%; max-width: 600px; border-radius: 5px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
                    <span id="close-modal" style="float: right; cursor: pointer; font-weight: bold;">&times;</span>
                    <h3 id="modal-title">Node Information</h3>
                    <div id="modal-content" style="max-height: 400px; overflow-y: auto;"></div>
                </div>
            </div>
            
            <div id="console-output" style="position: fixed; bottom: 10px; left: 10px; width: 400px; height: 200px; background: rgba(0,0,0,0.7); color: #0f0; padding: 10px; font-family: monospace; overflow: auto; z-index: 1000;">
                <div>Console Output:</div>
            </div>
            
            <div style="position: absolute; top: 10px; left: 10px; background: red; color: white; padding: 5px; z-index: 1000;">
                Test Element
            </div>
            
            <script>
            // Graph data
            const graphData = GRAPH_DATA_PLACEHOLDER;
            
            document.getElementById('node-count').textContent = graphData.nodes.length;
            document.getElementById('link-count').textContent = graphData.links.length;
            
            // Visualization code
            const width = window.innerWidth;
            const height = window.innerHeight;
            
            const svg = d3.select('#graph')
                .append('svg')
                .attr('width', width)
                .attr('height', height);
                
            // Create the simulation
            const simulation = d3.forceSimulation(graphData.nodes)
                .force('link', d3.forceLink(graphData.links).id(d => d.id).distance(100))
                .force('charge', d3.forceManyBody().strength(-200))
                .force('center', d3.forceCenter(width / 2, height / 2))
                .force('collide', d3.forceCollide().radius(30));
                
            // Create links
            const link = svg.append('g')
                .selectAll('line')
                .data(graphData.links)
                .enter().append('line')
                .attr('class', 'links');
                
            // Create link labels
            const linkLabel = svg.append('g')
                .selectAll('text')
                .data(graphData.links)
                .enter().append('text')
                .attr('class', 'link-labels')
                .text(d => d.label);
                
            // Create nodes
            const node = svg.append('g')
                .selectAll('circle')
                .data(graphData.nodes)
                .enter().append('circle')
                .attr('class', 'nodes')
                .attr('r', 5)
                .attr('fill', d => d.color || '#999')
                .call(d3.drag()
                    .on('start', dragstarted)
                    .on('drag', dragged)
                    .on('end', dragended));
                    
            // Create node labels
            const nodeLabel = svg.append('g')
                .selectAll('text')
                .data(graphData.nodes)
                .enter().append('text')
                .attr('class', 'node-labels')
                .attr('dx', 8)
                .attr('dy', 3)
                .text(d => d.label);
                
            // Add tooltips
            node.append('title')
                .text(d => `${d.label} (${d.type})`);
                
            // Update simulation on each tick
            simulation.on('tick', () => {
                link
                    .attr('x1', d => d.source.x)
                    .attr('y1', d => d.source.y)
                    .attr('x2', d => d.target.x)
                    .attr('y2', d => d.target.y);
                    
                node
                    .attr('cx', d => d.x)
                    .attr('cy', d => d.y);
                    
                nodeLabel
                    .attr('x', d => d.x)
                    .attr('y', d => d.y);
                    
                linkLabel
                    .attr('x', d => (d.source.x + d.target.x) / 2)
                    .attr('y', d => (d.source.y + d.target.y) / 2);
            });
            
            // Toggle labels
            document.getElementById('show-labels').addEventListener('change', function() {
                const visible = this.checked ? 'visible' : 'hidden';
                d3.selectAll('.node-labels').style('visibility', visible);
                d3.selectAll('.link-labels').style('visibility', visible);
            });
            
            // Zoom functionality
            const zoom = d3.zoom()
                .scaleExtent([0.1, 10])
                .on('zoom', (event) => {
                    svg.selectAll('g').attr('transform', event.transform);
                });
                
            svg.call(zoom);
            
            // Drag functions
            function dragstarted(event, d) {
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            }
            
            function dragged(event, d) {
                d.fx = event.x;
                d.fy = event.y;
            }
            
            function dragended(event, d) {
                if (!event.active) simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            }

            // Capture console output
            (function(){
                const consoleDiv = document.getElementById('console-output');
                const oldLog = console.log;
                const oldWarn = console.warn;
                const oldError = console.error;
                
                function addMessage(type, args) {
                    const msg = document.createElement('div');
                    msg.style.color = type === 'log' ? '#0f0' : type === 'warn' ? '#ff0' : '#f00';
                    msg.textContent = Array.from(args).join(' ');
                    consoleDiv.appendChild(msg);
                    consoleDiv.scrollTop = consoleDiv.scrollHeight;
                }
                
                console.log = function() {
                    oldLog.apply(console, arguments);
                    addMessage('log', arguments);
                };
                
                console.warn = function() {
                    oldWarn.apply(console, arguments);
                    addMessage('warn', arguments);
                };
                
                console.error = function() {
                    oldError.apply(console, arguments);
                    addMessage('error', arguments);
                };
            })();
            </script>
        </body>
        </html>
        """
        
        try:
            json_data = json.dumps(graph_data)
            print(f"JSON serialized successfully: {len(json_data)} characters")
            return html_template.replace('GRAPH_DATA_PLACEHOLDER', json_data)
        except Exception as e:
            print(f"ERROR serializing to JSON: {str(e)}")
            # Use minimal valid data as fallback
            return html_template.replace('GRAPH_DATA_PLACEHOLDER', '{"nodes":[],"edges":[]}')
    
    elif vis_type == "cytoscape":
        # Add debugging to verify graph_data structure
        print(f"Graph data contains {len(graph_data.get('nodes', []))} nodes and {len(graph_data.get('edges', []))} edges")
        
        # Cytoscape.js visualization with hub-centric layout
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Graph Visualization</title>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.24.0/cytoscape.min.js"></script>
            <style>
                body { margin: 0; padding: 0; font-family: Arial, sans-serif; }
                #cy { width: 100%; height: 90vh; background: #f8f9fa; }
                #controls { 
                    padding: 10px; 
                    background: #f0f0f0; 
                    display: flex;
                    flex-wrap: wrap;
                    gap: 10px;
                    align-items: center;
                }
                #stats {
                    font-weight: bold;
                    margin-right: 20px;
                }
                .control-group {
                    display: flex;
                    align-items: center;
                    gap: 5px;
                }
                button {
                    padding: 5px 10px;
                    cursor: pointer;
                }
                #filter-form {
                    position: absolute;
                    top: 50px;
                    right: 10px;
                    background: white;
                    padding: 15px;
                    border: 1px solid #ccc;
                    border-radius: 5px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    z-index: 100;
                    min-width: 250px;
                }
                #hub-info {
                    position: absolute;
                    bottom: 10px;
                    left: 10px;
                    background: rgba(255,255,255,0.8);
                    padding: 10px;
                    border-radius: 5px;
                    max-width: 250px;
                    max-height: 200px;
                    overflow: auto;
                    z-index: 100;
                }
            </style>
        </head>
        <body>
            <div id="controls">
                <div id="stats">Nodes: <span id="node-count">0</span> | Edges: <span id="edge-count">0</span></div>
                
                <div class="control-group">
                    <label for="layout-select">Layout:</label>
                    <select id="layout-select">
                        <option value="concentric" selected>Hub-Centric</option>
                        <option value="cose">Force-Directed</option>
                        <option value="breadthfirst">Breadth First</option>
                        <option value="circle">Circle</option>
                        <option value="grid">Grid</option>
                    </select>
                    <button id="apply-layout">Apply</button>
                </div>
                
                <div class="control-group">
                    <button id="fit-button">Fit View</button>
                    <label><input type="checkbox" id="show-labels" checked> Labels</label>
                    <label><input type="checkbox" id="highlight-hubs" checked> Highlight Hubs</label>
                </div>
                
                <div class="control-group">
                    <button id="show-filters">Adjust Filters</button>
                </div>
            </div>
            
            <div id="filter-form" style="display:none;">
                <h3>Graph Filters</h3>
                <form action="/" method="GET">
                    <label for="limit_nodes">Limit nodes (0 = no limit):</label>
                    <input type="number" id="limit_nodes" name="limit_nodes" min="0" value="0">
                    
                    <div style="margin-top: 10px;">
                        <label>
                            <input type="checkbox" name="aggregate_by_class"> 
                            Aggregate by class
                        </label>
                    </div>
                    
                    <div style="margin-top: 5px;">
                        <label>
                            <input type="checkbox" name="aggregate_edges"> 
                            Aggregate edges
                        </label>
                    </div>
                    
                    <button type="submit">Apply Filters</button>
                </form>
                <button id="close-filters" style="margin-top:5px;">Close</button>
            </div>
            
            <div id="hub-info" style="display:none;">
                <h3>Hub Nodes</h3>
                <div id="hub-list"></div>
            </div>
            
            <div id="cy"></div>
            
            <script>
            (function() {
                var graphData = GRAPH_DATA_PLACEHOLDER;
                
                // Update node/edge counts
                document.getElementById('node-count').textContent = graphData.nodes ? graphData.nodes.length : 0;
                document.getElementById('edge-count').textContent = graphData.edges ? graphData.edges.length : 0;
                
                // Check if we have nodes
                if (!graphData.nodes || graphData.nodes.length === 0) {
                    document.getElementById('cy').innerHTML = '<div style="padding: 20px; color: red; background: white;">No nodes found in graph data!</div>';
                } else {
                    // Initialize Cytoscape
                    var cy = window.cy = cytoscape({
                        container: document.getElementById('cy'),
                        elements: graphData,
                        style: [
                            {
                                selector: 'node',
                                style: {
                                    'background-color': 'data(color)',
                                    'label': 'data(label)',
                                    'width': 'data(size)',
                                    'height': 'data(size)',
                                    'text-valign': 'center',
                                    'text-halign': 'center',
                                    'font-size': '8px',
                                    'text-outline-width': 1,
                                    'text-outline-color': '#fff',
                                    'min-zoomed-font-size': '6px'
                                }
                            },
                            {
                                selector: 'edge',
                                style: {
                                    'width': 'data(width)',
                                    'line-color': '#ccc',
                                    'target-arrow-color': '#ccc',
                                    'target-arrow-shape': 'triangle',
                                    'curve-style': 'bezier',
                                    'label': 'data(label)',
                                    'font-size': '6px',
                                    'text-outline-width': 1,
                                    'text-outline-color': '#fff',
                                    'min-zoomed-font-size': '4px'
                                }
                            },
                            {
                                selector: '.hub',
                                style: {
                                    'border-width': 3,
                                    'border-color': '#ff0000',
                                    'text-outline-color': '#ff0000',
                                    'font-weight': 'bold'
                                }
                            }
                        ],
                        layout: {
                            name: 'concentric',
                            padding: 50,
                            animate: false,
                            // Sort nodes by their degree (most connected in the center)
                            concentric: function(node) {
                                return node.degree();
                            },
                            levelWidth: function() {
                                return 1;
                            }
                        },
                        wheelSensitivity: 0.3
                    });
                    
                    // Identify hub nodes (nodes with high connectivity)
                    var hubThreshold = 0.75; // Top 25% of nodes by connectivity
                    var hubNodes = [];
                    
                    // Calculate connectivity for each node
                    var degrees = cy.nodes().map(function(node) {
                        return { id: node.id(), degree: node.degree() };
                    });
                    
                    // Sort by degree (descending)
                    degrees.sort(function(a, b) {
                        return b.degree - a.degree;
                    });
                    
                    // Determine hub threshold
                    var hubMinDegree = 3; // Minimum degree to be considered a hub
                    if (degrees.length > 0) {
                        var hubIndex = Math.floor(degrees.length * (1 - hubThreshold));
                        hubMinDegree = Math.max(hubMinDegree, degrees[Math.min(hubIndex, degrees.length - 1)].degree);
                    }
                    
                    // Mark hub nodes
                    cy.nodes().forEach(function(node) {
                        if (node.degree() >= hubMinDegree) {
                            node.addClass('hub');
                            hubNodes.push({
                                id: node.id(),
                                label: node.data('label'),
                                degree: node.degree()
                            });
                        }
                    });
                    
                    // Display hub information
                    var hubInfo = document.getElementById('hub-info');
                    var hubList = document.getElementById('hub-list');
                    
                    if (hubNodes.length > 0) {
                        hubInfo.style.display = 'block';
                        
                        // Sort hub nodes by degree
                        hubNodes.sort(function(a, b) {
                            return b.degree - a.degree;
                        });
                        
                        // Display top 10 hubs
                        hubList.innerHTML = '';
                        hubNodes.slice(0, 10).forEach(function(hub) {
                            var hubItem = document.createElement('div');
                            hubItem.style.marginBottom = '5px';
                            hubItem.innerHTML = '<strong>' + hub.label + '</strong> (Connections: ' + hub.degree + ')';
                            hubItem.style.cursor = 'pointer';
                            hubItem.addEventListener('click', function() {
                                var node = cy.getElementById(hub.id);
                                cy.fit(node, 100);
                                cy.center(node);
                            });
                            hubList.appendChild(hubItem);
                        });
                    }
                    
                    // Fit the graph
                    cy.fit();
                    
                    // Setup event handlers
                    document.getElementById('apply-layout').addEventListener('click', function() {
                        var layoutName = document.getElementById('layout-select').value;
                        
                        var layoutOptions = {
                            name: layoutName,
                            animate: false
                        };
                        
                        // Special options for concentric layout
                        if (layoutName === 'concentric') {
                            layoutOptions.concentric = function(node) {
                                return node.degree();
                            };
                            layoutOptions.levelWidth = function() {
                                return 1;
                            };
                        }
                        
                        cy.layout(layoutOptions).run();
                    });
                    
                    document.getElementById('fit-button').addEventListener('click', function() {
                        cy.fit();
                    });
                    
                    document.getElementById('show-labels').addEventListener('change', function() {
                        cy.style()
                            .selector('node')
                            .style('label', this.checked ? 'data(label)' : '')
                            .update();
                        
                        cy.style()
                            .selector('edge')
                            .style('label', this.checked ? 'data(label)' : '')
                            .update();
                    });
                    
                    document.getElementById('highlight-hubs').addEventListener('change', function() {
                        if (this.checked) {
                            cy.nodes('.hub').style({
                                'border-width': 3,
                                'border-color': '#ff0000',
                                'text-outline-color': '#ff0000',
                                'font-weight': 'bold'
                            });
                        } else {
                            cy.nodes('.hub').style({
                                'border-width': 0,
                                'border-color': 'transparent',
                                'text-outline-color': '#fff',
                                'font-weight': 'normal'
                            });
                        }
                    });
                    
                    document.getElementById('show-filters').addEventListener('click', function() {
                        document.getElementById('filter-form').style.display = 'block';
                    });
                    
                    document.getElementById('close-filters').addEventListener('click', function() {
                        document.getElementById('filter-form').style.display = 'none';
                    });
                    
                    // Show node/edge information on click
                    cy.on('tap', 'node, edge', function(evt) {
                        var ele = evt.target;
                        var data = ele.data();
                        
                        // Create a simple popup with element details
                        var infoBox = document.createElement('div');
                        infoBox.style.position = 'fixed';
                        infoBox.style.top = '20%';
                        infoBox.style.left = '50%';
                        infoBox.style.transform = 'translateX(-50%)';
                        infoBox.style.background = 'white';
                        infoBox.style.padding = '15px';
                        infoBox.style.borderRadius = '5px';
                        infoBox.style.boxShadow = '0 0 10px rgba(0,0,0,0.3)';
                        infoBox.style.zIndex = '1000';
                        infoBox.style.maxWidth = '80%';
                        infoBox.style.maxHeight = '80%';
                        infoBox.style.overflow = 'auto';
                        
                        // Add close button
                        var closeBtn = document.createElement('button');
                        closeBtn.textContent = 'Close';
                        closeBtn.style.position = 'absolute';
                        closeBtn.style.top = '5px';
                        closeBtn.style.right = '5px';
                        closeBtn.addEventListener('click', function() {
                            document.body.removeChild(infoBox);
                        });
                        
                        // Add content
                        var title = document.createElement('h3');
                        title.textContent = data.label || 'Element Info';
                        
                        var connectionInfo = '';
                        if (ele.isNode()) {
                            connectionInfo = '<p>Connections: ' + ele.degree() + '</p>';
                        }
                        
                        var content = document.createElement('div');
                        content.innerHTML = connectionInfo;
                        
                        var dataPre = document.createElement('pre');
                        dataPre.style.whiteSpace = 'pre-wrap';
                        dataPre.style.maxHeight = '300px';
                        dataPre.style.overflow = 'auto';
                        dataPre.textContent = JSON.stringify(data, null, 2);
                        
                        // Build the box
                        infoBox.appendChild(closeBtn);
                        infoBox.appendChild(title);
                        infoBox.appendChild(content);
                        infoBox.appendChild(dataPre);
                        
                        // Add to document
                        document.body.appendChild(infoBox);
                    });
                    
                    // Clear info box when clicking on background
                    cy.on('tap', function(evt) {
                        if (evt.target === cy) {
                            var infoBoxes = document.querySelectorAll('div[style*="position: fixed"]');
                            infoBoxes.forEach(function(box) {
                                if (box.querySelector('pre')) { // Only remove our info boxes
                                    document.body.removeChild(box);
                                }
                            });
                        }
                    });
                }
            })();
            </script>
        </body>
        </html>
        """
        
        try:
            json_data = json.dumps(graph_data, default=str)
            print(f"JSON serialized successfully: {len(json_data)} characters")
            return html_template.replace('GRAPH_DATA_PLACEHOLDER', json_data)
        except Exception as e:
            print(f"ERROR serializing to JSON: {str(e)}")
            return html_template.replace('GRAPH_DATA_PLACEHOLDER', '{"nodes":[],"edges":[]}')
    
    else:
        raise ValueError(f"Unknown visualization type: {vis_type}")

def create_visualization(graph, output_file, vis_type='cytoscape', aggregate_by_class=False, limit_nodes=0):
    """
    Create an interactive HTML visualization of a knowledge graph
    
    Args:
        graph: RDF graph to visualize
        output_file: Path to save the HTML visualization
        vis_type: Type of visualization ('cytoscape' or 'd3')
        aggregate_by_class: Whether to aggregate nodes by class
        limit_nodes: Limit the number of nodes (0 for no limit)
        
    Returns:
        bool: True if successful, False otherwise
    """
    if len(graph) == 0:
        print("Warning: Graph is empty, nothing to visualize")
        return False
    
    print(f"Preparing visualization for graph with {len(graph)} triples")
    
    # Apply node limit if specified
    if limit_nodes > 0:
        graph = limit_graph_size(graph, limit_nodes)
    
    # Create visualization data based on the selected type
    if vis_type == 'd3':
        print("Preparing D3.js visualization data...")
        vis_data = create_d3_graph_data(graph)
    else:  # cytoscape
        print("Preparing Cytoscape.js visualization data...")
        vis_data = create_cytoscape_graph_data(graph, aggregate_by_class=aggregate_by_class)
    
    # Check if data was generated
    if not vis_data or len(vis_data.get('nodes', [])) == 0:
        print("Warning: No visualization data was generated")
        return False
    
    print(f"Created visualization data with {len(vis_data.get('nodes', []))} nodes and {len(vis_data.get('edges', []))} edges")
    
    # Create HTML template
    html_content = create_html_visualization(vis_data, vis_type)
    
    # Write to file
    try:
        with open(output_file, 'w') as f:
            f.write(html_content)
        print(f"Visualization saved to {output_file}")
        return True
    except Exception as e:
        print(f"Error saving visualization: {str(e)}")
        return False

def limit_graph_size(graph, max_nodes):
    """Limit the graph size by keeping only the most connected nodes"""
    from rdflib import URIRef, Graph
    
    # Get all unique subjects and objects that are URIs
    nodes = set()
    for s, p, o in graph:
        nodes.add(s)
        if isinstance(o, URIRef):
            nodes.add(o)
    
    if len(nodes) <= max_nodes:
        return graph
        
    print(f"Graph has {len(nodes)} nodes, limiting to {max_nodes}")
    
    # Get the most connected nodes by degree
    node_degrees = {}
    for node in nodes:
        # Count outgoing edges
        out_degree = len(list(graph.predicate_objects(node)))
        # Count incoming edges
        in_degree = len(list(graph.subject_predicates(node)))
        node_degrees[node] = out_degree + in_degree
    
    # Sort nodes by degree and keep the top N
    sorted_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)
    top_nodes = [node for node, _ in sorted_nodes[:max_nodes]]
    
    # Create a new graph with only the top nodes
    limited_graph = Graph()
    for s, p, o in graph:
        if s in top_nodes and (not isinstance(o, URIRef) or o in top_nodes):
            limited_graph.add((s, p, o))
    
    print(f"Limited graph to {len(limited_graph)} triples")
    return limited_graph

def main():
    """
    Main function to load a graph and create a visualization.
    """
    parser = argparse.ArgumentParser(description="Create visualizations from RDF knowledge graphs")
    parser.add_argument("--input", help="Path to input RDF file (Turtle, N-Quads, etc.)")
    parser.add_argument("--output", help="Path to save the HTML visualization")
    # Make these arguments optional with defaults
    parser.add_argument("--vis-type", choices=["cytoscape", "d3"], default="cytoscape", help="Visualization type")
    parser.add_argument("--aggregate-by-class", action="store_true", help="Aggregate nodes by class in visualization")
    parser.add_argument("--limit-nodes", type=int, default=0, help="Limit the number of nodes (0 for no limit)")
    parser.add_argument("--entity", help="URI of specific entity to visualize")
    parser.add_argument("--filter-types", nargs='+', help="Only include entities of these types")
    
    args = parser.parse_args()
    
    # Use default paths if not provided
    project_root = Path(__file__).parent.parent
    
    if args.input is None:
        args.input = os.path.join(project_root, "kg_output.nq")
        
    if args.output is None:
        args.output = os.path.join(project_root, "kg_visualization.html")
        
    # Load the graph
    graph = load_graph(args.input)
    
    # Apply filters if specified
    if args.entity:
        print(f"Extracting subgraph for entity: {args.entity}")
        # Use the more powerful version with default depth=1 to match original behavior
        graph = extract_subgraph(graph, args.entity, depth=1)
        print(f"Extracted subgraph with {len(graph)} triples")
    
    if args.filter_types:
        print(f"Filtering graph by types: {args.filter_types}")
        graph = filter_graph_by_type(graph, args.filter_types)
        print(f"Filtered graph to {len(graph)} triples")
    
    # Create the visualization
    success = create_visualization(
        graph,
        args.output,
        vis_type=args.vis_type,
        aggregate_by_class=args.aggregate_by_class,
        limit_nodes=args.limit_nodes
    )
    
    if success:
        print(f"Visualization successfully created at {args.output}")
    else:
        print("Failed to create visualization. See above for errors.")

if __name__ == "__main__":
    main() 