import json
import networkx as nx
import numpy as np

# Saving Optimization results to json
def save_optimization_results(filepath, sol):
    
    def serialize_var_dict(var_dict):
        out = {}
        for k, v in var_dict.items():
            # key can be Var, int, tuple, etc.
            if hasattr(k, "name"):
                key = k.name
            else:
                key = str(k)
            # value can be numpy/scip
            out[key] = float(v)
        return out

    def serialize_graph(G):
        return {
            "nodes": list(G.nodes()),
            "edges": [(u, v) for u, v in G.edges()]
        }

    results = {
        "variables": {
            "x": serialize_var_dict(sol["x"]),
            "y": serialize_var_dict(sol["y"]),
            "z": serialize_var_dict(sol["z"]),
            "u": serialize_var_dict(sol["u"]),
            "t": float(sol["t"])
        },
        "objective_components": {
            "c0": serialize_var_dict(sol["c0"]),
            "losses": serialize_var_dict(sol["losses"]),
            "c_eff": serialize_var_dict(sol["c_eff"])
        },
        "graphs": {
            "original": serialize_graph(sol["G_original"]),
            "contracted": serialize_graph(sol["G_contracted"])
        },
        "gap": sol.get("gap", None)
    }

    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {filepath}")


# Loading Optimization results from json
def load_optimization_results(filepath):
    """
    Load and deserialize SCIP optimization results from JSON.
    
    Returns:
        sol (dict): Dictionary matching the structure from build_and_solve
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Deserialize variable dictionaries
    # Keys might be strings like "(0, 1)" for tuples - need to parse
    def deserialize_var_dict(var_dict, key_type='int'):
        """
        Convert string keys back to proper types.
        key_type: 'int', 'tuple', or 'str'
        """
        out = {}
        for k_str, v in var_dict.items():
            if key_type == 'int':
                key = int(k_str)
            elif key_type == 'tuple':
                # Parse string like "(0, 1)" back to tuple
                key = eval(k_str)  # Safe here since we control the format
            else:
                key = k_str
            out[key] = float(v)
        return out
    
    # Reconstruct variable dictionaries
    x_vals = deserialize_var_dict(data["variables"]["x"], key_type='int')
    y_vals = deserialize_var_dict(data["variables"]["y"], key_type='tuple')
    z_vals = deserialize_var_dict(data["variables"]["z"], key_type='tuple')
    u_vals = deserialize_var_dict(data["variables"]["u"], key_type='tuple')
    t_val = float(data["variables"]["t"])
    
    # Reconstruct objective components
    c0 = deserialize_var_dict(data["objective_components"]["c0"], key_type='int')
    losses = deserialize_var_dict(data["objective_components"]["losses"], key_type='int')
    c_eff = deserialize_var_dict(data["objective_components"]["c_eff"], key_type='int')
    
    # Reconstruct graphs
    G_original = nx.Graph()
    G_original.add_nodes_from(data["graphs"]["original"]["nodes"])
    G_original.add_edges_from(data["graphs"]["original"]["edges"])
    
    G_contracted = nx.Graph()
    G_contracted.add_nodes_from(data["graphs"]["contracted"]["nodes"])
    G_contracted.add_edges_from(data["graphs"]["contracted"]["edges"])
    
    # Reconstruct sol dictionary (matching build_and_solve output)
    sol = {
        "model": None,  # Can't save SCIP model to JSON
        "x": x_vals,
        "z": z_vals,
        "y": y_vals,
        "u": u_vals,
        "t": t_val,
        "c0": c0,
        "losses": losses,
        "c_eff": c_eff,
        "G_original": G_original,
        "G_contracted": G_contracted,
        "fused_map": {},  # Not saved in your current code
        "gap": data.get("gap", None)
    }
    
    return sol