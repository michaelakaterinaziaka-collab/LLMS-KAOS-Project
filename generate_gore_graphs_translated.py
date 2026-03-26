import json
import os
from graphviz import Digraph
from unidecode import unidecode

INPUT_JSON = "learning_goals_and_relationships_KAOS_model_gemini_.json"
OUTPUT_DIR = "KAOS_goal_model_graphs_"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def wrap(text, max_words=4):
    words = text.split()
    return "\n".join(" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words))

def clean_text(text):
    if not isinstance(text, str):
        print(f"⚠️ Warning: goal is not string but {type(text)} – converting: {text}")
        text = json.dumps(text, ensure_ascii=False)
    return unidecode(text.strip().replace(".", "").replace(",", "").replace("(", "").replace(")", "")).lower()

def safe_id(text):
    return clean_text(text).replace(" ", "_")

def generate_graph(entry, index):
    dot = Digraph(format="png", engine="dot")
    dot.attr(rankdir="BT", fontname="Arial", splines="polyline", concentrate="true")
    dot.attr('node', fontname="Arial")
    dot.attr('edge', fontname="Arial")

    id_map = {}
    connected_nodes = set()
    extra_node_counter = 0

    behav_set = set(clean_text(g) for g in entry.get("Behavioural_goals", []))
    soft_set = set(clean_text(g) for g in entry.get("soft_goals", []))

    all_goals = entry.get("Behavioural_goals", []) + entry.get("soft_goals", [])
    for goal in all_goals:
        goal_clean = clean_text(goal)
        gid = safe_id(goal)
        id_map[goal_clean] = gid
        if goal_clean in behav_set:
            color = "#85C1E9"
            style = "filled"
        elif goal_clean in soft_set:
            color = "#D5F5E3"
            style = "filled,dashed"
        else:
            continue

        dot.node(gid, wrap(goal), shape="parallelogram", style=style, fillcolor=color)

    for rel in entry.get("goal_relationships", []):
        source = rel.get("goal")
        relation = rel.get("relation", "AND").upper()
        targets = (
            rel.get("decomposed_to")
            or rel.get("influences")
            or rel.get("supports")
        )

        if not source or not targets:
            continue

        if isinstance(targets, str):
            targets = [t.strip() for t in targets.split(",") if t.strip()]

        src_clean = clean_text(source)
        src_id = id_map.get(src_clean)
        if not src_id:
            if src_clean not in behav_set and src_clean not in soft_set:
                behav_set.add(src_clean)
                entry.setdefault("Behavioural_goals", []).append(source)
            src_id = safe_id(source)
            id_map[src_clean] = src_id
            if src_clean in behav_set:
                color = "#85C1E9"
                style = "filled"
            elif src_clean in soft_set:
                color = "#D5F5E3"
                style = "filled,dashed"
            dot.node(src_id, wrap(source), shape="parallelogram", style=style, fillcolor=color)

        if relation == "AND":
            connector_id = f"{relation.lower()}_{extra_node_counter}"
            dot.node(connector_id, label="", shape="circle", fixedsize="true", width="0.3", style="filled", fillcolor="yellow")

            for tgt in targets:
                tgt_clean = clean_text(tgt)
                tgt_id = id_map.get(tgt_clean)
                if not tgt_id:
                    if tgt_clean not in behav_set and tgt_clean not in soft_set:
                        behav_set.add(tgt_clean)
                        entry.setdefault("Behavioural_goals", []).append(tgt)
                    tgt_id = safe_id(tgt)
                    id_map[tgt_clean] = tgt_id
                    if tgt_clean in behav_set:
                        color = "#85C1E9"
                        style = "filled"
                    elif tgt_clean in soft_set:
                        color = "#D5F5E3"
                        style = "filled,dashed"
                    dot.node(tgt_id, wrap(tgt), shape="parallelogram", style=style, fillcolor=color)
                dot.edge(tgt_id, connector_id, arrowhead="none")
                connected_nodes.add(tgt_id)

            dot.edge(connector_id, src_id)
            connected_nodes.add(src_id)
            extra_node_counter += 1

        elif relation == "OR":
            for tgt in targets:
                tgt_clean = clean_text(tgt)
                tgt_id = id_map.get(tgt_clean)
                if not tgt_id:
                    if tgt_clean not in behav_set and tgt_clean not in soft_set:
                        behav_set.add(tgt_clean)
                        entry.setdefault("Behavioural_goals", []).append(tgt)
                    tgt_id = safe_id(tgt)
                    id_map[tgt_clean] = tgt_id
                    if tgt_clean in behav_set:
                        color = "#85C1E9"
                        style = "filled"
                    elif tgt_clean in soft_set:
                        color = "#D5F5E3"
                        style = "filled,dashed"
                    dot.node(tgt_id, wrap(tgt), shape="parallelogram", style=style, fillcolor=color)

                connector_id = f"{relation.lower()}_{extra_node_counter}"
                dot.node(connector_id, label="", shape="circle", fixedsize="true", width="0.3", style="filled", fillcolor="pink")
                dot.edge(tgt_id, connector_id, arrowhead="none")
                dot.edge(connector_id, src_id)

                connected_nodes.update([tgt_id, connector_id, src_id])
                extra_node_counter += 1

        else:
            for tgt in targets:
                tgt_clean = clean_text(tgt)
                tgt_id = id_map.get(tgt_clean)
                if not tgt_id:
                    if tgt_clean not in behav_set and tgt_clean not in soft_set:
                        behav_set.add(tgt_clean)
                        entry.setdefault("Behavioural_goals", []).append(tgt)
                    tgt_id = safe_id(tgt)
                    id_map[tgt_clean] = tgt_id
                    if tgt_clean in behav_set:
                        color = "#85C1E9"
                        style = "filled"
                    elif tgt_clean in soft_set:
                        color = "#D5F5E3"
                        style = "filled,dashed"
                    dot.node(tgt_id, wrap(tgt), shape="parallelogram", style=style, fillcolor=color)
                dot.edge(tgt_id, src_id)
                connected_nodes.update([src_id, tgt_id])

    for rel in entry.get("goal_relationships", []):
        if rel.get("relation", "").lower() == "conflict":
            source = rel.get("goal")
            targets = rel.get("hinders") or []

            if isinstance(targets, str):
                targets = [t.strip() for t in targets.split(",") if t.strip()]

            for tgt in targets:
                src_id = id_map.get(clean_text(source), safe_id(source))
                tgt_id = id_map.get(clean_text(tgt), safe_id(tgt))

                conflict_node_id = f"conflict_{extra_node_counter}"
                dot.attr('node', fontname='Segoe UI Emoji')
                dot.node(conflict_node_id, label="⚡", shape="plaintext", fontcolor="red", fontsize="20")

                dot.attr('edge')
                dot.edge(src_id, conflict_node_id, arrowhead="none")
                dot.edge(conflict_node_id, tgt_id, arrowhead="none")
                dot.attr('edge', style="solid", color="black", penwidth="1")

                extra_node_counter += 1

    output_path = os.path.join(OUTPUT_DIR, f"kaos_diagram_{index+1}")
    dot.render(output_path, cleanup=True)
    print(f" Created: {output_path}.png")

with open(INPUT_JSON, encoding="utf-8") as f:
    entries = json.load(f)
    print(f" Found {len(entries)} records in JSON")
    for i, entry in enumerate(entries):
        try:
            generate_graph(entry, i)
        except Exception as e:
            print(f" Error in diagram {i+1}: {e}")
