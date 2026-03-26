import pandas as pd
import google.generativeai as genai
import json
import time
import re
import networkx as nx

def normalize_goal_item(item):
    if isinstance(item, dict):
        return json.dumps(item, ensure_ascii=False)  
    return str(item).strip()

def normalize_goal_list(goal_list):
    if not isinstance(goal_list, list):
        return []
    return [normalize_goal_item(g) for g in goal_list if isinstance(g, (str, dict))]

def clean_goal_relationships(rel_list):
    if not isinstance(rel_list, list):
        return []
    cleaned = []
    for rel in rel_list:
        if isinstance(rel, dict):
            cleaned_rel = {}
            for k, v in rel.items():
                key = normalize_goal_item(k)
                if isinstance(v, list):
                    value = normalize_goal_list(v)
                elif isinstance(v, dict):
                    value = [normalize_goal_item(v)]
                else:
                    value = normalize_goal_item(v)
                cleaned_rel[key] = value
            cleaned.append(cleaned_rel)
    return cleaned

def filter_relationships_by_existing_goals(parsed):
    allowed_goals = set(parsed.get("Behavioural_goals", []) + parsed.get("soft_goals", []))
    filtered = []

    for rel in parsed.get("goal_relationships", []):
        if not isinstance(rel, dict):
            continue
        source = rel.get("goal")
        rel_type = rel.get("relation", "").lower()

        if rel_type in ("and", "or"):
            targets = rel.get("decomposed_to", [])
        elif rel_type == "conflict":
            targets = rel.get("hinders", [])
        else:
            continue

        if isinstance(targets, str):
            targets = [targets]

        if source in allowed_goals and all(t in allowed_goals for t in targets):
            filtered.append(rel)

    parsed["goal_relationships"] = filtered
    return parsed


def extract_connected_to_main_goal(relationships, main_goal):
    G = nx.Graph()
    for rel in relationships:
        if not isinstance(rel, dict):
            continue
        src = rel.get("goal")
        rel_type = rel.get("relation", "").lower()
        if rel_type in ("and", "or"):
            targets = rel.get("decomposed_to", [])
        elif rel_type == "conflict":
            targets = rel.get("hinders", [])
        else:
            continue
        if isinstance(targets, str):
            targets = [targets]
        for tgt in targets:
            G.add_edge(src, tgt)
    if main_goal in G:
        return nx.node_connected_component(G, main_goal)
    else:
        return set()

# API setup
genai.configure(api_key="API_KEY")

# Read the CSV file
input_file = "sections_from_books.csv"
df = pd.read_csv(input_file)

# Build the LLM prompt
def build_prompt(text):
    return f"""
You are an expert in analyzing educational goals that concern the student who reads them and in KAOS modeling methodologies.

You are given a passage from a university textbook. The task is to identify and categorize:

Behavioural Goals: What will the student learn/be able to do afterwards?
Definition: Clear, precise, and measurable goals.
Rule: They must be implementable through actions (operations) or assigned agents.
ATTENTION: Do not include any Behavioural Goal that is not directly or indirectly connected with the main goal through a relationship (AND, OR, Conflict).

Soft Goals: What does the student wish to better understand or improve?
Definition: Subjective, vague, or hard-to-measure goals.
Rule: They cannot be definitively satisfied or rejected.
Rule: Usually used for non-functional requirements.
Rule: They are handled through satisficing (satisfying as much as possible).
ATTENTION: Do not include any Soft Goal that is not directly or indirectly connected with the main goal through a relationship (AND, OR, Conflict).

Important: Goals always concern the student. Goals may be explicitly stated or inferred from the content. If the text describes concepts, processes, or phenomena, infer the goals based on what the student learns from them. Avoid extracting duplicate goals. Extract goals under the MAIN RULE that each extracted goal must belong to a FULLY COHERENT goal network starting from the main goal of the section. If this is not the case for a goal you identify, do NOT include it in the output.

Goal Relationships: What relationships exist between the goals?

First, identify the main goal of the passage. It is the most general goal, usually a behavioural one, and summarizes what is sought overall in the section.

Analyze the main goal into subgoals, based on the appropriate relation, from the following (AND, OR, Conflict):

Refinement Analysis:

AND-analysis: All subgoals must be achieved for the main goal to be satisfied.
(i.e., the main goal necessarily depends on all of them.)

OR-analysis: It suffices that one or more subgoals are satisfied for the main goal to be achieved.
(i.e., the main goal is satisfied alternatively.)

Rule: The analysis must maintain semantic coherence, i.e., the subgoals must be meaningfully related to the main goal.
So, there cannot be a subgoal that is not connected through one of the three relations with the main goal.
Repeat the analysis recursively:

If a subgoal is complex or general, analyze it further into subgoals.

Create as many levels as needed until you have a complete hierarchical model. Attention:

It is STRICTLY FORBIDDEN to extract goals (Behavioural or Soft) that do not participate in some relation with other goals that are coherently connected to the main one.

That is, do NOT include in the output goals that:

Are not directly or indirectly connected with the main goal, and

Are not included in any relation (analysis or conflict).

If you detect goals that seem important but are not logically related to others, simply do not include them in the output.

Your output must include ONLY goals that are within a SINGLE and COHERENT network starting from the main goal. Any goal outside this is omitted.

If you create a goal that has no relation with another, it is wrong.

Conflict Relationships (Conflict):

Definition: Two goals are in opposition or hinder each other’s achievement. It is not enough that the goals are different, the pursuit of one must actively obstruct the other. Before assigning a conflict relation, think:

If one goal is achieved, does the other become harder or impossible?

Is there a contradiction in their achievement? If not, do not assign conflict.
For example, the goal “fast application of statistical methods through software” conflicts with the goal “in-depth understanding of the theory behind the methods”, since emphasis on practical use reduces motivation for theoretical understanding, and conversely, deepening in theory slows down practical application.

Rule: Conflicts must be explicitly modeled so that possible trade-offs are recognized.

Remember:
All goals must come from the analysis of the main goal.
If they are not related to it, they are discarded and must not be included in the output. There must be a coherent connection through some relation among all the goals extracted.

ATTENTION:
It is FORBIDDEN to create multiple independent sub-networks of goals. All goals (Behavioural and Soft) MUST be related, directly or indirectly, to the main goal, forming a chain of dependencies that does not break anywhere. For example: If you identify 3 goals A, B, C, they must be coherently connected (e.g., A with B and C, or A with B and B with C, etc.). If A connects with B and neither connects with C, do not mention C in the output.

If a goal cannot be integrated into this unified model through relations, simply omit it completely and do NOT include it in the output.

Prohibition: Do not mention any goal (Behavioural or Soft) that is not connected with a relation (AND, OR, or conflict) with another goal.

Specifically: if a goal cannot be integrated into a coherent hierarchical network starting from the main goal, do NOT mention it at all in the output.

Do not over-analyze. Limit the number of goals and levels only to what is absolutely necessary for the semantic coverage of the original main goal. If further analysis does not contribute essentially to clarity, avoid it.

Present the output in JSON format using exclusively the following phrases/fields (do not introduce others):

"the goal"

"analyzed into goals"

"with relation": one of "AND", "OR", "conflict"

"blocks the goal"

Do not use any other phrasing for the relations.

The output must be of the form:

{
  "Behavioural_goals": [
    "Understanding the basic concepts of the theory",
    "Definition of the main term",
    "Distinguishing concepts",
    "Fast application"
  ],
  "soft_goals": [
    "Enhancing the ability to evaluate alternative approaches",
    "Encouraging further engagement with the history of philosophy",
    "In-depth understanding"
  ],
  "goal_relationships": [
    {
      "the goal": "Understanding the basic concepts of the theory",
      "analyzed into goals": ["Definition of the main term", "Distinguishing concepts"],
      "with relation": "AND"
    },
    {
      "the goal": "Definition of the main term",
      "analyzed into goals": ["Definition of keywords", "Usage examples"],
      "with relation": "OR"
    },
    {
      "the goal": "Fast application",
      "blocks the goal": ["In-depth understanding"],
      "with relation": "conflict"
    }
  ],
  "agents": ["The student"]
}

If there are no goals, write:
{"Behavioural_goals": [], "soft_goals": [], "goal_relationships": [], "agents": ["The student"]}

###Text for analysis:
{text}
"""

def clean_json_string(response_text):
    response_text = response_text.strip()
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]
    response_text = re.sub(r'\.{3}\s*$', '', response_text)
    if response_text.strip().startswith('['):
        response_text = '{ "functional_goals": ' + response_text.strip() + ' }'
    return response_text

model = genai.GenerativeModel("gemini-2.0-flash")

results = []

for i, row in df.iterrows():
    filename = str(row.get('filename', '')).replace('%', '%%')
    header = str(row.get('header', '')).replace('%', '%%')
    print(f"{i+1}/{len(df)} | File: {filename} | Section: {header}")

    try:
        prompt = build_prompt(str(row["section"]))
        response = model.generate_content(prompt)
        content = response.text.strip()
        print(f"Model response:\n{content}")

        if content:
            try:
                content_cleaned = clean_json_string(content)
                parsed = json.loads(content_cleaned)
                parsed = filter_relationships_by_existing_goals(parsed)

            except json.JSONDecodeError:
                print(" Invalid JSON! Returned:\n", content)
                parsed = {
                    "Behavioural_goals": [],
                    "soft_goals": [],
                    "goal_relationships": [],
                    "agents": ["Student"]
                }
        else:
            print(" Empty response. Skipping.")
            parsed = {
                "Behavioural_goals": [],
                "soft_goals": [],
                "goal_relationships": [],
                "agents": ["Student"]
            }

        relationships = parsed.get("goal_relationships", [])
        main_goal = relationships[0].get("goal") if relationships else ""
        connected_goals = extract_connected_to_main_goal(relationships, main_goal)

        parsed["Behavioural_goals"] = [g for g in parsed.get("Behavioural_goals", []) if g in connected_goals]
        parsed["soft_goals"] = [g for g in parsed.get("soft_goals", []) if g in connected_goals]

    except Exception as e:
        print(" Model error:", e)
        parsed = {
            "Behavioural_goals": [],
            "soft_goals": [],
            "goal_relationships": [],
            "agents": ["Student"]
        }

    results.append({
        "filename": row["filename"],
        "header": row["header"],
        "place": row["place"],
        "Behavioural_goals": normalize_goal_list(parsed.get("Behavioural_goals") or parsed.get("behavioural_goals", [])),
        "soft_goals": normalize_goal_list(parsed.get("soft_goals", [])),
        "goal_relationships": clean_goal_relationships(parsed.get("goal_relationships", [])),
        "agents": normalize_goal_list(parsed.get("agents", ["Student"])),
    })

    time.sleep(2.5)

with open("learning_goals_and_relationships_KAOS_model_gemini_.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(" Completed KAOS goal extraction for the student!")
