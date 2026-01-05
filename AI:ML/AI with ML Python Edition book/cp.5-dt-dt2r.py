# dt to rules
def dt_to_rules(model):
    rules = []
    
    # Identify leaf nodes (nodes that are NOT parents of anyone)
    all_indices = range(len(model.parent))
    parent_indices = set(p - 1 for p in model.parent if p > 0)
    leaf_indices = [i for i in all_indices if i not in parent_indices]

    for leaf in leaf_indices:
        conditions = []
        curr = leaf
        
        # Walk up the tree from leaf to root
        while curr > 0:
            p_idx = model.parent[curr] - 1
            attr_name = model.node[p_idx]
            branch_val = model.branch[curr - 1]
            conditions.append(f"({attr_name} == '{branch_val}')")
            curr = p_idx
        
        # Reverse because we walked backwards
        rule_str = " AND ".join(reversed(conditions))
        result = model.node[leaf]
        rules.append(f"IF {rule_str} THEN Class = {result}")
        
    return rules
    
# rules to dt 
class RuleToDT:
    def __init__(self, rules, all_features):
        self.rules = rules # List of strings
        self.features = all_features
        
    def parse_rule(self, rule_str):
        # Very basic parser: extract conditions and result
        # Expected format: "IF (Color == 'Red') AND (Size == 'Small') THEN Class = 1"
        parts = rule_str.replace("IF ", "").split(" THEN ")
        conds = parts[0].split(" AND ")
        result = parts[1].split(" = ")[1]
        
        parsed_conds = {}
        for c in conds:
            # strip parentheses and split
            c = c.replace("(", "").replace(")", "")
            key, val = c.split(" == ")
            parsed_conds[key] = val.strip("'")
            
        return parsed_conds, result

    def generate_tree(self):
        # In practice, converting raw rules back to a tree often requires 
        # ID3/C4.5 logic using the rules as 'perfect' synthetic data.
        print("To convert rules to DT, we map the rules back into a tabular")
        print("format and re-run the ID3 algorithm to find the optimal structure.")
        
        # 1. Create synthetic data from rules
        # 2. Run ID3(synthetic_X, synthetic_T, features)
        pass