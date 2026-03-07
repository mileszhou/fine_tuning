import os

# ==== Reserved words ====
RESERVED_TOP_LEVEL = {
    "_outputs", "_results", "_stuff",
    "src", "res", "lib", "bin",
    "python", "pytorch", "notebook", "vsbasic"
}

# Folders considered internal (hidden from deliverable)
INTERNAL_FOLDERS = {"src", "res", "lib", "bin"}

# Output folders must start with "_"
OUTPUT_PREFIX = "_"

# ==== Enforcement function ====
def check_node(node_path):
    issues = []

    # List all top-level folders in the node
    entries = [d for d in os.listdir(node_path) if os.path.isdir(os.path.join(node_path, d))]
    
    for entry in entries:
        # 1. Reserved word check
        if entry not in RESERVED_TOP_LEVEL:
            issues.append(f"❌ Invalid top-level folder: '{entry}' not a reserved word")
        
        # 2. Output prefix check
        if entry.startswith(OUTPUT_PREFIX):
            if entry not in {"_outputs", "_results", "_stuff"}:
                issues.append(f"⚠ {_outputs} prefix used for unknown folder: {entry}")
        
        # 3. Internal folder content check (optional: detect non-source files)
        if entry in INTERNAL_FOLDERS:
            path = os.path.join(node_path, entry)
            for root, dirs, files in os.walk(path):
                for f in files:
                    if f.endswith(".ipynb") and entry in {"src", "lib", "bin"}:
                        issues.append(f"❌ Notebook found inside internal folder '{entry}': {f}")
    
    # 4. Success
    if not issues:
        print(f"✅ Node '{node_path}' is compliant")
    else:
        print(f"⚠ Node '{node_path}' compliance issues:")
        for i in issues:
            print("  " + i)

# ==== Example Usage ====
if __name__ == "__main__":
    node_dir = input("Enter the path of the node to check: ").strip()
    if not os.path.isdir(node_dir):
        print("Invalid path")
    else:
        check_node(node_dir)
