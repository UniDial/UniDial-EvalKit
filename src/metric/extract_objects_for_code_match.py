import ast
import re

# Define the keys we expect in the output dictionary.
OBJECTS = [
    "function",
    "function argument",
    "function docstring",
    "function try",
    "function assert",
    "function annotation",
    "function decorator",
    "class",
    "class decorator",
    "method",
    "method docstring",
    "method try",
    "method assert",
    "method annotation",
    "method decorator",
    "attribute",
    "variable",
    "import",
    "comment",
]


def extract_decorator_id(decorator):
    """Extracts a name or attribute from a decorator node."""
    if isinstance(decorator, ast.Name):
        return decorator.id
    elif isinstance(decorator, ast.Attribute):
        return decorator.attr
    elif isinstance(decorator, ast.Call):
        return extract_decorator_id(decorator.func)
    return None


def get_decorator_ids(decorators):
    """Returns a list of decorator names from a list of decorator nodes."""
    ids = []
    for dec in decorators:
        dec_id = extract_decorator_id(dec)
        if dec_id:
            ids.append(dec_id)
    return ids


def extract_annotations(node):
    """Collect annotations from a FunctionDef or Method node."""
    # Collect annotations on arguments.
    annotations = [arg.annotation for arg in node.args.args if arg.annotation]
    # Collect the return annotation (if any) only once.
    if node.returns:
        annotations.append(node.returns)
    return annotations


def extract_try_assert(nodes):
    """Return lists of Try and Assert nodes from a list of nodes (e.g. function body)."""
    tries = [n for n in nodes if isinstance(n, ast.Try)]
    asserts = [n for n in nodes if isinstance(n, ast.Assert)]
    return tries, asserts


def extract_self_attributes(nodes):
    """
    Look for assignments in __init__ (or other methods) of the form
       self.attr = value
    and return a list of attribute names.
    """
    attributes = []
    for stmt in nodes:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    attributes.append(target.attr)
    return attributes


def extract_variable_names(node):
    """
    Given an assignment (Assign or AnnAssign) node, extract the variable names
    (but not attributes).
    """
    names = []
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                names.append(target.id)
            elif isinstance(target, ast.Tuple):
                names.extend(elt.id for elt in target.elts if isinstance(elt, ast.Name))
    elif isinstance(node, ast.AnnAssign):
        target = node.target
        if isinstance(target, ast.Name):
            names.append(target.id)
        elif isinstance(target, ast.Tuple):
            names.extend(elt.id for elt in target.elts if isinstance(elt, ast.Name))
    return names


class PythonObjectExtractor(ast.NodeVisitor):
    """
    AST visitor that collects functions, classes, methods, assignments,
    and imports into a dictionary.
    """

    def __init__(self):
        self.objects = {obj: [] for obj in OBJECTS}
        # Keep track of the current class (if any)
        self.current_class = None
        super().__init__()

    def visit_FunctionDef(self, node):
        if self.current_class is None:
            # Top-level (or nested non-method) function.
            self.objects["function"].append([node.name])
            self.objects["function argument"].append([arg.arg for arg in node.args.args])
            doc = ast.get_docstring(node)
            self.objects["function docstring"].append([doc] if doc else [])
            tries, asserts = extract_try_assert(node.body)
            self.objects["function try"].append(tries)
            self.objects["function assert"].append(asserts)
            self.objects["function annotation"].append(extract_annotations(node))
            self.objects["function decorator"].append(get_decorator_ids(node.decorator_list))
        else:
            # Function defined inside a class: treat as a method.
            if node.name == "__init__":
                # In __init__, look for self-assignments (instance attributes)
                attributes = extract_self_attributes(node.body)
                self.objects["attribute"].append(attributes)
            elif not node.name.startswith("__"):
                self.objects["method"].append([node.name])
                doc = ast.get_docstring(node)
                self.objects["method docstring"].append([doc] if doc else [])
                tries, asserts = extract_try_assert(node.body)
                self.objects["method try"].append(tries)
                self.objects["method assert"].append(asserts)
                self.objects["method annotation"].append(extract_annotations(node))
                self.objects["method decorator"].append(get_decorator_ids(node.decorator_list))
        # Continue recursing into the node.
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.objects["class"].append([node.name])
        self.objects["class decorator"].append(get_decorator_ids(node.decorator_list))
        # Set the current class context and visit its children.
        prev_class = self.current_class
        self.current_class = node
        self.generic_visit(node)
        self.current_class = prev_class

    def visit_Assign(self, node):
        for name in extract_variable_names(node):
            self.objects["variable"].append([name])
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        for name in extract_variable_names(node):
            self.objects["variable"].append([name])
        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            self.objects["import"].append(alias.name)
        self.generic_visit(node)


def get_all_objects(python_text_code):
    """
    Returns a dictionary with all objects (functions, classes, methods, etc.)
    extracted from the Python code string.
    """
    tree = ast.parse(python_text_code)
    extractor = PythonObjectExtractor()
    extractor.visit(tree)
    # Collect comments from the source code.
    extractor.objects["comment"] = [re.findall(r"#.*", python_text_code)]
    return extractor.objects
