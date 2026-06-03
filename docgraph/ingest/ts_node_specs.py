from __future__ import annotations

_SPECS: dict[str, frozenset[str]] = {
    "python": frozenset({"function_definition", "class_definition", "decorated_definition"}),
    "javascript": frozenset(
        {"function_declaration", "class_declaration", "method_definition", "arrow_function"}
    ),
    "typescript": frozenset(
        {
            "function_declaration",
            "class_declaration",
            "method_definition",
            "interface_declaration",
        }
    ),
    "go": frozenset({"function_declaration", "method_declaration", "type_declaration"}),
    "rust": frozenset(
        {"function_item", "impl_item", "struct_item", "enum_item", "trait_item"}
    ),
    "java": frozenset(
        {
            "method_declaration",
            "class_declaration",
            "interface_declaration",
            "constructor_declaration",
        }
    ),
    "csharp": frozenset(
        {"method_declaration", "class_declaration", "interface_declaration"}
    ),
    "cpp": frozenset({"function_definition", "class_specifier", "struct_specifier"}),
    "c": frozenset({"function_definition", "struct_specifier"}),
    "ruby": frozenset({"method", "class", "module"}),
    "php": frozenset(
        {"function_definition", "method_declaration", "class_declaration"}
    ),
}


def split_node_types(language: str) -> frozenset[str]:
    return _SPECS.get(language, frozenset())
