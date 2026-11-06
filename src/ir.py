"""Typed intermediate representation for the supported Aether source subset."""

from __future__ import annotations

from dataclasses import dataclass, replace

from src.frontend import (
    Expr,
    ExprStmt,
    FunctionDecl,
    LetStmt,
    Module,
    ReturnStmt,
    SourceSpan,
    TypeDecl,
)
from src.semantic import (
    AetherType,
    Diagnostic,
    SemanticResult,
    UnitConversion,
    analyze,
    resolve_type,
)


@dataclass(frozen=True)
class IRMatchArm:
    variant: str
    binding: str | None
    value: IRExpression
    span: SourceSpan


@dataclass(frozen=True)
class IRExpression:
    kind: str
    span: SourceSpan
    type: AetherType
    value: str = ""
    children: tuple[IRExpression, ...] = ()
    arms: tuple[IRMatchArm, ...] = ()
    conversion: UnitConversion | None = None


@dataclass(frozen=True)
class IRStatement:
    kind: str
    span: SourceSpan
    name: str = ""
    binding_kind: str = ""
    expression: IRExpression | None = None


@dataclass(frozen=True)
class IRParameter:
    name: str
    type: AetherType


@dataclass(frozen=True)
class IRFunction:
    name: str
    parameters: tuple[IRParameter, ...]
    return_type: AetherType
    has_return_annotation: bool
    declared_effects: tuple[str, ...]
    required_effects: tuple[str, ...]
    body: tuple[IRStatement, ...]
    span: SourceSpan
    external_abi: str | None = None
    external_library: str | None = None


@dataclass(frozen=True)
class IRTypeDeclaration:
    name: str
    fields: tuple[tuple[str, AetherType], ...]
    span: SourceSpan


@dataclass(frozen=True)
class IRModule:
    functions: tuple[IRFunction, ...]
    types: tuple[IRTypeDeclaration, ...]

    def function(self, name: str) -> IRFunction:
        for function in self.functions:
            if function.name == name:
                return function
        raise KeyError(f"Unknown Aether function {name!r}")


class IRLoweringError(ValueError):
    def __init__(self, diagnostics: tuple[Diagnostic, ...]) -> None:
        self.diagnostics = diagnostics
        codes = ", ".join(
            getattr(diagnostic, "code", "unknown")
            for diagnostic in diagnostics
        )
        super().__init__(f"Cannot lower a module with semantic errors: {codes}")


def lower_module(module: Module, semantics: SemanticResult | None = None) -> IRModule:
    """Lower a semantically valid supported-subset module into typed IR."""
    semantics = semantics or analyze(module)
    if any(diagnostic.severity == "error" for diagnostic in semantics.diagnostics):
        raise IRLoweringError(semantics.diagnostics)

    expression_types = {span: type_ for span, type_ in semantics.expression_types}
    conversions = {conversion.span: conversion for conversion in semantics.conversions}

    def lower_expression(expression: Expr) -> IRExpression:
        children = tuple(lower_expression(child) for child in expression.children)
        if (
            expression.kind == "binary"
            and expression.value == "|>"
            and len(children) == 2
            and children[1].kind in {"call", "identifier"}
            and children[1].type.name == "Unknown"
        ):
            children = (
                children[0],
                replace(
                    children[1],
                    type=expression_types.get(
                        expression.span,
                        AetherType("Unknown"),
                    ),
                ),
            )
        expression_type = expression_types.get(expression.span)
        if expression_type is None and expression.kind == "literal":
            if expression.value.startswith(("'", '"')):
                expression_type = AetherType("str")
            elif expression.value in {"true", "false"}:
                expression_type = AetherType("bool")
            elif expression.value == "none":
                expression_type = AetherType("None")
            else:
                expression_type = AetherType(
                    "f32"
                    if any(char in expression.value for char in ".eE")
                    else "i32"
                )
        arms = tuple(
            IRMatchArm(
                arm.variant,
                arm.binding,
                lower_expression(arm.value),
                arm.span,
            )
            for arm in expression.arms
        )
        return IRExpression(
            expression.kind,
            expression.span,
            expression_type or AetherType("Unknown"),
            expression.value,
            children,
            arms,
            conversions.get(expression.span),
        )

    def untyped_expressions(
        expression: IRExpression,
        *,
        callee: bool = False,
    ) -> list[Diagnostic]:
        errors: list[Diagnostic] = []
        if not callee and expression.type.name == "Unknown":
            errors.append(
                Diagnostic(
                    "E_IR_UNTYPED",
                    f"Cannot lower untyped {expression.kind} expression",
                    expression.span,
                )
            )
        if expression.kind == "call":
            if expression.children:
                errors.extend(untyped_expressions(expression.children[0], callee=True))
                for child in expression.children[1:]:
                    errors.extend(untyped_expressions(child))
        elif expression.kind == "member":
            if expression.children:
                errors.extend(untyped_expressions(expression.children[0]))
        else:
            for child in expression.children:
                errors.extend(untyped_expressions(child))
        for arm in expression.arms:
            errors.extend(untyped_expressions(arm.value))
        return errors

    functions: list[IRFunction] = []
    types: list[IRTypeDeclaration] = []
    effect_summaries = {summary.function: summary for summary in semantics.effects}
    for declaration in module.declarations:
        if isinstance(declaration, TypeDecl):
            types.append(
                IRTypeDeclaration(
                    declaration.name,
                    tuple(
                        (field.name, resolve_type(field.type))
                        for field in declaration.fields
                    ),
                    declaration.span,
                )
            )
            continue
        if not isinstance(declaration, FunctionDecl):
            continue
        body: list[IRStatement] = []
        for statement in declaration.body:
            if isinstance(statement, LetStmt):
                body.append(
                    IRStatement(
                        "let",
                        statement.span,
                        statement.name,
                        statement.binding_kind,
                        lower_expression(statement.value),
                    )
                )
            elif isinstance(statement, ReturnStmt):
                body.append(
                    IRStatement(
                        "return",
                        statement.span,
                        expression=(
                            lower_expression(statement.value)
                            if statement.value is not None
                            else None
                        ),
                    )
                )
            elif isinstance(statement, ExprStmt):
                body.append(
                    IRStatement(
                        "expression",
                        statement.span,
                        expression=lower_expression(statement.expression),
                    )
                )
        effect_summary = effect_summaries.get(declaration.name)
        functions.append(
            IRFunction(
                declaration.name,
                tuple(
                    IRParameter(
                        parameter.name,
                        resolve_type(parameter.type),
                    )
                    for parameter in declaration.parameters
                ),
                (
                    resolve_type(declaration.return_type)
                    if declaration.return_type is not None
                    else AetherType("Unknown")
                ),
                declaration.return_type is not None,
                effect_summary.declared if effect_summary else (),
                effect_summary.required if effect_summary else (),
                tuple(body),
                declaration.span,
                declaration.external_abi,
                declaration.external_library,
            )
        )
    ir_module = IRModule(tuple(functions), tuple(types))
    errors = tuple(
        diagnostic
        for function in ir_module.functions
        for statement in function.body
        if statement.expression is not None
        for diagnostic in untyped_expressions(statement.expression)
    )
    if errors:
        raise IRLoweringError(errors)
    return ir_module
