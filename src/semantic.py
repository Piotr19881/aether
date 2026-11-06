"""Initial Aether type, tensor-shape, unit, and effect analysis."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import re

from src.frontend import (
    Diagnostic,
    Expr,
    ExprStmt,
    FunctionDecl,
    LetStmt,
    Module,
    ReturnStmt,
    SourceSpan,
    TypeDecl,
    TypeExpr,
)


_NUMERIC_TYPES = {
    "i8", "i16", "i32", "i64", "i128",
    "u8", "u16", "u32", "u64", "u128",
    "f16", "bf16", "f32", "f64", "fp8", "i4", "t2",
}
_INTEGER_TYPES = {
    "i8", "i16", "i32", "i64", "i128",
    "u8", "u16", "u32", "u64", "u128", "usize",
}
_FLOAT_TYPES = {"f16", "bf16", "f32", "f64", "fp8"}
_FFI_INTEGER_TYPES = {
    "i8", "i16", "i32", "i64",
    "u8", "u16", "u32", "u64",
}
_FFI_TYPES = _FFI_INTEGER_TYPES | {"f32", "f64"}
@dataclass(frozen=True)
class UnitDefinition:
    dimensions: tuple[tuple[str, int], ...]
    scale: Fraction = Fraction(1)
    offset: Fraction = Fraction(0)
    kind: str = ""
    pi_power: int = 0


_UNITS: dict[str, UnitDefinition] = {
    "mm": UnitDefinition((("length", 1),), Fraction(1, 1000)),
    "cm": UnitDefinition((("length", 1),), Fraction(1, 100)),
    "m": UnitDefinition((("length", 1),)),
    "km": UnitDefinition((("length", 1),), Fraction(1000)),
    "um": UnitDefinition((("length", 1),), Fraction(1, 1_000_000)),
    "nm": UnitDefinition((("length", 1),), Fraction(1, 1_000_000_000)),
    "V": UnitDefinition((("voltage", 1),)),
    "mV": UnitDefinition((("voltage", 1),), Fraction(1, 1000)),
    "uV": UnitDefinition((("voltage", 1),), Fraction(1, 1_000_000)),
    "A": UnitDefinition((("current", 1),)),
    "mA": UnitDefinition((("current", 1),), Fraction(1, 1000)),
    "s": UnitDefinition((("time", 1),)),
    "ms": UnitDefinition((("time", 1),), Fraction(1, 1000)),
    "us": UnitDefinition((("time", 1),), Fraction(1, 1_000_000)),
    "ns": UnitDefinition((("time", 1),), Fraction(1, 1_000_000_000)),
    "Hz": UnitDefinition((("time", -1),)),
    "kHz": UnitDefinition((("time", -1),), Fraction(1000)),
    "kg": UnitDefinition((("mass", 1),)),
    "g": UnitDefinition((("mass", 1),), Fraction(1, 1000)),
    "K": UnitDefinition((("temperature", 1),), kind="absolute_temperature"),
    "degC": UnitDefinition(
        (("temperature", 1),),
        offset=Fraction(5463, 20),
        kind="absolute_temperature",
    ),
    "degF": UnitDefinition(
        (("temperature", 1),),
        Fraction(5, 9),
        Fraction(45967, 180),
        "absolute_temperature",
    ),
    "N": UnitDefinition((("length", 1), ("mass", 1), ("time", -2))),
    "Pa": UnitDefinition((("length", -1), ("mass", 1), ("time", -2))),
    "kPa": UnitDefinition(
        (("length", -1), ("mass", 1), ("time", -2)),
        Fraction(1000),
    ),
    "J": UnitDefinition((("length", 2), ("mass", 1), ("time", -2))),
    "W": UnitDefinition((("length", 2), ("mass", 1), ("time", -3))),
    "rad": UnitDefinition((("angle", 1),)),
    "deg": UnitDefinition((("angle", 1),), Fraction(1, 180), pi_power=1),
}
_EFFECTS = {"io", "net", "fs", "ffi"}


@dataclass(frozen=True)
class AetherType:
    """Resolved type with optional static shape and canonical unit dimensions."""

    name: str
    arguments: tuple[AetherType, ...] = ()
    dimensions: tuple[int, ...] = ()
    unit: tuple[tuple[str, int], ...] = ()
    unit_scale: Fraction = Fraction(1)
    unit_offset: Fraction = Fraction(0)
    unit_symbol: str = ""
    unit_kind: str = ""
    unit_pi_power: int = 0

    @property
    def is_numeric(self) -> bool:
        return self.name in _NUMERIC_TYPES

    @property
    def element_type(self) -> AetherType | None:
        if self.name == "Tensor" and self.arguments:
            return self.arguments[0]
        return None

    def display(self) -> str:
        if self.name == "Tensor" and self.arguments:
            dimensions = ",".join(str(value) for value in self.dimensions)
            unit = self.unit_symbol or _unit_display(
                self.unit,
                self.unit_scale,
                self.unit_pi_power,
            )
            suffix = f",{unit}" if unit else ""
            return f"Tensor<{self.arguments[0].display()},[{dimensions}]{suffix}>"
        if self.arguments:
            return f"{self.name}<" + ",".join(
                argument.display() for argument in self.arguments
            ) + ">"
        return self.name


@dataclass(frozen=True)
class EffectSummary:
    function: str
    declared: tuple[str, ...]
    required: tuple[str, ...]


@dataclass(frozen=True)
class UnitConversion:
    span: SourceSpan
    multiplier: Fraction
    addend: Fraction
    pi_power: int
    target_unit: str


@dataclass(frozen=True)
class SemanticResult:
    expression_types: tuple[tuple[SourceSpan, AetherType], ...]
    effects: tuple[EffectSummary, ...]
    conversions: tuple[UnitConversion, ...]
    diagnostics: tuple[Diagnostic, ...]


@dataclass(frozen=True)
class BuiltinOperation:
    parameters: tuple[AetherType, ...]
    result: AetherType
    effect: str


def _result_type(value: AetherType, error: str) -> AetherType:
    return AetherType("Result", (value, AetherType(error)))


_BUILTIN_OPERATIONS = {
    "io_write": BuiltinOperation(
        (AetherType("str"),),
        _result_type(AetherType("void"), "IoError"),
        "io",
    ),
    "io_read_line": BuiltinOperation(
        (),
        _result_type(AetherType("str"), "IoError"),
        "io",
    ),
    "fs_read_text": BuiltinOperation(
        (AetherType("str"),),
        _result_type(AetherType("str"), "FsError"),
        "fs",
    ),
    "fs_write_text": BuiltinOperation(
        (AetherType("str"), AetherType("str")),
        _result_type(AetherType("void"), "FsError"),
        "fs",
    ),
    "fs_exists": BuiltinOperation(
        (AetherType("str"),),
        _result_type(AetherType("bool"), "FsError"),
        "fs",
    ),
    "net_get": BuiltinOperation(
        (AetherType("str"),),
        _result_type(AetherType("str"), "NetError"),
        "net",
    ),
    "net_post": BuiltinOperation(
        (AetherType("str"), AetherType("str")),
        _result_type(AetherType("str"), "NetError"),
        "net",
    ),
}


def _unit_display(
    unit: tuple[tuple[str, int], ...],
    scale: Fraction = Fraction(1),
    pi_power: int = 0,
) -> str:
    if not unit and scale == 1 and pi_power == 0:
        return ""
    preferred = {
        (("length", 1),): ("m", Fraction(1)),
        (("mass", 1),): ("kg", Fraction(1)),
        (("temperature", 1),): ("K", Fraction(1)),
        (("time", 1),): ("s", Fraction(1)),
        (("time", -1),): ("Hz", Fraction(1)),
        (("voltage", 1),): ("V", Fraction(1)),
        (("current", 1),): ("A", Fraction(1)),
        (("angle", 1),): ("rad", Fraction(1)),
    }
    if unit in preferred and scale == preferred[unit][1] and pi_power == 0:
        return preferred[unit][0]
    dimensions = "*".join(
        dimension if exponent == 1 else f"{dimension}^{exponent}"
        for dimension, exponent in unit
    ) or "1"
    factor = ""
    if scale != 1:
        factor = str(scale)
    if pi_power:
        pi_factor = "pi" if pi_power == 1 else f"pi^{pi_power}"
        factor = f"{factor}*{pi_factor}" if factor else pi_factor
    return dimensions if not factor else f"{dimensions}*{factor}"


def _combine_units(
    left: tuple[tuple[str, int], ...],
    right: tuple[tuple[str, int], ...],
    sign: int,
) -> tuple[tuple[str, int], ...]:
    result = dict(left)
    for dimension, exponent in right:
        result[dimension] = result.get(dimension, 0) + sign * exponent
    return tuple(sorted((dimension, exponent) for dimension, exponent in result.items() if exponent))


def _resolve_type(type_expr: TypeExpr) -> AetherType:
    if type_expr.name == "[]":
        return AetherType(
            "Shape",
            dimensions=tuple(
                int(value) for value in type_expr.dimensions if value.isdecimal()
            ),
        )
    arguments = tuple(_resolve_type(argument) for argument in type_expr.arguments)
    if type_expr.name == "Tensor" and len(type_expr.arguments) >= 2:
        shape_expr = type_expr.arguments[1]
        dimensions = tuple(
            int(value) for value in shape_expr.dimensions if value.isdecimal()
        ) if shape_expr.name == "[]" else ()
        unit = UnitDefinition(())
        unit_symbol = ""
        if len(type_expr.arguments) >= 3:
            unit_symbol = type_expr.arguments[2].name
            unit = _UNITS.get(unit_symbol, UnitDefinition(()))
        return AetherType(
            "Tensor",
            (arguments[0],),
            dimensions,
            unit.dimensions,
            unit.scale,
            unit.offset,
            unit_symbol,
            unit.kind,
            unit.pi_power,
        )
    return AetherType(type_expr.name, arguments)


def _same_shape(left: AetherType, right: AetherType) -> bool:
    return left.name == right.name == "Tensor" and left.dimensions == right.dimensions


def _compatible(left: AetherType, right: AetherType) -> bool:
    if left.name == "Tensor" or right.name == "Tensor":
        return (
            _same_shape(left, right)
            and left.arguments == right.arguments
            and _same_unit(left, right)
        )
    if left.is_numeric and right.is_numeric:
        return left.name == right.name and _same_unit(left, right)
    return left == right


def _same_unit(left: AetherType, right: AetherType) -> bool:
    return (
        left.unit == right.unit
        and left.unit_scale == right.unit_scale
        and left.unit_offset == right.unit_offset
        and left.unit_kind == right.unit_kind
        and left.unit_pi_power == right.unit_pi_power
    )


def _with_unit(value: AetherType, unit_symbol: str) -> AetherType:
    definition = _UNITS[unit_symbol]
    return AetherType(
        value.name,
        value.arguments,
        value.dimensions,
        definition.dimensions,
        definition.scale,
        definition.offset,
        unit_symbol,
        definition.kind,
        definition.pi_power,
    )


def _combine_unit_types(
    left: AetherType,
    right: AetherType,
    sign: int,
) -> tuple[tuple[tuple[str, int], ...], Fraction, int]:
    return (
        _combine_units(left.unit, right.unit, sign),
        left.unit_scale * (right.unit_scale if sign > 0 else 1 / right.unit_scale),
        left.unit_pi_power + sign * right.unit_pi_power,
    )


class _Analyzer:
    def __init__(self, module: Module) -> None:
        self.module = module
        self.functions = {
            declaration.name: declaration
            for declaration in module.declarations
            if isinstance(declaration, FunctionDecl)
        }
        self.types = {
            declaration.name: declaration
            for declaration in module.declarations
            if isinstance(declaration, TypeDecl)
        }
        self.diagnostics: list[Diagnostic] = []
        self.expression_types: list[tuple[SourceSpan, AetherType]] = []
        self.conversions: list[UnitConversion] = []
        self.current_return_type: AetherType | None = None

    def error(self, code: str, message: str, span: SourceSpan) -> None:
        self.diagnostics.append(Diagnostic(code, message, span))

    def analyze(self) -> SemanticResult:
        summaries = self._analyze_effects()
        for function in self.functions.values():
            if function.name in _BUILTIN_OPERATIONS:
                self.error(
                    "E_RESERVED_NAME",
                    f"Function name {function.name!r} is reserved for a built-in operation",
                    function.span,
                )
            if function.external_abi is not None:
                self._validate_extern_function(function)
        for declaration in self.types.values():
            for field in declaration.fields:
                self._validate_type(field.type)
        for function in self.functions.values():
            self._analyze_function(function)
        return SemanticResult(
            tuple(self.expression_types),
            summaries,
            tuple(self.conversions),
            tuple(self.diagnostics),
        )

    def _validate_extern_function(self, function: FunctionDecl) -> None:
        if function.external_abi != "C":
            self.error(
                "E_EXTERN_ABI",
                f"Unsupported foreign ABI {function.external_abi!r}",
                function.span,
            )
        link_attributes = [
            attribute for attribute in function.attributes
            if attribute.name == "link"
        ]
        if len(link_attributes) != 1 or function.external_library is None:
            self.error(
                "E_EXTERN_LINK",
                'An extern "C" declaration requires exactly one @link("library")',
                function.span,
            )
        elif not re.fullmatch(r"[A-Za-z0-9_.+-]+", function.external_library):
            self.error(
                "E_EXTERN_LINK",
                "Library name must be a simple linker name, not a path",
                link_attributes[0].span,
            )
        if function.return_type is None:
            self.error(
                "E_EXTERN_RETURN_TYPE",
                "Extern declarations require an explicit return type (use void for no value)",
                function.span,
            )
        for parameter in function.parameters:
            if not self._is_ffi_type(parameter.type, allow_void=False):
                self.error(
                    "E_FFI_TYPE",
                    f"Type {parameter.type.name!r} is not supported in a C ABI parameter",
                    parameter.type.span,
                )
        if function.return_type is not None and not self._is_ffi_type(
            function.return_type,
            allow_void=True,
        ):
            self.error(
                "E_FFI_TYPE",
                f"Type {function.return_type.name!r} is not supported as a C ABI return type",
                function.return_type.span,
            )

    @staticmethod
    def _is_ffi_type(type_expr: TypeExpr, *, allow_void: bool) -> bool:
        if type_expr.arguments or type_expr.dimensions:
            return False
        if type_expr.name == "void":
            return allow_void
        return type_expr.name in _FFI_TYPES

    def _validate_type(self, type_expr: TypeExpr) -> None:
        for argument in type_expr.arguments:
            self._validate_type(argument)
        if type_expr.name == "Result":
            if len(type_expr.arguments) != 2:
                self.error(
                    "E_RESULT_TYPE",
                    "Result type requires a value type and an error type",
                    type_expr.span,
                )
            elif type_expr.arguments[1].name not in {
                "IoError",
                "FsError",
                "NetError",
            }:
                self.error(
                    "E_RESULT_ERROR_TYPE",
                    "Result error type must be IoError, FsError, or NetError",
                    type_expr.arguments[1].span,
                )
            return
        if type_expr.name != "Tensor":
            return
        if len(type_expr.arguments) < 2:
            self.error(
                "E_TENSOR_SHAPE",
                "Tensor type requires an element type and a static shape",
                type_expr.span,
            )
            return
        if len(type_expr.arguments) > 3:
            self.error(
                "E_TENSOR_TYPE",
                "Tensor type accepts an element type, shape, and optional unit",
                type_expr.span,
            )
        element_type = type_expr.arguments[0]
        if element_type.name not in _NUMERIC_TYPES:
            self.error(
                "E_TENSOR_ELEMENT_TYPE",
                "Tensor element type must be a supported numeric type",
                element_type.span,
            )
        shape = type_expr.arguments[1]
        if shape.name != "[]":
            self.error(
                "E_TENSOR_SHAPE",
                "Tensor shape must use bracketed dimensions",
                shape.span,
            )
            return
        for dimension in shape.dimensions:
            if not dimension.isdecimal() or int(dimension) <= 0:
                self.error(
                    "E_TENSOR_DIMENSION",
                    f"Tensor dimension {dimension!r} must be a positive integer",
                    shape.span,
                )
        if len(type_expr.arguments) == 3:
            unit = type_expr.arguments[2]
            if unit.name not in _UNITS or unit.arguments:
                self.error(
                    "E_UNKNOWN_UNIT",
                    f"Unknown or unsupported unit {unit.name!r}",
                    unit.span,
                )

    def _analyze_effects(self) -> tuple[EffectSummary, ...]:
        declared = {
            name: {
                attribute.name
                for attribute in function.attributes
                if attribute.name in _EFFECTS
            }
            for name, function in self.functions.items()
        }
        calls: dict[str, list[tuple[str, SourceSpan]]] = {
            name: [] for name in self.functions
        }
        required = {name: set() for name in self.functions}
        for name, function in self.functions.items():
            if function.external_abi is not None:
                required[name].add("ffi")
            for statement_index, statement in enumerate(function.body):
                expression = _statement_expression(statement)
                if expression is not None:
                    for node in _walk(expression):
                        if node.kind == "call" and node.children[0].kind == "identifier":
                            callee = node.children[0].value
                            if callee in self.functions:
                                calls[name].append((callee, node.span))
                            elif callee in _BUILTIN_OPERATIONS:
                                required[name].add(
                                    _BUILTIN_OPERATIONS[callee].effect
                                )

        changed = True
        while changed:
            changed = False
            for caller, callees in calls.items():
                for callee, _ in callees:
                    new_effects = required[caller] | required[callee]
                    if new_effects != required[caller]:
                        required[caller] = new_effects
                        changed = True

        for function_name in self.functions:
            missing = required[function_name] - declared[function_name]
            if missing:
                self.error(
                    "E_EFFECT_VIOLATION",
                    f"Function {function_name!r} must declare effect(s): "
                    + ", ".join(f"@{effect}" for effect in sorted(missing)),
                    self.functions[function_name].span,
                )
        return tuple(
            EffectSummary(
                name,
                tuple(sorted(declared[name])),
                tuple(sorted(required[name])),
            )
            for name in sorted(self.functions)
        )

    def _analyze_function(self, function: FunctionDecl) -> None:
        for parameter in function.parameters:
            self._validate_type(parameter.type)
        environment = {
            parameter.name: _resolve_type(parameter.type)
            for parameter in function.parameters
        }
        expected_return = (
            _resolve_type(function.return_type)
            if function.return_type is not None
            else None
        )
        self.current_return_type = expected_return
        if function.return_type is not None:
            self._validate_type(function.return_type)
        result_bindings: dict[str, SourceSpan] = {}
        handled_results: set[str] = set()
        reachable = True
        for statement in function.body:
            was_reachable = reachable
            statement_expression = _statement_expression(statement)
            if was_reachable and statement_expression is not None:
                referenced = {
                    node.value
                    for node in _walk(statement_expression)
                    if node.kind == "identifier" and node.value in handled_results
                }
                for name in sorted(referenced):
                    self.error(
                        "E_RESULT_REUSE",
                        f"Result binding {name!r} was already consumed",
                        statement.span,
                    )
            if isinstance(statement, LetStmt):
                inferred = self._infer(statement.value, environment)
                if statement.annotation is not None:
                    self._validate_type(statement.annotation)
                annotation = (
                    _resolve_type(statement.annotation)
                    if statement.annotation is not None
                    else None
                )
                if annotation is not None and inferred is not None and not _compatible(annotation, inferred):
                    self.error(
                        "E_TYPE_MISMATCH",
                        f"Binding {statement.name!r} declares {annotation.display()} "
                        f"but receives {inferred.display()}",
                        statement.span,
                    )
                environment[statement.name] = annotation or inferred or AetherType("Unknown")
                if reachable and inferred is not None and inferred.name == "Result":
                    result_bindings[statement.name] = statement.span
                elif reachable and statement.value.kind == "identifier":
                    source_name = statement.value.value
                    if source_name in result_bindings:
                        result_bindings.pop(source_name)
                        if inferred is not None and inferred.name == "Result":
                            result_bindings[statement.name] = statement.span
            elif isinstance(statement, ReturnStmt):
                actual = (
                    self._infer(statement.value, environment)
                    if statement.value is not None
                    else AetherType("void")
                )
                propagates_result = (
                    expected_return is not None
                    and expected_return.name == "Result"
                    and len(expected_return.arguments) == 2
                    and actual == expected_return.arguments[0]
                    and statement.value is not None
                    and any(
                        node.kind == "try"
                        for node in _walk(statement.value)
                    )
                )
                if (
                    expected_return is not None
                    and actual is not None
                    and not _compatible(expected_return, actual)
                    and not propagates_result
                ):
                    self.error(
                        "E_RETURN_TYPE",
                        f"Function {function.name!r} returns {actual.display()}, "
                        f"expected {expected_return.display()}",
                        statement.span,
                    )
                if statement.value is not None:
                    consumed = _consumed_result_names(statement.value, "return")
                    if was_reachable:
                        for name in consumed:
                            if name in result_bindings:
                                result_bindings.pop(name)
                                handled_results.add(name)
                reachable = False
            elif isinstance(statement, ExprStmt):
                actual = self._infer(statement.expression, environment)
                if actual is not None and actual.name == "Result":
                    self.error(
                        "E_UNHANDLED_RESULT",
                        "Result value must be returned, matched, propagated with '?', or bound and used",
                        statement.span,
                    )
            if isinstance(statement, LetStmt):
                consumed = _consumed_result_names(statement.value, "bind")
            elif isinstance(statement, ExprStmt):
                consumed = _consumed_result_names(statement.expression, "expression")
            else:
                consumed = set()
            for name in (consumed if was_reachable else ()):
                if name in result_bindings:
                    result_bindings.pop(name)
                    handled_results.add(name)
                    if (
                        isinstance(statement, LetStmt)
                        and statement.value.kind == "identifier"
                        and statement.value.value == name
                        and environment.get(statement.name, AetherType("Unknown")).name
                        == "Result"
                    ):
                        result_bindings[statement.name] = statement.span

        for name, span in result_bindings.items():
            self.error(
                "E_UNHANDLED_RESULT",
                f"Result binding {name!r} is never handled, propagated, or returned",
                span,
            )
        self.current_return_type = None

    def _record(self, expression: Expr, value: AetherType | None) -> AetherType | None:
        if value is not None:
            self.expression_types.append((expression.span, value))
        return value

    def _infer(
        self,
        expression: Expr | None,
        environment: dict[str, AetherType],
        pipeline_input: AetherType | None = None,
    ) -> AetherType | None:
        if expression is None:
            return None
        if expression.kind == "literal":
            value = expression.value
            if value in {"true", "false"}:
                return self._record(expression, AetherType("bool"))
            if value == "none":
                return self._record(expression, AetherType("None"))
            if value.startswith(("'", '"')):
                return self._record(expression, AetherType("str"))
            return self._record(
                expression,
                AetherType("f32" if any(char in value for char in ".eE") else "i32"),
            )
        if expression.kind == "identifier":
            return self._record(expression, environment.get(expression.value))
        if expression.kind == "group":
            value = self._infer(expression.children[0], environment)
            return self._record(expression, value)
        if expression.kind == "try":
            result_type = self._infer(expression.children[0], environment)
            if result_type is None:
                return None
            if result_type.name != "Result" or len(result_type.arguments) != 2:
                self.error(
                    "E_TRY_TYPE",
                    "The '?' operator requires a Result<T, E> value",
                    expression.span,
                )
                return None
            if (
                self.current_return_type is None
                or self.current_return_type.name != "Result"
                or len(self.current_return_type.arguments) != 2
            ):
                self.error(
                    "E_TRY_CONTEXT",
                    "The '?' operator is only valid in a function returning Result<T, E>",
                    expression.span,
                )
                return None
            source_error = result_type.arguments[1]
            return_error = self.current_return_type.arguments[1]
            if source_error != return_error:
                self.error(
                    "E_TRY_ERROR_TYPE",
                    f"Cannot propagate {source_error.display()} from a function returning "
                    f"{return_error.display()}",
                    expression.span,
                )
                return None
            return self._record(expression, result_type.arguments[0])
        if expression.kind == "match":
            result_type = self._infer(expression.children[0], environment)
            if result_type is None:
                return None
            if result_type.name != "Result" or len(result_type.arguments) != 2:
                self.error(
                    "E_MATCH_TYPE",
                    "Result match requires a Result<T, E> value",
                    expression.span,
                )
                return None
            seen: set[str] = set()
            branch_types: list[AetherType] = []
            for arm in expression.arms:
                if arm.variant in seen:
                    self.error(
                        "E_MATCH_DUPLICATE_ARM",
                        f"Duplicate {arm.variant} arm",
                        arm.span,
                    )
                seen.add(arm.variant)
                arm_environment = dict(environment)
                if arm.binding is not None:
                    payload_index = 0 if arm.variant == "Ok" else 1
                    arm_environment[arm.binding] = result_type.arguments[payload_index]
                inferred_arm = self._infer(arm.value, arm_environment)
                if inferred_arm is not None:
                    branch_types.append(inferred_arm)
            if seen != {"Ok", "Err"}:
                missing = {"Ok", "Err"} - seen
                self.error(
                    "E_MATCH_NON_EXHAUSTIVE",
                    "Result match must include " + " and ".join(sorted(missing)),
                    expression.span,
                )
            if not branch_types:
                return None
            result = branch_types[0]
            for branch_type in branch_types[1:]:
                if not _compatible(result, branch_type):
                    self.error(
                        _mismatch_code(result, branch_type),
                        "Result match arms must produce compatible types",
                        expression.span,
                    )
                    return None
            return self._record(expression, result)
        if expression.kind == "member":
            base = self._infer(expression.children[0], environment)
            if expression.value == "?.":
                self.error(
                    "E_OPTIONAL_MEMBER_UNSUPPORTED",
                    "Optional member access is not supported until an option type is defined",
                    expression.span,
                )
                return None
            member = expression.children[1].value
            declaration = self.types.get(base.name) if base is not None else None
            if declaration is None:
                return self._record(expression, None)
            field = next((field for field in declaration.fields if field.name == member), None)
            if field is None:
                self.error(
                    "E_UNKNOWN_FIELD",
                    f"Type {base.name!r} has no field {member!r}",
                    expression.children[1].span,
                )
                return None
            return self._record(expression, _resolve_type(field.type))
        if expression.kind == "index":
            base = self._infer(expression.children[0], environment)
            index_type = self._infer(expression.children[1], environment)
            if index_type is not None and index_type.name not in _INTEGER_TYPES:
                self.error("E_INDEX_TYPE", "Tensor index must be an integer", expression.children[1].span)
            if base is not None and base.name == "Tensor" and base.arguments:
                dimensions = base.dimensions[1:] if len(base.dimensions) > 1 else ()
                element = base.arguments[0]
                result = (
                    AetherType(
                        "Tensor",
                        (element,),
                        dimensions,
                        base.unit,
                        base.unit_scale,
                        base.unit_offset,
                        base.unit_symbol,
                        base.unit_kind,
                        base.unit_pi_power,
                    )
                    if dimensions
                    else AetherType(
                        element.name,
                        element.arguments,
                        unit=base.unit,
                        unit_scale=base.unit_scale,
                        unit_offset=base.unit_offset,
                        unit_symbol=base.unit_symbol,
                        unit_kind=base.unit_kind,
                        unit_pi_power=base.unit_pi_power,
                    )
                )
                return self._record(expression, result)
            return self._record(expression, None)
        if expression.kind == "unary":
            operand = self._infer(expression.children[0], environment)
            if operand is not None and expression.value in {"+", "-"} and not operand.is_numeric:
                self.error("E_TYPE_OPERATOR", f"Unary {expression.value} requires a numeric value", expression.span)
                return None
            if operand is not None and expression.value == "!" and operand.name != "bool":
                self.error("E_TYPE_OPERATOR", "Unary ! requires bool", expression.span)
                return None
            return self._record(expression, AetherType("bool") if expression.value == "!" else operand)
        if expression.kind == "binary":
            if expression.value == "|>":
                left = self._infer(expression.children[0], environment)
                if left is None:
                    return self._record(expression, None)
                right_expression = expression.children[1]
                if right_expression.kind == "call":
                    value = self._infer_call(
                        right_expression,
                        environment,
                        pipeline_input=left,
                    )
                elif right_expression.kind == "identifier":
                    call = Expr(
                        "call",
                        right_expression.span,
                        children=(right_expression,),
                    )
                    value = self._infer_call(call, environment, pipeline_input=left)
                else:
                    value = self._infer(right_expression, environment, left)
                return self._record(expression, value)
            left = self._infer(expression.children[0], environment)
            right = self._infer(expression.children[1], environment)
            return self._record(
                expression,
                self._infer_binary(expression, left, right),
            )
        if expression.kind == "call":
            return self._record(
                expression,
                self._infer_call(expression, environment, pipeline_input),
            )
        return None

    def _infer_binary(
        self,
        expression: Expr,
        left: AetherType | None,
        right: AetherType | None,
    ) -> AetherType | None:
        if left is None or right is None:
            return None
        operator = expression.value
        if operator in {"&&", "||"}:
            if left.name != "bool" or right.name != "bool":
                self.error(
                    "E_TYPE_OPERATOR",
                    f"Operator {operator} requires bool operands",
                    expression.span,
                )
                return None
            return AetherType("bool")
        if operator in {"==", "!=", "<", ">", "<=", ">="}:
            if not _compatible(left, right):
                self.error(
                    _mismatch_code(left, right),
                    f"Cannot compare {left.display()} with {right.display()}",
                    expression.span,
                )
            return AetherType("bool")
        if operator in {"+", "-"}:
            if left.unit_kind == "absolute_temperature" or right.unit_kind == "absolute_temperature":
                self.error(
                    "E_TEMPERATURE_ARITHMETIC",
                    "Arithmetic on absolute temperatures requires an explicit conversion or delta type",
                    expression.span,
                )
                return None
            valid_left = left.is_numeric or left.name == "Tensor"
            valid_right = right.is_numeric or right.name == "Tensor"
            if not valid_left or not valid_right:
                self.error(
                    "E_TYPE_OPERATOR",
                    f"Operator {operator} requires numeric or tensor operands",
                    expression.span,
                )
                return None
            if left.name == "Tensor" and right.is_numeric:
                if (
                    left.arguments[0].name == right.name
                    and _same_unit(left, right)
                ):
                    return left
            if right.name == "Tensor" and left.is_numeric:
                if (
                    right.arguments[0].name == left.name
                    and _same_unit(right, left)
                ):
                    return right
            if not _compatible(left, right):
                self.error(
                    _mismatch_code(left, right),
                    f"Operator {operator} cannot combine {left.display()} and {right.display()}",
                    expression.span,
                )
                return None
            if left.name == "Tensor":
                return left
            return _numeric_result(left, right)
        if operator in {"*", "/"}:
            if (
                left.unit_kind == "absolute_temperature"
                or right.unit_kind == "absolute_temperature"
            ):
                self.error(
                    "E_TEMPERATURE_ARITHMETIC",
                    "Absolute temperatures cannot be multiplied or divided",
                    expression.span,
                )
                return None
            if left.name == "Tensor" and right.is_numeric:
                if left.arguments[0].name != right.name:
                    self.error(
                        "E_TYPE_OPERATOR",
                        "Tensor and scalar element types must match; convert explicitly",
                        expression.span,
                    )
                    return None
                unit, scale, pi_power = _combine_unit_types(
                    left,
                    right,
                    1 if operator == "*" else -1,
                )
                return AetherType(
                    "Tensor",
                    left.arguments,
                    left.dimensions,
                    unit,
                    scale,
                    unit_pi_power=pi_power,
                )
            if right.name == "Tensor" and left.is_numeric and operator == "*":
                if right.arguments[0].name != left.name:
                    self.error(
                        "E_TYPE_OPERATOR",
                        "Tensor and scalar element types must match; convert explicitly",
                        expression.span,
                    )
                    return None
                unit, scale, pi_power = _combine_unit_types(right, left, 1)
                return AetherType(
                    "Tensor",
                    right.arguments,
                    right.dimensions,
                    unit,
                    scale,
                    unit_pi_power=pi_power,
                )
            if left.is_numeric and right.is_numeric:
                if left.name != right.name:
                    self.error(
                        "E_TYPE_OPERATOR",
                        f"Operator {operator} requires matching numeric types, got "
                        f"{left.name} and {right.name}",
                        expression.span,
                    )
                    return None
                unit, scale, pi_power = _combine_unit_types(
                    left,
                    right,
                    1 if operator == "*" else -1,
                )
                return AetherType(
                    left.name,
                    unit=unit,
                    unit_scale=scale,
                    unit_pi_power=pi_power,
                )
        if operator == "%" and left.is_numeric and right.is_numeric:
            if not _compatible(left, right):
                self.error(
                    _mismatch_code(left, right),
                    "Modulo operands must have identical types and units",
                    expression.span,
                )
                return None
            if left.unit_kind == "absolute_temperature":
                self.error(
                    "E_TEMPERATURE_ARITHMETIC",
                    "Modulo is not defined for absolute temperatures",
                    expression.span,
                )
                return None
            return left
        if operator == "**" and left.is_numeric and right.is_numeric:
            if right.unit or right.unit_kind or left.unit or left.unit_kind:
                self.error(
                    "E_UNIT_POWER",
                    "Exponentiation of unitful values is not yet supported",
                    expression.span,
                )
                return None
            if left.name != right.name:
                self.error(
                    "E_TYPE_OPERATOR",
                    "Exponentiation operands must have matching numeric types",
                    expression.span,
                )
                return None
            return left
        self.error(
            "E_TYPE_OPERATOR",
            f"Operator {operator} is not defined for {left.display()} and {right.display()}",
            expression.span,
        )
        return None

    def _infer_call(
        self,
        expression: Expr,
        environment: dict[str, AetherType],
        pipeline_input: AetherType | None = None,
    ) -> AetherType | None:
        callee = expression.children[0]
        if callee.kind != "identifier":
            self._infer(callee, environment)
            return None
        name = callee.value
        if name == "convert":
            raw_arguments = list(expression.children[1:])
            expected_arity = 2 if pipeline_input is None else 1
            if len(raw_arguments) != expected_arity:
                self.error(
                    "E_BUILTIN_ARITY",
                    "convert expects a value and unit string"
                    if pipeline_input is None
                    else "Pipeline convert expects one unit string",
                    expression.span,
                )
                return None
            if pipeline_input is not None:
                value_type = pipeline_input
                unit_expression = raw_arguments[0]
            else:
                value_type = self._infer(raw_arguments[0], environment)
                unit_expression = raw_arguments[1]
            if value_type is None or unit_expression is None:
                return None
            if unit_expression.kind != "literal" or not unit_expression.value.startswith(("'", '"')):
                self.error(
                    "E_BUILTIN_TYPE",
                    "convert unit must be a quoted unit name",
                    unit_expression.span,
                )
                return None
            target_name = unit_expression.value[1:-1]
            target = _UNITS.get(target_name)
            if target is None:
                self.error(
                    "E_UNKNOWN_UNIT",
                    f"Unknown unit {target_name!r}",
                    unit_expression.span,
                )
                return None
            if value_type.name != "Tensor" and not value_type.is_numeric:
                self.error(
                    "E_BUILTIN_TYPE",
                    "convert requires a numeric scalar or tensor",
                    expression.span,
                )
                return None
            if value_type.unit != target.dimensions:
                self.error(
                    "E_UNIT_MISMATCH",
                    f"Cannot convert {value_type.display()} to {target_name}",
                    expression.span,
                )
                return None
            if (
                value_type.unit_kind != target.kind
                and (value_type.unit_kind or target.kind)
            ):
                self.error(
                    "E_UNIT_MISMATCH",
                    "Conversion between absolute and non-absolute units is not supported",
                    expression.span,
                )
                return None
            self.conversions.append(
                UnitConversion(
                    expression.span,
                    value_type.unit_scale / target.scale,
                    (value_type.unit_offset - target.offset) / target.scale,
                    value_type.unit_pi_power - target.pi_power,
                    target_name,
                )
            )
            return _with_unit(value_type, target_name)

        arguments = [
            self._infer(argument, environment)
            for argument in expression.children[1:]
        ]
        if pipeline_input is not None:
            arguments.insert(0, pipeline_input)
        if any(argument is None for argument in arguments):
            return None
        concrete = [argument for argument in arguments if argument is not None]

        function = self.functions.get(name)
        if function is not None:
            if len(concrete) != len(function.parameters):
                self.error(
                    "E_CALL_ARITY",
                    f"Function {name!r} expects {len(function.parameters)} arguments, got {len(concrete)}",
                    expression.span,
                )
            for parameter, argument in zip(function.parameters, concrete):
                expected = _resolve_type(parameter.type)
                if not _compatible(expected, argument):
                    self.error(
                        "E_ARGUMENT_TYPE",
                        f"Argument {parameter.name!r} expects {expected.display()}, got {argument.display()}",
                        expression.span,
                    )
            return _resolve_type(function.return_type) if function.return_type else None

        operation = _BUILTIN_OPERATIONS.get(name)
        if operation is not None:
            if len(concrete) != len(operation.parameters):
                return self._builtin_arity_error(
                    name,
                    len(operation.parameters),
                    expression.span,
                )
            for index, (expected, actual) in enumerate(
                zip(operation.parameters, concrete),
                start=1,
            ):
                if not _compatible(expected, actual):
                    self.error(
                        "E_ARGUMENT_TYPE",
                        f"Argument {index} to {name} expects "
                        f"{expected.display()}, got {actual.display()}",
                        expression.span,
                    )
            return operation.result

        if name in {"mean", "max_abs", "sum"}:
            if len(concrete) != 1:
                return self._builtin_arity_error(name, 1, expression.span)
            argument = concrete[0]
            if argument.name != "Tensor" or not argument.arguments:
                return self._builtin_type_error(
                    name,
                    "expects a tensor argument",
                    expression.span,
                )
            element = argument.arguments[0]
            if name == "mean" and element.name not in _FLOAT_TYPES:
                element = AetherType("f32")
            return AetherType(
                element.name,
                element.arguments,
                unit=argument.unit,
                unit_scale=argument.unit_scale,
                unit_offset=argument.unit_offset,
                unit_symbol=argument.unit_symbol,
                unit_kind=argument.unit_kind,
                unit_pi_power=argument.unit_pi_power,
            )
        if name in {"scale", "subtract"}:
            if len(concrete) != 2:
                return self._builtin_arity_error(name, 2, expression.span)
            return self._infer_binary(
                Expr("binary", expression.span, "*" if name == "scale" else "-", ()),
                concrete[0],
                concrete[1],
            )
        if name == "abs":
            if len(concrete) != 1:
                return self._builtin_arity_error(name, 1, expression.span)
            if concrete[0].name != "Tensor" and not concrete[0].is_numeric:
                return self._builtin_type_error(
                    name,
                    "expects a numeric scalar or tensor",
                    expression.span,
                )
            return concrete[0]
        if name in {"min", "max"}:
            if len(concrete) != 2:
                return self._builtin_arity_error(name, 2, expression.span)
            if not _compatible(concrete[0], concrete[1]):
                self.error(
                    _mismatch_code(concrete[0], concrete[1]),
                    f"{name} arguments must have matching types, shapes, and units",
                    expression.span,
                )
                return None
            if concrete[0].name != "Tensor" and not concrete[0].is_numeric:
                return self._builtin_type_error(
                    name,
                    "expects numeric or tensor arguments",
                    expression.span,
                )
            return concrete[0]
        if name == "len":
            if len(concrete) != 1:
                return self._builtin_arity_error(name, 1, expression.span)
            if concrete[0].name != "Tensor":
                return self._builtin_type_error(name, "expects a tensor", expression.span)
            return AetherType("i64")
        return None

    def _builtin_arity_error(
        self,
        name: str,
        expected: int,
        span: SourceSpan,
    ) -> None:
        self.error(
            "E_BUILTIN_ARITY",
            f"{name} expects {expected} argument(s)",
            span,
        )
        return None

    def _builtin_type_error(
        self,
        name: str,
        message: str,
        span: SourceSpan,
    ) -> None:
        self.error("E_BUILTIN_TYPE", f"{name} {message}", span)
        return None


def _numeric_result(left: AetherType, right: AetherType) -> AetherType:
    order = {"i4": 0, "t2": 0, "i8": 1, "u8": 1, "i16": 2, "u16": 2,
             "f16": 3, "bf16": 3, "i32": 4, "u32": 4, "f32": 5,
             "i64": 6, "u64": 6, "f64": 7, "i128": 8, "u128": 8, "fp8": 3}
    name = max((left.name, right.name), key=lambda item: order.get(item, 0))
    return AetherType(
        name,
        unit=left.unit,
        unit_scale=left.unit_scale,
        unit_offset=left.unit_offset,
        unit_symbol=left.unit_symbol,
        unit_kind=left.unit_kind,
        unit_pi_power=left.unit_pi_power,
    )


def _consumed_result_names(expression: Expr, context: str) -> set[str]:
    """Return only bindings consumed in supported Result-handling positions."""
    def direct_name(value: Expr) -> str | None:
        while value.kind == "group" and value.children:
            value = value.children[0]
        return value.value if value.kind == "identifier" else None

    def guaranteed(value: Expr) -> set[str]:
        consumed: set[str] = set()
        if value.kind == "try" and value.children:
            name = direct_name(value.children[0])
            if name is not None:
                consumed.add(name)
            consumed.update(guaranteed(value.children[0]))
            return consumed
        if value.kind == "match" and value.children:
            name = direct_name(value.children[0])
            if name is not None:
                consumed.add(name)
            consumed.update(guaranteed(value.children[0]))
            if value.arms:
                arm_consumption = [guaranteed(arm.value) for arm in value.arms]
                consumed.update(set.intersection(*arm_consumption))
            return consumed
        if value.kind == "binary" and value.value in {"&&", "||"}:
            return guaranteed(value.children[0])
        for child in value.children:
            consumed.update(guaranteed(child))
        return consumed

    consumed = guaranteed(expression)
    if context in {"return", "bind", "expression"}:
        if context in {"return", "bind"}:
            name = direct_name(expression)
            if name is not None:
                consumed.add(name)
    return consumed


def _mismatch_code(left: AetherType, right: AetherType) -> str:
    if left.name == "Tensor" and right.name == "Tensor":
        if left.dimensions != right.dimensions:
            return "E_SHAPE_MISMATCH"
        if left.arguments != right.arguments:
            return "E_TYPE_MISMATCH"
    if left.unit != right.unit or not _same_unit(left, right):
        return "E_UNIT_MISMATCH"
    return "E_TYPE_MISMATCH"


def _statement_expression(statement):
    if isinstance(statement, LetStmt):
        return statement.value
    if isinstance(statement, ReturnStmt):
        return statement.value
    if isinstance(statement, ExprStmt):
        return statement.expression
    return None


def _walk(expression: Expr):
    yield expression
    for child in expression.children:
        yield from _walk(child)
    for arm in expression.arms:
        yield from _walk(arm.value)


def analyze(module: Module) -> SemanticResult:
    """Analyze the supported subset without executing code or loading AI models."""
    return _Analyzer(module).analyze()


def resolve_type(type_expr: TypeExpr) -> AetherType:
    """Resolve a syntax-level type using the prototype's built-in unit table."""
    return _resolve_type(type_expr)
