"""Stable parse, compile, and execution entry points for the Aether prototype."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Sequence

from src.build import compile_c99_object, link_c99_executable
from src.c99 import C99LoweringError, emit_c99_module
from src.frontend import Diagnostic, SourceSpan, parse_source
from src.interpreter import (
    EffectHandler,
    ExternalFunctionHandler,
    Interpreter,
)
from src.ir import IRLoweringError, IRModule, lower_module
from src.parser import ParseError


class CompilationError(ValueError):
    """Compilation stopped because parsing, analysis, or lowering failed."""

    def __init__(self, diagnostics: tuple[Diagnostic, ...]) -> None:
        self.diagnostics = diagnostics
        summary = "; ".join(
            f"{diagnostic.code}: {diagnostic.message}"
            for diagnostic in diagnostics
        )
        super().__init__(summary or "Aether compilation failed")


def compile_source(source: str) -> IRModule:
    """Parse, analyze, and lower source, refusing all reported errors."""
    try:
        parsed = parse_source(source)
    except ParseError as error:
        diagnostic = Diagnostic(
            "E_LEXICAL",
            str(error),
            SourceSpan(error.offset, error.offset, error.line, error.column),
        )
        raise CompilationError((diagnostic,)) from error

    errors = tuple(
        diagnostic
        for diagnostic in parsed.diagnostics
        if diagnostic.severity == "error"
    )
    if errors:
        raise CompilationError(errors)
    try:
        return lower_module(parsed.ast)
    except IRLoweringError as error:
        raise CompilationError(error.diagnostics) from error


def emit_c99(source: str) -> str:
    """Compile source and emit C99 for the checked scalar backend subset."""
    module = compile_source(source)
    try:
        return emit_c99_module(module)
    except C99LoweringError as error:
        raise CompilationError(error.diagnostics) from error


def _emit_c99_executable_source(source: str) -> str:
    module = compile_source(source)
    try:
        return emit_c99_module(module, include_entry_point=True)
    except C99LoweringError as error:
        raise CompilationError(error.diagnostics) from error


def build_object(
    source: str,
    output_path: str | os.PathLike[str],
    *,
    compiler: str | os.PathLike[str] | Sequence[str] | None = None,
) -> Path:
    """Compile supported Aether source to a host C99 object file.

    This invokes a C compiler with strict C99 flags but does not link an
    executable or load any native libraries.
    """
    generated = emit_c99(source)
    return compile_c99_object(
        generated,
        output_path,
        compiler=compiler,
    )


def build_executable(
    source: str,
    output_path: str | os.PathLike[str],
    *,
    compiler: str | os.PathLike[str] | Sequence[str] | None = None,
) -> Path:
    """Build a host executable from source containing fn main() -> i32 or void."""
    generated = _emit_c99_executable_source(source)
    return link_c99_executable(
        generated,
        output_path,
        compiler=compiler,
    )


def run_source(
    source: str,
    function_name: str,
    *arguments: object,
    effect_handlers: Mapping[str, EffectHandler] | None = None,
    external_handlers: Mapping[str, ExternalFunctionHandler] | None = None,
    max_call_depth: int = 128,
    **keywords: object,
) -> object:
    """Compile source and invoke one function with explicit effect handlers."""
    module = compile_source(source)
    interpreter = Interpreter(
        module,
        effect_handlers=effect_handlers,
        external_handlers=external_handlers,
        max_call_depth=max_call_depth,
    )
    return interpreter.run(function_name, *arguments, **keywords)
