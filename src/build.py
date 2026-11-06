"""Host C99 toolchain driver for compiling Aether-generated C to an object."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Sequence


_C99_FLAGS = ("-std=c99", "-pedantic", "-Wall", "-Wextra", "-Werror", "-c")
_COMPILER_TIMEOUT_SECONDS = 120


class C99ToolchainError(RuntimeError):
    """Raised when no usable C99 compiler command can be found or started."""


@dataclass(frozen=True)
class C99BuildError(RuntimeError):
    """Compiler failure with the full command and captured diagnostics."""

    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    def __str__(self) -> str:
        details = self.stderr.strip() or self.stdout.strip() or "(no compiler output)"
        return (
            f"C99 compiler exited with status {self.returncode}: "
            f"{subprocess.list2cmdline(self.command)}\n{details}"
        )


def _resolve_compiler(
    compiler: str | os.PathLike[str] | Sequence[str] | None,
) -> tuple[str, ...]:
    if compiler is None:
        for name in ("cc", "gcc", "clang"):
            path = shutil.which(name)
            if path is not None:
                return (path,)
        zig = shutil.which("zig")
        if zig is not None:
            return (zig, "cc")
        raise C99ToolchainError(
            "No supported C compiler found (looked for cc, gcc, clang, or zig cc)"
        )

    if isinstance(compiler, (str, os.PathLike)):
        command = (os.fspath(compiler),)
    else:
        command = tuple(os.fspath(part) for part in compiler)
    if not command or any(not part for part in command):
        raise ValueError("compiler must be a non-empty command")

    executable = command[0]
    if not Path(executable).is_file() and shutil.which(executable) is None:
        raise C99ToolchainError(
            f"C99 compiler executable {executable!r} was not found"
        )
    return command


def compile_c99_object(
    c_source: str,
    output_path: str | os.PathLike[str],
    *,
    compiler: str | os.PathLike[str] | Sequence[str] | None = None,
) -> Path:
    """Compile a C99 translation unit to an object without invoking a linker.

    The object is written atomically after a successful compiler run. Compiler
    arguments are passed directly to the process without a shell.
    """
    command_prefix = _resolve_compiler(compiler)
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=".aether-c99-",
        dir=output.parent,
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        source_file = temporary / "module.c"
        temporary_object = temporary / "module.o"
        source_file.write_text(c_source, encoding="utf-8", newline="\n")
        command = (
            *command_prefix,
            *_C99_FLAGS,
            str(source_file),
            "-o",
            str(temporary_object),
        )
        try:
            result = subprocess.run(
                command,
                cwd=temporary,
                capture_output=True,
                text=True,
                check=False,
                timeout=_COMPILER_TIMEOUT_SECONDS,
            )
        except FileNotFoundError as error:
            raise C99ToolchainError(
                f"Unable to start C99 compiler {command_prefix[0]!r}: {error}"
            ) from error
        except subprocess.TimeoutExpired as error:
            stdout = (
                error.stdout.decode(errors="replace")
                if isinstance(error.stdout, bytes)
                else error.stdout or ""
            )
            stderr = (
                error.stderr.decode(errors="replace")
                if isinstance(error.stderr, bytes)
                else error.stderr or ""
            )
            raise C99BuildError(
                command,
                -1,
                stdout,
                stderr or "Compiler timed out",
            ) from error

        if result.returncode != 0:
            raise C99BuildError(
                command,
                result.returncode,
                result.stdout,
                result.stderr,
            )
        if not temporary_object.is_file():
            raise C99ToolchainError(
                "C99 compiler reported success but did not produce an object file"
            )
        os.replace(temporary_object, output)
    return output
