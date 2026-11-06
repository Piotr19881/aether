import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import uuid

from src.api import CompilationError, build_object, emit_c99, run_source
from src.build import C99BuildError, C99ToolchainError
from src.build import compile_c99_object


_STRICT_C99_FLAGS = ("-std=c99", "-pedantic", "-Wall", "-Wextra", "-Werror")


class C99BackendTests(unittest.TestCase):
    def compiler(self) -> tuple[str, ...]:
        compiler = next(
            (
                (path,)
                for name in ("cc", "gcc", "clang")
                if (path := shutil.which(name)) is not None
            ),
            None,
        )
        if compiler is None:
            zig = shutil.which("zig")
            if zig is not None:
                compiler = (zig, "cc")
        if compiler is None:
            message = "No C99 compiler is installed"
            if os.environ.get("AETHER_REQUIRE_C99_COMPILER") == "1":
                self.fail(f"{message}; this test environment requires one")
            self.skipTest(message)
        return compiler

    def compile_program(
        self,
        generated: str,
        harness: str,
    ) -> Path:
        directory = Path(__file__).resolve().parent / f".c99-build-{uuid.uuid4().hex}"
        directory.mkdir()
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        source_path = directory / "generated.c"
        executable_path = directory / "generated"
        source_path.write_text(generated + "\n" + harness, encoding="utf-8")
        result = subprocess.run(
            [
                *self.compiler(),
                *_STRICT_C99_FLAGS,
                str(source_path),
                "-o",
                str(executable_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"C99 compilation failed:\n{result.stdout}\n{result.stderr}",
        )
        return executable_path

    def run_program(
        self,
        executable_path: Path,
        *arguments: str,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(executable_path), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
            cwd=executable_path.parent,
        )

    def compile_and_run(
        self,
        generated: str,
        harness: str,
        *arguments: str,
    ) -> subprocess.CompletedProcess[str]:
        executable = self.compile_program(generated, harness)
        return self.run_program(executable, *arguments)

    def test_generated_c99_compiles_when_a_c_compiler_is_available(self) -> None:
        compiler = self.compiler()
        generated = emit_c99(
            "fn add(left: i32, right: i32) -> i32 { return left + right; }\n"
            "fn divide(left: i32, right: i32) -> i32 { return left / right; }\n"
        )
        directory = Path(__file__).resolve().parent / f".c99-build-{uuid.uuid4().hex}"
        directory.mkdir()
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        source_path = directory / "generated.c"
        object_path = directory / "generated.o"
        source_path.write_text(generated, encoding="utf-8")
        subprocess.run(
            [
                *compiler,
                *_STRICT_C99_FLAGS,
                "-c",
                str(source_path),
                "-o",
                str(object_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_build_object_writes_a_compiled_object_without_linking(self) -> None:
        compiler = self.compiler()
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = (
                Path(temporary_directory)
                / "artifacts with spaces"
                / "module.o"
            )
            actual_path = build_object(
                "fn add(left: i32, right: i32) -> i32 { return left + right; }\n",
                output_path,
                compiler=compiler,
            )

            self.assertEqual(actual_path, output_path.resolve())
            self.assertTrue(actual_path.is_file())
            self.assertGreater(actual_path.stat().st_size, 0)

    def test_build_object_preserves_artifact_on_compiler_failure(self) -> None:
        compiler = self.compiler()
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "existing.o"
            output_path.write_bytes(b"previous object")
            with self.assertRaises(C99BuildError):
                compile_c99_object(
                    "this is not valid C;\n",
                    output_path,
                    compiler=compiler,
                )
            self.assertEqual(output_path.read_bytes(), b"previous object")

    def test_build_object_reports_missing_compiler(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaises(C99ToolchainError):
                compile_c99_object(
                    "int value(void) { return 1; }\n",
                    Path(temporary_directory) / "module.o",
                    compiler=("aether-missing-c-compiler",),
                )

    def test_emits_deterministic_c99_with_checked_integer_arithmetic(self) -> None:
        source = (
            "fn add(left: i32, right: i32) -> i32 { return left + right; }\n"
            "fn twice(value: i32) -> i32 { return add(value, value); }\n"
        )

        generated = emit_c99(source)

        self.assertEqual(generated, emit_c99(source))
        self.assertIn("#include <stdint.h>", generated)
        self.assertIn("INT32_MAX - right", generated)
        self.assertIn("INT32_MIN - right", generated)
        self.assertIn("aether_fn_616464", generated)
        self.assertIn("aether_fn_7477696365", generated)
        self.assertIn("aether_i32_add", generated)

    def test_checks_division_remainder_and_unsigned_overflow(self) -> None:
        generated = emit_c99(
            "fn divide(left: i64, right: i64) -> i64 { return left / right; }\n"
            "fn remainder(left: i32, right: i32) -> i32 { return left % right; }\n"
            "fn add(left: u8, right: u8) -> u8 { return left + right; }\n"
        )

        self.assertIn("left == INT64_MIN && right == -1", generated)
        self.assertIn("left == INT32_MIN && right == -1", generated)
        self.assertIn("UINT8_MAX - right", generated)

    def test_preserves_short_circuit_and_pipeline_call_order(self) -> None:
        generated = emit_c99(
            "fn accept(value: i32, offset: i32) -> i32 { return value + offset; }\n"
            "fn positive(value: i32) -> bool { return value > 0; }\n"
            "fn both(left: bool, value: i32) -> bool {\n"
            "  return left && positive(value);\n"
            "}\n"
            "fn pipeline(value: i32, offset: i32) -> i32 {\n"
            "  return value |> accept(offset);\n"
            "}\n"
        )

        self.assertIn("if (aether_tmp_0)", generated)
        self.assertIn("aether_fn_616363657074", generated)
        self.assertIn("aether_fn_706f736974697665", generated)
        self.assertIn("aether_fn_616363657074(aether_tmp_0, aether_tmp_1)", generated)

    def test_handles_signed_minimum_literal_without_overflow(self) -> None:
        generated = emit_c99(
            "fn min_value() -> i32 { return -2147483648; }\n"
        )

        self.assertIn("return INT32_MIN;", generated)

    def test_rejects_literals_outside_c99_type_ranges(self) -> None:
        for source in (
            "fn too_large() -> i32 { return 2147483648; }\n",
            "fn too_small() -> i32 { return -2147483649; }\n",
            "fn float_overflow() -> f32 { return 1e39; }\n",
        ):
            with self.subTest(source=source):
                with self.assertRaises(CompilationError) as raised:
                    emit_c99(source)
                self.assertIn(
                    "E_C99_LITERAL_RANGE",
                    {diagnostic.code for diagnostic in raised.exception.diagnostics},
                )

    def test_emits_void_functions_and_float_division_guard(self) -> None:
        generated = emit_c99(
            "fn noop() -> void { return; }\n"
            "fn ratio(left: f32, right: f32) -> f32 { return left / right; }\n"
        )

        self.assertIn("void aether_fn_6e6f6f70(void)", generated)
        self.assertIn("if (right == 0.0) aether_trap();", generated)
        self.assertIn("float aether_f32_div", generated)

    def test_compiled_program_matches_interpreter_for_scalar_features(self) -> None:
        source = (
            "fn add(left: i32, right: i32) -> i32 { return left + right; }\n"
            "fn accept(value: i32, offset: i32) -> i32 { return value + offset; }\n"
            "fn pipeline(value: i32, offset: i32) -> i32 { "
            "return value |> accept(offset); }\n"
            "fn danger(value: i32) -> bool { return value / 0 > 0; }\n"
            "fn short_and() -> bool { return false && danger(1); }\n"
            "fn short_or() -> bool { return true || danger(1); }\n"
            "fn half() -> f32 { return 0.5; }\n"
            "fn min_value() -> i32 { return -2147483648; }\n"
            "fn unused_parameter(value: i32) -> i32 { return 7; }\n"
            "fn noop() -> void { return; }\n"
            "fn call_noop() -> void { noop(); return; }\n"
        )
        generated = emit_c99(source)
        harness = r"""
#include <stdio.h>
int main(void) {
    printf("%d\n", (int)aether_fn_616464(9, -4));
    printf("%d\n", (int)aether_fn_706970656c696e65(8, 6));
    printf("%d\n", (int)aether_fn_73686f72745f616e64());
    printf("%d\n", (int)aether_fn_73686f72745f6f72());
    printf("%.9g\n", (double)aether_fn_68616c66());
    printf("%d\n", (int)aether_fn_6d696e5f76616c7565());
    printf("%d\n", (int)aether_fn_756e757365645f706172616d65746572(3));
    aether_fn_63616c6c5f6e6f6f70();
    puts("void");
    return 0;
}
"""
        result = self.compile_and_run(generated, harness)
        self.assertEqual(result.returncode, 0, result.stderr)
        actual = result.stdout.splitlines()
        expected = [
            str(run_source(source, "add", 9, -4)),
            str(run_source(source, "pipeline", 8, 6)),
            str(int(run_source(source, "short_and"))),
            str(int(run_source(source, "short_or"))),
            format(run_source(source, "half"), ".9g"),
            str(run_source(source, "min_value")),
            str(run_source(source, "unused_parameter", 3)),
            "void",
        ]
        self.assertEqual(actual, expected)

    def test_compiled_integer_boundaries_match_interpreter(self) -> None:
        source = (
            "fn add_i64(left: i64, right: i64) -> i64 { return left + right; }\n"
            "fn sub_i64(left: i64, right: i64) -> i64 { return left - right; }\n"
            "fn mul_i64(left: i64, right: i64) -> i64 { return left * right; }\n"
            "fn add_u64(left: u64, right: u64) -> u64 { return left + right; }\n"
            "fn mul_u64(left: u64, right: u64) -> u64 { return left * right; }\n"
            "fn div_i32(left: i32, right: i32) -> i32 { return left / right; }\n"
            "fn rem_i32(left: i32, right: i32) -> i32 { return left % right; }\n"
        )
        harness = r"""
#include <stdio.h>
int main(void) {
    printf("%lld\n", (long long)aether_fn_6164645f693634(INT64_MIN, 0));
    printf("%lld\n", (long long)aether_fn_6164645f693634(INT64_MAX, 0));
    printf("%lld\n", (long long)aether_fn_7375625f693634(INT64_MIN, 0));
    printf("%lld\n", (long long)aether_fn_6d756c5f693634(INT64_MIN, 1));
    printf("%lld\n", (long long)aether_fn_6d756c5f693634(0, INT64_MIN));
    printf("%llu\n", (unsigned long long)aether_fn_6164645f753634(UINT64_MAX, 0));
    printf("%llu\n", (unsigned long long)aether_fn_6d756c5f753634(UINT64_MAX, 1));
    printf("%d\n", (int)aether_fn_6469765f693332(-7, 3));
    printf("%d\n", (int)aether_fn_72656d5f693332(-7, 3));
    return 0;
}
"""
        result = self.compile_and_run(emit_c99(source), harness)
        self.assertEqual(result.returncode, 0, result.stderr)
        test_cases = (
            ("add_i64", (-(1 << 63), 0)),
            ("add_i64", ((1 << 63) - 1, 0)),
            ("sub_i64", (-(1 << 63), 0)),
            ("mul_i64", (-(1 << 63), 1)),
            ("mul_i64", (0, -(1 << 63))),
            ("add_u64", ((1 << 64) - 1, 0)),
            ("mul_u64", ((1 << 64) - 1, 1)),
            ("div_i32", (-7, 3)),
            ("rem_i32", (-7, 3)),
        )
        expected = [
            str(run_source(source, name, *arguments))
            for name, arguments in test_cases
        ]
        self.assertEqual(result.stdout.splitlines(), expected)

    def test_compiled_arithmetic_traps_invalid_operations(self) -> None:
        source = (
            "fn add_i32(left: i32, right: i32) -> i32 { return left + right; }\n"
            "fn add_u8(left: u8, right: u8) -> u8 { return left + right; }\n"
            "fn mul_i64(left: i64, right: i64) -> i64 { return left * right; }\n"
            "fn div_i32(left: i32, right: i32) -> i32 { return left / right; }\n"
            "fn rem_i32(left: i32, right: i32) -> i32 { return left % right; }\n"
            "fn neg_i64(value: i64) -> i64 { return -value; }\n"
            "fn div_f32(left: f32, right: f32) -> f32 { return left / right; }\n"
        )
        harness = r"""
#include <string.h>
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    if (strcmp(argv[1], "signed_add") == 0)
        (void)aether_fn_6164645f693332(INT32_MAX, 1);
    else if (strcmp(argv[1], "unsigned_add") == 0)
        (void)aether_fn_6164645f7538((uint8_t)255, (uint8_t)1);
    else if (strcmp(argv[1], "signed_multiply") == 0)
        (void)aether_fn_6d756c5f693634(INT64_MAX, 2);
    else if (strcmp(argv[1], "signed_divide_zero") == 0)
        (void)aether_fn_6469765f693332(1, 0);
    else if (strcmp(argv[1], "signed_divide_min") == 0)
        (void)aether_fn_6469765f693332(INT32_MIN, -1);
    else if (strcmp(argv[1], "signed_remainder_zero") == 0)
        (void)aether_fn_72656d5f693332(1, 0);
    else if (strcmp(argv[1], "signed_remainder_min") == 0)
        (void)aether_fn_72656d5f693332(INT32_MIN, -1);
    else if (strcmp(argv[1], "signed_negate_min") == 0)
        (void)aether_fn_6e65675f693634(INT64_MIN);
    else if (strcmp(argv[1], "float_divide_zero") == 0)
        (void)aether_fn_6469765f663332(1.0f, 0.0f);
    else return 3;
    return 0;
}
"""
        generated = emit_c99(source)
        executable = self.compile_program(generated, harness)
        for case in (
            "signed_add",
            "unsigned_add",
            "signed_multiply",
            "signed_divide_zero",
            "signed_divide_min",
            "signed_remainder_zero",
            "signed_remainder_min",
            "signed_negate_min",
            "float_divide_zero",
        ):
            with self.subTest(operation=case):
                result = self.run_program(executable, case)
                self.assertNotEqual(result.returncode, 0, result.stdout)

    def test_rejects_features_outside_backend_subset_with_diagnostics(self) -> None:
        cases = (
            (
                "fn tensor(value: Tensor<f32, [2]>) -> Tensor<f32, [2]> "
                "{ return value; }\n",
                "E_C99_TYPE",
            ),
            (
                "fn power(value: i32, exponent: i32) -> i32 "
                "{ return value ** exponent; }\n",
                "E_C99_OPERATOR",
            ),
            (
                "fn text() -> str { return \"hello\"; }\n",
                "E_C99_TYPE",
            ),
            (
                "fn unit(value: Tensor<f32, [2], m>) -> f32 { return 1.0; }\n",
                "E_C99_TYPE",
            ),
            (
                "fn result(value: Result<i32, IoError>) -> i32 { return 1; }\n",
                "E_C99_TYPE",
            ),
            (
                "@io fn read() -> Result<str, IoError> { return io_read_line(); }\n",
                "E_C99_EFFECT",
            ),
            (
                "@ffi @link(\"sensor\") extern \"C\" fn read() -> i32;\n",
                "E_C99_FFI",
            ),
            (
                "fn missing_return(value: i32) -> i32 { let copy = value; }\n",
                "E_C99_RETURN",
            ),
            (
                "type Pair { left: i32, right: i32 }\n"
                "fn value() -> i32 { return 1; }\n",
                "E_C99_TYPE_DECL",
            ),
        )
        for source, expected_code in cases:
            with self.subTest(expected_code=expected_code):
                with self.assertRaises(CompilationError) as raised:
                    emit_c99(source)
                self.assertIn(
                    expected_code,
                    {diagnostic.code for diagnostic in raised.exception.diagnostics},
                )


if __name__ == "__main__":
    unittest.main()
