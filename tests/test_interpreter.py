from pathlib import Path
import unittest

from src.frontend import parse_source
from src.api import CompilationError, compile_source, run_source
from src.interpreter import (
    AetherRuntimeError,
    Interpreter,
    Quantity,
    ResultValue,
)
from src.ir import IRLoweringError, lower_module


class TypedIRTests(unittest.TestCase):
    def test_lowering_preserves_expression_types_and_spans(self) -> None:
        parsed = parse_source(
            "fn add(left: i32, right: i32) -> i32 { return left + right; }\n"
        )
        lowered = lower_module(parsed.ast)
        function = lowered.function("add")
        expression = function.body[0].expression

        self.assertEqual(function.return_type.name, "i32")
        self.assertEqual(expression.type.name, "i32")
        self.assertEqual(expression.span.line, 1)

    def test_ir_retains_declared_and_transitive_effects(self) -> None:
        parsed = parse_source(
            "@io\n"
            "fn read() -> Result<str, IoError> { return io_read_line(); }\n"
            "@io\n"
            "fn wrapper() -> Result<str, IoError> { return read(); }\n"
        )
        wrapper = lower_module(parsed.ast).function("wrapper")

        self.assertEqual(wrapper.declared_effects, ("io",))
        self.assertEqual(wrapper.required_effects, ("io",))

    def test_ir_preserves_extern_abi_library_and_signature(self) -> None:
        parsed = parse_source(
            '@ffi\n@link("sensor")\n'
            'extern "C" fn read_channel(channel: u32) -> i32;\n'
        )
        self.assertEqual(parsed.diagnostics, ())
        external = lower_module(parsed.ast).function("read_channel")

        self.assertEqual(external.external_abi, "C")
        self.assertEqual(external.external_library, "sensor")
        self.assertEqual(external.required_effects, ("ffi",))
        self.assertEqual(external.parameters[0].type.name, "u32")
        self.assertEqual(external.return_type.name, "i32")
        self.assertEqual(external.body, ())

    def test_ir_preserves_unit_conversion_metadata_and_match_payload_types(self) -> None:
        parsed = parse_source(
            "fn to_fahrenheit(value: Tensor<f32, [1], degC>) "
            "-> Tensor<f32, [1], degF> { return convert(value, \"degF\"); }\n"
            "fn inspect(value: Result<str, IoError>) -> str {\n"
            "  return match value { Ok(text) => text, Err(_) => \"failed\" };\n"
            "}\n"
        )
        self.assertEqual(parsed.diagnostics, ())
        module = lower_module(parsed.ast)
        conversion = module.function("to_fahrenheit").body[0].expression
        match = module.function("inspect").body[0].expression

        self.assertEqual(conversion.type.unit_symbol, "degF")
        self.assertIsNotNone(conversion.conversion)
        self.assertEqual(match.type.name, "str")
        self.assertEqual(match.children[0].type.name, "Result")
        self.assertEqual(match.arms[0].value.type.name, "str")

    def test_lowering_refuses_semantic_errors(self) -> None:
        parsed = parse_source(
            "fn invalid(left: bool, right: bool) { let value = left + right; }\n"
        )

        with self.assertRaises(IRLoweringError):
            lower_module(parsed.ast)

    def test_lowering_refuses_an_expression_without_an_inferred_type(self) -> None:
        parsed = parse_source(
            "fn caller() { return callee(); }\n"
            "fn callee() { return 1; }\n"
        )

        with self.assertRaises(IRLoweringError) as raised:
            lower_module(parsed.ast)
        self.assertIn("E_IR_UNTYPED", {item.code for item in raised.exception.diagnostics})


class InterpreterTests(unittest.TestCase):
    def test_executes_numeric_functions_and_user_calls(self) -> None:
        parsed = parse_source(
            "fn add(left: i32, right: i32) -> i32 { return left + right; }\n"
            "fn twice(value: i32) -> i32 { return add(value, value); }\n"
            "fn divide(left: i32, right: i32) -> i32 { return left / right; }\n"
            "fn operations(left: i32, right: i32) -> i32 {\n"
            "  let remainder = left % right;\n"
            "  return -(remainder + left ** 2);\n"
            "}\n"
            "fn logic(left: bool, right: bool) -> bool {\n"
            "  return left && right || !left;\n"
            "}\n"
            "fn fail() -> bool { return 1 / 0 == 0; }\n"
            "fn short_circuit() -> bool { return false && fail(); }\n"
            "fn stats(values: Tensor<f32, [2]>) -> f32 { return sum(values); }\n"
            "fn average(values: Tensor<i32, [2]>) -> f32 { return mean(values); }\n"
            "fn size(values: Tensor<f32, [2]>) -> i64 { return len(values); }\n"
            "fn extremes(values: Tensor<f32, [2]>) -> Tensor<f32, [2]> {\n"
            "  return max(abs(values), values);\n"
            "}\n"
        )
        self.assertEqual(parsed.diagnostics, ())
        interpreter = Interpreter.from_source(parsed.ast)

        self.assertEqual(interpreter.run("twice", 21), 42)
        self.assertEqual(interpreter.run("divide", -3, 2), -1)
        self.assertEqual(interpreter.run("operations", -5, 2), -24)
        self.assertTrue(interpreter.run("logic", True, True))
        self.assertFalse(interpreter.run("logic", True, False))
        self.assertTrue(interpreter.run("logic", False, False))
        self.assertFalse(interpreter.run("short_circuit"))
        self.assertEqual(interpreter.run("stats", [2.0, 4.0]), 6.0)
        self.assertEqual(interpreter.run("average", [1, 2]), 1.5)
        self.assertEqual(interpreter.run("size", [2.0, 4.0]), 2)
        self.assertEqual(interpreter.run("extremes", [-2.0, 1.0]), [2.0, 1.0])

    def test_executes_tensor_pipeline(self) -> None:
        parsed = parse_source(
            "fn center(values: Tensor<f32, [2]>) -> Tensor<f32, [2]> {\n"
            "  let average = mean(values);\n"
            "  return values |> subtract(average);\n"
            "}\n"
        )

        self.assertEqual(parsed.diagnostics, ())
        self.assertEqual(Interpreter.from_source(parsed.ast).run("center", [1.0, 3.0]), [-1.0, 1.0])

    def test_executes_sensor_example_end_to_end(self) -> None:
        source_path = Path(__file__).parents[1] / "examples" / "sensor.hl"
        with source_path.open("r", encoding="utf-8", newline="") as source_file:
            parsed = parse_source(source_file.read())
        self.assertEqual(parsed.diagnostics, ())

        samples = [[float(row * 4 + column) for column in range(4)] for row in range(4)]
        result = Interpreter.from_source(parsed.ast).run(
            "normalize_voltage",
            {"samples": samples, "timestamp": 1},
        )

        self.assertEqual(len(result), 4)
        self.assertAlmostEqual(result[0][0], -1.0)
        self.assertAlmostEqual(result[3][3], 1.0)

    def test_match_handles_injected_result_and_effects_are_opt_in(self) -> None:
        parsed = parse_source(
            "@io\n"
            "fn read_or_default() -> str {\n"
            "  let result = io_read_line();\n"
            "  return match result { Ok(value) => value, Err(error) => \"fallback\" };\n"
            "}\n"
        )
        self.assertEqual(parsed.diagnostics, ())

        with self.assertRaisesRegex(AetherRuntimeError, "no injected handler"):
            Interpreter.from_source(parsed.ast).run("read_or_default")
        with self.assertRaisesRegex(AetherRuntimeError, "invalid success value"):
            Interpreter.from_source(
                parsed.ast,
                effect_handlers={"io_read_line": lambda: ResultValue.ok(False)},
            ).run("read_or_default")
        interpreter = Interpreter.from_source(
            parsed.ast,
            effect_handlers={
                "io_read_line": lambda: ResultValue.err("offline")
            },
        )
        self.assertEqual(interpreter.run("read_or_default"), "fallback")

    def test_result_parameters_validate_success_payloads(self) -> None:
        parsed = parse_source(
            "fn read(result: Result<str, IoError>) -> str {\n"
            "  return match result { Ok(value) => value, Err(_) => \"fallback\" };\n"
            "}\n"
        )
        self.assertEqual(parsed.diagnostics, ())
        interpreter = Interpreter.from_source(parsed.ast)

        self.assertEqual(interpreter.run("read", ResultValue.ok("ok")), "ok")
        with self.assertRaisesRegex(TypeError, "Ok payload expects str"):
            interpreter.run("read", ResultValue.ok(12))

    def test_question_operator_propagates_error_and_wraps_success(self) -> None:
        parsed = parse_source(
            "@io\n"
            "fn read() -> Result<str, IoError> { return io_read_line(); }\n"
            "@io\n"
            "fn wrapper() -> Result<str, IoError> {\n"
            "  let value = read();\n"
            "  return value?;\n"
            "}\n"
        )
        self.assertEqual(parsed.diagnostics, ())
        failing = Interpreter.from_source(
            parsed.ast,
            effect_handlers={"io_read_line": lambda: ResultValue.err("EOF")},
        ).run("wrapper")
        succeeding = Interpreter.from_source(
            parsed.ast,
            effect_handlers={"io_read_line": lambda: ResultValue.ok("hello")},
        ).run("wrapper")

        self.assertEqual(failing, ResultValue.err("EOF"))
        self.assertEqual(succeeding, ResultValue.ok("hello"))

    def test_executes_explicit_unit_conversion(self) -> None:
        parsed = parse_source(
            "fn to_meters(value: Tensor<f32, [2], mm>) "
            "-> Tensor<f32, [2], m> { return convert(value, \"m\"); }\n"
        )
        result = Interpreter.from_source(parsed.ast).run("to_meters", [1000.0, 2500.0])

        self.assertIsInstance(result, Quantity)
        self.assertEqual(result.type.unit_symbol, "m")
        self.assertEqual(result.value, [1.0, 2.5])

    def test_pipeline_conversion_and_tensor_unit_cancellation(self) -> None:
        parsed = parse_source(
            "fn to_meters(value: Tensor<f32, [2], mm>) "
            "-> Tensor<f32, [2], m> { return value |> convert(\"m\"); }\n"
            "fn normalize(value: Tensor<f32, [2], V>) "
            "-> Tensor<f32, [2]> {\n"
            "  return value |> scale(1.0 / max_abs(value));\n"
            "}\n"
        )
        self.assertEqual(parsed.diagnostics, ())
        interpreter = Interpreter.from_source(parsed.ast)

        converted = interpreter.run("to_meters", [1000.0, 2500.0])
        normalized = interpreter.run("normalize", [1.0, 3.0])
        self.assertIsInstance(converted, Quantity)
        self.assertEqual(converted.type.unit_symbol, "m")
        self.assertEqual(converted.value, [1.0, 2.5])
        self.assertEqual(normalized, [1.0 / 3.0, 1.0])

    def test_executes_affine_and_symbolic_pi_unit_conversions(self) -> None:
        parsed = parse_source(
            "fn fahrenheit(value: Tensor<f32, [1], degC>) "
            "-> Tensor<f32, [1], degF> { return convert(value, \"degF\"); }\n"
            "fn radians(value: Tensor<f32, [1], deg>) "
            "-> Tensor<f32, [1], rad> { return convert(value, \"rad\"); }\n"
        )
        interpreter = Interpreter.from_source(parsed.ast)

        fahrenheit = interpreter.run("fahrenheit", [0.0])
        radians = interpreter.run("radians", [180.0])
        self.assertAlmostEqual(fahrenheit.value[0], 32.0)
        self.assertAlmostEqual(radians.value[0], 3.141592653589793)

    def test_checks_runtime_tensor_shapes_and_division_by_zero(self) -> None:
        parsed = parse_source(
            "fn first(values: Tensor<f32, [2]>) -> f32 { return values[0]; }\n"
            "fn negative(values: Tensor<f32, [2]>) -> f32 { return values[-1]; }\n"
            "fn divide(left: f32, right: f32) -> f32 { return left / right; }\n"
            "fn missing() -> f32 {}\n"
            "fn recurse() -> i32 { return recurse(); }\n"
        )
        interpreter = Interpreter.from_source(parsed.ast)

        with self.assertRaisesRegex(TypeError, "expects tensor shape"):
            interpreter.run("first", [1.0])
        with self.assertRaisesRegex(AetherRuntimeError, "Division by zero"):
            interpreter.run("divide", 1.0, 0.0)
        with self.assertRaisesRegex(AetherRuntimeError, "Negative index"):
            interpreter.run("negative", [1.0, 2.0])
        with self.assertRaisesRegex(AetherRuntimeError, "completed without returning"):
            interpreter.run("missing")
        with self.assertRaisesRegex(AetherRuntimeError, "Maximum call depth"):
            Interpreter.from_source(parsed.ast, max_call_depth=4).run("recurse")

    def test_tensor_inputs_validate_rectangular_shape_and_element_types(self) -> None:
        parsed = parse_source(
            "fn accept(values: Tensor<f32, [2, 2]>) -> f32 { return mean(values); }\n"
        )
        interpreter = Interpreter.from_source(parsed.ast)

        with self.assertRaisesRegex(TypeError, "rectangular"):
            interpreter.run("accept", [[1.0], [2.0, 3.0]])
        with self.assertRaisesRegex(TypeError, r"values\[3\] expects f32"):
            interpreter.run("accept", [[1.0, 2.0], [3.0, "bad"]])

    def test_struct_arguments_validate_fields_and_field_types(self) -> None:
        parsed = parse_source(
            "type Point { x: f32, y: f32, }\n"
            "fn x(point: Point) -> f32 { return point.x; }\n"
        )
        interpreter = Interpreter.from_source(parsed.ast)

        self.assertEqual(interpreter.run("x", {"x": 2.0, "y": 3.0}), 2.0)
        with self.assertRaisesRegex(TypeError, "invalid Point fields"):
            interpreter.run("x", {"x": 2.0})
        with self.assertRaisesRegex(TypeError, "point.x expects f32"):
            interpreter.run("x", {"x": "bad", "y": 3.0})

    def test_public_api_compiles_and_runs_clean_source_with_injected_effects(self) -> None:
        pure_source = "fn double(value: i32) -> i32 { return value * 2; }\n"
        self.assertEqual(run_source(pure_source, "double", 21), 42)

        effect_source = (
            "@io\n"
            "fn read() -> Result<str, IoError> { return io_read_line(); }\n"
            "@io\n"
            "fn wrapper() -> Result<str, IoError> {\n"
            "  let value = read();\n"
            "  return value?;\n"
            "}\n"
        )
        self.assertEqual(
            run_source(
                effect_source,
                "wrapper",
                effect_handlers={
                    "io_read_line": lambda: ResultValue.ok("ready")
                },
            ),
            ResultValue.ok("ready"),
        )
        self.assertEqual(compile_source(pure_source).function("double").name, "double")

    def test_extern_calls_only_use_explicit_handlers(self) -> None:
        source = (
            '@ffi\n@link("sensor")\n'
            'extern "C" fn read_channel(channel: u32) -> i32;\n'
            "@ffi\n"
            "fn sample(channel: u32) -> i32 { return read_channel(channel); }\n"
            '@ffi\n@link("sensor")\n'
            'extern "C" fn reset() -> void;\n'
        )
        module = compile_source(source)
        interpreter = Interpreter(module)

        with self.assertRaisesRegex(AetherRuntimeError, "does not load native libraries"):
            interpreter.run("sample", 4)
        with self.assertRaisesRegex(TypeError, "expects u32"):
            Interpreter(
                module,
                external_handlers={"read_channel": lambda channel: channel},
            ).run("sample", -1)
        with self.assertRaisesRegex(TypeError, "expects u32"):
            Interpreter(
                module,
                external_handlers={"read_channel": lambda channel: channel},
            ).run("sample", 2**32)
        with self.assertRaisesRegex(AetherRuntimeError, "violates i32"):
            Interpreter(
                module,
                external_handlers={"read_channel": lambda channel: "not an integer"},
            ).run("sample", 4)
        with self.assertRaisesRegex(AetherRuntimeError, "violates void"):
            Interpreter(
                module,
                external_handlers={
                    "read_channel": lambda channel: channel,
                    "reset": lambda: "unexpected",
                },
            ).run("reset")

        calls = []
        result = run_source(
            source,
            "sample",
            7,
            external_handlers={
                "read_channel": lambda channel: channel * 3,
                "reset": lambda: calls.append("reset"),
            },
        )
        self.assertEqual(result, 21)
        self.assertIsNone(
            run_source(
                source,
                "reset",
                external_handlers={
                    "read_channel": lambda channel: channel,
                    "reset": lambda: calls.append("reset"),
                },
            )
        )
        self.assertEqual(calls, ["reset"])

    def test_public_api_rejects_parse_and_semantic_errors(self) -> None:
        with self.assertRaises(CompilationError) as syntax_error:
            compile_source('fn broken() { return "unterminated; }\n')
        self.assertEqual(syntax_error.exception.diagnostics[0].code, "E_LEXICAL")

        with self.assertRaises(CompilationError) as semantic_error:
            compile_source("fn broken() { let x = true + false; }\n")
        self.assertIn(
            "E_TYPE_OPERATOR",
            {item.code for item in semantic_error.exception.diagnostics},
        )

        with self.assertRaises(CompilationError) as lowering_error:
            compile_source(
                "fn caller() { return callee(); }\n"
                "fn callee() { return 1; }\n"
            )
        self.assertIn(
            "E_IR_UNTYPED",
            {item.code for item in lowering_error.exception.diagnostics},
        )


if __name__ == "__main__":
    unittest.main()
