import unittest
from fractions import Fraction

from src.frontend import parse_source, render_module
from src.semantic import analyze


class SemanticTests(unittest.TestCase):
    def test_sensor_pipeline_infers_tensor_units_and_cancels_normalization_unit(self) -> None:
        source = (
            "type Frame { samples: Tensor<f32, [4, 4], V>, }\n"
            "fn normalize(frame: Frame) -> Tensor<f32, [4, 4]> {\n"
            "  let centered = frame.samples |> subtract(mean(frame.samples));\n"
            "  return centered |> scale(1.0 / max_abs(centered));\n"
            "}\n"
        )
        result = parse_source(source)
        self.assertEqual(result.diagnostics, ())

        semantic = analyze(result.ast)
        return_expr = result.ast.declarations[1].body[-1].value
        inferred = dict(semantic.expression_types)[return_expr.span]
        self.assertEqual(inferred.name, "Tensor")
        self.assertEqual(inferred.dimensions, (4, 4))
        self.assertEqual(inferred.unit, ())

    def test_unit_and_shape_mismatches_are_diagnosed(self) -> None:
        result = parse_source(
            "fn add(a: Tensor<f32, [2, 3], V>, "
            "b: Tensor<f32, [2, 3], mm>) {\n"
            "  let invalid = a + b;\n"
            "}\n"
        )
        codes = [item.code for item in result.diagnostics]

        self.assertIn("E_UNIT_MISMATCH", codes)

    def test_tensor_shape_mismatch_has_dedicated_diagnostic(self) -> None:
        result = parse_source(
            "fn add(a: Tensor<f32, [2, 3]>, b: Tensor<f32, [3, 2]>) {\n"
            "  let invalid = a + b;\n"
            "}\n"
        )

        self.assertIn("E_SHAPE_MISMATCH", [item.code for item in result.diagnostics])

    def test_unit_scales_are_not_implicitly_mixed(self) -> None:
        result = parse_source(
            "fn add(a: Tensor<f32, [2], m>, b: Tensor<f32, [2], mm>) {\n"
            "  let invalid = a + b;\n"
            "}\n"
        )

        self.assertIn("E_UNIT_MISMATCH", [item.code for item in result.diagnostics])

    def test_explicit_conversion_changes_tensor_unit_and_scale(self) -> None:
        result = parse_source(
            "fn convert_length(value: Tensor<f32, [2], mm>) "
            "-> Tensor<f32, [2], m> {\n"
            "  return value |> convert(\"m\");\n"
            "}\n"
        )

        self.assertEqual(result.diagnostics, ())
        returned = result.ast.declarations[0].body[0].value
        inferred = dict(analyze(result.ast).expression_types)[returned.span]
        self.assertEqual(inferred.unit_symbol, "m")
        self.assertEqual(inferred.unit_scale.numerator, 1)
        self.assertEqual(inferred.unit_scale.denominator, 1)
        conversion = analyze(result.ast).conversions[0]
        self.assertEqual(conversion.multiplier.numerator, 1)
        self.assertEqual(conversion.multiplier.denominator, 1000)
        self.assertEqual(conversion.addend, 0)

    def test_explicit_celsius_to_fahrenheit_conversion_is_affine(self) -> None:
        result = parse_source(
            "fn convert_temperature(value: Tensor<f32, [1], degC>) "
            "-> Tensor<f32, [1], degF> {\n"
            "  return convert(value, \"degF\");\n"
            "}\n"
        )

        self.assertEqual(result.diagnostics, ())
        conversion = analyze(result.ast).conversions[0]
        self.assertEqual(conversion.multiplier, Fraction(9, 5))
        self.assertEqual(conversion.addend, Fraction(32))

    def test_degree_to_radian_conversion_keeps_pi_symbolic(self) -> None:
        result = parse_source(
            "fn convert_angle(value: Tensor<f32, [1], deg>) "
            "-> Tensor<f32, [1], rad> {\n"
            "  return convert(value, \"rad\");\n"
            "}\n"
        )

        self.assertEqual(result.diagnostics, ())
        conversion = analyze(result.ast).conversions[0]
        self.assertEqual(conversion.multiplier, Fraction(1, 180))
        self.assertEqual(conversion.pi_power, 1)

    def test_temperature_arithmetic_requires_explicit_temperature_semantics(self) -> None:
        result = parse_source(
            "fn add(a: Tensor<f32, [2], degC>, "
            "b: Tensor<f32, [2], degC>) {\n"
            "  let invalid = a + b;\n"
            "}\n"
        )

        self.assertIn(
            "E_TEMPERATURE_ARITHMETIC",
            [item.code for item in result.diagnostics],
        )

    def test_unknown_units_and_invalid_builtin_calls_are_reported(self) -> None:
        result = parse_source(
            "fn invalid(value: Tensor<f32, [2], furlong>) {\n"
            "  let average = mean();\n"
            "}\n"
        )
        codes = [item.code for item in result.diagnostics]

        self.assertIn("E_UNKNOWN_UNIT", codes)
        self.assertIn("E_BUILTIN_ARITY", codes)

    def test_non_numeric_addition_is_rejected(self) -> None:
        result = parse_source("fn invalid(a: bool, b: bool) { let value = a + b; }\n")

        self.assertIn("E_TYPE_OPERATOR", [item.code for item in result.diagnostics])

    def test_tensor_dimensions_must_be_positive_static_integers(self) -> None:
        result = parse_source("fn invalid(value: Tensor<f32, [0, 3]>) {}\n")

        self.assertIn("E_TENSOR_DIMENSION", [item.code for item in result.diagnostics])

    def test_effect_requirements_propagate_through_calls(self) -> None:
        result = parse_source(
            "@net\n"
            "fn request() -> Result<str, NetError> {\n"
            "  let response = net_get(\"https://example.test/data\"); return response;\n"
            "}\n"
            "fn caller() { return request(); }\n"
            "@net\n"
            "fn authorized() { return request(); }\n"
        )

        violations = [item for item in result.diagnostics if item.code == "E_EFFECT_VIOLATION"]
        self.assertEqual(len(violations), 1)
        self.assertIn("@net", violations[0].message)
        summaries = {item.function: item for item in analyze(result.ast).effects}
        self.assertEqual(summaries["request"].required, ("net",))
        self.assertEqual(summaries["caller"].required, ("net",))

    def test_effectful_builtins_require_declared_effects(self) -> None:
        result = parse_source(
            "fn invalid() { let content = fs_read_text(\"data.txt\"); return content; }\n"
            "@fs\n"
            "fn authorized() { let content = fs_read_text(\"data.txt\"); return content; }\n"
        )

        violations = [item for item in result.diagnostics if item.code == "E_EFFECT_VIOLATION"]
        self.assertEqual(len(violations), 1)
        self.assertIn("@fs", violations[0].message)

    def test_pure_annotation_rejects_direct_effectful_operations(self) -> None:
        result = parse_source(
            "@pure\n"
            "fn invalid() -> Result<str, NetError> {\n"
            "  let response = net_get(\"https://example.test/data\"); return response;\n"
            "}\n"
        )

        self.assertIn("E_EFFECT_VIOLATION", [item.code for item in result.diagnostics])

    def test_effects_propagate_across_recursive_call_graphs(self) -> None:
        result = parse_source(
            "@net\n"
            "fn fetch() -> Result<str, NetError> {\n"
            "  let response = net_get(\"https://example.test/data\"); caller(); return response;\n"
            "}\n"
            "fn caller() { return fetch(); }\n"
        )

        self.assertEqual(
            [item.code for item in result.diagnostics].count("E_EFFECT_VIOLATION"),
            1,
        )
        summaries = {item.function: item for item in analyze(result.ast).effects}
        self.assertEqual(summaries["caller"].required, ("net",))

    def test_ffi_effect_is_required_and_propagates_through_calls(self) -> None:
        result = parse_source(
            '@ffi\n@link("sensor")\n'
            'extern "C" fn read_channel(channel: u32) -> i32;\n'
            "fn unauthorized(channel: u32) -> i32 { return read_channel(channel); }\n"
            "@ffi\n"
            "fn authorized(channel: u32) -> i32 { return read_channel(channel); }\n"
        )

        violations = [
            item for item in result.diagnostics
            if item.code == "E_EFFECT_VIOLATION"
        ]
        self.assertEqual(len(violations), 1)
        self.assertIn("@ffi", violations[0].message)
        summaries = {item.function: item for item in analyze(result.ast).effects}
        self.assertEqual(summaries["read_channel"].required, ("ffi",))
        self.assertEqual(summaries["authorized"].required, ("ffi",))
        self.assertEqual(summaries["unauthorized"].required, ("ffi",))

    def test_ffi_rejects_missing_effect_link_and_non_scalar_types(self) -> None:
        missing_ffi = parse_source(
            '@link("sensor")\nextern "C" fn read() -> i32;\n'
        )
        missing_link = parse_source(
            '@ffi\nextern "C" fn read() -> i32;\n'
        )
        unsupported = parse_source(
            '@ffi\n@link("sensor")\n'
            'extern "C" fn read(value: Tensor<f32, [2]>) -> Result<i32, IoError>;\n'
        )

        self.assertIn(
            "E_EXTERN_EFFECT",
            [item.code for item in missing_ffi.diagnostics],
        )
        self.assertIn(
            "E_EXTERN_LINK",
            [item.code for item in missing_link.diagnostics],
        )
        self.assertGreaterEqual(
            [item.code for item in unsupported.diagnostics].count("E_FFI_TYPE"),
            2,
        )

    def test_ffi_rejects_pure_or_malformed_library_declarations(self) -> None:
        pure_extern = parse_source(
            '@pure\n@ffi\n@link("sensor")\n'
            'extern "C" fn read() -> i32;\n'
        )
        path_library = parse_source(
            '@ffi\n@link("../sensor")\n'
            'extern "C" fn read() -> i32;\n'
        )
        link_on_normal_function = parse_source(
            '@link("sensor")\nfn read() -> i32 { return 1; }\n'
        )

        self.assertIn(
            "E_CONFLICTING_EFFECTS",
            [item.code for item in pure_extern.diagnostics],
        )
        self.assertIn("E_EXTERN_LINK", [item.code for item in path_library.diagnostics])
        self.assertIn(
            "E_EXTERN_ATTRIBUTE",
            [item.code for item in link_on_normal_function.diagnostics],
        )

    def test_result_rejects_unsupported_error_types(self) -> None:
        result = parse_source("fn invalid() -> Result<str, CustomError> {}\n")

        self.assertIn(
            "E_RESULT_ERROR_TYPE",
            [item.code for item in result.diagnostics],
        )

    def test_match_result_is_exhaustive_and_types_payloads(self) -> None:
        result = parse_source(
            "@io\n"
            "fn read_or_default() -> str {\n"
            "  let result = io_read_line();\n"
            "  return match result {\n"
            "    Ok(value) => value,\n"
            "    Err(error) => \"fallback\",\n"
            "  };\n"
            "}\n"
        )

        self.assertEqual(result.diagnostics, ())
        returned = result.ast.declarations[0].body[-1].value
        self.assertEqual(returned.kind, "match")
        self.assertEqual(dict(analyze(result.ast).expression_types)[returned.span].name, "str")

    def test_match_result_requires_both_variants_and_compatible_arms(self) -> None:
        result = parse_source(
            "@io\n"
            "fn invalid() -> Result<str, IoError> {\n"
            "  let value = io_read_line();\n"
            "  return match value { Ok(text) => text };\n"
            "}\n"
        )

        self.assertIn(
            "E_MATCH_NON_EXHAUSTIVE",
            [item.code for item in result.diagnostics],
        )

    def test_match_result_rejects_duplicate_variants_and_mismatched_arm_types(self) -> None:
        result = parse_source(
            "@io\n"
            "fn invalid() -> str {\n"
            "  let value = io_read_line();\n"
            "  return match value {\n"
            "    Ok(text) => text,\n"
            "    Ok(other) => other,\n"
            "    Err(error) => false,\n"
            "  };\n"
            "}\n"
        )
        codes = [item.code for item in result.diagnostics]

        self.assertIn("E_MATCH_DUPLICATE_ARM", codes)
        self.assertIn("E_TYPE_MISMATCH", codes)

    def test_match_and_try_survive_ast_canonical_roundtrip(self) -> None:
        source = (
            "@io\n"
            "fn read() -> Result<str, IoError> { return io_read_line(); }\n"
            "@io\n"
            "fn wrapper() -> str {\n"
            "  let result = read();\n"
            "  return match result { Ok(value) => value, Err(error) => \"fallback\" };\n"
            "}\n"
        )
        parsed = parse_source(source)
        canonical = render_module(parsed.ast)
        reparsed = parse_source(canonical)

        self.assertEqual(parsed.diagnostics, ())
        self.assertEqual(reparsed.diagnostics, ())
        self.assertIn("match result", canonical)

    def test_question_operator_propagates_matching_error_type(self) -> None:
        result = parse_source(
            "@io\n"
            "fn read_value() -> Result<str, IoError> { return io_read_line(); }\n"
            "@io\n"
            "fn wrapper() -> Result<str, IoError> { return read_value()?; }\n"
        )

        self.assertEqual(result.diagnostics, ())
        propagated = result.ast.declarations[1].body[0].value
        self.assertEqual(dict(analyze(result.ast).expression_types)[propagated.span].name, "str")

    def test_question_operator_rejects_wrong_context_and_error_type(self) -> None:
        result = parse_source(
            "@io\n"
            "fn read_value() -> Result<str, IoError> { return io_read_line(); }\n"
            "fn invalid_context() -> str { return read_value()?; }\n"
            "@io\n"
            "fn invalid_error() -> Result<str, FsError> { return read_value()?; }\n"
        )
        codes = [item.code for item in result.diagnostics]

        self.assertIn("E_TRY_CONTEXT", codes)
        self.assertIn("E_TRY_ERROR_TYPE", codes)

    def test_result_values_cannot_be_silently_discarded(self) -> None:
        result = parse_source(
            "@io\n"
            "fn invalid() { io_read_line(); }\n"
            "@io\n"
            "fn also_invalid() { let ignored = io_read_line(); }\n"
        )

        self.assertEqual(
            [item.code for item in result.diagnostics].count("E_UNHANDLED_RESULT"),
            2,
        )

    def test_result_binding_requires_a_handling_use_not_any_reference(self) -> None:
        result = parse_source(
            "@io\n"
            "fn invalid() {\n"
            "  let value = io_read_line();\n"
            "  let unrelated = value == value;\n"
            "}\n"
            "@io\n"
            "fn handled() -> str {\n"
            "  let value = io_read_line();\n"
            "  let alias = value;\n"
            "  return match alias { Ok(text) => text, Err(error) => \"fallback\" };\n"
            "}\n"
        )

        unhandled = [
            item for item in result.diagnostics
            if item.code == "E_UNHANDLED_RESULT"
        ]
        self.assertEqual(len(unhandled), 1)
        self.assertIn("value", unhandled[0].message)

    def test_result_in_short_circuited_match_is_not_definitely_handled(self) -> None:
        result = parse_source(
            "@io\n"
            "fn maybe() -> bool {\n"
            "  let value = io_read_line();\n"
            "  return true || match value {\n"
            "    Ok(_) => false,\n"
            "    Err(_) => false,\n"
            "  };\n"
            "}\n"
        )

        self.assertIn("E_UNHANDLED_RESULT", [item.code for item in result.diagnostics])

    def test_result_use_after_return_does_not_count_as_handling(self) -> None:
        result = parse_source(
            "@io\n"
            "fn maybe() -> str {\n"
            "  let value = io_read_line();\n"
            "  return \"done\";\n"
            "  return match value { Ok(text) => text, Err(_) => \"fallback\" };\n"
            "}\n"
        )

        self.assertIn("E_UNHANDLED_RESULT", [item.code for item in result.diagnostics])

    def test_result_binding_cannot_be_consumed_more_than_once(self) -> None:
        result = parse_source(
            "@io\n"
            "fn invalid() -> str {\n"
            "  let value = io_read_line();\n"
            "  let alias = value;\n"
            "  let second_alias = alias;\n"
            "  let first = match second_alias {\n"
            "    Ok(text) => text,\n"
            "    Err(_) => \"first fallback\",\n"
            "  };\n"
            "  return match value {\n"
            "    Ok(text) => text,\n"
            "    Err(_) => \"second fallback\",\n"
            "  };\n"
            "}\n"
        )

        self.assertIn("E_RESULT_REUSE", [item.code for item in result.diagnostics])

    def test_optional_member_access_is_rejected_without_an_option_type(self) -> None:
        result = parse_source(
            "type Point { x: f32, }\n"
            "fn read(point: Point) -> f32 { return point?.x; }\n"
        )

        self.assertIn(
            "E_OPTIONAL_MEMBER_UNSUPPORTED",
            [item.code for item in result.diagnostics],
        )

    def test_coalescing_operator_is_rejected_until_option_semantics_exist(self) -> None:
        result = parse_source(
            "fn fallback(value: bool) -> bool { return value ?? false; }\n"
        )

        self.assertIn("E_TYPE_OPERATOR", [item.code for item in result.diagnostics])

    def test_effectful_builtin_signatures_return_typed_results(self) -> None:
        result = parse_source(
            "@io\n"
            "fn read_input() -> Result<str, IoError> { return io_read_line(); }\n"
            "@io\n"
            "fn write_output(text: str) -> Result<void, IoError> {\n"
            "  return io_write(text);\n"
            "}\n"
            "@fs\n"
            "fn read_file(path: str) -> Result<str, FsError> {\n"
            "  return fs_read_text(path);\n"
            "}\n"
            "@fs\n"
            "fn write_file(path: str, text: str) -> Result<void, FsError> {\n"
            "  return fs_write_text(path, text);\n"
            "}\n"
            "@fs\n"
            "fn file_exists(path: str) -> Result<bool, FsError> {\n"
            "  return fs_exists(path);\n"
            "}\n"
            "@net\n"
            "fn get_data(url: str) -> Result<str, NetError> { return net_get(url); }\n"
            "@net\n"
            "fn post_data(url: str, body: str) -> Result<str, NetError> {\n"
            "  return net_post(url, body);\n"
            "}\n"
        )

        self.assertEqual(result.diagnostics, ())

    def test_effectful_builtin_argument_and_result_types_are_checked(self) -> None:
        result = parse_source(
            "@fs\n"
            "fn invalid() -> Result<bool, FsError> {\n"
            "  return fs_read_text(false);\n"
            "}\n"
        )
        codes = [item.code for item in result.diagnostics]

        self.assertIn("E_ARGUMENT_TYPE", codes)
        self.assertIn("E_RETURN_TYPE", codes)

    def test_return_and_argument_types_are_checked(self) -> None:
        result = parse_source(
            "fn takes(value: i32) -> i32 { return value; }\n"
            "fn caller() -> bool { return takes(true); }\n"
        )
        codes = [item.code for item in result.diagnostics]

        self.assertIn("E_ARGUMENT_TYPE", codes)
        self.assertIn("E_RETURN_TYPE", codes)

    def test_numeric_argument_types_are_not_silently_coerced(self) -> None:
        result = parse_source(
            "fn takes(value: i32) { return value; }\n"
            "fn caller() { takes(1.0); }\n"
        )

        self.assertIn("E_ARGUMENT_TYPE", [item.code for item in result.diagnostics])


if __name__ == "__main__":
    unittest.main()
