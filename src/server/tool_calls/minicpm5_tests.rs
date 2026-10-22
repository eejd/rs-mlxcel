// Copyright 2025-2026 Lablup Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::str::FromStr as _;

use serde_json::{Value, json};

use super::try_minicpm5;
use crate::server::tool_calls::{ToolCallFormat, parse_tool_calls};
use crate::server::types::request::{FunctionDefinition, Tool};

fn tool(name: &str, properties: Value) -> Tool {
    Tool {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: name.to_string(),
            description: None,
            parameters: Some(json!({"type": "object", "properties": properties})),
        },
    }
}

/// The tool set of the quantbench `tool_call` / `tool_arg` / `tool_parallel`
/// cells the stored model outputs were generated against.
fn qualify_tools() -> Vec<Tool> {
    vec![
        tool(
            "get_weather",
            json!({"city": {"type": "string"},
                   "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}),
        ),
        tool(
            "add_numbers",
            json!({"a": {"type": "integer"}, "b": {"type": "integer"}}),
        ),
        tool(
            "search_docs",
            json!({"query": {"type": "string"}, "limit": {"type": "integer"}}),
        ),
        tool(
            "echo_args",
            json!({"text": {"type": "string"},
                   "count": {"type": "integer"},
                   "flag": {"type": "boolean"},
                   "items": {"type": "array", "items": {"type": "string"}},
                   "nested": {"type": "object"}}),
        ),
    ]
}

fn args(call: &crate::server::tool_calls::ParsedToolCall) -> Value {
    serde_json::from_str(&call.arguments).expect("arguments are a JSON object")
}

fn parse(text: &str) -> crate::server::tool_calls::ToolCallParseResult {
    try_minicpm5(text, Some(&qualify_tools())).expect("a MiniCPM5 call")
}

// -- the stored model outputs ------------------------------------------------

/// The 60 `tool_call` / `tool_arg` / `tool_parallel` outputs MiniCPM5-2B-4bit
/// wrote under mlxcel in the T0 engine runs (py-quantbench#217 and #287), where
/// all 60 stayed in `content` (eejd/rs-mlxcel#19), each with the call the
/// quantbench case expects. Run 1, run 2 and the repeat of run 2 wrote the same
/// 60 strings.
const ENGINE_RUN_OUTPUTS: &str = include_str!("fixtures/minicpm5_engine_run_outputs.jsonl");

#[test]
fn engine_run_outputs_parse_to_the_expected_typed_calls() {
    let tools = qualify_tools();
    let mut seen = 0;
    for line in ENGINE_RUN_OUTPUTS.lines() {
        let row: Value = serde_json::from_str(line).unwrap();
        let (ability, case, raw) = (
            row["ability"].as_str().unwrap(),
            row["case"].as_str().unwrap(),
            row["raw"].as_str().unwrap(),
        );
        let parsed = parse_tool_calls(raw, Some(&tools));
        assert_eq!(parsed.format, Some(ToolCallFormat::Minicpm5), "{case}");
        assert_eq!(parsed.content, "", "{case}: the markup is not content");
        let expect = &row["expect"];
        if ability == "tool_parallel" {
            let mut cities: Vec<String> = parsed
                .tool_calls
                .iter()
                .map(|call| {
                    assert_eq!(call.name, "get_weather", "{case}");
                    args(call)["city"].as_str().unwrap().to_string()
                })
                .collect();
            cities.sort();
            let mut want: Vec<String> = expect["cities"]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| c.as_str().unwrap().to_string())
                .collect();
            want.sort();
            assert_eq!(cities, want, "{case}");
        } else {
            assert_eq!(parsed.tool_calls.len(), 1, "{case}");
            assert_eq!(parsed.tool_calls[0].name, expect["name"], "{case}");
            // `Value` equality tells `17` from `17.0` and `"17"` from `17`.
            assert_eq!(args(&parsed.tool_calls[0]), expect["args"], "{case}");
        }
        seen += 1;
    }
    assert_eq!(seen, 60);
}

// -- typing by the request schema ---------------------------------------------

#[test]
fn scalar_arguments_are_typed_by_their_declaration() {
    let result = parse(
        r#"<function name="echo_args"><param name="text">hand summer</param><param name="count">17</param><param name="flag">True</param></function>"#,
    );
    assert_eq!(result.format, Some(ToolCallFormat::Minicpm5));
    assert_eq!(result.tool_calls[0].name, "echo_args");
    let arguments = args(&result.tool_calls[0]);
    assert_eq!(
        arguments,
        json!({"text": "hand summer", "count": 17, "flag": true})
    );
    assert!(arguments["count"].is_i64());
}

#[test]
fn string_typed_parameter_keeps_text_that_reads_as_another_type() {
    let result = parse(
        r#"<function name="echo_args"><param name="text">02134</param></function><function name="echo_args"><param name="text">True</param></function><function name="echo_args"><param name="text">None</param></function>"#,
    );
    let texts: Vec<Value> = result
        .tool_calls
        .iter()
        .map(|c| args(c)["text"].clone())
        .collect();
    assert_eq!(texts, [json!("02134"), json!("True"), json!("None")]);
}

#[test]
fn list_and_object_arguments_arrive_as_python_literals_and_are_typed() {
    // The template renders a non-string argument with Python's `str()`.
    let result = parse(
        r#"<function name="echo_args"><param name="text">list</param><param name="items">['age', 'computer', 'issue']</param></function><function name="echo_args"><param name="text">obj</param><param name="nested">{'unit': 'a', 'depth': 2, 'on': True, 'none': None, 'xs': [1, 2.5]}</param></function>"#,
    );
    assert_eq!(
        args(&result.tool_calls[0]),
        json!({"text": "list", "items": ["age", "computer", "issue"]})
    );
    assert_eq!(
        args(&result.tool_calls[1]),
        json!({"text": "obj",
               "nested": {"unit": "a", "depth": 2, "on": true, "none": null, "xs": [1, 2.5]}})
    );
}

#[test]
fn json_literals_and_python_string_escapes_are_read() {
    let result = parse(
        r#"<function name="echo_args"><param name="items">["x", "y"]</param><param name="nested">{"k": [1, {"z": null}]}</param></function><function name="echo_args"><param name="items">["it's", 'say "hi"', 'a\nb', 'back\\slash', 'caf\u00e9']</param></function>"#,
    );
    assert_eq!(
        args(&result.tool_calls[0]),
        json!({"items": ["x", "y"], "nested": {"k": [1, {"z": null}]}})
    );
    assert_eq!(
        args(&result.tool_calls[1])["items"],
        json!(["it's", "say \"hi\"", "a\nb", "back\\slash", "caf\u{e9}"])
    );
}

#[test]
fn a_dict_for_an_array_parameter_stays_text() {
    let result =
        parse(r#"<function name="echo_args"><param name="items">{'a': 1}</param></function>"#);
    assert_eq!(args(&result.tool_calls[0]), json!({"items": "{'a': 1}"}));
}

#[test]
fn an_unparseable_literal_stays_text() {
    let result = parse(
        r#"<function name="echo_args"><param name="items">[unquoted, words]</param><param name="nested">{'a': }</param></function>"#,
    );
    assert_eq!(
        args(&result.tool_calls[0]),
        json!({"items": "[unquoted, words]", "nested": "{'a': }"})
    );
}

#[test]
fn deeply_nested_literal_is_bounded() {
    let depth = 200_000;
    let nested = format!("{}{}", "[".repeat(depth), "]".repeat(depth));
    let text =
        format!(r#"<function name="echo_args"><param name="items">{nested}</param></function>"#);
    // Neither a stack overflow nor a typed array: past the depth cap the value is text.
    let result = parse(&text);
    assert!(args(&result.tool_calls[0])["items"].is_string());
}

#[test]
fn undeclared_parameters_take_the_loose_literal_rules() {
    let result = try_minicpm5(
        r#"<function name="anything"><param name="n">17</param><param name="f">2.5</param><param name="b">True</param><param name="z">None</param><param name="l">[1, 'a']</param><param name="s">plain words</param></function>"#,
        None,
    )
    .unwrap();
    assert_eq!(
        args(&result.tool_calls[0]),
        json!({"n": 17, "f": 2.5, "b": true, "z": null, "l": [1, "a"], "s": "plain words"})
    );
}

#[test]
fn integer_typed_parameter_takes_a_zero_fraction_float_and_pads_are_trimmed() {
    let result = parse(
        r#"<function name="add_numbers"><param name="a">
 5.0 </param><param name="b"> 344
</param></function>"#,
    );
    let arguments = args(&result.tool_calls[0]);
    assert_eq!(arguments, json!({"a": 5, "b": 344}));
    assert!(arguments["a"].is_i64());
}

// -- values: bare markup and CDATA -------------------------------------------

#[test]
fn bare_value_may_hold_unescaped_markup() {
    // Not well-formed XML; the model writes it this way (stored output markup-03).
    let result = parse(
        r#"<function name="echo_args"><param name="text">if student < way and official > organization then <ok> & done</param></function>"#,
    );
    assert_eq!(
        args(&result.tool_calls[0])["text"],
        "if student < way and official > organization then <ok> & done"
    );
}

#[test]
fn cdata_value_is_verbatim_including_indentation() {
    let result = parse(
        "<function name=\"echo_args\"><param name=\"text\"><![CDATA[def canyon(x):\n    return x + 1]]></param></function>",
    );
    assert_eq!(
        args(&result.tool_calls[0])["text"],
        "def canyon(x):\n    return x + 1"
    );
}

#[test]
fn cdata_payload_may_contain_the_closing_tags_and_cdata_end() {
    let result = parse(
        r#"<function name="echo_args"><param name="text"><![CDATA[a </param> b </function> c ]]> d]]></param><param name="count">3</param></function>"#,
    );
    assert_eq!(result.tool_calls.len(), 1);
    assert_eq!(
        args(&result.tool_calls[0]),
        json!({"text": "a </param> b </function> c ]]> d", "count": 3})
    );
}

#[test]
fn cdata_is_not_trimmed_but_surrounding_whitespace_is_ignored() {
    let result = parse(
        "<function name=\"echo_args\"><param name=\"text\">\n  <![CDATA[  padded  ]]>\n</param></function>",
    );
    assert_eq!(args(&result.tool_calls[0])["text"], "  padded  ");
}

#[test]
fn bare_value_containing_a_function_closer_is_not_cut_short() {
    let result = parse(
        r#"<function name="echo_args"><param name="text">close it with </function> please</param><param name="count">1</param></function>"#,
    );
    assert_eq!(
        args(&result.tool_calls[0]),
        json!({"text": "close it with </function> please", "count": 1})
    );
}

#[test]
fn bare_string_loses_only_one_framing_newline_per_side() {
    let result = parse(
        "<function name=\"echo_args\"><param name=\"text\">\n\nkeep blank above\n</param></function>",
    );
    assert_eq!(args(&result.tool_calls[0])["text"], "\nkeep blank above");
}

// -- calls --------------------------------------------------------------------

#[test]
fn parallel_calls_are_read_in_order_whatever_separates_them() {
    for separator in ["", "\n", " \n\n "] {
        let text = format!(
            r#"<function name="get_weather"><param name="city">Havana</param></function>{separator}<function name="get_weather"><param name="city">Dublin</param></function>"#
        );
        let result = parse(&text);
        let cities: Vec<Value> = result
            .tool_calls
            .iter()
            .map(|c| args(c)["city"].clone())
            .collect();
        assert_eq!(cities, [json!("Havana"), json!("Dublin")], "{separator:?}");
        assert_eq!(result.content, "");
    }
}

#[test]
fn text_around_the_calls_is_content() {
    let result = parse(
        r#"Let me check. <function name="get_weather"><param name="city">Lima</param></function> And then
<function name="get_weather"><param name="city">Oslo</param></function>
Done."#,
    );
    assert_eq!(result.tool_calls.len(), 2);
    assert_eq!(result.content, "Let me check.\nAnd then\nDone.");
}

#[test]
fn zero_argument_and_self_closing_calls() {
    let result = parse(
        r#"<function name="get_time"></function><function name="ping"/><function name="pong" />"#,
    );
    let names: Vec<&str> = result.tool_calls.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, ["get_time", "ping", "pong"]);
    assert!(result.tool_calls.iter().all(|c| c.arguments == "{}"));
}

#[test]
fn attributes_are_read_tolerantly() {
    let result = parse(
        r#"<function  data-name="x"  name = 'get_weather' id="1"><param id="p" name='city'>Lima</param><param name=unit>celsius</param></function><function name=add_numbers><param name="a">1</param></function>"#,
    );
    assert_eq!(result.tool_calls[0].name, "get_weather");
    assert_eq!(
        args(&result.tool_calls[0]),
        json!({"city": "Lima", "unit": "celsius"})
    );
    assert_eq!(result.tool_calls[1].name, "add_numbers");
}

#[test]
fn a_later_duplicate_parameter_replaces_the_earlier() {
    let result = parse(
        r#"<function name="add_numbers"><param name="a">1</param><param name="a">2</param></function>"#,
    );
    assert_eq!(args(&result.tool_calls[0]), json!({"a": 2}));
}

#[test]
fn stray_text_and_nameless_params_inside_a_call_are_skipped() {
    let result = parse(
        r#"<function name="add_numbers"><param id="p">9</param><param name="a">1</param> more noise<param name="b">2</param></function>"#,
    );
    assert_eq!(args(&result.tool_calls[0]), json!({"a": 1, "b": 2}));
}

#[test]
fn a_call_cut_off_inside_a_value_is_not_salvaged() {
    // The tool would run without the argument that was being written.
    let text = r#"<function name="search_docs"><param name="query">prefix cache</param><param name="limit">2</param><param name="extra">cut o"#;
    assert!(try_minicpm5(text, Some(&qualify_tools())).is_none());
    let text = r#"<function name="search_docs"><param name="query">prefix cache</param><param name="limit"#;
    assert!(try_minicpm5(text, Some(&qualify_tools())).is_none());
}

#[test]
fn a_call_that_ends_after_a_complete_parameter_is_a_call() {
    // Only the `</function>` is missing, as when the model stops there.
    for tail in ["", "\n", " "] {
        let text = format!(
            r#"<function name="search_docs"><param name="query">prefix cache</param><param name="limit">2</param>{tail}"#
        );
        let result = parse(&text);
        assert_eq!(
            args(&result.tool_calls[0]),
            json!({"query": "prefix cache", "limit": 2})
        );
    }
    let truncated = r#"<function name="search_docs"><param name="query">a</param></func"#;
    assert!(try_minicpm5(truncated, Some(&qualify_tools())).is_none());
}

#[test]
fn call_missing_its_closer_ends_where_the_next_call_starts() {
    let result = parse(
        r#"<function name="get_weather"><param name="city">Lima</param><function name="get_weather"><param name="city">Oslo</param></function>"#,
    );
    let cities: Vec<Value> = result
        .tool_calls
        .iter()
        .map(|c| args(c)["city"].clone())
        .collect();
    assert_eq!(cities, [json!("Lima"), json!("Oslo")]);
}

#[test]
fn writing_about_the_syntax_is_not_a_call() {
    for text in [
        r#"You open a call with <function name="get_weather"> and close it."#,
        "<function> alone, <function name=\"\"></function>, <functional name=\"x\"></functional>",
        "plain answer",
        "",
    ] {
        assert!(
            try_minicpm5(text, Some(&qualify_tools())).is_none(),
            "{text:?}"
        );
    }
}

#[test]
fn a_call_cut_off_in_its_opening_tag_stays_text() {
    let result = parse(
        r#"Checking. <function name="get_weather"><param name="city">Lima</param></function><function name="get_wea"#,
    );
    assert_eq!(result.tool_calls.len(), 1);
    assert_eq!(result.content, "Checking.\n<function name=\"get_wea");
}

#[test]
fn a_body_that_is_not_parameters_is_not_a_call() {
    // A JSON body in the MiniCPM5 wrapper must not become a call with empty
    // arguments (it would run the tool without them), and an XML snippet or a
    // sentence after an opener is the model writing, not calling.
    for text in [
        r#"<function name="get_weather">{"city": "Lima"}</function>"#,
        "<function name=\"foo\">\n  <body/>\n</function>",
        r#"Example: <function name="x"> opens a block."#,
        r#"<function name="x">noise<param name="a">1</param></function>"#,
        r#"<function name="x"><params>1</params></function>"#,
        r#"<function name="x"><param>1</param></function>"#,
    ] {
        assert!(
            try_minicpm5(text, Some(&qualify_tools())).is_none(),
            "{text:?}"
        );
    }
}

#[test]
fn a_json_body_in_the_wrapper_is_left_as_content_by_the_dispatcher() {
    let tools = qualify_tools();
    let text = r#"<function name="get_weather">{"city": "Lima"}</function>"#;
    let result = parse_tool_calls(text, Some(&tools));
    assert!(!result.has_tool_calls());
    assert_eq!(result.content, text);
}

#[test]
fn a_prose_opener_does_not_hide_a_later_call() {
    let result = parse(
        r#"You write <function name="x"> to open a call, as in <function name="get_weather"><param name="city">Lima</param></function>"#,
    );
    assert_eq!(result.tool_calls.len(), 1);
    assert_eq!(
        result.content,
        "You write <function name=\"x\"> to open a call, as in"
    );
}

#[test]
fn a_quote_in_an_attribute_hides_a_closing_bracket() {
    let result = parse(r#"<function name="a>b"><param name="c">1</param></function>"#);
    assert_eq!(result.tool_calls[0].name, "a>b");
}

#[test]
fn a_function_namespaced_like_the_tool_gets_that_tools_schema() {
    let result =
        parse(r#"<function name="functions.add_numbers"><param name="a">7</param></function>"#);
    assert_eq!(args(&result.tool_calls[0]), json!({"a": 7}));
}

// -- dispatch -----------------------------------------------------------------

#[test]
fn dispatch_drops_calls_to_undeclared_functions() {
    let text = r#"<function name="rm_rf"><param name="path">/</param></function>"#;
    let tools = qualify_tools();
    let result = parse_tool_calls(text, Some(&tools));
    assert!(!result.has_tool_calls());
    // Without a tool list nothing is declared, so nothing is filtered.
    assert_eq!(parse_tool_calls(text, None).tool_calls[0].name, "rm_rf");
}

#[test]
fn dispatch_reads_the_call_after_a_thinking_block() {
    let tools = qualify_tools();
    let result = parse_tool_calls(
        "<think>I should call <function name=\"add_numbers\"> maybe</think>\n\n<function name=\"add_numbers\"><param name=\"a\">2</param><param name=\"b\">3</param></function>",
        Some(&tools),
    );
    assert_eq!(result.format, Some(ToolCallFormat::Minicpm5));
    assert_eq!(result.tool_calls.len(), 1);
    assert_eq!(args(&result.tool_calls[0]), json!({"a": 2, "b": 3}));
}

#[test]
fn equals_sign_function_formats_still_go_to_their_own_parsers() {
    let tools = vec![tool("f", json!({"a": {"type": "integer"}}))];
    let qwen = parse_tool_calls(
        "<tool_call>\n<function=f>\n<parameter=a>\n1\n</parameter>\n</function>\n</tool_call>",
        Some(&tools),
    );
    assert_eq!(qwen.format, Some(ToolCallFormat::Qwen3Coder));
    let functionary = parse_tool_calls(r#"<function=f>{"a": 1}</function>"#, Some(&tools));
    assert_eq!(functionary.format, Some(ToolCallFormat::FunctionaryV31));
}

#[test]
fn format_name_round_trips() {
    assert_eq!(ToolCallFormat::Minicpm5.as_str(), "minicpm5");
    assert_eq!(
        ToolCallFormat::from_str("minicpm5"),
        Ok(ToolCallFormat::Minicpm5)
    );
}

#[test]
fn a_json_call_written_inside_a_parameter_is_not_taken_for_a_call() {
    let tools = qualify_tools();
    let result = parse_tool_calls(
        r#"<function name="echo_args"><param name="text">{"name": "add_numbers", "arguments": {"a": 1, "b": 2}}</param></function>"#,
        Some(&tools),
    );
    assert_eq!(result.format, Some(ToolCallFormat::Minicpm5));
    assert_eq!(result.tool_calls.len(), 1);
    assert_eq!(result.tool_calls[0].name, "echo_args");
    assert_eq!(
        args(&result.tool_calls[0])["text"],
        r#"{"name": "add_numbers", "arguments": {"a": 1, "b": 2}}"#
    );
}

// -- non-ASCII input ------------------------------------------------------------

#[test]
fn python_literals_with_multibyte_whitespace_between_tokens_are_read() {
    // U+00A0, U+0085 and U+3000 are `char::is_whitespace` and 2 or 3 bytes long.
    for space in ["\u{a0}", "\u{85}", "\u{3000}", "\u{a0}\u{3000}"] {
        let result = parse(&format!(
            "<function name=\"echo_args\"><param name=\"items\">['a',{space}'b' {space}]</param><param name=\"nested\">{{'k':{space}1,{space}'z':{space}[{space}2{space}]}}</param></function>"
        ));
        assert_eq!(
            args(&result.tool_calls[0]),
            json!({"items": ["a", "b"], "nested": {"k": 1, "z": [2]}}),
            "{space:?}"
        );
    }
}

#[test]
fn multibyte_text_in_names_values_cdata_and_literals_is_kept() {
    let result = parse(
        "<function name=\"echo_args\"><param name=\"text\">caf\u{e9} e\u{301} \u{1f600} \u{4e2d}\u{6587}</param></function><function name=\"echo_args\"><param name=\"text\"><![CDATA[\u{1f600} < \u{e9}]]></param><param name=\"items\">['\u{1f600}', '\u{e9}', '\\u00e9', '\\x41']</param></function>",
    );
    assert_eq!(
        args(&result.tool_calls[0])["text"],
        "caf\u{e9} e\u{301} \u{1f600} \u{4e2d}\u{6587}"
    );
    assert_eq!(args(&result.tool_calls[1])["text"], "\u{1f600} < \u{e9}");
    assert_eq!(
        args(&result.tool_calls[1])["items"],
        json!(["\u{1f600}", "\u{e9}", "\u{e9}", "A"])
    );
}

#[test]
fn a_hex_escape_must_be_hex_digits() {
    // `from_str_radix` alone would read `+123` as U+0123.
    let result = parse(
        r#"<function name="echo_args"><param name="items">['\u+123']</param></function><function name="echo_args"><param name="items">['\x4z']</param></function>"#,
    );
    assert_eq!(args(&result.tool_calls[0])["items"], "['\\u+123']");
    assert_eq!(args(&result.tool_calls[1])["items"], "['\\x4z']");
}

/// A deterministic xorshift stream: the fuzz tests must fail the same way twice.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// The pieces fuzz inputs are built from: the grammar's own tokens, in whole and
/// in part, next to multi-byte, combining and 4-byte characters.
const PIECES: &[&str] = &[
    "<function name=\"",
    "<function  id=1 name='",
    "<function name=get_weather>",
    "<function",
    "\">",
    "'>",
    "/>",
    ">",
    "<param name=\"",
    "<param name='",
    "<param id=\"p\">",
    "<param",
    "</param>",
    "</function>",
    "</func",
    "<![CDATA[",
    "]]>",
    "<![CD",
    "get_weather",
    "echo_args",
    "add_numbers",
    "city",
    "items",
    "nested",
    "text",
    "count",
    " ",
    "\n",
    "\t",
    "\u{a0}",
    "\u{85}",
    "\u{3000}",
    "\u{e9}",
    "e\u{301}",
    "\u{1f600}",
    "\u{4e2d}",
    "[",
    "]",
    "{",
    "}",
    "(",
    ")",
    ":",
    ",",
    "'",
    "\"",
    "\\",
    "\\u00e9",
    "\\x4",
    "\\n",
    "True",
    "None",
    "1",
    "-2.5",
    "x",
    "<",
    "&",
    "/",
    "=",
    "<think>",
    "</think>",
];

fn fuzz_text(rng: &mut Rng) -> String {
    (0..rng.below(40))
        .map(|_| PIECES[rng.below(PIECES.len())])
        .collect()
}

#[test]
fn no_input_makes_the_parser_panic_or_emit_a_bad_call() {
    let tools = qualify_tools();
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut seeds: Vec<String> = ENGINE_RUN_OUTPUTS
        .lines()
        .map(|l| {
            serde_json::from_str::<Value>(l).unwrap()["raw"]
                .as_str()
                .unwrap()
                .to_string()
        })
        .collect();
    for _ in 0..20_000 {
        seeds.push(fuzz_text(&mut rng));
    }
    for (n, text) in seeds.iter().enumerate() {
        // Fixtures are also cut at every character boundary.
        let cuts: Vec<usize> = if n < 60 {
            text.char_indices()
                .map(|(i, _)| i)
                .chain([text.len()])
                .collect()
        } else {
            vec![text.len()]
        };
        for cut in cuts {
            let cut_text = &text[..cut];
            for with_tools in [Some(&tools[..]), None] {
                if let Some(result) = try_minicpm5(cut_text, with_tools) {
                    assert!(!result.tool_calls.is_empty(), "{cut_text:?}");
                    for call in &result.tool_calls {
                        assert!(!call.name.is_empty(), "{cut_text:?}");
                        assert!(
                            serde_json::from_str::<Value>(&call.arguments)
                                .unwrap()
                                .is_object(),
                            "{cut_text:?}"
                        );
                    }
                }
            }
        }
    }
}

// -- cost -------------------------------------------------------------------------

#[test]
fn a_value_of_many_cdata_sections_is_read_in_one_pass() {
    // 300k sections: re-searching the far `</param>` after each was quadratic
    // (seconds for 200k); one pass is milliseconds.
    let sections = "<![CDATA[]]>".repeat(300_000);
    let text = format!(
        r#"<function name="echo_args"><param name="text">{sections}</param><param name="count">2</param></function>"#
    );
    let start = std::time::Instant::now();
    let result = parse(&text);
    assert!(start.elapsed() < std::time::Duration::from_secs(5));
    assert_eq!(args(&result.tool_calls[0])["count"], 2);
}

#[test]
fn a_run_of_unclosed_openers_costs_a_bounded_number_of_rescans() {
    let text = r#"<function name="echo_args"><param name="text">x"#.repeat(100_000);
    let start = std::time::Instant::now();
    assert!(try_minicpm5(&text, Some(&qualify_tools())).is_none());
    assert!(start.elapsed() < std::time::Duration::from_secs(5));
}

#[test]
fn words_after_a_parameter_before_the_next_call_make_the_first_one_text() {
    // Nothing says where that call ended, so it is not run; the next one is.
    let result = parse(
        r#"<function name="get_weather"><param name="city">Lima</param> and then <function name="get_weather"><param name="city">Oslo</param></function>"#,
    );
    assert_eq!(result.tool_calls.len(), 1);
    assert_eq!(args(&result.tool_calls[0]), json!({"city": "Oslo"}));
    assert!(
        result
            .content
            .starts_with(r#"<function name="get_weather"><param name="city">Lima"#)
    );
}
