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

use super::StreamFilter;

const CALL: &str = r#"<function name="get_weather"><param name="city">Lima</param></function>"#;
/// A CDATA payload that holds both closers a plain scan would stop at.
const CDATA_CALL: &str = "<function name=\"echo_args\"><param name=\"text\"><![CDATA[a </param> b </function> c <![CDATA[ ]]></param><param name=\"count\">3</param></function>";

struct Streamed {
    content: String,
    reasoning: String,
    suppressed: usize,
    consumed: usize,
}

fn stream(mut filter: StreamFilter, fragments: &[&str]) -> Streamed {
    let mut streamed = Streamed {
        content: String::new(),
        reasoning: String::new(),
        suppressed: 0,
        consumed: 0,
    };
    let mut add = |out: super::FilterOutput| {
        streamed
            .content
            .push_str(out.content.as_deref().unwrap_or(""));
        streamed
            .reasoning
            .push_str(out.reasoning.as_deref().unwrap_or(""));
        streamed.suppressed += out.suppressed_positions;
        streamed.consumed += out.consumed_positions;
    };
    for fragment in fragments {
        add(filter.feed(fragment));
    }
    add(filter.flush());
    streamed
}

/// `text` cut into fragments of `size` characters.
fn chunks(text: &str, size: usize) -> Vec<String> {
    text.chars()
        .collect::<Vec<_>>()
        .chunks(size)
        .map(|c| c.iter().collect())
        .collect()
}

fn stream_chunked(text: &str, size: usize) -> Streamed {
    let owned = chunks(text, size);
    let fragments: Vec<&str> = owned.iter().map(String::as_str).collect();
    stream(StreamFilter::new(), &fragments)
}

#[test]
fn a_call_is_suppressed_and_the_text_around_it_is_content() {
    let out = stream(
        StreamFilter::new(),
        &[&format!("Let me check. {CALL}Done.")],
    );
    assert_eq!(out.content, "Let me check. Done.");
}

#[test]
fn the_markup_never_leaks_wherever_the_fragments_split() {
    let text = format!("A{CALL}\n{CDATA_CALL}\n{CALL}\nB");
    for size in 1..=17 {
        let out = stream_chunked(&text, size);
        assert_eq!(out.content, "AB", "fragments of {size} characters");
    }
}

#[test]
fn only_the_close_outside_cdata_ends_the_call() {
    // The first `</function>` is inside the payload: ending the call there would
    // show ` c <![CDATA[ ]]>...` to the client.
    let out = stream_chunked(&format!("before {CDATA_CALL} after"), 1);
    assert_eq!(out.content, "before after");
}

#[test]
fn whitespace_between_and_after_calls_is_not_content() {
    let out = stream_chunked(&format!("{CALL} \n\n{CALL}\n"), 1);
    assert_eq!(out.content, "");
}

#[test]
fn text_after_a_call_keeps_its_own_leading_text() {
    let out = stream_chunked(&format!("{CALL}\n  then words"), 3);
    assert_eq!(out.content, "then words");
}

#[test]
fn a_call_cut_off_by_the_end_of_generation_stays_hidden() {
    for text in [
        r#"x<function name="a"><param name="p">1"#,
        r#"x<function name="a"><param name="p"><![CDATA[never closed </function>"#,
        r#"x<function name="a"><param name="p">1</param></func"#,
        r#"x<function name="a"><param name="p"><![CDA"#,
    ] {
        assert_eq!(stream_chunked(text, 1).content, "x", "{text:?}");
    }
}

#[test]
fn position_accounting_covers_every_fragment_of_a_call() {
    let text = format!("A{CALL}B");
    let out = stream_chunked(&text, 1);
    assert_eq!(out.content, "AB");
    // `<function name=` and `</function>` are matched delimiters; the rest of
    // the call is dropped without placeholders for the route to emit.
    assert_eq!(
        out.suppressed,
        "<function name=".len() + "</function>".len()
    );
    assert_eq!(out.consumed, text.chars().count());
}

#[test]
fn text_that_only_resembles_a_call_passes_through() {
    for text in [
        "use <functional name=x> and <function=f>{\"a\": 1}</function> as shown",
        "an unfinished <function na",
        "a <function name> without the equals sign",
    ] {
        assert_eq!(stream_chunked(text, 1).content, text);
    }
}

#[test]
fn a_call_written_about_inside_reasoning_does_not_end_the_reasoning() {
    let out = stream_chunked(
        "<think>I would write <function name=\"x\"> here</think>The answer.",
        2,
    );
    assert_eq!(out.reasoning, "I would write <function name=\"x\"> here");
    assert_eq!(out.content, "The answer.");
}

#[test]
fn the_stored_engine_run_outputs_leave_no_content() {
    let fixture = include_str!("fixtures/minicpm5_engine_run_outputs.jsonl");
    let mut seen = 0;
    for line in fixture.lines() {
        let row: serde_json::Value = serde_json::from_str(line).unwrap();
        let raw = row["raw"].as_str().unwrap();
        assert_eq!(stream_chunked(raw, 5).content, "", "{}", row["case"]);
        seen += 1;
    }
    assert_eq!(seen, 60);
}
