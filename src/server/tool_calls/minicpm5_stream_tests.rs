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
use crate::server::tool_calls::minicpm5::try_minicpm5;

const CALL: &str = r#"<function name="get_weather"><param name="city">Lima</param></function>"#;
/// A CDATA payload that holds both closers a plain scan would stop at.
const CDATA_CALL: &str = "<function name=\"echo_args\"><param name=\"text\"><![CDATA[a </param> b </function> c <![CDATA[ ]]></param><param name=\"count\">3</param></function>";

struct Streamed {
    content: String,
    reasoning: String,
    suppressed: usize,
    consumed: usize,
    /// What the fragments before the final flush had emitted.
    before_flush: String,
}

fn stream(mut filter: StreamFilter, fragments: &[&str]) -> Streamed {
    let mut streamed = Streamed {
        content: String::new(),
        reasoning: String::new(),
        suppressed: 0,
        consumed: 0,
        before_flush: String::new(),
    };
    let add = |out: super::FilterOutput, streamed: &mut Streamed| {
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
        add(filter.feed(fragment), &mut streamed);
    }
    streamed.before_flush = streamed.content.clone();
    add(filter.flush(), &mut streamed);
    streamed
}

fn armed() -> StreamFilter {
    StreamFilter::new().with_minicpm5_calls(true)
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
    stream(armed(), &fragments)
}

/// What the non-streaming parser leaves as content, up to whitespace, which
/// the parser trims around a call and the stream does not add.
fn without_whitespace(text: &str) -> String {
    text.chars().filter(|c| !c.is_whitespace()).collect()
}

fn parsed_content(text: &str) -> String {
    try_minicpm5(text, None).map_or_else(|| text.to_string(), |result| result.content)
}

#[test]
fn a_call_is_suppressed_and_the_text_around_it_is_content() {
    let out = stream(armed(), &[&format!("Let me check. {CALL}Done.")]);
    assert_eq!(out.content, "Let me check. Done.");
}

#[test]
fn the_markup_never_leaks_wherever_the_fragments_split() {
    let text = format!("A{CALL}\n{CDATA_CALL}\n{CALL}\nB");
    for size in 1..=17 {
        let out = stream_chunked(&text, size);
        assert_eq!(out.content, "AB", "fragments of {size} characters");
        // Each call was decided when its last fragment arrived, not at the
        // end of the stream: the text after the last one was already out.
        assert_eq!(out.before_flush, "AB", "fragments of {size} characters");
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
fn a_bare_function_closer_inside_a_value_does_not_end_the_call() {
    // The parser reads parameter by parameter; the stream asks the same scanner.
    let text = r#"a<function name="x"><param name="p">close </function> mid</param></function>b"#;
    for size in [1, 3, 100] {
        assert_eq!(stream_chunked(text, size).content, "ab", "size {size}");
    }
}

#[test]
fn whitespace_between_and_after_calls_is_not_content() {
    let out = stream_chunked(&format!("{CALL} \n\n{CALL}\n"), 1);
    assert_eq!(out.content, "");
}

#[test]
fn text_after_a_call_keeps_its_own_words() {
    let out = stream_chunked(&format!("{CALL}\n  then words"), 3);
    assert_eq!(out.content, "then words");
    assert_eq!(out.before_flush, "then words");
}

#[test]
fn openers_the_parser_reads_in_any_attribute_order_are_suppressed_too() {
    for call in [
        r#"<function  name="x"><param name="p">1</param></function>"#,
        r#"<function id="1" name="x"><param name="p">1</param></function>"#,
        "<function\tname='x'><param name=p>1</param></function>",
        r#"<function name=x><param  name="p">1</param></function>"#,
        r#"<function name="x"/>"#,
        r#"<function name="x"></function>"#,
    ] {
        assert_eq!(stream_chunked(&format!("<<{call}>>"), 1).content, "<<>>");
    }
}

#[test]
fn a_call_cut_off_by_the_end_of_generation_is_shown_as_the_text_it_is() {
    // Not a call (nothing is salvaged from it), so the non-streaming content
    // keeps it; the stream releases it at the end instead of dropping it.
    for text in [
        r#"x<function name="a"><param name="p">1"#,
        r#"x<function name="a"><param name="p"><![CDATA[never closed </function>"#,
        r#"x<function name="a"><param name="p">1</param></func"#,
        r#"x<function name="a"><param name="p"><![CDA"#,
        r#"x<function name="a"#,
        r#"x<function name="a"><par"#,
    ] {
        for size in [1, 4] {
            assert_eq!(stream_chunked(text, size).content, text, "{text:?}");
        }
    }
}

#[test]
fn a_call_that_only_lacks_its_closer_is_a_call_at_the_end_of_the_stream() {
    let text = r#"x<function name="a"><param name="p">1</param>"#;
    assert_eq!(stream_chunked(text, 1).content, "x");
}

#[test]
fn prose_after_an_opener_is_released_at_once_not_at_the_end() {
    let out = stream_chunked(
        r#"Example: <function name="x"> opens a block, then more."#,
        1,
    );
    assert_eq!(
        out.content,
        r#"Example: <function name="x"> opens a block, then more."#
    );
    // Decided at the first word, so the stream carried it all before the flush.
    assert_eq!(out.before_flush, out.content);
}

#[test]
fn a_prose_opener_does_not_hide_a_later_call() {
    let out = stream_chunked(
        &format!(r#"Write <function name="x"> to open a call, as in {CALL}"#),
        2,
    );
    assert_eq!(
        out.content,
        r#"Write <function name="x"> to open a call, as in "#
    );
}

#[test]
fn an_xml_snippet_and_a_json_body_stay_content() {
    for text in [
        "Here is the config:\n```xml\n<function name=\"foo\">\n  <body/>\n</function>\n```\nDone.",
        r#"<function name="get_weather">{"city": "Lima"}</function>"#,
    ] {
        for size in [1, 7, 1000] {
            assert_eq!(stream_chunked(text, size).content, text, "{text:?}");
        }
    }
}

#[test]
fn a_filter_that_is_not_armed_passes_everything_through() {
    // No tools, or `tool_choice: none`: nothing is parsed as a call, so
    // nothing may be removed from what the model wrote.
    for text in [
        CALL.to_string(),
        CDATA_CALL.to_string(),
        "x<function name=\"a\"><param name=\"p\">1".to_string(),
        format!("Here: {CALL} and more."),
    ] {
        let owned = chunks(&text, 3);
        let fragments: Vec<&str> = owned.iter().map(String::as_str).collect();
        assert_eq!(stream(StreamFilter::new(), &fragments).content, text);
        assert_eq!(
            stream(StreamFilter::new().with_minicpm5_calls(false), &fragments).content,
            text
        );
    }
}

#[test]
fn a_partial_opener_at_a_fragment_end_is_only_held_when_armed() {
    let mut armed = armed();
    assert_eq!(armed.feed("a <func").content.as_deref(), Some("a "));
    let mut unarmed = StreamFilter::new();
    assert_eq!(unarmed.feed("a <func").content.as_deref(), Some("a <func"));
}

#[test]
fn position_accounting_covers_every_fragment_of_a_call() {
    let text = format!("A{CALL}B");
    let out = stream_chunked(&text, 1);
    assert_eq!(out.content, "AB");
    // Every token of the call is a suppressed position, so a caller that keeps
    // one queue entry per token drops exactly the call's.
    assert_eq!(out.suppressed, CALL.chars().count());
    assert_eq!(out.consumed, text.chars().count());
}

#[test]
fn text_that_only_resembles_a_call_passes_through() {
    for text in [
        "use <functional name=x> and <function=f>{\"a\": 1}</function> as shown",
        "an unfinished <function na",
        "a <function> alone and <function\u{a0}",
    ] {
        assert_eq!(stream_chunked(text, 1).content, text);
    }
}

#[test]
fn a_call_written_about_inside_reasoning_does_not_end_the_reasoning() {
    let out = stream(
        armed(),
        &["<think>I would write <function name=\"x\"> here</think>The answer."],
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
        for size in [1, 5] {
            assert_eq!(stream_chunked(raw, size).content, "", "{}", row["case"]);
        }
        seen += 1;
    }
    assert_eq!(seen, 60);
}

#[test]
fn a_long_call_is_read_about_once_not_once_per_token() {
    // A 600 KB value in 1-character fragments: re-reading the call at each
    // would take minutes.
    let body = "x".repeat(600_000);
    let text = format!(r#"<function name="f"><param name="p">{body}</param></function>tail"#);
    let start = std::time::Instant::now();
    let out = stream_chunked(&text, 1);
    assert_eq!(out.content, "tail");
    assert!(start.elapsed() < std::time::Duration::from_secs(20));
}

// -- the stream and the parser agree ---------------------------------------------

struct Rng(u64);

impl Rng {
    fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % n as u64) as usize
    }
}

const PIECES: &[&str] = &[
    "<function name=\"get_weather\">",
    "<param name=\"city\">Lima</param>",
    "<param name=\"text\"><![CDATA[a</param>b</function>]]></param>",
    "<function name=\"x\"></function>",
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
    "get_weather",
    "echo_args",
    "city",
    "x",
    " ",
    "\n",
    "\u{a0}",
    "\u{e9}",
    "\u{1f600}",
    "[",
    "]",
    "{",
    "}",
    ",",
    "'",
    "\"",
    "1",
    "<",
    "&",
    "/",
    "=",
    "word",
];

#[test]
fn the_stream_leaves_the_content_the_parser_leaves_whatever_the_fragments() {
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    let mut disagreements = Vec::new();
    let mut with_calls = 0;
    for _ in 0..20_000 {
        let text: String = (0..rng.below(40))
            .map(|_| PIECES[rng.below(PIECES.len())])
            .collect();
        with_calls += usize::from(try_minicpm5(&text, None).is_some());
        let expected = without_whitespace(&parsed_content(&text));
        let chars: Vec<char> = text.chars().collect();
        // Whole, one character at a time, and cut at random places.
        let mut cuts: Vec<Vec<String>> = vec![vec![text.clone()], chunks(&text, 1)];
        let mut random = Vec::new();
        let mut i = 0;
        while i < chars.len() {
            let n = (1 + rng.below(9)).min(chars.len() - i);
            random.push(chars[i..i + n].iter().collect());
            i += n;
        }
        cuts.push(random);
        for cut in cuts {
            let fragments: Vec<&str> = cut.iter().map(String::as_str).collect();
            let streamed = without_whitespace(&stream(armed(), &fragments).content);
            if streamed != expected {
                disagreements.push(format!("{text:?}: stream {streamed:?} parser {expected:?}"));
            }
        }
    }
    assert!(
        disagreements.is_empty(),
        "{:#?}",
        &disagreements[..disagreements.len().min(5)]
    );
    // The corpus has to reach the call paths, or agreement proves nothing.
    assert!(with_calls > 2_000, "only {with_calls} inputs held a call");
}
