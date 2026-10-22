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

//! MiniCPM5 tool-call parser.
//!
//! The MiniCPM5 chat template (`openbmb/MiniCPM5-*`) teaches the model one
//! call grammar, with no outer wrapper:
//!
//! ```text
//! <function name="get_weather"><param name="city">Bogota</param></function>
//! <function name="get_weather"><param name="city"><![CDATA[a < b]]></param></function>
//! ```
//!
//! Parallel calls are consecutive `<function>` blocks. A value that contains
//! `<`, `&` or a newline is wrapped in a CDATA section; everything else is
//! written bare, and the template renders a non-string argument with Python's
//! `str()` (`True`, `['a', 'b']`, `{'k': 1}`), so list and object values arrive
//! as Python literals rather than JSON.
//!
//! The scan is tolerant on purpose and is not an XML parser. A model writes
//! `<`, `>` and `&` unescaped inside a bare value (a strict XML parse rejects
//! the call), so a value ends at the first `</param>` outside a CDATA section,
//! and a call is read parameter by parameter rather than by first locating its
//! `</function>`, so a literal `</function>` inside a value cannot truncate it.
//!
//! What is a call, and what is prose, is decided in one place, [`scan_call`],
//! which the non-streaming parser ([`try_minicpm5`]) and the streaming filter
//! (`stream_filter::StreamFilter`) both ask, so the two cannot disagree about
//! where a call ends:
//!
//! * the opener is `<function` followed by whitespace and a quoted or bare
//!   `name` attribute (any attribute order, any further attributes);
//! * the body starts, after optional whitespace, with `<param` or `</function>`
//!   (or the tag is self-closing). Anything else (`<function name="x"> opens a
//!   block`, an XML snippet, a JSON body) is prose and stays in the content;
//! * the call ends at its `</function>`, or where the next call opens, or at
//!   the end of the text, but only in the last two cases when nothing but
//!   whitespace follows the last complete `</param>`. A generation cut off
//!   inside a value or a tag is not a call: it is not salvaged, because a tool
//!   would run with whichever required argument was cut off.
//!
//! Values are typed by the request's tool schema, through the same coercion the
//! MiniMax M2 and Qwen3-Coder parsers use: a `string`-typed parameter keeps its
//! raw text, integer / number / boolean follow the declared type, and a
//! parameter the schema does not describe takes the loose literal rules. Array
//! and object values (and undeclared bracketed ones) are parsed as JSON first
//! and as a Python literal second.
//!
//! Nothing here slices at an arithmetic byte offset: every index comes from a
//! search for an ASCII marker or from `char::len_utf8`, so any input, however
//! it is cut, is valid UTF-8 at every boundary used.
//!
//! Used by: tool_calls::parser, tool_calls::stream_filter

use serde_json::{Map, Value};

use super::formats::{
    M3Type, coerce_xml_param, kv_param_schema, minimax_m3_function_schema, minimax_m3_schema_type,
    schema_type_admits_null,
};
use super::types::{ParsedToolCall, ToolCallFormat, ToolCallParseResult};
use crate::server::types::request::Tool;

/// The opener of a call, which is only an opener when whitespace follows, so
/// `<functional>` and the equals-sign `<function=NAME>` of Qwen3-Coder /
/// Functionary never match.
pub(super) const CALL_OPENER: &str = "<function";
const FUNCTION_CLOSE: &str = "</function>";
const PARAM_OPEN: &str = "<param";
const PARAM_CLOSE: &str = "</param>";
const CDATA_OPEN: &str = "<![CDATA[";
const CDATA_CLOSE: &str = "]]>";

/// Bounds on what a single model output can make the parser build, as in the
/// other XML grammars: real models emit a handful of parallel calls.
const MAX_PARAMS_PER_CALL: usize = 1024;
/// Openers considered in one output, calls and prose mentions together; the
/// text after the last one is content. Bounds the rescans a hostile run of
/// unclosed openers could cause.
pub(super) const MAX_OPENERS: usize = 1024;
/// An opening tag is a name and a few attributes; one that has not closed after
/// this many bytes is not one.
const MAX_OPEN_TAG_BYTES: usize = 4096;
/// Deepest list / dict nesting the Python-literal reader follows. The reader is
/// recursive and its input is untrusted model output, so the depth is capped
/// instead of left to the stack.
const MAX_LITERAL_DEPTH: usize = 64;

/// Try parsing the MiniCPM5 `<function name=..><param name=..>..</param></function>`
/// grammar. `None` when the text holds no call.
///
/// `tools` supplies the per-function schema each value is typed by; without it
/// the loose literal rules apply. Calls to functions the request did not
/// declare are not filtered here: [`super::parser::parse_tool_calls`] does that
/// for every format. Whatever is outside the calls, including every opener
/// [`scan_call`] declined, is returned as `content`, with the whitespace around
/// the calls trimmed.
pub fn try_minicpm5(text: &str, tools: Option<&[Tool]>) -> Option<ToolCallParseResult> {
    let mut calls: Vec<ParsedToolCall> = Vec::new();
    let mut content: Vec<&str> = Vec::new();
    let mut cursor = 0usize;
    let mut search_from = 0usize;

    for _ in 0..MAX_OPENERS {
        let Some(start) = find_call_opener(text, search_from) else {
            break;
        };
        match scan_call(&text[start..], true, tools) {
            CallScan::Call {
                name,
                arguments,
                end,
            } => {
                push_content(&mut content, &text[cursor..start]);
                calls.push(ParsedToolCall { name, arguments });
                cursor = start + end;
                search_from = cursor;
            }
            CallScan::Prose | CallScan::Pending { .. } => search_from = start + CALL_OPENER.len(),
        }
    }

    if calls.is_empty() {
        return None;
    }
    push_content(&mut content, &text[cursor..]);
    Some(ToolCallParseResult {
        format: Some(ToolCallFormat::Minicpm5),
        tool_calls: calls,
        content: content.join("\n"),
        reasoning_content: None,
    })
}

fn push_content<'a>(content: &mut Vec<&'a str>, segment: &'a str) {
    let segment = segment.trim();
    if !segment.is_empty() {
        content.push(segment);
    }
}

/// What [`scan_call`] made of the text that starts at a call opener.
pub(super) enum CallScan {
    /// A call that spans the first `end` bytes.
    Call {
        name: String,
        arguments: String,
        end: usize,
    },
    /// Not a call: the opener is ordinary text.
    Prose,
    /// Not decidable yet: more text could still make it a call or prose. Only
    /// when the text is not known to be complete. `body` says the opening tag
    /// and the first token of the body are settled and what is awaited is the
    /// end of the call, which only a `</function>` or `<function` can bring.
    Pending { body: bool },
}

/// The first call opener at or after `from`: `<function` and a whitespace
/// character.
pub(super) fn find_call_opener(text: &str, from: usize) -> Option<usize> {
    find_tag_open(text, CALL_OPENER, from)
}

/// Read the call that starts at the opener at the beginning of `text`.
///
/// `at_end` says that nothing follows `text`: a call that is cut off, or text
/// that cannot become one, is [`CallScan::Prose`], never [`CallScan::Pending`].
/// Otherwise an undecided prefix is `Pending`.
pub(super) fn scan_call(text: &str, at_end: bool, tools: Option<&[Tool]>) -> CallScan {
    let undecided = if at_end {
        CallScan::Prose
    } else {
        CallScan::Pending { body: false }
    };
    let tag = match read_open_tag(text, CALL_OPENER.len()) {
        Ok(tag) => tag,
        Err(TagEnd::TooLong) => return CallScan::Prose,
        Err(TagEnd::Unfinished) => return undecided,
    };
    let Some(name) = attribute(tag.attrs, "name").filter(|n| !n.is_empty()) else {
        return CallScan::Prose;
    };
    let call = |arguments: Map<String, Value>, end: usize| CallScan::Call {
        name: name.clone(),
        arguments: Value::Object(arguments).to_string(),
        end,
    };
    if tag.self_closing {
        return call(Map::new(), tag.end);
    }

    let body = &text[tag.end..];
    let lead = body.len() - body.trim_start().len();
    let rest = &body[lead..];
    if rest.starts_with(FUNCTION_CLOSE) {
        return call(Map::new(), tag.end + lead + FUNCTION_CLOSE.len());
    }
    let starts_param = rest.starts_with(PARAM_OPEN)
        && rest[PARAM_OPEN.len()..]
            .chars()
            .next()
            .is_some_and(char::is_whitespace);
    if !starts_param {
        // `<par` and `</funct` may still become the first token of a call.
        let may_grow = PARAM_OPEN.starts_with(rest) || FUNCTION_CLOSE.starts_with(rest);
        return if may_grow { undecided } else { CallScan::Prose };
    }

    let body = read_body(text, tag.end, minimax_m3_function_schema(tools, &name));
    match body.stop {
        Stop::Closed => call(body.arguments, body.end),
        Stop::NextCall if body.clean => call(body.arguments, body.end),
        Stop::EndOfText if body.clean && at_end => call(body.arguments, body.end),
        Stop::EndOfText if !at_end => CallScan::Pending { body: true },
        _ => CallScan::Prose,
    }
}

/// An opening tag: its attribute text, the index just past its `>`, and whether
/// it was written `<tag .../>`.
struct OpenTag<'a> {
    attrs: &'a str,
    end: usize,
    self_closing: bool,
}

/// The next `<name` at or after `from` that is followed by whitespace.
fn find_tag_open(text: &str, open: &str, mut from: usize) -> Option<usize> {
    while let Some(rel) = text[from..].find(open) {
        let at = from + rel;
        let after = at + open.len();
        if text[after..]
            .chars()
            .next()
            .is_some_and(char::is_whitespace)
        {
            return Some(at);
        }
        from = after;
    }
    None
}

/// Why an opening tag could not be read.
enum TagEnd {
    /// The text ended before the tag's `>`.
    Unfinished,
    /// The tag runs past [`MAX_OPEN_TAG_BYTES`]: it is not a tag, whatever
    /// follows.
    TooLong,
}

/// Read an opening tag whose attribute text starts at `attrs_start`, up to its
/// `>`. A `>` inside a quoted attribute value does not end the tag.
fn read_open_tag(text: &str, attrs_start: usize) -> Result<OpenTag<'_>, TagEnd> {
    let mut quote: Option<char> = None;
    for (i, c) in text[attrs_start..].char_indices() {
        if i > MAX_OPEN_TAG_BYTES {
            return Err(TagEnd::TooLong);
        }
        match quote {
            Some(q) if c == q => quote = None,
            Some(_) => {}
            None if c == '"' || c == '\'' => quote = Some(c),
            None if c == '>' => {
                let attrs = &text[attrs_start..attrs_start + i];
                let trimmed = attrs.trim_end();
                let self_closing = trimmed.ends_with('/');
                return Ok(OpenTag {
                    attrs: if self_closing {
                        &trimmed[..trimmed.len() - 1]
                    } else {
                        attrs
                    },
                    end: attrs_start + i + 1,
                    self_closing,
                });
            }
            None => {}
        }
    }
    Err(TagEnd::Unfinished)
}

/// The value of attribute `key` in an opening tag's attribute text: quoted with
/// `"` or `'`, or bare up to the next whitespace. Names match whole words, so
/// `data-name=` is not `name=`.
fn attribute(attrs: &str, key: &str) -> Option<String> {
    let bytes = attrs.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        let name_start = i;
        while i < bytes.len() && !bytes[i].is_ascii_whitespace() && !matches!(bytes[i], b'=' | b'/')
        {
            i += 1;
        }
        let name = &attrs[name_start..i];
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() || bytes[i] != b'=' {
            // A valueless attribute, or a stray `/`.
            i += usize::from(name.is_empty());
            continue;
        }
        i += 1;
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        let value = match bytes.get(i) {
            Some(&q) if q == b'"' || q == b'\'' => {
                let value_start = i + 1;
                let len = attrs[value_start..].find(q as char)?;
                i = value_start + len + 1;
                &attrs[value_start..value_start + len]
            }
            _ => {
                let value_start = i;
                while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
                    i += 1;
                }
                &attrs[value_start..i]
            }
        };
        if name == key {
            return Some(value.to_string());
        }
    }
    None
}

/// The position of the next occurrence of something, searched again only once
/// the cursor has moved past the last answer or the end of the span searched
/// has moved. A search that found nothing is not repeated while neither has.
#[derive(Default)]
struct Lookahead {
    at: Option<usize>,
    searched_to: Option<usize>,
}

impl Lookahead {
    fn find(
        &mut self,
        pos: usize,
        limit: usize,
        search: impl Fn(usize) -> Option<usize>,
    ) -> Option<usize> {
        if self.searched_to != Some(limit) || self.at.is_some_and(|at| at < pos) {
            self.at = search(pos);
            self.searched_to = Some(limit);
        }
        self.at
    }
}

/// How a call's body ended.
enum Stop {
    /// At its `</function>`.
    Closed,
    /// Where the next call opens: the model left out a `</function>`.
    NextCall,
    /// At the end of the text, inside a value or tag or after the last one.
    EndOfText,
}

/// What one call's body held.
struct Body {
    arguments: Map<String, Value>,
    /// Index just past `</function>`, at the next call's opener, or the end of
    /// the text.
    end: usize,
    stop: Stop,
    /// Nothing but whitespace lies between the last complete `</param>` (or
    /// the opening tag) and where the body stopped.
    clean: bool,
}

/// Read a call's parameters from `from` (just past the opening tag) up to its
/// `</function>`. A later duplicate of a parameter replaces the earlier one,
/// as in the other XML grammars.
fn read_body(text: &str, from: usize, fn_schema: Option<&Value>) -> Body {
    let mut arguments = Map::new();
    let mut pos = from;
    // Just past the last complete parameter, or the opening tag.
    let mut settled = from;

    // Where the next `<function` / `</function>` / `<param` lies. The last two
    // are searched only up to the next call, so a call that was never closed
    // cannot make every later one rescan the whole text; each answer is kept
    // until the cursor passes it, so a body is scanned once however many
    // parameters it has.
    let mut next_call = Lookahead::default();
    let mut next_close = Lookahead::default();
    let mut next_param = Lookahead::default();

    let (stop, end) = loop {
        if arguments.len() >= MAX_PARAMS_PER_CALL {
            // A call this long is not one a model wrote for a tool.
            break (Stop::EndOfText, text.len());
        }
        let call = next_call.find(pos, text.len(), |from| {
            find_tag_open(text, CALL_OPENER, from)
        });
        let limit = call.unwrap_or(text.len());
        let close = next_close.find(pos, limit, |from| {
            text[from..limit].find(FUNCTION_CLOSE).map(|r| from + r)
        });
        let param = next_param.find(pos, limit, |from| {
            find_tag_open(&text[..limit], PARAM_OPEN, from)
        });
        let Some(at) = [close, param, call].into_iter().flatten().min() else {
            // Neither a parameter nor an end: the generation stopped here.
            break (Stop::EndOfText, text.len());
        };
        if Some(at) == close {
            break (Stop::Closed, at + FUNCTION_CLOSE.len());
        }
        if Some(at) == call {
            // The next call starts before this one was closed: the model left
            // out a `</function>`. The caller reads the next one from here.
            break (Stop::NextCall, at);
        }
        let Ok(tag) = read_open_tag(text, at + PARAM_OPEN.len()) else {
            break (Stop::EndOfText, text.len());
        };
        let name = attribute(tag.attrs, "name").filter(|n| !n.is_empty());
        let (raw, after) = if tag.self_closing {
            ("", tag.end)
        } else if let Some(end) = find_param_close(text, tag.end) {
            (&text[tag.end..end], end + PARAM_CLOSE.len())
        } else {
            // A value cut off before its `</param>` is never salvaged.
            break (Stop::EndOfText, text.len());
        };
        if let Some(name) = name {
            let value = typed_value(raw, kv_param_schema(fn_schema, &name));
            arguments.insert(name, value);
        }
        pos = after;
        settled = after;
    };

    let clean = text[settled.min(end)..end].trim().is_empty();
    Body {
        arguments,
        end,
        stop,
        clean,
    }
}

/// The first `</param>` at or after `from` that is not inside a CDATA section,
/// whose payload may legitimately contain that text. `None` when there is none,
/// or when a CDATA section never closes (its end is unknowable).
///
/// The closer is searched once and again only when a CDATA section swallowed
/// it, so a value of many sections is read in one pass.
fn find_param_close(text: &str, from: usize) -> Option<usize> {
    let mut i = from;
    let mut close = i + text[i..].find(PARAM_CLOSE)?;
    loop {
        let Some(cdata) = text[i..close].find(CDATA_OPEN) else {
            return Some(close);
        };
        let payload = i + cdata + CDATA_OPEN.len();
        i = payload + text[payload..].find(CDATA_CLOSE)? + CDATA_CLOSE.len();
        if i > close {
            close = i + text[i..].find(PARAM_CLOSE)?;
        }
    }
}

/// The text a `<param>` carries. A value that is one CDATA section (whitespace
/// around it is ignored) yields its payload verbatim, up to the last `]]>`: the
/// template wraps without escaping, so a payload may itself contain `]]>`. Any
/// other value is the text as written, minus the one newline a model sometimes
/// frames it with.
fn param_text(raw: &str) -> (&str, bool) {
    let probe = raw.trim();
    if let Some(payload) = probe
        .strip_prefix(CDATA_OPEN)
        .and_then(|p| p.strip_suffix(CDATA_CLOSE))
    {
        return (payload, true);
    }
    let raw = raw
        .strip_prefix("\r\n")
        .or_else(|| raw.strip_prefix('\n'))
        .unwrap_or(raw);
    let raw = raw
        .strip_suffix("\r\n")
        .or_else(|| raw.strip_suffix('\n'))
        .unwrap_or(raw);
    (raw, false)
}

/// A parameter's JSON value, typed by its declared schema.
fn typed_value(raw: &str, schema: Option<&Value>) -> Value {
    let (text, cdata) = param_text(raw);
    let declared = schema.and_then(minimax_m3_schema_type);
    let keeps_raw_text =
        declared == Some(M3Type::Str) && !schema.is_some_and(schema_type_admits_null);
    if keeps_raw_text {
        return Value::String(text.to_string());
    }
    // A CDATA payload is exactly what was written; a bare value may carry
    // padding that is not part of it.
    let text = if cdata { text } else { text.trim() };

    // `None` is how Python's `str()` writes a null argument.
    if text == "None" {
        return Value::Null;
    }
    // The opening bracket already fixes the shape a literal can have, so a
    // successful read is the declared array or object.
    let bracketed = match declared {
        Some(M3Type::Array) => text.starts_with('['),
        Some(M3Type::Object) => text.starts_with('{'),
        None => text.starts_with(['[', '{']),
        Some(_) => false,
    };
    if bracketed && let Some(value) = parse_literal(text) {
        return value;
    }
    coerce_xml_param(text, schema)
}

/// A JSON document, else a Python literal (`ast.literal_eval`'s data subset).
fn parse_literal(text: &str) -> Option<Value> {
    if let Ok(value) = serde_json::from_str::<Value>(text) {
        return Some(value);
    }
    let mut reader = Literal { text, pos: 0 };
    let value = reader.value(0)?;
    reader.skip_ws();
    (reader.pos == text.len()).then_some(value)
}

/// Recursive-descent reader for Python's list / dict / str / number / `True` /
/// `False` / `None` literals.
struct Literal<'a> {
    text: &'a str,
    pos: usize,
}

impl Literal<'_> {
    fn peek(&self) -> Option<char> {
        self.text.get(self.pos..)?.chars().next()
    }

    fn skip_ws(&mut self) {
        while let Some(c) = self.peek().filter(|c| c.is_whitespace()) {
            self.pos += c.len_utf8();
        }
    }

    fn eat(&mut self, c: char) -> bool {
        self.skip_ws();
        if self.peek() == Some(c) {
            self.pos += c.len_utf8();
            true
        } else {
            false
        }
    }

    fn value(&mut self, depth: usize) -> Option<Value> {
        if depth > MAX_LITERAL_DEPTH {
            return None;
        }
        self.skip_ws();
        match self.peek()? {
            '[' => self.list(depth),
            '{' => self.dict(depth),
            '\'' | '"' => self.string().map(Value::String),
            _ => self.scalar(),
        }
    }

    /// Consume `,` after an element and report whether another may follow; a
    /// missing comma is only valid when `close` ends the container.
    fn more(&mut self, close: char) -> Option<bool> {
        if self.eat(',') {
            return Some(true);
        }
        self.skip_ws();
        (self.peek() == Some(close)).then_some(false)
    }

    fn list(&mut self, depth: usize) -> Option<Value> {
        self.pos += 1;
        let mut items = Vec::new();
        while !self.eat(']') {
            items.push(self.value(depth + 1)?);
            self.more(']')?;
        }
        Some(Value::Array(items))
    }

    fn dict(&mut self, depth: usize) -> Option<Value> {
        self.pos += 1;
        let mut map = Map::new();
        while !self.eat('}') {
            let key = match self.value(depth + 1)? {
                Value::String(s) => s,
                // JSON object keys are strings; Python's other hashable
                // scalars keep their written form.
                key @ (Value::Number(_) | Value::Bool(_)) => key.to_string(),
                _ => return None,
            };
            if !self.eat(':') {
                return None;
            }
            map.insert(key, self.value(depth + 1)?);
            self.more('}')?;
        }
        Some(Value::Object(map))
    }

    /// A quoted string with Python's escapes (a single- or double-quoted
    /// literal; triple quotes are not data a tool call carries).
    fn string(&mut self) -> Option<String> {
        let quote = self.peek()?;
        self.pos += 1;
        let mut out = String::new();
        loop {
            let c = self.peek()?;
            self.pos += c.len_utf8();
            if c == quote {
                return Some(out);
            }
            if c != '\\' {
                out.push(c);
                continue;
            }
            let e = self.peek()?;
            self.pos += e.len_utf8();
            match e {
                'n' => out.push('\n'),
                't' => out.push('\t'),
                'r' => out.push('\r'),
                'b' => out.push('\u{8}'),
                'f' => out.push('\u{c}'),
                '0' => out.push('\0'),
                'x' => out.push(self.hex_char(2)?),
                'u' => out.push(self.hex_char(4)?),
                '\n' => {}
                // `\\`, `\'`, `\"` and an escape Python leaves as written.
                '\\' | '\'' | '"' => out.push(e),
                other => {
                    out.push('\\');
                    out.push(other);
                }
            }
        }
    }

    fn hex_char(&mut self, digits: usize) -> Option<char> {
        let hex = self.text.get(self.pos..self.pos + digits)?;
        if !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
            return None;
        }
        let code = u32::from_str_radix(hex, 16).ok()?;
        self.pos += digits;
        char::from_u32(code)
    }

    fn scalar(&mut self) -> Option<Value> {
        let start = self.pos;
        while self
            .peek()
            .is_some_and(|c| c.is_alphanumeric() || matches!(c, '_' | '.' | '+' | '-'))
        {
            self.pos += self.peek()?.len_utf8();
        }
        match &self.text[start..self.pos] {
            "True" | "true" => Some(Value::Bool(true)),
            "False" | "false" => Some(Value::Bool(false)),
            "None" | "null" => Some(Value::Null),
            word => word
                .parse::<i64>()
                .map(Value::from)
                .or_else(|_| word.parse::<u64>().map(Value::from))
                .ok()
                .or_else(|| {
                    word.parse::<f64>()
                        .ok()
                        .and_then(serde_json::Number::from_f64)
                        .map(Value::Number)
                }),
        }
    }
}

#[cfg(test)]
#[path = "minicpm5_tests.rs"]
mod tests;
