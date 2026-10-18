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

//! Whole-prompt prompt-cache hits on the KV branch (lablup/mlxcel#1760,
//! eejd/rs-mlxcel#4).
//!
//! A client that replays an identical prompt gets a cache hit covering every
//! prompt token. Prefill still has to forward the last one to produce a
//! sampling logit, so the adopt must install `len - 1` tokens and leave the
//! last for prefill. Before the fix the adopt installed all `len` and admission
//! only backed the prefill cursor off, so the last token landed in the cache
//! twice (and, under the NAX tile-padded prefill, the mask no longer matched
//! the cache: `[broadcast_shapes] Shapes (32,303) and (1,32,32,304)`).
//!
//! These tests use a real (tiny) Llama, the family whose natural backend is the
//! dense KV cache, and read the restored per-layer cache offset itself.

use super::*;
use mlxcel_core::generate::SamplingConfig;
use mlxcel_core::weights::WeightMap;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, mpsc};
use std::time::Duration;

use super::prompt_cache::whole_prompt_reuse_cap;
use crate::LoadedModel;
use crate::models::Llama3Model;
use crate::models::llama3::ModelArgs;
use crate::server::config::{
    DecodeStorageBackend, PreemptionPolicy, PromptCacheRequestContext, ReasoningBudgetOverride,
};
use crate::server::model_provider::GenerateEvent;
use crate::server::prompt_cache::{PromptCacheConfig, PromptCacheStore, key::MultimodalDigest};
use crate::server::state::BatchMetrics;
use crate::tokenizer::MlxcelTokenizer;

const VOCAB: usize = 8;
const EOS_ID: i32 = 7;

fn tiny_args() -> ModelArgs {
    serde_json::from_str(
        r#"{
            "model_type": "llama",
            "hidden_size": 64,
            "num_hidden_layers": 1,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "head_dim": 16,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "tie_word_embeddings": false,
            "max_position_embeddings": 256,
            "vocab_size": 8
        }"#,
    )
    .expect("tiny llama args")
}

/// Deterministic pseudo-random values in [-0.5, 0.5).
fn filled(shape: &[i32]) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let n: i32 = shape.iter().product();
    let mut state: u32 = 0x9E37_79B9;
    let data: Vec<f32> = (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        })
        .collect();
    mlxcel_core::from_slice_f32(&data, shape)
}

fn ones(shape: &[i32]) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let n: i32 = shape.iter().product();
    mlxcel_core::from_slice_f32(&vec![1.0; n as usize], shape)
}

fn tiny_llama() -> Llama3Model {
    let args = tiny_args();
    let (hidden, ff, vocab) = (64, 128, VOCAB as i32);
    let mut w = WeightMap::new();
    w.insert("model.embed_tokens.weight".into(), filled(&[vocab, hidden]));
    w.insert("model.norm.weight".into(), ones(&[hidden]));
    w.insert("lm_head.weight".into(), filled(&[vocab, hidden]));
    let p = "model.layers.0";
    w.insert(format!("{p}.input_layernorm.weight"), ones(&[hidden]));
    w.insert(
        format!("{p}.post_attention_layernorm.weight"),
        ones(&[hidden]),
    );
    for name in ["q_proj", "k_proj", "v_proj", "o_proj"] {
        w.insert(
            format!("{p}.self_attn.{name}.weight"),
            filled(&[hidden, hidden]),
        );
    }
    w.insert(format!("{p}.mlp.gate_proj.weight"), filled(&[ff, hidden]));
    w.insert(format!("{p}.mlp.up_proj.weight"), filled(&[ff, hidden]));
    w.insert(format!("{p}.mlp.down_proj.weight"), filled(&[hidden, ff]));
    Llama3Model::from_weights(&w, &args).expect("tiny llama loads")
}

fn test_store() -> Arc<PromptCacheStore> {
    Arc::new(PromptCacheStore::with_config(PromptCacheConfig::new(
        true,
        1 << 24,
        32,
        Duration::from_secs(600),
        4,
    )))
}

/// A scheduler shaped like `--n-parallel 1` serving: dense KV storage, prompt
/// cache installed.
fn scheduler(store: Arc<PromptCacheStore>) -> BatchScheduler {
    let (_tx, rx) = mpsc::channel();
    let sched = BatchScheduler::with_config(
        LoadedModel::Llama(tiny_llama()),
        MlxcelTokenizer::stub(),
        vec![EOS_ID],
        rx,
        4,
        8,
        Arc::new(BatchMetrics::new()),
        Arc::new(BatchObservability::new()),
        0,
        false,
        PreemptionPolicy::default(),
        1,
        DecodeStorageBackend::Dense,
    )
    .with_prompt_cache(Some(store));
    install_thread_local_default_stream(sched.generation_stream.as_ref());
    sched
}

fn cache_ctx() -> PromptCacheRequestContext {
    PromptCacheRequestContext {
        model_id: "tiny-llama".to_string(),
        lora_id: None,
        template_sig: "tpl-sig-v1".to_string(),
        session_key: "session-1".to_string(),
        mm_digest: MultimodalDigest::empty(),
        history_prompt: None,
        history_prefix_tokens: None,
    }
}

fn options() -> ServerGenerateOptions {
    ServerGenerateOptions {
        n_indent: 0,
        t_max_predict_ms: None,
        reasoning_budget_message: None,
        retention: Default::default(),
        dry_breaker_strings: None,
        logit_bias: Vec::new(),
        logit_bias_texts: Vec::new(),
        post_sampling_probs: false,
        max_tokens: 1,
        sampling: SamplingConfig::greedy(),
        stop_sequences: None,
        // The tiny model may sample EOS at once; `max_tokens == 1` with
        // `ignore_eos` is what makes the stored entry one token longer than the
        // prompt.
        ignore_eos: true,
        priority: RequestPriority::Normal,
        lora_scales: None,
        logprobs: Default::default(),
        reasoning_budget: ReasoningBudgetOverride::InheritServerDefault,
        thinking_enter_block_on_start: false,
        reasoning_control: None,
        prompt_cache_ctx: Some(cache_ctx()),
        structured: None,
        grammar: None,
        image_soft_tokens: None,
    }
}

fn enqueue(sched: &mut BatchScheduler, prompt_tokens: Vec<i32>) -> mpsc::Receiver<GenerateEvent> {
    let (tx, rx) = mpsc::channel();
    sched.enqueue_request(
        "prompt".to_string(),
        Some(prompt_tokens),
        options(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        tx,
        Arc::new(AtomicBool::new(false)),
        true,
    );
    rx
}

/// Tokens the tiny model can actually embed (`vocab_size == 8`).
fn prompt(n: usize) -> Vec<i32> {
    (0..n).map(|i| (i % 6) as i32).collect()
}

/// Run one request to completion so the scheduler donates its KV cache
/// (`prompt ++ generated`) to the prompt cache, and return what was stored.
fn donate_turn(sched: &mut BatchScheduler, first: &[i32]) -> Vec<i32> {
    let rx = enqueue(sched, first.to_vec());
    for _ in 0..8 {
        match sched.decide_action() {
            BatchSchedulerAction::Prefill(id) => sched.execute_prefill(id),
            BatchSchedulerAction::Decode(ids) => sched.execute_decode_step(&ids),
            BatchSchedulerAction::Idle => break,
            other => panic!("unexpected scheduler action {other:?}"),
        }
        if sched.active_batch.is_empty() && sched.prefill_queue.is_empty() {
            break;
        }
    }
    let generated = loop {
        match rx.recv_timeout(Duration::from_secs(5)) {
            Ok(GenerateEvent::Token(_, _))
            | Ok(GenerateEvent::TokenWithLogprobs(_, _, _))
            | Ok(GenerateEvent::Prefill(_)) => {}
            Ok(GenerateEvent::Done(result)) => break result.generated_token_ids,
            Ok(GenerateEvent::Error(err)) => panic!("unexpected generation error: {err}"),
            Err(err) => panic!("generation did not finish: {err}"),
        }
    };
    let mut stored = first.to_vec();
    stored.extend(generated);
    stored
}

/// Offset (tokens held) of the first layer of an adopted sequence.
fn adopted_offset(sched: &BatchScheduler, seq_id: SequenceId) -> i32 {
    sched
        .cache_pool
        .get(seq_id)
        .expect("adopted sequence is active")
        .caches[0]
        .offset
}

#[test]
fn a_whole_prompt_hit_leaves_the_last_prompt_token_to_prefill() {
    assert_eq!(whole_prompt_reuse_cap(40, 40), 39, "whole prompt: stop short");
    assert_eq!(whole_prompt_reuse_cap(41, 40), 39, "entry longer than prompt");
    assert_eq!(whole_prompt_reuse_cap(32, 40), 32, "a prefix hit is untouched");
    assert_eq!(whole_prompt_reuse_cap(0, 0), 0, "empty prompt does not underflow");
}

/// The exact-entry shape: the request is the stored conversation itself, so the
/// store matches the whole entry without truncating. The cap must turn that
/// into a truncating adopt of `len - 1`, and admission must then prefill exactly
/// the one remaining token and report the same count as cached.
#[test]
fn whole_entry_hit_adopts_len_minus_one_and_prefills_one_token() {
    let mut sched = scheduler(test_store());
    let replay = donate_turn(&mut sched, &prompt(40));

    let _rx = enqueue(&mut sched, replay.clone());
    let queued = sched
        .prefill_queue
        .dequeue()
        .expect("the replay is queued for prefill");
    assert_eq!(queued.prefill_start_offset, replay.len() - 1);
    assert_eq!(
        queued.already_cached_tokens,
        replay.len() - 1,
        "`cached_tokens` never claims the token prefill re-runs"
    );
    assert_eq!(
        adopted_offset(&sched, queued.seq_id),
        (replay.len() - 1) as i32
    );
}
