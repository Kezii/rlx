// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pipeline-parallel forward orchestration.
//!
//! Each rank owns a contiguous block of layers (see [`crate::partition`])
//! and implements [`BlockRunner`] for it. [`PipelineCoordinator`] drives
//! one forward pass as a relay over a [`ProcessGroup`]:
//!
//! ```text
//!   rank world-1 (First):  tokens ─embed→ layers ─hidden→  send → world-2
//!   rank r       (Middle):           recv ← layers ─hidden→ send → r-1
//!   rank 0       (Last):             recv ← layers → norm+head → logits → sample
//!                                                    └─ broadcast token → all ranks
//! ```
//!
//! The sampled token is broadcast from rank 0 so every rank can append it
//! and stay in lockstep for the next step. The coordinator is model-
//! agnostic — only [`BlockRunner`] knows how to run a block.
//!
//! Teardown contract: after the last step, call [`PipelineCoordinator::barrier`]
//! before dropping the group, so no rank tears down its transport while a
//! peer is still mid-transfer.

use crate::partition::{BlockRole, LayerSplit, block_role};
use anyhow::{Result, bail};
use rlx_driver::ProcessGroup;

/// Tag for pipeline hidden-state messages (below `TAG_RESERVED_BASE`, so
/// it never collides with collective/barrier traffic).
const TAG_PIPE_HIDDEN: u32 = 100;

/// Input handed to a [`BlockRunner`].
pub enum BlockInput<'a> {
    /// Token ids — only the First/Single block receives these (it embeds).
    Tokens(&'a [u32]),
    /// Hidden states `[batch * seq * hidden]` from the previous block.
    Hidden(&'a [f32]),
}

/// Output produced by a [`BlockRunner`].
pub enum BlockOutput {
    /// Hidden states to forward to the next block.
    Hidden(Vec<f32>),
    /// Final logits (last token), produced only by the Last/Single block.
    Logits(Vec<f32>),
}

impl BlockOutput {
    fn into_hidden(self) -> Result<Vec<f32>> {
        match self {
            BlockOutput::Hidden(v) => Ok(v),
            BlockOutput::Logits(_) => bail!("block produced Logits where Hidden was expected"),
        }
    }
    fn into_logits(self) -> Result<Vec<f32>> {
        match self {
            BlockOutput::Logits(v) => Ok(v),
            BlockOutput::Hidden(_) => bail!("block produced Hidden where Logits was expected"),
        }
    }
}

/// A model's per-rank layer block. One implementation per model family
/// (e.g. `Qwen3PipelineStage`); the coordinator drives it.
pub trait BlockRunner {
    /// This rank's role, which dictates the input/output kinds it handles.
    fn role(&self) -> BlockRole;

    /// Run this block. The First/Single block receives [`BlockInput::Tokens`]
    /// and embeds; others receive [`BlockInput::Hidden`]. The Last/Single
    /// block returns [`BlockOutput::Logits`]; others return
    /// [`BlockOutput::Hidden`].
    fn run(&mut self, input: BlockInput<'_>) -> Result<BlockOutput>;
}

/// Per-step choices for [`PipelineCoordinator::forward_step_with`].
#[derive(Debug, Clone, Copy)]
pub struct StepOptions {
    /// Give up on the next block's hidden state after this long. `None`
    /// waits forever, which is only right for a rank waiting on its leader.
    pub timeout: Option<std::time::Duration>,
    /// Broadcast the sampled token to every rank at the end of the step.
    ///
    /// Right for [`PipelineCoordinator::generate`], where the broadcast *is* how the other
    /// ranks learn the token. Wrong for a caller that samples above the
    /// pipeline and tells the other ranks itself — there it is a second
    /// message saying what the first already said.
    pub broadcast_token: bool,
}

impl Default for StepOptions {
    fn default() -> Self {
        Self {
            timeout: None,
            broadcast_token: true,
        }
    }
}

/// Drives pipeline-parallel forward passes over a process group.
pub struct PipelineCoordinator {
    group: ProcessGroup,
}

impl PipelineCoordinator {
    pub fn new(group: ProcessGroup) -> Self {
        Self { group }
    }

    pub fn rank(&self) -> u32 {
        self.group.rank()
    }
    pub fn world_size(&self) -> u32 {
        self.group.world_size()
    }
    pub fn is_leader(&self) -> bool {
        self.group.is_leader()
    }
    pub fn group(&self) -> &ProcessGroup {
        &self.group
    }

    /// Synchronize all ranks. Call once after the final step before
    /// dropping the group (teardown contract above).
    pub fn barrier(&self) -> Result<()> {
        Ok(self.group.barrier()?)
    }

    /// Run one pipeline forward pass over `token_ids` (the full sequence so
    /// far, prefill-style) and return the next token.
    ///
    /// `sample` is invoked only on the Last/Single rank, with the final
    /// logits; its result is broadcast so every rank returns the same
    /// token. `token_ids` is only read by the First/Single rank, but all
    /// ranks must call this in lockstep.
    pub fn forward_step(
        &self,
        runner: &mut dyn BlockRunner,
        token_ids: &[u32],
        sample: impl FnOnce(&[f32]) -> u32,
    ) -> Result<u32> {
        self.forward_step_within(runner, token_ids, sample, None)
    }

    /// As [`Self::forward_step`], but giving up if the next block's hidden
    /// state does not arrive within `timeout`.
    ///
    /// Worth using on the rank that is serving requests. A machine that was
    /// unplugged, slept, or lost its network does not close its sockets, so
    /// a plain receive on it waits forever — and a server blocked forever on
    /// a peer looks to its client exactly like a model that is thinking.
    /// With a bound it becomes an error that names the rank that went quiet.
    ///
    /// Only the hidden-state receive is bounded. A rank waiting on the one
    /// *above* it in the chain is waiting on the leader, which is idle
    /// between requests by design — timing that out would end the job every
    /// time nobody asked anything for a while.
    pub fn forward_step_within(
        &self,
        runner: &mut dyn BlockRunner,
        token_ids: &[u32],
        sample: impl FnOnce(&[f32]) -> u32,
        timeout: Option<std::time::Duration>,
    ) -> Result<u32> {
        self.forward_step_with(
            runner,
            token_ids,
            sample,
            StepOptions {
                timeout,
                broadcast_token: true,
            },
        )
    }

    /// As [`Self::forward_step`], with the per-step choices spelled out.
    pub fn forward_step_with(
        &self,
        runner: &mut dyn BlockRunner,
        token_ids: &[u32],
        sample: impl FnOnce(&[f32]) -> u32,
        opts: StepOptions,
    ) -> Result<u32> {
        let timeout = opts.timeout;
        let world = self.group.world_size();
        let rank = self.group.rank();

        // Defensive: the runner's role must agree with its rank.
        let role = block_role(rank, world);
        if runner.role() != role {
            bail!(
                "BlockRunner role {:?} disagrees with rank {rank}/{world} role {role:?}",
                runner.role()
            );
        }

        let mut sampled: Option<u32> = None;
        match role {
            BlockRole::Single => {
                let logits = runner.run(BlockInput::Tokens(token_ids))?.into_logits()?;
                sampled = Some(sample(&logits));
            }
            BlockRole::First => {
                let hidden = runner.run(BlockInput::Tokens(token_ids))?.into_hidden()?;
                self.group.send_f32(rank - 1, TAG_PIPE_HIDDEN, &hidden)?;
            }
            BlockRole::Middle => {
                let hidden_in = self.recv_hidden(rank + 1, timeout)?;
                let hidden = runner.run(BlockInput::Hidden(&hidden_in))?.into_hidden()?;
                self.group.send_f32(rank - 1, TAG_PIPE_HIDDEN, &hidden)?;
            }
            BlockRole::Last => {
                let hidden_in = self.recv_hidden(rank + 1, timeout)?;
                let logits = runner.run(BlockInput::Hidden(&hidden_in))?.into_logits()?;
                sampled = Some(sample(&logits));
            }
        }

        // Broadcast the token from rank 0 (the Last/Single block) to all.
        if !opts.broadcast_token {
            // The caller moves the token itself, so this would be a second
            // message carrying what the first already said. One extra
            // round trip per token is nothing on a cable and the dominant
            // cost over a relayed WAN link, where it doubles the per-token
            // latency.
            return Ok(sampled.unwrap_or(0));
        }
        let mut tok = [sampled.unwrap_or(0) as f32];
        self.group.broadcast(0, &mut tok)?;
        Ok(tok[0] as u32)
    }

    /// Take the hidden state from `from`, optionally bounded.
    fn recv_hidden(&self, from: u32, timeout: Option<std::time::Duration>) -> Result<Vec<f32>> {
        match timeout {
            None => Ok(self.group.recv_f32(from, TAG_PIPE_HIDDEN)?),
            Some(t) => match self.group.recv_f32_timeout(from, TAG_PIPE_HIDDEN, t)? {
                Some(h) => Ok(h),
                None => bail!(
                    "rank {from} did not send its hidden state within {t:?}: \
                     that machine has stopped responding"
                ),
            },
        }
    }

    /// Generate up to `max_tokens`, appending each sampled token to
    /// `tokens` (seeded with the prompt). `sample` runs on the leader;
    /// `should_stop` ends early (e.g. on EOS) and — because the token is
    /// broadcast — fires identically on every rank, so all stay in
    /// lockstep. Returns the newly generated ids and barriers at the end.
    ///
    /// This is the prefill-recompute loop (each step re-runs the whole
    /// sequence); a KV-cached decode loop is the follow-up.
    pub fn generate(
        &self,
        runner: &mut dyn BlockRunner,
        tokens: &mut Vec<u32>,
        max_tokens: usize,
        mut sample: impl FnMut(&[f32]) -> u32,
        mut should_stop: impl FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        let start = tokens.len();
        for _ in 0..max_tokens {
            let tok = self.forward_step(runner, tokens, |l| sample(l))?;
            tokens.push(tok);
            if should_stop(tok) {
                break;
            }
        }
        self.barrier()?;
        Ok(tokens[start..].to_vec())
    }
}

impl LayerSplit {
    /// Check that every rank in `group` was given this same division.
    ///
    /// Worth a round trip before loading any weights, because this is the
    /// one disagreement the rest of the design cannot catch. A rank's share
    /// of the layers decides which weights it reads and how many layers its
    /// graph has, and any share is internally consistent: two ranks given
    /// different divisions both load, both compile, and both run. The
    /// pipeline then puts some layer through twice and skips another, every
    /// shape agreeing, and answers plausible nonsense with nothing
    /// anywhere to say so.
    ///
    /// Compares [`LayerSplit::fingerprint`] rather than the counts, so one
    /// all-gather of a single number settles it however many ranks there
    /// are. Every rank learns of a disagreement, not just rank 0 — a rank
    /// that stayed silent would sit waiting for peers that have already
    /// given up.
    pub fn agree(&self, group: &ProcessGroup) -> Result<()> {
        let mine = self.fingerprint();
        let all = group.all_gather(&[mine as f32])?;
        let disagree: Vec<String> = all
            .iter()
            .enumerate()
            .filter(|&(_, f)| *f as u32 != mine)
            .map(|(rank, f)| format!("rank {rank} has {:#08x}", *f as u32))
            .collect();
        if !disagree.is_empty() {
            bail!(
                "the ranks were given different layer splits — this rank has \
                 {mine:#08x}, {}. Every rank needs the same division; without \
                 it the model would run some layers twice and others never, \
                 and answer nonsense with nothing to show it.",
                disagree.join(", ")
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::partition::pipeline_layer_range;
    use rlx_driver::NetTransport;
    use std::net::{Ipv4Addr, SocketAddr, TcpListener};
    use std::ops::Range;
    use std::sync::Arc;
    use std::thread;

    /// Toy model used to verify the relay end-to-end without real weights.
    ///
    ///   embed(t):  h[i] = t + i
    ///   layer g:   h[i] += g + 1
    ///   last:      logits = h ; token = round(logits[0])
    ///
    /// After all `L` layers, `h[0] = t + 0 + sum_{g=0..L}(g+1) = t + L(L+1)/2`.
    /// So the token a correct pipeline yields is `t + L(L+1)/2` — and it only
    /// comes out right if every layer is applied exactly once, in order.
    struct MockStage {
        role: BlockRole,
        layers: Range<usize>,
        hidden: usize,
    }

    impl BlockRunner for MockStage {
        fn role(&self) -> BlockRole {
            self.role
        }
        fn run(&mut self, input: BlockInput<'_>) -> Result<BlockOutput> {
            let mut h: Vec<f32> = match input {
                BlockInput::Tokens(ids) => {
                    let t = *ids.last().unwrap() as f32;
                    (0..self.hidden).map(|i| t + i as f32).collect()
                }
                BlockInput::Hidden(hv) => hv.to_vec(),
            };
            for g in self.layers.clone() {
                for x in h.iter_mut() {
                    *x += (g + 1) as f32;
                }
            }
            Ok(match self.role {
                BlockRole::Last | BlockRole::Single => BlockOutput::Logits(h),
                _ => BlockOutput::Hidden(h),
            })
        }
    }

    /// Spin up `world` loopback ranks (ephemeral ports, no collisions) and
    /// run `body(ProcessGroup)` on a thread each.
    fn run_group<F>(world: u32, body: F)
    where
        F: Fn(ProcessGroup) + Send + Sync + 'static,
    {
        let listeners: Vec<TcpListener> = (0..world)
            .map(|_| TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap())
            .collect();
        let addrs: Vec<SocketAddr> = listeners.iter().map(|l| l.local_addr().unwrap()).collect();
        let body = Arc::new(body);
        let handles: Vec<_> = listeners
            .into_iter()
            .enumerate()
            .map(|(rank, listener)| {
                let addrs = addrs.clone();
                let body = body.clone();
                thread::spawn(move || {
                    let t =
                        NetTransport::from_listener(rank as u32, world, listener, addrs, 1 << 20)
                            .expect("build transport");
                    body(ProcessGroup::new(Arc::new(t)));
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn ranks_given_the_same_split_agree() {
        run_group(3, |group| {
            let split = LayerSplit::even(30, 3);
            split.agree(&group).expect("a split every rank shares");
        });
    }

    #[test]
    fn a_rank_given_a_different_split_is_caught_on_every_rank() {
        // The failure this exists for, and the reason every rank has to
        // hear about it: a rank that noticed nothing would wait for peers
        // that have already failed and gone.
        run_group(3, |group| {
            let split = if group.rank() == 1 {
                LayerSplit::parse("5,20,5").unwrap()
            } else {
                LayerSplit::even(30, 3)
            };
            let err = split
                .agree(&group)
                .expect_err("rank 1 holds a different division")
                .to_string();
            assert!(
                err.contains("different layer splits"),
                "rank {}: {err}",
                group.rank()
            );
        });
    }

    #[test]
    fn an_uneven_split_still_agrees_with_itself() {
        // 28 across 3 does not divide, and the remainder convention is the
        // thing most likely to differ between two ways of computing the
        // same split.
        run_group(3, |group| {
            let split = LayerSplit::even(28, 3);
            assert_eq!(split.counts, vec![9, 9, 10]);
            split.agree(&group).expect("one function, one answer");
        });
    }

    fn run_pipeline_case(world: u32, num_layers: usize, hidden: usize, token: u32) {
        let expected = token as usize + num_layers * (num_layers + 1) / 2;
        run_group(world, move |group| {
            let rank = group.rank();
            let mut stage = MockStage {
                role: block_role(rank, world),
                layers: pipeline_layer_range(num_layers, rank, world),
                hidden,
            };
            let coord = PipelineCoordinator::new(group);
            let tok = coord
                .forward_step(&mut stage, &[token], |logits| logits[0].round() as u32)
                .unwrap();
            assert_eq!(tok as usize, expected, "rank {rank}");
            // Two steps to exercise repeated relays staying in lockstep.
            let tok2 = coord
                .forward_step(&mut stage, &[token], |logits| logits[0].round() as u32)
                .unwrap();
            assert_eq!(tok2 as usize, expected, "rank {rank} step 2");
            coord.barrier().unwrap();
        });
    }

    #[test]
    fn pipeline_even_split_matches_serial() {
        run_pipeline_case(4, 16, 8, 5);
    }

    #[test]
    fn pipeline_uneven_split_matches_serial() {
        run_pipeline_case(3, 10, 6, 7);
    }

    #[test]
    fn pipeline_two_ranks() {
        run_pipeline_case(2, 5, 4, 11);
    }

    #[test]
    fn single_rank_runs_whole_model() {
        run_group(1, |group| {
            let mut stage = MockStage {
                role: BlockRole::Single,
                layers: 0..6,
                hidden: 4,
            };
            let coord = PipelineCoordinator::new(group);
            let tok = coord
                .forward_step(&mut stage, &[9], |logits| logits[0].round() as u32)
                .unwrap();
            assert_eq!(tok as usize, 9 + 6 * 7 / 2);
        });
    }

    #[test]
    fn generate_loops_and_appends() {
        // Mock token = last_token + S where S = L(L+1)/2. Generating from
        // [T] gives [T+S, T+2S, T+3S] (each step feeds the previous token).
        let world = 2u32;
        let num_layers = 5usize;
        let hidden = 4usize;
        let prompt0 = 11u32;
        let s = (num_layers * (num_layers + 1) / 2) as u32;
        run_group(world, move |group| {
            let rank = group.rank();
            let mut stage = MockStage {
                role: block_role(rank, world),
                layers: pipeline_layer_range(num_layers, rank, world),
                hidden,
            };
            let coord = PipelineCoordinator::new(group);
            let mut tokens = vec![prompt0];
            let produced = coord
                .generate(
                    &mut stage,
                    &mut tokens,
                    3,
                    |l| l[0].round() as u32,
                    |_| false,
                )
                .unwrap();
            assert_eq!(
                produced,
                vec![prompt0 + s, prompt0 + 2 * s, prompt0 + 3 * s],
                "rank {rank}"
            );
            assert_eq!(tokens.len(), 4, "prompt + 3 generated");
        });
    }

    #[test]
    fn generate_stops_early_on_should_stop() {
        run_group(2, |group| {
            let rank = group.rank();
            let mut stage = MockStage {
                role: block_role(rank, 2),
                layers: pipeline_layer_range(4, rank, 2),
                hidden: 4,
            };
            let coord = PipelineCoordinator::new(group);
            let mut tokens = vec![1u32];
            // Stop after the first generated token regardless of value.
            let produced = coord
                .generate(
                    &mut stage,
                    &mut tokens,
                    10,
                    |l| l[0].round() as u32,
                    |_| true,
                )
                .unwrap();
            assert_eq!(produced.len(), 1, "rank {rank}: stopped after one token");
        });
    }
}
