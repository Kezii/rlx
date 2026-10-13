// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! What a compiled schedule asks the **host** to do, and where those asks can
//! be batched instead of serialising the graph.
//!
//! A host step is a hole in the GPU schedule: the executor ends its compute
//! pass, submits, reads the step's inputs back, computes on the CPU, and writes
//! the result into the arena before the next pass. Each one is a sync point, and
//! consecutive host steps pay for one each even when they have nothing to do
//! with each other.
//!
//! This module answers two questions the executor previously could not:
//!
//! 1. **What** is on the host — which step kinds, how many, and how many sync
//!    points they cost. Callers had one bit (`requires_host_execution`), which
//!    is why a conv graph on browser WebGPU could only report *"contains
//!    host-executed ops (GGUF dequant, LSTM, collectives, FFT-host, …)"* —
//!    a list of what it might have been rather than what it was.
//! 2. **Which of them are independent**, and could therefore run together in
//!    one sync point rather than one each. Nothing may be batched on a hunch:
//!    two host steps can only share a sync point if neither reads a byte range
//!    the other writes, so a step whose regions cannot be named is a barrier.
//!
//! Region knowledge is deliberately partial. [`Conv2dHost`](Step::Conv2dHost)
//! and [`BufferCopy`](Step::BufferCopy) carry explicit arena offsets and full
//! shapes, so their reads and writes are exact; every other host step reports
//! [`HostIo::Unknown`] and plans as a barrier. Adding a kind is one match arm,
//! and until then the conservative answer is the correct one — a wrong
//! independence claim would reorder reads and writes into the same arena bytes
//! and produce plausible, wrong numbers.

use super::step::{Step, step_is_tail_host, step_name, step_runs_on_host};
use std::collections::BTreeMap;

/// A byte range in the arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Region {
    pub off: u64,
    pub bytes: u64,
}

impl Region {
    fn overlaps(self, other: Self) -> bool {
        self.off < other.off.saturating_add(other.bytes)
            && other.off < self.off.saturating_add(self.bytes)
    }
}

/// The arena bytes a host step touches, when they can be named.
#[derive(Clone, Debug)]
pub enum HostIo {
    Known {
        reads: Vec<Region>,
        writes: Vec<Region>,
    },
    /// Regions unknown — must be treated as aliasing everything.
    Unknown,
}

const F32: u64 = 4;

/// Arena bytes `step` reads and writes, or [`HostIo::Unknown`].
pub(crate) fn host_io(step: &Step) -> HostIo {
    match step {
        Step::Conv2dHost {
            n,
            c_in,
            c_out,
            h,
            w,
            h_out,
            w_out,
            kh,
            kw,
            groups,
            in_byte_off,
            w_byte_off,
            out_byte_off,
            ..
        } => {
            let g = (*groups).max(1) as u64;
            let input = u64::from(*n) * u64::from(*c_in) * u64::from(*h) * u64::from(*w) * F32;
            let weights =
                u64::from(*c_out) * (u64::from(*c_in) / g) * u64::from(*kh) * u64::from(*kw) * F32;
            let output =
                u64::from(*n) * u64::from(*c_out) * u64::from(*h_out) * u64::from(*w_out) * F32;
            HostIo::Known {
                reads: vec![
                    Region {
                        off: *in_byte_off,
                        bytes: input,
                    },
                    Region {
                        off: *w_byte_off,
                        bytes: weights,
                    },
                ],
                writes: vec![Region {
                    off: *out_byte_off,
                    bytes: output,
                }],
            }
        }
        Step::BufferCopy {
            src_byte_off,
            dst_byte_off,
            bytes,
        } => HostIo::Known {
            reads: vec![Region {
                off: *src_byte_off,
                bytes: u64::from(*bytes),
            }],
            writes: vec![Region {
                off: *dst_byte_off,
                bytes: u64::from(*bytes),
            }],
        },
        _ => HostIo::Unknown,
    }
}

/// Host steps that may run **together** in one sync point.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostBatch {
    /// Schedule indices, in order.
    pub steps: Vec<usize>,
    /// Proven free of read/write conflicts, so these may run concurrently.
    pub parallel: bool,
}

/// A run of consecutive host steps with no GPU work between them — one sync
/// point, however many batches it needs internally.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostGroup {
    pub batches: Vec<HostBatch>,
}

impl HostGroup {
    fn steps(&self) -> usize {
        self.batches.iter().map(|b| b.steps.len()).sum()
    }
}

/// The host side of a compiled schedule.
#[derive(Clone, Debug, Default)]
pub struct HostPlan {
    pub total_steps: usize,
    pub host_steps: usize,
    pub tail_host_steps: usize,
    pub groups: Vec<HostGroup>,
    pub kinds: BTreeMap<&'static str, usize>,
}

impl HostPlan {
    /// GPU→host→GPU transitions the schedule pays for.
    pub fn sync_points(&self) -> usize {
        self.groups.len()
    }

    /// Host steps that share a sync point with at least one other.
    pub fn parallel_steps(&self) -> usize {
        self.groups
            .iter()
            .flat_map(|g| &g.batches)
            .filter(|b| b.parallel)
            .map(|b| b.steps.len())
            .sum()
    }

    /// Sync points this schedule would need if every host step ran on its own.
    pub fn serial_sync_points(&self) -> usize {
        self.groups.iter().map(HostGroup::steps).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.host_steps == 0 && self.tail_host_steps == 0
    }

    /// One line naming what is on the host — for an error a user has to act on.
    pub fn summary(&self) -> String {
        if self.is_empty() {
            return "no host-executed steps".into();
        }
        let kinds = self
            .kinds
            .iter()
            .map(|(k, n)| {
                if *n > 1 {
                    format!("{k}×{n}")
                } else {
                    (*k).to_string()
                }
            })
            .collect::<Vec<_>>()
            .join(", ");
        let mut s = format!(
            "{} of {} steps run on the host in {} sync point{} ({kinds})",
            self.host_steps + self.tail_host_steps,
            self.total_steps,
            self.sync_points(),
            if self.sync_points() == 1 { "" } else { "s" },
        );
        if self.parallel_steps() > 1 {
            s.push_str(&format!(
                "; {} are independent ({} sync points if run one at a time)",
                self.parallel_steps(),
                self.serial_sync_points()
            ));
        }
        if self.tail_host_steps > 0 {
            s.push_str(&format!("; {} run after the graph", self.tail_host_steps));
        }
        s
    }
}

/// Name of a host step kind, for diagnostics.
///
/// Delegates to [`step_name`], the per-variant label the trace layer already
/// maintains. A hand-written list here looked tidier and was worse: it fell
/// back to a generic `"host-step"` for every variant it had not been taught,
/// so a conv model's three host steps reported as `host-step×3` — hiding
/// exactly the names the plan exists to surface.
pub(crate) fn host_step_kind(step: &Step) -> &'static str {
    step_name(step)
}

/// Plan the host side of `schedule`.
pub fn plan(schedule: &[Step]) -> HostPlan {
    let mut out = HostPlan {
        total_steps: schedule.len(),
        ..Default::default()
    };

    let mut i = 0;
    while i < schedule.len() {
        let step = &schedule[i];
        if step_is_tail_host(step) {
            out.tail_host_steps += 1;
            *out.kinds.entry(host_step_kind(step)).or_default() += 1;
            i += 1;
            continue;
        }
        if !step_runs_on_host(step) {
            i += 1;
            continue;
        }

        // A maximal run of consecutive host steps: one GPU sync point, however
        // it has to be split internally.
        let start = i;
        while i < schedule.len()
            && step_runs_on_host(&schedule[i])
            && !step_is_tail_host(&schedule[i])
        {
            *out.kinds.entry(host_step_kind(&schedule[i])).or_default() += 1;
            out.host_steps += 1;
            i += 1;
        }

        let mut batches: Vec<HostBatch> = Vec::new();
        let mut cur: Vec<usize> = Vec::new();
        let mut cur_reads: Vec<Region> = Vec::new();
        let mut cur_writes: Vec<Region> = Vec::new();
        let flush = |batches: &mut Vec<HostBatch>, cur: &mut Vec<usize>| {
            if !cur.is_empty() {
                batches.push(HostBatch {
                    parallel: cur.len() > 1,
                    steps: std::mem::take(cur),
                });
            }
        };

        for idx in start..i {
            match host_io(&schedule[idx]) {
                HostIo::Unknown => {
                    // Cannot be proven independent of anything, so it is its
                    // own sync point and ends the batch being accumulated.
                    flush(&mut batches, &mut cur);
                    cur_reads.clear();
                    cur_writes.clear();
                    batches.push(HostBatch {
                        steps: vec![idx],
                        parallel: false,
                    });
                }
                HostIo::Known { reads, writes } => {
                    // Conflict = this step writes what the batch touches, or
                    // reads what the batch writes. Write-write counts too: two
                    // steps writing the same bytes have an order.
                    let conflicts = writes.iter().any(|w| {
                        cur_reads
                            .iter()
                            .chain(cur_writes.iter())
                            .any(|r| r.overlaps(*w))
                    }) || reads
                        .iter()
                        .any(|r| cur_writes.iter().any(|w| w.overlaps(*r)));
                    if conflicts {
                        flush(&mut batches, &mut cur);
                        cur_reads.clear();
                        cur_writes.clear();
                    }
                    cur.push(idx);
                    cur_reads.extend(reads);
                    cur_writes.extend(writes);
                }
            }
        }
        flush(&mut batches, &mut cur);
        if !batches.is_empty() {
            out.groups.push(HostGroup { batches });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::MatmulCompute;

    fn conv(in_off: u64, w_off: u64, out_off: u64) -> Step {
        // 1×1×4×4 input, 1 output channel, 3×3 kernel, stride 1, no pad.
        Step::Conv2dHost {
            n: 1,
            c_in: 1,
            c_out: 1,
            h: 4,
            w: 4,
            h_out: 2,
            w_out: 2,
            kh: 3,
            kw: 3,
            sh: 1,
            sw: 1,
            ph: 0,
            pw: 0,
            dh: 1,
            dw: 1,
            groups: 1,
            in_byte_off: in_off,
            w_byte_off: w_off,
            out_byte_off: out_off,
        }
    }

    fn gpu_step() -> Step {
        Step::Matmul {
            m: 1,
            k: 1,
            n: 1,
            a_off_f32: 0,
            b_off_f32: 0,
            c_off_f32: 0,
            batch: 1,
            a_batch_stride: 0,
            b_batch_stride: 0,
            c_batch_stride: 0,
            has_bias: 0,
            bias_off_f32: 0,
            act_id: 0xFFFF,
            b_is_param: false,
            compute_precision: MatmulCompute::F32,
        }
    }

    #[test]
    fn a_pure_gpu_schedule_has_no_host_plan() {
        let p = plan(&[gpu_step(), gpu_step()]);
        assert!(p.is_empty());
        assert_eq!(p.sync_points(), 0);
        assert_eq!(p.summary(), "no host-executed steps");
    }

    /// Two convs over disjoint arena ranges have no ordering between them, so
    /// they belong in one sync point instead of paying for one each.
    #[test]
    fn disjoint_convs_share_one_batch() {
        // input 64B at 0, weights 36B at 1024, output 16B at 2048.
        let a = conv(0, 1024, 2048);
        let b = conv(4096, 5120, 6144);
        let p = plan(&[a, b]);

        assert_eq!(p.host_steps, 2);
        assert_eq!(p.sync_points(), 1, "one GPU interlude, not two");
        assert_eq!(p.groups[0].batches.len(), 1);
        assert!(p.groups[0].batches[0].parallel);
        assert_eq!(p.parallel_steps(), 2);
        assert_eq!(p.serial_sync_points(), 2, "what it would cost unbatched");
        assert!(
            p.summary().contains("conv2d_host×2"),
            "summary should name the step kind: {}",
            p.summary()
        );
    }

    /// The dependent case must NOT be batched: the second conv reads what the
    /// first writes, and running them together would read the stale bytes.
    #[test]
    fn a_conv_consuming_the_previous_output_stays_ordered() {
        let first = conv(0, 1024, 2048);
        // Reads at 2048 — exactly the first conv's output.
        let second = conv(2048, 3072, 4096);
        let p = plan(&[first, second]);

        assert_eq!(p.sync_points(), 1);
        assert_eq!(
            p.groups[0].batches.len(),
            2,
            "write→read must not be batched"
        );
        assert!(p.groups[0].batches.iter().all(|b| !b.parallel));
        assert_eq!(p.parallel_steps(), 0);
    }

    /// Partial overlap counts: the regions are byte ranges, not slot ids.
    #[test]
    fn partially_overlapping_regions_conflict() {
        let first = conv(0, 1024, 2048);
        // Output is 16 bytes at 2048; read 8 bytes into it.
        let second = conv(2056, 3072, 4096);
        let p = plan(&[first, second]);
        assert_eq!(p.groups[0].batches.len(), 2, "overlap at 2056..2064 missed");
    }

    /// GPU work between host steps is a second sync point by definition.
    #[test]
    fn gpu_work_between_host_steps_splits_the_groups() {
        let p = plan(&[conv(0, 1024, 2048), gpu_step(), conv(4096, 5120, 6144)]);
        assert_eq!(p.host_steps, 2);
        assert_eq!(p.sync_points(), 2, "a GPU step between them is a boundary");
        assert!(p.groups.iter().all(|g| g.batches.len() == 1));
        assert_eq!(
            p.parallel_steps(),
            0,
            "different sync points cannot be parallel"
        );
    }
}
