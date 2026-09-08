# Local CombLlama reproduction status

This workspace is reproducing the official `shijuzhao/Comb` CombLlama path.
The authoritative upstream revision is:

```text
25bb50823ab5998d0caa55014ec07771ca9fba9a
```

## Active run

- Training root: `/data3/junhaohu/checkpoints/Comb_official_reproduction_true_tp2_stage0`
- Base checkpoint: Llama-3.1-8B-Instruct revision `0e9e39f249a16976918f6564b8830bc894c89659`
- Parallelism: true tensor parallelism, TP=4; read the live physical GPU IDs
  from `watchdog_status.json`
- Optimizer state: DeepSpeed ZeRO stage 0
- Logical global batch size: 32
- Save interval: 1,000 optimizer steps
- Retention: latest two full checkpoints
- Final optimizer step: 265,287
- Disk safety floor: 300 GiB free, including headroom for the next checkpoint
- Current committed checkpoint: use the atomic `latest_repro.json` pointer in
  the training root
- Current state: actively training beyond the latest committed checkpoint
  under the supervisor, watchdog, strict checkpoint auditor, and milestone controller

The run migrated from the audited TP=1 step-20,000 checkpoint into a universal
checkpoint and resumed with true TP=2. At step 89,000 it passed the
restart-noise-calibrated transition gate and continued with the paper's TP=4
hardware configuration. Paired transition tests, complete checkpoint-state
audits, the recovery audit, and the continuous official-schedule audit bind the
training semantics across both migrations.

The live source of truth is `watchdog_status.json` in the training root. A quick
read-only status check is:

```bash
sed -n '1,220p' \
  /data3/junhaohu/checkpoints/Comb_official_reproduction_true_tp2_stage0/watchdog_status.json
```

Every committed TP=4 checkpoint is scanned by the strict auditor. Each audit
checks the structure/cursor and performs a complete finite scan of all four TP
ranks' model, FP32 master, `exp_avg`, and `exp_avg_sq` tensors. It also requires
all 336 expected model shards and all 112 optimizer shards per state to be
distinct, with only the three explicitly replicated embedding and LM-head
parameters bit-identical. Read the latest accepted step and report paths from
`strict_checkpoint_auditor_status.json` in the TP4 revalidation evidence root.

## Rejected TP4 transition at step 88,000

A bounded TP2-to-TP4 transition was pre-registered before step 88,000 and was
allowed to alter the main run only if the candidate passed every numerical,
state, data-coordinate, and throughput gate.  The TP4 candidate used the same
step-88,000 model and AdamW state, the same 100 batches, and the same learning
rates.  It produced a 1.3423x median step speedup and passed its full four-rank
checkpoint/state audit.  It was nevertheless rejected because its mean and
maximum absolute loss deltas were 0.001727 and 0.010135, above the pre-registered
limits of 0.000731 and 0.003025.

A post-decision calibration check then compared the independently resumed TP2
run against the same TP2 reference.  That same-TP replay also exceeded both
limits (mean 0.001847, maximum 0.007628).  Thus the pre-registered decision is
preserved, but the failed loss gate by itself is not evidence that TP4 was
implemented incorrectly: the gate was tighter than observed TP2 replay
variability.  The calibration report SHA-256 is
`c676c48f538b8ef94d8943864998be1eaa6e3f6f93da552a9d5630f12dfc712f`.

The main loss log was rolled back to step 88,000, the original TP2 checkpoint
was restored, and official TP2 training resumed from the next exact batch.  The
rejected candidate checkpoint and its temporary Universal conversion were
deleted after preserving the protocol, paired trajectories, audit, and rollback
report.  Their durable root is:

```text
/data3/junhaohu/checkpoints/Comb_official_tp4_transition_step_88000
```

The protocol, acceptance audit, and rollback-report SHA-256 values are
`b6884c5a75163a6170088e5628b1944d4a2803cc125b2cd35226e575cb16c331`,
`59940a45f5ccf1d7ddf95dc3342bbe2ffcf44ca5a70d29706c7f2bea53915d57`,
and `d229d8a2ab2b81c58f7d132fcf5089efd1a6c714b9b58a85bce35b6a329c456e`.

## Passed TP4 revalidation at step 89,000

A second transition was pre-registered before step 89,000 with three independent
100-update arms over the exact same Natural-Instructions batches and learning
rates: a TP=2 reference, a second TP=2 restart-noise control, and a TP=4
candidate.  This prevents ordinary restart nondeterminism from being mistaken
for a TP-layout error.  The TP=2 repeat mean/maximum absolute loss deltas were
`0.0019918` and `0.0102144`.  The TP=4 deltas were `0.0023435` and `0.0104743`,
inside the pre-registered noise-envelope limits of `0.0030877` and `0.0158216`.
The first-step delta was `0.0007338` (limit `0.001`), and TP=4 achieved a
`1.33723x` median-step speedup (required minimum `1.05x`).

The candidate step-89,100 checkpoint passed its four-rank structure, finite
model/master-weight/Adam-state, ZeRO-0, replica, cursor, and recovery checks.
The frozen controller then failed only because the final audit script's file
mode had unintentionally become `000`; restoring mode `0644` left the
pre-registered content hash unchanged, and all six related unit tests passed.
The audit was rerun successfully without rerunning any training arm.

The accepted candidate was committed to the formal training root and training
continued with TP=4. The 100k cross-TP milestone, recovery-chain audit, and
subsequent strict four-rank checkpoint audits passed. Live progress is reported
only by `watchdog_status.json` and `latest_repro.json`.

- Revalidation root:
  `/data3/junhaohu/checkpoints/Comb_official_tp4_revalidation_step_89000`
- Protocol SHA-256:
  `67bd5b3bc4dd3be137f24f75ae0530ab0be9f1d5db67f8d2dcb282cfe847d59b`
- Acceptance report SHA-256:
  `bfc92cdee4683100ddc14836182a458ed82d3e85db3fa16674f7e57026c90903`
- Checkpoint/replica audit SHA-256:
  `0d4b8433eae89020feef74868117447d33310c6d583eb189bee41ba5475533db`,
  `d5033a4aaec525cc41fdd8158874fdae60ca52bf05d8c014e4442f8997955162`

## Source fidelity

The current Git HEAD is the official revision above. Of the 88 official files
under `comb/`, `data/`, and `training/`, 81 are byte-identical and the other
seven exactly match the reviewed compatibility-patch hashes. There are no
unregistered changes to tracked official source. In addition, all nine files
in the restart supervisor's source lock match, and the three training sources
recorded by the active rank-0 process still match their launch-time hashes.

The durable source-fidelity report is:

```text
/data3/junhaohu/checkpoints/Comb_official_reproduction_true_tp2_stage0/source_fidelity_live_current.json
```

Its SHA-256 is
`c021b68cd763d4c0b0338e1900d745481aba435ffd5d8356f0a5017e5bc749ed`.

## Paper/code training-recipe discrepancy

The paper says that AdamW uses linear warmup followed by cosine decay. The
released `training/ds_llama_config.json`, however, specifies DeepSpeed
`WarmupDecayLR` with `total_num_steps=8,000,000`; the installed implementation
performs warmup followed by linear decay. The released config also omits
`gradient_clipping`, which means DeepSpeed's default value is 0.0, while the
paper gives no numeric clipping value. The active run follows the released
code exactly for optimizer and scheduler semantics. It does not silently
replace the released schedule based on the paper prose.

The durable comparison report is:

```text
/data3/junhaohu/checkpoints/Comb_official_reproduction_true_tp2_stage0/paper_code_training_recipe_audit.json
```

Its SHA-256 is
`e0ebd769e89bf1ea89e4acb199c616a591e59a2d6efe59f4117e67d3e7d1a0fc`.

## Training-loss logging semantics

The two live loss records intentionally have different per-step semantics.
`training_loss.csv` records the final microbatch loss in each optimizer update,
after `engine.step()` returns. DeepSpeed's
`Train/Samples/train_loss` TensorBoard scalar sums the four losses after each
has been divided by `gradient_accumulation_steps=4`, so it is the four-
microbatch mean. DeepSpeed emits that scalar immediately before incrementing
the optimizer-step/sample counters; consequently its derived optimizer-step
label is one lower than the CSV label for the same update. Long-window means
agree, but individual points are not expected to be equal. TensorBoard is the
less noisy source for convergence and plateau decisions; the CSV remains the
authoritative source for schedule/cursor continuity.

## Observed validation and PIC trend

The fixed held-out validation NLL decreased monotonically across the completed
true-TP milestones: 0.535114 at step 21,000, 0.466968 at step 30,000,
0.442678 at step 50,000, and 0.434392 at step 60,000. The step-60,000 value is
within 0.002142 NLL of the official reference on the same fixed selection.

The distinct-wrong-context NLL gap (larger means the predicted target depends
more strongly on the supplied context) changed as follows:

| Selection | step 21,000 | step 30,000 | step 50,000 | step 60,000 | step 100,000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| generic | 0.059163 | 0.126171 | 0.184198 | 0.181534 | 0.245386 |
| Natural-Instructions | 0.362608 | 0.768959 | 0.893339 | 0.905761 | 1.094168 |
| Natural-Instructions short-context | 0.659576 | 0.945290 | 1.193368 | 1.272808 | 1.399963 |

At step 100,000, held-out validation NLL reached `0.417909`, and all four
predeclared directional signals improved relative to step 50,000. The completed
150k and 200k fixed-panel results are summarized in
`pic_compact_trajectory_100000_150000_200000.json` in the training root. These
intermediate results support continued progress but are not treated as proof of
final paper reproduction; the remaining full NLL/PIC milestone is step 265,287.

## Evaluation-set integrity

The fixed 256-example SQuAD-v2 validation selection is disjoint from all
130,319 examples in the official SQuAD training cache by both source ID and a
SHA-256 signature of `(context, question)`. The dataset fingerprints and the
teacher manifest also match their frozen values. The durable report is:

```text
/data3/junhaohu/checkpoints/Comb_validation/squad_v2_validation_llama_teacher_vllm0101_tp2_seed42_256/training_validation_disjoint_audit.json
```

Its SHA-256 is
`7729edbe8353de950dbfa97d033f5abfcd7217e4c985b038e21c90c26d5b1c18`.

The 64-example generic, Natural-Instructions, and short-context
Natural-Instructions PIC selections were also re-audited. Every paired wrong
context is tokenwise distinct, the pair indices form a permutation, and the
correct/wrong context-length multisets are identical with zero mean signed
length difference. This prevents a systematic context-length change from
masquerading as context dependence. The report is:

```text
/data3/junhaohu/comb/benchmarks/results/context_dependency_pairing_audit.json
```

Its SHA-256 is
`3823416de1662ee59c58a392985e8af2505d4525ae3cf25bae98da8ad2b6380f`.

## Remaining training-data caches

The official order is SQuAD, Natural-Instructions, XSum, then
Super-Natural-Instructions. XSum and Super-NI bucket caches were precomputed at
low CPU/IO priority while the run was still in Natural-Instructions, preventing
a long preprocessing stall at the later dataset transition.

- XSum: 204,041 cached rows, 4 rows above the 16,384-token bucket ceiling,
  273,739,176 bytes;
- Super-Natural-Instructions: 1,990,915 cached rows, no excluded rows,
  1,288,626,009 bytes.

Both manifests and every parquet file were structurally checked and SHA-256
hashed. The durable report is:

```text
/data3/junhaohu/checkpoints/Comb_official_reproduction_true_tp2_stage0/remaining_bucket_cache_precompute.json
```

The official bucket manifests plus DeepSpeed's cross-bucket gradient-
accumulation phase derive the training target exactly:

- SQuAD ends at optimizer step 4,073;
- Natural-Instructions ends at 196,691;
- XSum ends at 203,069;
- Super-Natural-Instructions ends at 265,287 with accumulation phase zero.

Within Natural-Instructions, the training-log `bucket` field is the parquet
file index, not the conceptual length-bucket index. Conceptual bucket 0
(1--256 context tokens) is split across parquet files 0 through 5:
`bucket_2.parquet` ends at optimizer step 102,378, `bucket_3.parquet` ends at
135,146, `bucket_4.parquet` ends at 167,914, and `bucket_5.parquet` ends at
172,139. Thus both the 100k (`file_index=2`) and 150k (`file_index=4`)
evaluations were reached while training in the same 1--256-token conceptual
bucket. The first transition to conceptual bucket 1 (257--512 tokens) occurs at
step 172,140. The original 100k-to-150k paired report mislabeled the two file
indices as length buckets; its numerical results remain valid, and the durable
metadata/interpretation erratum is:

```text
/data3/junhaohu/checkpoints/Comb_official_reproduction_true_tp2_stage0/paired_context_significance_100000_vs_150000_erratum_1.json
```

This is one ordered pass through the four official datasets, not repeated
epochs. The reproducible derivation is recorded in:

```text
/data3/junhaohu/checkpoints/Comb_official_reproduction_true_tp2_stage0/official_data_schedule_audit.json
```

## Supervisors

The following tmux sessions are expected to remain active:

- `comb_official_true_tp4_stage0_supervisor`: training and crash recovery
- `comb_official_true_tp4_stage0_watchdog`: loss/GPU/disk/checkpoint monitoring
- `comb_official_true_tp4_stage0_milestones`: 150k/200k/final evaluation
- `comb_official_tp4_strict_checkpoint_auditor_supervisor`: per-1k TP checkpoint audits
- `comb_official_remaining_superni_transition_sequence`: remaining data-boundary audits
- `comb_official_final_paired_fixed_panel_analysis`: final fixed-panel NLL/PIC comparison
- `comb_official_true_tp4_final_reproduction`: final paper evaluation

The 100k pre-registered audit has two idempotent paths: the dedicated watcher
above and the milestone runner loaded fresh at step 100,000.  The latter also
publishes the bounded 100k progress summary before invoking the hash-locked
audit, so losing the standalone watcher cannot leave the final evaluator
waiting on a missing audit artifact.  Both paths use atomic JSON publication
and validate the unchanged protocol SHA-256.

Each TP rank uses four DataLoader workers. Their process command lines resemble
the trainer command, but they are children of the four active rank processes and
do not represent duplicate training launches.

## Immutable protocols

- 100k NLL/PIC protocol:
  `/data3/junhaohu/checkpoints/Comb_official_milestone_100000_true_tp_eval/evaluation_protocol_pre_100000.json`
- 100k protocol SHA-256:
  `b0b97ba71881bec9d098f330e8ca659eb9d3dc7735de0d4b7b6bc0d3e217d6b2`
- 200k interpretation-protocol metadata erratum:
  `/data3/junhaohu/checkpoints/Comb_official_reproduction_true_tp2_stage0/milestone_200000_interpretation_protocol_erratum_1_pre_200000.json`
- Final performance protocol:
  `/data3/junhaohu/checkpoints/Comb_official_reproduction_true_tp2_stage0/final_performance_protocol_pre_265287.json`
- Final performance protocol SHA-256:
  `656e2178497d94a44e2583e47d6fe696bbac0b1d8d45117051d936f15f8d3e21`
- Final serving-compatibility amendment:
  `/data3/junhaohu/checkpoints/Comb_official_reproduction_true_tp2_stage0/final_performance_protocol_amendment_1_pre_265287.json`
- Amendment SHA-256:
  `4f409f8410cfecb05312ddb24f1b9862db8cd078a783204c5f1f1540f133b1b9`

Do not modify files hashed by either protocol or the amendment. The amendment
records a serving-only concurrency compatibility patch; it changes neither the
model architecture, training trajectory, PIC tensors, benchmark workload, nor
numerical thresholds. The 100k audit watcher validates its protocol at that
milestone; the final evaluator validates the final protocol and amendment
before waiting and again before consuming the checkpoint.

The 100k audit now directly binds every primary result to the frozen protocol:
256 validation examples and 24,182 target tokens, the complete frozen teacher
manifest, all three 64-example PIC selections (including exact selected indices,
filters, and target-token totals inherited from the hash-locked 50k comparisons),
the four finite primary values, and both the 50k and official-reference values.
It also validates comparison steps/shapes and the summary's directional rule.
The frozen protocol itself was not changed. Four focused regression tests cover
the strengthened audit, including token-count, selection, and baseline drift.
The real step-100,000 pre-registered audit passed all 69 checks with no failed
fields and bound every result to the unchanged protocol and frozen inputs.

## Long-context capacity preflight

Before the pre-registered 100k evaluation, the real 81k checkpoint was loaded
with true TP=2 on two A100 80GB GPUs and exercised with a 16,255-token context
(9,095 non-padding context tokens), a 512-token query, forward, backward, and
one AdamW update. Both ranks produced the same finite loss, all tensor shapes
matched, and the minimum peak-reserved headroom was 17,678,729,216 bytes. The
enhanced report additionally proves that each rank's restored Adam counter
advanced from 81,000 to 81,001 and that trainable BF16 parameters changed,
while the DeepSpeed engine/global checkpoint cursor remained at 81,000. This
was the early acceptance capacity check; the milestone controller subsequently
repeated the hardened check at the committed 100k checkpoint. The final completion
audit treats missing, invalid, or malformed 100k/150k/final capacity and milestone
evidence as explicit failed gates instead of terminating with an exception.
It also rehashes the final model files, the five LongBench inputs, and every
recorded evaluator source; a stale input manifest cannot satisfy completion.
The final result artifacts are rehashed against the pre-registered audit, and
offline/online/PIC outputs must bind to the current model and benchmark input
manifest, so stale or post-hoc replaced outputs cannot satisfy completion.

- Report:
  `/data3/junhaohu/checkpoints/Comb_official_milestone_81000_true_tp_capacity_preflight_v2/long_context_capacity_preflight_step_00081000.json`
- Report SHA-256:
  `77dba5635e1282819d115201d78d574d54f0cab856c21da1fe36b9d52e6252a2`

## Native TP to HF export equivalence

The milestone exporter previously defaulted to the released ZeRO-2 config even
though the active real-TP checkpoints use ZeRO-0. The exporter now defaults to
`ds_llama_true_tp_stage0_config.json`, and the milestone controller passes that
config explicitly. A real 81k checkpoint was consolidated with TP=2 and loaded
strictly as a 12,045,391,888-parameter, five-shard HF model. On the same
16,255-token row, native TP loss was `1.1981171369552612` and HF loss was
`1.1986156702041626`, an absolute delta of `0.0004985332489013672` in BF16.
The final milestone must repeat both the native long-context check and this
HF numerical-equivalence check with a maximum allowed loss delta of `0.001`.
For the current TP=4 run, the controller pauses training and requires four idle
devices from the configured milestone candidates. Temporary GPU scarcity is
retryable: the checkpoint remains pinned while the controller waits, so it
cannot be lost to the two-checkpoint retention policy. The
controller now creates a verified, same-filesystem hard-linked snapshot
immediately after each non-final capacity milestone commits, before waiting for
host memory or idle GPUs. Capacity testing and export share that snapshot, and
controller restarts recognize it even after the trainer's two-checkpoint
retention has removed the original directory. This closes the remaining case
where prolonged GPU scarcity could otherwise make a 100k/150k checkpoint
impossible to evaluate. Snapshot release is idempotent and occurs only after
all milestone post-processing gates pass; a controller restart also releases
an already-completed snapshot. Cleanup requires the exact expected path,
matching optimizer-step metadata, and a tree containing no symbolic links.
Snapshot construction itself is implemented by
`training/pin_checkpoint_snapshot.py`: it validates the exact tag and filesystem,
rejects links/non-regular files and stale staging directories, creates an atomic
hard-linked snapshot, fsyncs its parent, and is idempotent after restart. Tests
verify that every pinned file shares its inode with the committed checkpoint.
Malformed capacity-report field types are handled as explicit invalid evidence
rather than crashing the milestone controller.

- Equivalence report:
  `/data3/junhaohu/checkpoints/Comb_official_milestone_81000_true_tp_export_equivalence/hf_export_equivalence_step_00081000.json`

  The corresponding 81k consolidated HF weights were an intermediate
  acceptance artifact and were removed on 2026-08-21 after every recorded
  shard identity was reverified. The equivalence and strict-loading reports
  remain available, together with `hf_step_00081000.REMOVED.json`; current
  training, later milestones, and the final reproduction do not consume the
  removed checkpoint.

## Online serving calibration

The official async API originally submitted concurrent requests to the same
synchronous vLLM `LLM` through the unrestricted default executor. Under real
load, vLLM batched multiple `pic_request_id` values and the single-PIC forward
path failed with an ambiguous multi-element Tensor boolean, killing EngineCore.
The API now owns one dedicated single-worker executor per synchronous COMB
engine. HTTP requests remain asynchronous and queued, while model invocations
respect the one-engine/one-PIC-at-a-time contract.

The repaired official released checkpoint completed the frozen Figure-10-style
grid at 1, 2, 3, 4, and 5 requests/second. Every prefix and COMB request
succeeded, COMB TTFT was lower at every rate, and the peak COMB/prefix prompt-
token-throughput ratio was 2.682922, above the pre-registered 2.5 threshold.
The immutable report is:

```text
/data3/junhaohu/comb/benchmarks/results/official_checkpoint_online_serialized_full/online_full_audit.json
```

Its SHA-256 is
`d0dd35ecb2bce522d82c0f4b1c6b5fc46ac57102aa37e24395ef78cd17c9329d`.
The current-evaluator gate now checks this online evidence in addition to the
official-checkpoint LongBench score calibration.

## Automatic milestones and final evidence

At 100k and 150k the controller first pins and validates the committed checkpoint,
pauses training, runs the native long-context capacity test with TP=4 on four
idle GPUs, and then exports and evaluates it. Durable hard-linked snapshots
prevent checkpoint retention from invalidating an in-flight evaluation. If
four GPUs are temporarily unavailable, the checkpoint remains pinned and the
controller retries without advancing or corrupting training. At 200k it runs
the standard milestone evaluation. At step 265,287 it additionally runs:

- strict DeepSpeed and HF checkpoint validation;
- TP-rank replica validation and full trajectory/recovery audits;
- five LongBench datasets, 200 examples each;
- cache-hit and cache-miss TTFT;
- online TTFT and throughput under fixed request rates;
- PIC reuse and reordered-context separation;
- 75% Llama KV-cache-size reduction check;
- exact numerical no-context equivalence to the frozen Llama backbone;
- frozen/trainable parameter and weight-statistics audits;
- explicit Natural-Instructions/LongBench source and selected-instance overlap audit;
- a final requirement-by-requirement completion audit.

`reproduction_complete=true` is written only after the final completion audit
passes. The final manifest will be:

```text
/data3/junhaohu/comb/benchmarks/results/trained_final_true_tp_checkpoint/final_true_tp_reproduction_manifest.json
```

## Training/evaluation overlap disclosure

The official Natural-Instructions training pool is not source-dataset-disjoint
from the paper's LongBench selection. It contains tasks derived from HotpotQA,
Multi-News, and SAMSum. MuSiQue and 2WikiMQA have no corresponding task name in
the 757-task training inventory.

The fixed 200-example selections for the three overlapping sources were
scanned against the related 65,042 Natural-Instructions rows. Under the
declared exact Unicode/case/whitespace/punctuation-normalized question and
answer matching rules, the selected-instance overlap is zero. This proves no
exact match under those rules; it does not claim semantic or near-duplicate
disjointness. The durable report is:

```text
/data3/junhaohu/comb/benchmarks/results/longbench_training_overlap_audit.json
```

The final evaluator reruns this audit and the completion gate requires zero
selected-instance exact matches while preserving the source-overlap disclosure.
The audit source is included in the final benchmark identity manifest.

The final benchmark identity directly hashes 78 evaluator and engineering
sources. In addition to the pre-registered metric implementations, this now
includes the HF and vLLM CombLlama implementations, PIC local-cache and IPC
code, TP training/export adapter and config, milestone/capacity/snapshot
orchestration, and the context/validation calculation and comparison scripts.
The frozen performance protocol remains unchanged and still independently
checks its original source subset and numerical rules.
An automated subset check additionally requires every source locked by the
main protocol and its serving-compatibility amendment to appear in this final
manifest.

## Scope

The current run covers the CombLlama and prefix-caching arms of the paper. It
does not claim reproduction of CombDeepSeek, the third-party CacheBlend/EPIC/
BlockAttention arms, Figure 6's BlockAttention comparison, or equality to the
paper's reported 2,966 A100-GPU-hours.

The paper PDF used by the frozen protocol has SHA-256
`e1f3d874f337263943b756527f63f3b1fca564f69f48ff0f9211e315d0ef50f7`.
Its current figure numbering is authoritative. The upstream
`benchmarks/README.md` still says that the offline runner corresponds to
Figures 4/6/7; that text predates the current PDF numbering and is not used to
define the reproduction gate.

## Tests

The current CPU regression suites pass:

- `/data3/junhaohu/anaconda3/envs/comb/bin/python -m unittest discover -s training -p 'test_*.py'`: 154 tests
- `/data3/junhaohu/anaconda3/envs/comb/bin/python -m unittest discover -s benchmarks -p 'test_*.py'`: 58 tests

Run the two discovery suites sequentially. The frozen API-concurrency test
uses a deliberately short wall-clock deadline and can time out under artificial
CPU contention if both import-heavy suites are launched concurrently; its
protocol-declared isolated command and the complete sequential benchmark suite
both pass.

The real-GPU acceptance suite has additionally covered TP2/TP4 loading,
long-context execution, official-checkpoint accuracy/TTFT, online concurrency,
PIC reuse, and no-context numerical equivalence on A100-80GB GPUs.
