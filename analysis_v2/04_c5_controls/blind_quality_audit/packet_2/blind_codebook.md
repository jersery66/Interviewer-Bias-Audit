# D-P blinded audit codebook (formal version)

## Domain definitions

- `anhedonia_interest`：兴趣、愉快感或动机下降；必须有被试语言的明确支持。
- `appetite_weight`：食欲或体重的改变。
- `concentration_psychomotor`：注意力、集中、决策困难，或精神运动性迟缓/激越。
- `depressed_mood`：悲伤、低落、情绪低沉或类似的抑郁心境表达。
- `functioning_impairment`：工作、学习、社交或日常生活功能受影响。
- `mental_health_history`：既往心理健康诊断、治疗、咨询、住院或相关病史。
- `protective_or_absent_symptom`：明确的保护因素，或明确否认/不存在某症状；不能自动转译为另一症状域阳性。
- `self_worth_guilt`：无价值感、自责、过度内疚或强烈自我贬低。
- `sleep_fatigue_energy`：睡眠异常、疲倦或精力下降。
- `suicide_self_harm`：自杀想法、死亡愿望、自伤行为或自伤想法。

## Domain label

- `1`: the visible participant text contains sufficiently explicit support for
  the domain and at least one exact quote plus line number is recorded.
- `0`: no supporting evidence is found in the visible participant text.  This
  is not a clinical claim that the symptom is absent.
- `NS`: meaning, context, or domain assignment cannot be determined reliably.
  Short answers such as `yes` or `sometimes` without the interviewer question,
  indirect evidence, and unresolved domain boundaries are NS.

## Count bin

Count independent evidence events or distinct symptom manifestations, not
repeated wording of the same manifestation: `0`, `1`, `2`, or `3+`.  A compound
sentence may count as two only when it contains two distinct manifestations
under this rule.  Identical repetition counts once.  For NS, count may be left
blank; the cell is excluded from the primary binary comparison.

## Evidence, line number, and polarity

Copy the participant's exact words; do not replace them with a summary.  Line
numbers are 1-based and may be comma- or semicolon-separated.  Use
`present` for positive symptom evidence, `denied/protective` for an explicit
denial or protective statement, and `ambiguous` when the text cannot support a
reliable polarity.  `protective_or_absent_symptom` can be positive for a
protective or explicit absence statement, but that statement must not also be
automatically coded as `suicide_self_harm=1`.

## Freeze rule

Reviewers may discuss the five training cases only.  Once formal scoring starts,
this codebook cannot be changed based on formal results.  Disagreements are
resolved after the independent metrics are calculated.  Raw reviewer files
remain immutable.
