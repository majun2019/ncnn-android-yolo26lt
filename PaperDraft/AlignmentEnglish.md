# Alignment Plan for the English Electronics Manuscript

## 1. Objective

This document defines the alignment rules for the English version of the current manuscript before submission to Electronics. It focuses on three types of consistency problems:

- grammar and academic writing style,
- terminology consistency and naming unification,
- Markdown structure constraints and maintainability.

All rules in this document are governed by the three-contribution logic defined in Section 2. Any statement that conflicts with that logic must be revised first.

The goal of this round is not to produce the final journal-template manuscript immediately. The goal is to prepare a stable English baseline that is stylistically consistent, terminologically controlled, structurally clear, and easy to revise further for journal submission.

## 2. Core Logical Principles

### 2.1 Hierarchy of the Three Contributions

The three contributions of this paper have a clear logical hierarchy:

| Contribution | Type | Role |
| --- | --- | --- |
| Contribution 1: unified deployment architecture | propose | An independent system contribution that does not rely on the SafeHat case to exist |
| Contribution 2: five-stage diagnosis workflow | propose | An independent method contribution that does not rely on the SafeHat case to exist |
| Contribution 3: SafeHat main case | validate | An end-to-end case used to validate Contributions 1 and 2 |

Contribution 3 is a validation contribution. It is not a third method contribution independent of the first two.

### 2.2 Narrative Order Constraint: Methods First

Every part of the manuscript must follow this order:

> First present the methods (Contributions 1 and 2), then use the SafeHat case to validate them (Contribution 3).

The following reversed order is not allowed:

> First describe the SafeHat implementation, then summarize the methods from that implementation.

Specific constraints:

- The abstract, Section 1, and Section 6 must not introduce the SafeHat case before the method contributions.
- If a subsection contains both method statements and case validation, the method statement must come first.
- The SafeHat case functions as a validation scenario, not as the origin from which the methods are derived.

### 2.3 Verb Usage Constraints

| Target | Preferred verbs | Prohibited patterns |
| --- | --- | --- |
| Contribution 1: deployment architecture | propose, design, implement | built from the project, summarized from the project, derived from the case |
| Contribution 2: diagnosis workflow | propose, organize, provide | generalized from the case, discovered during practice |
| Contribution 3: SafeHat case | validate, provide closed-loop validation, use ... as the main case | propose |
| Overall framework of the paper | propose | build |

### 2.4 Prohibited Narrative Patterns

The following patterns violate the hierarchy in Section 2.1 and must be corrected whenever they appear:

1. Reversed narration: describe what the SafeHat project did first and only then conclude that the paper proposes method X.
2. Isolated validation: report SafeHat results without stating which contribution is being validated.
3. Contribution confusion: use validate for Contributions 1 or 2, or use propose for the SafeHat case itself.
4. Reversed conclusion: place the SafeHat case before the method contributions in Section 6.

### 2.5 Criteria for Avoiding a Technically Explanatory Article

No paragraph in the manuscript should read as a technically explanatory article—that is, text whose primary function is to explain how a system works rather than to advance a method claim. The following four criteria must be applied section by section:

1. **Method claim first**: The primary statement of each section must be what method this paper proposes and what problem it addresses, not how the system operates or what the algorithm steps are.
2. **Technical detail embedded in the method narrative**: Algorithm pseudocode, formula derivations, and parameter descriptions must appear as supporting evidence embedded within the method claim narrative, not as standalone technical documentation paragraphs.
3. **Deletability test**: If a paragraph can be deleted without impairing the reader's understanding of the method claim, the paragraph is likely pure technical explanation and should be condensed or merged with the motivation statement.
4. **Opening-sentence constraint for Section 3 subsections**: No subsection of Section 3 may open with a sentence of the form *The system consists of …* or *The original implementation works as follows …*; the opening sentence must state the problem the contribution addresses and the solution proposed.

### 2.6 Overall Scientific Narrative (Per-Section Anchor Checklist)

The manuscript must sustain a three-phase scientific narrative throughout: **propose the method → validate algorithmically → confirm experimentally**. It must never read as a technically explanatory article. The role of each section is mapped as follows:

| Section | Scientific narrative role | Key checkpoint |
| --- | --- | --- |
| Abstract | Present C1/C2/C3 → describe the architecture and workflow formally → report on-device numbers | Method claims must appear before on-device numbers; Contribution 3 must use validate, not propose |
| Section 1 (Introduction) | Propose the methods (C1/C2/C3) in standard contribution-list format | Contributions 1 and 2 must use propose; Contribution 3 must use provide closed-loop validation; no experimental results in this section |
| Section 2 (Related Work) | Establish the research gaps that motivate C1 and C2 | No new contribution claims; the research gaps must correspond explicitly to the contribution background in Section 1 |
| Section 3 (C1) | Lead with the method claim (*To address … this paper proposes a unified deployment architecture …*) → validate through Figure 1 and the architecture subsystems | Must not open with a project implementation description; Section 3.1 must state the design goals before expanding on the architecture |
| Section 4.1 (C2 workflow) | Lead with the method claim (*This paper further proposes a five-stage diagnosis workflow …*) → formalize the five stages | The five stages must match the Contribution 2 wording in Section 1; all five stages must be listed |
| Sections 4.2–4.5 (C2 fault evidence) | Walk each fault through the five-stage workflow → root-cause analysis → repair → regression validation | Each subsection must state the correspondence between the fault and the workflow stages before presenting the fault symptoms and repair outcome |
| Section 5.1 (C3 SafeHat) | Experimental confirmation: SafeHat provides end-to-end closed-loop validation of C1 and C2 | Must explicitly state which contribution is being validated; do not use propose for any case content |
| Section 5.2 | Experimental confirmation: five-task on-device coverage validates the unified scheduling claim of C1 | Must state the correspondence between the coverage validation and the unified scheduling claim in C1 |
| Sections 5.3–5.4 | Experimental confirmation: on-device latency characterization + scope and limitations boundary statement | Numbers must be measured values; a limitations statement is required to prevent over-generalization |
| Section 6 (Conclusion) | Unified close: C1/C2 method contributions → architecture and workflow → SafeHat validation → limitations → future work | Experimental result paragraphs must not appear before method contribution paragraphs |

## 3. Scope

This plan mainly applies to the following manuscript files:

- Paper.md
- paper outline files related to the journal manuscript
- manuscript notes directly used for paper writing

The following materials are not subject to the same strict writing constraints in this round:

- log files,
- raw dataset notes,
- build outputs and auto-generated artifacts,
- temporary text in code comments or experiment scripts.

## 4. Alignment Principles

### 4.1 Manuscript Positioning for Electronics

The English manuscript should satisfy the following requirements:

- It should read as an engineering research paper rather than a project showcase.
- Contributions, methods, experiments, and conclusions must have clearly separated roles and must follow the hierarchy in Section 2.1.
- Contributions 1 and 2 should consistently use propose as the core verb, while Contribution 3 should use validate as the core verb.
- Each conclusion should be tied to a specific experiment, table, figure, or observed error pattern whenever possible.
- Avoid exaggerated claims, over-generalized conclusions, and strong evaluative language without evidence.
- Use one preferred wording for each core concept throughout the manuscript.
- Use one English variant consistently across the manuscript; American English is preferred for consistency.

### 4.2 Out of Scope in This Round

This round does not directly address:

- final polishing of the full English manuscript,
- reference formatting,
- reformatting into the official journal template,
- final figure and table cross-reference mapping in the submission template.

### 4.3 Electronics (MDPI) Format Conventions to Apply During Translation

The following format conventions must be applied during the English translation pass to avoid large-scale reformatting later:

- Use **Figure X** (not Fig. X) for all figure references; MDPI requires the full word.
- Use **Section 3** (capitalized, not *section 3* or *Sec. 3*) for all internal section cross-references.
- Equation numbers use the format **(n)**, right-aligned.
- Table captions appear **above** the table; figure captions appear **below** the figure.

## 5. Priority Alignment Targets for the English Version

The following issues should be aligned first in the English manuscript.

### 5.1 Unify the Description of the Research Object

The current manuscript may mix expressions such as:

- resource-constrained Android devices,
- resource-limited devices,
- low-resource settings,
- low-cost, widely available Android terminals.

Recommended usage:

- Use resource-constrained Android devices as the default term for the research object.
- Use low-resource settings when discussing deployment constraints or operating conditions.
- Use low-cost, widely available Android devices when discussing accessibility, affordability, or deployment reach.

Operational definition of resource-constrained Android devices:

> In-service mobile devices that use general-purpose mobile CPU cores, do not rely on a dedicated neural acceleration path, and deliver fp32 inference throughput on 640×640 input substantially below the real-time frame-rate threshold; low-cost, widely available mid-range or low-end phones are the typical representatives.

Three verifiable dimensions:

1. Compute structure: general-purpose CPU cores, typically ARM big.LITTLE, with NCNN running without NPU or other dedicated accelerators.
2. Precision constraint: fp32, without quantization.
3. Throughput constraint: inference throughput on 640×640 input remains below the real-time frame-rate threshold; in this paper the measured level is about 2–3 FPS.

The test device HUAWEI P20 Pro with Kirin 970 satisfies all three conditions, and the latency data in Section 5.3 and Table 10 serve as the empirical basis.

Usage constraints:

- The first occurrence of resource-constrained Android devices in Section 1 must include or immediately follow the operational definition.
- Later mentions must not claim that a device is resource-constrained only by brand or model name; the statement must remain tied to the definition above.
- The rationale for NCNN should emphasize its lightweight C++ runtime and mature ARM CPU support, not Vulkan GPU support as the primary reason for selection. Vulkan remains available at the framework level, but the experiments in this paper are conducted on the CPU path.

### 5.2 Unify On-Device Wording

The manuscript may use several variants for the deployment location concept, such as device-side, edge-side, or terminal-oriented wording. In the English version, wording should be normalized as follows:

- Prefer on-device when discussing implementation, inference flow, deployment results, or validation behavior.
- Edge inference may remain in background discussion, but it should not replace the default reference to on-device execution.
- Avoid terminal as the default noun; use device unless hardware form factor is the specific point.

### 5.3 Unify Model Export and Post-Processing Terms

Current or potential variants include:

- param-bin,
- param bin,
- param/bin,
- Top-K,
- top-k,
- One-to-Many,
- O2M.

Recommended usage:

- Use param/bin file pair.
- Use Top-K.
- Write One-to-Many (O2M) at first mention, then use O2M when the context is clear.

### 5.4 Reduce Formulaic Academic Phrasing

The English manuscript should avoid formulaic or template-like expressions such as:

- furthermore,
- overall,
- it should be noted that,
- this is precisely,
- of great significance,
- significantly improves, unless the significance claim is supported,
- a reusable systematic method,
- a more operational system method,
- to this end, when used as a generic connector rather than with a clear and specific referent in the preceding sentence,
- in this paper, we … when this phrase opens more than one consecutive sentence or paragraph,
- these results indicate that, when used as a formulaic framing device without a specific metric or observation following immediately.

Replacement principles:

- Replace abstract praise with concrete objects, conditions, and evidence.
- Prefer statements of the form what device, what task, what metric, and what observed result.
- Use formulations such as the experiments show or we observe only when the sentence is tied to actual evidence.

### 5.5 Remove Meta-Statements About Paper Positioning

The English manuscript should not explain to readers what kind of paper it is in a meta-discursive way. Typical patterns that should be removed include:

- The paper is positioned as ...
- This paper should be understood as ...
- The overall positioning of this paper is ...
- This paper should not be viewed as ... but rather as ...

Replacement principles:

- Replace The paper is positioned as X with This paper studies X or This paper focuses on X.
- Replace This paper should be understood as the proposal and validation of X with This paper proposes and validates X.
- Replace statements about paper type with direct statements about research scope, content, or contribution boundaries.

### 5.6 Contribution Order and Verb Constraints

This subsection applies the rules in Section 2 directly to Paper.md.

Checklist for narrative order after each revision of the abstract, Section 1, and Section 6:

1. Does the abstract present the methods first and the validation case afterward?
2. Does the contribution list in Section 1 follow the order Contribution 1 -> Contribution 2 -> Contribution 3?
3. Does Section 6 discuss the deployment architecture and diagnosis workflow before the SafeHat validation case?
4. In each fault-mode subsection, does the diagnosis workflow step appear before the SafeHat-specific instance or result?

Checklist for verbs:

- Do Contributions 1 and 2 use propose instead of build, generalize, or derive from the project?
- Does Contribution 3 use validate or provide closed-loop validation instead of propose?
- Does the conclusion explicitly state that the SafeHat case validates the proposed methods or an equivalent formulation?

## 6. Tool Stack and Role Separation

### 6.1 markdownlint

Responsibilities:

- enforce heading hierarchy,
- enforce list and blank-line conventions,
- enforce fenced code block format,
- keep Markdown structure stable and maintainable.

Not responsible for:

- grammar in English,
- terminology unification,
- academic style judgments.

### 6.2 cspell

Responsibilities:

- manage technical spelling,
- fix capitalization and hyphenation for abbreviations, model names, file names, and framework names,
- reduce inconsistency in English technical terms.

Priority items include:

- NCNN,
- YOLO26,
- SafeHat,
- Vulkan,
- OpenCV,
- Ultralytics,
- OBB,
- PPE,
- Top-K.

### 6.3 Vale

Responsibilities:

- detect formulaic academic phrasing,
- detect non-preferred terminology,
- flag promotional or weakly supported claims.

Vale should function as the main checker for style and terminology in the English manuscript.

## 7. Terminology Consistency Rules

### 7.1 Preferred Terms

| Category | Preferred wording | Note |
| --- | --- | --- |
| Research object | resource-constrained Android devices | default term throughout the paper |
| Constraint setting | low-resource settings | used for deployment constraints and operating conditions |
| Deployment location | on-device | default wording for implementation, inference, and validation |
| File pair | param/bin file pair | do not mix variants |
| Candidate filtering | Top-K | fixed capitalization |
| Output assignment strategy | One-to-Many (O2M) | full form at first mention |
| Inference engine | NCNN | uppercase fixed |
| Case dataset | SafeHat | capitalization fixed |
| Core verb for Contributions 1 and 2 | propose | do not use build or derive |
| Core verb for Contribution 3 | validate / provide closed-loop validation | do not use propose |
| Dimension notation | 640×640, 224×224 | use Unicode × (U+00D7), not the letter x |
| Numeric range separator | 321.3–430.5 ms | use en-dash –, not a hyphen-minus - |

### 7.2 Usage Rules

- Keep one default wording for one concept.
- At first mention of a mechanism or strategy, use a direct English formulation and include the abbreviation only when needed.
- Define abbreviations twice: once in the abstract at first occurrence, and again at first mention in the body text, following MDPI Electronics practice. Both the abstract and the body are self-contained in MDPI layout.
- Do not alternate among synonyms within the same subsection.
- Keep terminology consistent across body text, figure captions, table captions, abstract, and conclusion.
- File names, model names, and script names may preserve their actual spelling.
- Method sections for Contributions 1 and 2 must not make SafeHat a prerequisite for the methods to exist.

### 7.3 Compound Modifier Hyphenation

The following compound expressions must be hyphenated when used as attributive modifiers before a noun. They are written without a hyphen when used as predicative complements or as standalone noun phrases.

| Expression | Hyphenated (before noun) | Not hyphenated |
| --- | --- | --- |
| on device | on-device inference | inference is performed on device |
| resource constrained | resource-constrained Android devices | — |
| end to end | end-to-end validation | validated end to end |
| five stage | five-stage workflow | — |
| cross backend | cross-backend comparison | comparison across backends |

Apply this rule consistently across body text, figure captions, table captions, abstract, and conclusion.

## 8. Grammar and Style Rules

### 8.1 Sentence-Level Rules

- One sentence should preferably carry one main claim.
- Avoid long chains of stacked modifiers.
- Conclusion sentences should include the condition, object, or evidence source whenever possible.
- Avoid generic evaluative sentences without a clear subject.

### 8.2 Academic Style Rules

- Use strong words such as significant only when a metric, comparison, or error range supports the claim.
- Avoid repetitive connectors such as overall and furthermore as paragraph fillers.
- Replace this is precisely with a more concrete causal or problem-defining statement.
- Remove meta-statements about what type of paper this is; state what the paper studies and what it contributes.
- The narrative order must follow the methods-first principle in Section 2.2.

### 8.3 Suggested Replacement Patterns

Replace expressions such as:

- of great significance,
- significantly improves,
- a reusable systematic method,

with one of the following patterns when appropriate:

- validates the feasibility of the method on a specific device and task,
- improves metric X relative to baseline Y under condition Z,
- provides a complete implementation path from export and deployment to diagnosis.

### 8.4 Voice and Tense Rules

- Prefer **neutral third-person or non-personal formulations** for contribution statements, method descriptions, and summary sentences in the target Electronics manuscript. Preferred patterns include *The main contributions of this study are as follows*, *This study proposes*, *The proposed framework integrates*, and *Table 5 shows*. Avoid author-centered openings such as *We make three contributions* as the default form.
- Use **simple present tense** for contribution claims and method descriptions: *This study proposes a unified architecture that resolves …*, *A five-stage workflow is proposed …*, *The proposed framework integrates …*
- Use **simple past tense** for experimental operations and measurements: *Latency was measured on …*, *Samples were collected …*
- Use **present tense** for statements that describe the paper's current content: *Section 3 describes …*, *Table 5 shows …*
- First-person forms are not prohibited, but they should be used sparingly and should not dominate the abstract, the introduction contribution list, or the conclusion.
- Do not mix tenses within the same sentence when describing a single experimental or method context.

## 9. Markdown Structure Constraints

### 9.1 Heading Hierarchy

- The document should contain only one level-1 heading.
- Main sections should use level-2 headings.
- Level-3 headings should be used only for internal subdivision when necessary.

### 9.2 Paragraphs and Lists

- Leave blank lines around headings.
- Leave blank lines before and after lists.
- Keep list items structurally parallel where possible.
- Do not split consecutive short statements into excessive bullet lists.

### 9.3 Figures, Tables, and Code Blocks

- Use a consistent format for figure titles and table titles.
- Use fenced code blocks with backticks.
- Add language labels to code blocks whenever practical; use text for pseudo-output or pipeline sketches.
- Keep display equations as standalone blocks rather than embedding them inside list items when that would interrupt readability.

### 9.4 Items Not Enforced in This Round

To avoid unnecessary noise, the following items are not enforced in this round:

- maximum line length,
- bare links in references,
- emphasis style inside figure and table notes.

## 10. How the Automatic Checks Should Be Applied

This round should use the following division of responsibilities:

- .markdownlint.jsonc: Markdown structural boundaries,
- cspell.json: English technical terms, abbreviations, and spelling allowlist,
- .vale.ini and project Vale style rules: style and terminology reminders for English academic style and preferred terms.

### 10.1 First-Pass Vale Rule Set

The first Vale rule set should focus on three categories:

| Rule file | Purpose | Level |
| --- | --- | --- |
| PreferredTerms.yml | unify preferred terms and detect common variants | warning |
| TemplatePhrases.yml | flag formulaic transition words and stock phrases | suggestion |
| ClaimStrength.yml | flag strong claims without explicit evidence support | suggestion |

All three rule files should be placed under `.vale/styles/ElectronicsEnglish/` and referenced in `.vale.ini` accordingly.

These rules should first cover the high-frequency issues already visible in the current manuscript rather than attempting full stylistic coverage immediately.

## 11. Recommended Execution Order

### 11.1 Phase 1: Pre-Translation Checks on the Chinese Manuscript (Paper.md)

1. Check the abstract, Section 1, and Section 6 of Paper.md against the logical rules in Section 2 to confirm the methods-first order and contribution hierarchy are correct before translation begins.
2. Fix structural constraints in Paper.md so heading hierarchy, code blocks, and list conventions are stable.
3. Freeze English terminology and abbreviation choices so translation decisions are consistent from the start.

### 11.2 Phase 2: During-Translation Alignment

4. Apply the Electronics format conventions in Section 4.3 as each section is translated.
5. Use Vale to clean formulaic phrases and non-preferred wording in translated sections.
6. Translate Paper.md section by section, checking the §5 alignment targets after each section.
7. After completing the full draft, run the acceptance criteria checklist in Section 12 before moving to submission-format refinement.

## 12. Acceptance Criteria

The manuscript can be considered ready for pre-submission English polishing when all of the following conditions are met:

- The abstract, Section 1, and Section 6 follow the methods-first order in Section 2.2.
- Contributions 1 and 2 consistently use propose, and Contribution 3 consistently uses validate.
- Core concepts no longer show obvious wording drift.
- Heading hierarchy, lists, and fenced code blocks remain structurally stable.
- Formulaic academic phrases and abstract claim language are clearly reduced.
- Conclusion paragraphs rely more on observed results than on generic evaluation.
- Later English-polishing work will not first require another round of large-scale terminology recovery.

## 13. Baseline Files Already Generated in This Round

The following baseline files have already been created:

- .markdownlint.jsonc
- cspell.json
- .vale.ini
- .vale/styles/ElectronicsEnglish/PreferredTerms.yml
- .vale/styles/ElectronicsEnglish/TemplatePhrases.yml
- .vale/styles/ElectronicsEnglish/ClaimStrength.yml

> **Note**: The English manuscript rules live under `.vale/styles/ElectronicsEnglish/`. This directory is separate from the Chinese-manuscript rules under `.vale/styles/ElectronicsChinese/`. The English PreferredTerms and TemplatePhrases rule files must reference English-specific preferred terms and English stock phrases rather than reusing the Chinese-rule files.

These files form the first automatic-check skeleton for the manuscript. If the next step is direct manuscript revision, Paper.md should be checked first, and the rules should be tightened incrementally according to actual warning density rather than made overly strict from the start.

## 14. Locked Content for Cross-Section Consistency

This section records key content that has already been revised and fixed. Later edits to the abstract, body, conclusion, and captions should remain consistent with this section. Terminology and numbers recorded here must not be changed locally without synchronizing other occurrences.

### 14.1 Main Contributions (Section 1, finalized on 2026-05-12)

> The following wording uses neutral third-person or non-personal contribution statements to match the preferred journal tone in Section 8.4. If a local grammatical context requires *This study proposes …* or *The proposed workflow …*, that form is also acceptable; all occurrences must remain consistent within the same section.

1. A unified multi-task vision deployment architecture is proposed for resource-constrained Android devices to resolve asset naming conflicts, interface semantic divergence, and task-scheduling coupling under coexisting E2E, One-to-Many, and Legacy output paths across five task types, and to realize unified loading, unified scheduling, and consistent interface management within the Java–JNI–C++–NCNN pipeline.
2. A five-stage consistency diagnosis workflow is proposed for multi-task mobile deployment. Through anomaly logging, intermediate-output inspection, cross-backend comparison, structure tracing, and regression validation, the workflow supports structured localization and repair of four representative fault classes: preprocessing mismatch, redundant activation, coordinate semantic misinterpretation, and OBB layout misjudgment.
3. SafeHat 10-class PPE detection is used as the main case to provide end-to-end closed-loop validation from hard-negative mining and model fine-tuning to on-device diagnosis and regression, and five-task runtime baselines are reported on the Kirin 970 CPU path, with mean latency of 321.3–430.5 ms for the four 640×640 tasks and 33.6 ms for classification at 224×224, thereby providing a reproducible reference for deployment evaluation on resource-constrained Android devices.

### 14.2 Contribution-to-Body Mapping

| Contribution | Section | Contribution type | Logical role |
| --- | --- | --- | --- |
| Contribution 1: unified deployment architecture | Section 3 | system design contribution | propose |
| Contribution 2: five-stage diagnosis workflow | Section 4 | method contribution | propose |
| Contribution 3: case validation and latency baseline | Section 5 | empirical contribution | validates Contributions 1 and 2 |

### 14.3 Latency Baseline Numbers

- Mean CPU latency for the four 640×640 tasks, detection, segmentation, pose, and OBB: 321.3–430.5 ms.
- Mean CPU latency for classification at 224×224 input: 33.6 ms.
- Test device: HUAWEI P20 Pro, Kirin 970, NCNN CPU fp32 arm64-v8a, measured on 2026-05-07.
