# Attribute Type Inference from Names — code and dataset

Code and anonymized data for the paper *Attribute Type Inference from Names Using
Rule-Based and LLM-Assisted Techniques with Application to Automated Student
Assessment* (M. Kocev, M. Todorovikj, Faculty of Computer Science and
Engineering, Ss. Cyril and Methodius University in Skopje).

The system assigns one of four SQL types — `INTEGER`, `NUMERIC`, `DATE`, `TEXT` —
to a database attribute **from its name alone**, using priority-ordered
multilingual regular expressions (English, Macedonian Latin and Cyrillic) with a
local LLM (Qwen2.5:3b via Ollama) as a fallback for names no rule matches. It is
part of a pipeline that grades student ER diagrams automatically.

## What is here

| Path | Contents |
|---|---|
| `src/type_inference.py` | The implementation used for the paper: transliteration, the four keyword regexes and their priority order, the Ollama fallback with its exact prompts, and the SQLAlchemy schema builder |
| `data/attribute_labels.csv` | 2,625 distinct attribute names with their manually assigned type label (the ground truth, annotated per unique name) |
| `data/attribute_predictions.csv` | 14,076 extracted attributes with the prediction of every evaluated method |
| `data/diagrams/` | the 520 ER diagrams themselves (`dev001.xml` … `test080.xml`), anonymized draw.io XML |
| `analysis/` | Scripts that reproduce the paper's tables and analyses from `data/` |

### `attribute_predictions.csv` columns

| Column | Meaning |
|---|---|
| `split` | `dev` (440 diagrams) or `test` (80 unseen diagrams) |
| `diagram` | opaque diagram id (`dev001`, `test012`, …) |
| `domain` | application domain of the exam task (e.g. `marathon`, `ski race`) |
| `attribute` | attribute name as resolved by the pipeline |
| `language` | `lat_mk`, `cyr_mk` or `eng` |
| `label` | ground-truth type |
| `tier` | which tier decided: `regex`, `llm`, or `fk` (foreign key, typed structurally and excluded from scoring) |
| `regex`, `llm_3b`, `llm_7b`, `hybrid_3b`, `hybrid_7b` | prediction of each evaluated configuration |
| `llm_label`, `llm_conf`, `llm_reason` | raw LLM answer, its self-reported confidence (before the 0.70 threshold) and the justification it stated |
| `table`, `siblings` | table name and sibling attribute names the LLM saw (filled for attributes that reached the LLM tier) |

Rows are in the original processing order.

`llm_3b` comes from a separate LLM-only run, so on one test attribute it
differs from the answer the hybrid logged for the same name (246 vs 247 correct
on the 320 attributes the rules leave unresolved). Both numbers are correct for
their own configuration.

## Reproducing the paper

```bash
pip install -r requirements.txt
python analysis/reproduce_tables.py      # Tables 2 and 3, per-tier and per-language accuracy
python analysis/vocabulary_overlap.py    # Section 5.2 overlap, NUMERIC failure analysis
python analysis/numeric_mitigations.py   # confidence-threshold replay
python analysis/finer_types.py           # how often finer SQL types would apply
python analysis/ml_baselines.py          # TF-IDF + LR/SVM baselines
python analysis/significance.py          # exact McNemar tests against the baselines
```

`reproduce_tables.py` returns the published numbers exactly: 97.40% / 93.57%
accuracy, 83.40% / 74.58% regex coverage, and the per-tier accuracies
(7,457/7,476 and 1,276/1,490 on development; 931/939 and 247/320 on the test
set).

Running the type inference itself additionally needs a local
[Ollama](https://ollama.com) server with `qwen2.5:3b` pulled; the rule tier runs
without it.

## Data provenance and privacy

The attributes come from ER diagrams submitted by students of the
[Databases (Бази на податоци, F23L3W004)](https://finki.ukim.mk/subject/F23L3W004/)
course at FCSE, Ss. Cyril and Methodius University in Skopje: 440 diagrams used
for development and 80 later, unseen diagrams used as a held-out test set.

The diagrams are released in anonymized form. File names, which contained
student names and index numbers, are replaced by opaque ids that match the
`diagram` column of the CSV; the `<mxfile>` editor metadata (`host`, `agent`,
`version`, `etag`, `modified`, `type`) is stripped; and any student name or
index known from the original file names is redacted from the text the diagrams
display, in both Latin and Cyrillic spelling. The modelled content is kept as
drawn, so attribute names are the students' own wording, including
misspellings, and labels such as ``ime i prezime`` or ``email`` are fields
students modelled, not personal data.

Grades, submission metadata and any other record about the students are not
part of this release.

## Paper

The LaTeX sources of the paper, together with the review-response history (one
pull request per reviewer point), are kept in a separate repository by the
authors and can be shared on request.

## Citation

Please cite the paper; the reference will be added here once it appears.
