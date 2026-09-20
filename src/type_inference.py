"""Attribute type inference: transliteration, priority-ordered rules, LLM fallback.

This is the implementation used for the results in the paper, extracted from the
grading pipeline notebook (cells defining the IR classes, the Ollama fallback
with its prompts, the regex tier, and the SQLAlchemy schema builder).
Requires `pip install requests sqlalchemy` and, for the fallback tier, a local
Ollama server with the qwen2.5:3b model.
"""

import re
import os
import numpy as np

#class for entities

class Entity(object):
    id = ""
    attributes = []
    primary_key = []
    foreign_keys = []

    def __init__(self, id, attributes, primary_key, foreign_keys):
        self.id = id
        self.attributes = attributes
        self.primary_key = primary_key
        self.foreign_keys = foreign_keys

    def print(self):
        print("id: ", self.id)
        print("attribute IDs: ", self.attributes)
        print('primary_key: ', self.primary_key)
        print('foreign_keys: ')
        for fk in self.foreign_keys:
            fk.print()
            print()


#class for foreign keys
class Foreign_key(object):
    attributes = []
    entity = ''
    not_null = False

    def __init__(self, attributes, entity, not_null):
        self.attributes = attributes
        self.entity = entity
        self.not_null = not_null

    def print(self):
        print("Attributes: ", self.attributes)
        print("Entity: ", self.entity)
        print("Not null: ", self.not_null)


#class for inheritence objects
class Inheritence(object):
    parent_id = ''
    children_ids = []
    total_participation = False

    def __init__(self, parent_id, children_ids, total_participation):
        self.parent_id = parent_id
        self.children_ids = children_ids
        self.total_participation = total_participation

    def print(self):
        print("Parent: ", self.parent_id)
        print("Children: ", self.children_ids)
        print("Total participation: ", self.total_participation)


#class for 1:1 with both entities with total participation and N:M relationships with total participation entities
class EntityRelationshipNotNull(object):
    entity_id = ''
    relationship_id = ''

    def __init__(self, entity_id, relationship_id):
        self.entity_id = entity_id
        self.relationship_id = relationship_id

    def print(self):
        print("Entity id: ", self.entity_id)
        print("Relationship id: ", self.relationship_id)

# ============================================================
# OLLAMA FALLBACK
# YOU NEED OLLAMA INSTALLED AND RUNNING FOR FALLBACK
# Example:
#   ollama run qwen2.5:3b
# ============================================================

import json
import re
from typing import Optional, List, Dict, Any
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:3b"  # change to qwen2.5:7b if you want a stronger local model

TYPE_INFERENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {
            "type": "string",
            "enum": ["text", "integer", "numeric", "date"]
        },
        "confidence": {
            "type": "number"
        },
        "reason": {
            "type": "string"
        }
    },
    "required": ["type", "confidence", "reason"]
}

def tokenize_label(text: str) -> List[str]:
    if not text:
        return []
    return [t for t in re.split(r'[_\s\-]+', text.lower()) if t]

def safe_json_loads(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # CHANGED: keep a fallback extractor in case the model wraps JSON in extra text
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise

def infer_type_with_ollama(
    attr: str,
    table_name: Optional[str] = None,
    sibling_attributes: Optional[List[str]] = None,
    attr_cyrillic: Optional[str] = None,
    extra_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    sibling_attributes = sibling_attributes or []
    extra_context = extra_context or {}

    normalized_attr = attr.strip().lower()
    tokens = tokenize_label(normalized_attr)

    system_prompt = """
You are a database schema type inference assistant.

Your task is to infer the most likely SQL-like type of a database attribute.

═══ ALLOWED OUTPUT TYPES ═══

- text
- integer
- numeric
- date

═══ TYPE DEFINITIONS ═══

integer   — identifiers, ids, counters, counts, ranks, years, minute counts,
            whole-number codes, order/sequence numbers
numeric   — decimal or measured quantities: price, amount, cost, score,
            percentage, ratio, weight, height, distance, balance, deposit, total
date      — dates and date-like fields: birth dates, created/updated/deleted
            dates, validity dates, expiry dates
text      — names, categories, descriptions, brands, phone numbers, serial
            numbers, passport/account numbers, free text, statuses, time-of-day
            labels, time periods, durations, timestamps

═══ CRITICAL RULES ═══

1. CLASSIFY EACH ATTRIBUTE INDEPENDENTLY.
   Siblings tell you what the entity is and does — not what type to assign.
   If siblings are all TEXT, that does not push an ambiguous attribute toward
   TEXT. Classify based on what the attribute itself semantically represents.
   Example: siblings [name, category, brand] are all TEXT, but "unit_price"
   is still NUMERIC regardless.

2. EXPAND AMBIGUOUS NAMES BEFORE CLASSIFYING.
   Attribute names may be in English, Macedonian Cyrillic, or Macedonian Latin
   transliteration. They may contain misspellings, abbreviations, acronyms, or
   concatenated words. Before classifying, mentally expand the name:
   - Common suffixes: "_id"→identifier, "_dt"/"_d"→date, "_amt"→amount,
     "_pct"→percentage, "_qty"→quantity, "_no"→number/code, "_cd"→code
   - Concatenated: "startd"→"start date", "endtm"→"end time"
   - Misspelled: "adress"→address, "produtc"→product
   Classify the *expanded* meaning using all available context.

3. TIME IS TEXT, NOT DATE.
   Fields representing time-of-day, durations, or time periods (e.g. "start
   time", "end time", "shift", "period") must be classified as TEXT.
   Only fields representing a calendar date qualify as DATE.
   Carefully check fields like "start_d" or "end_d" — they could be
   "start date" (DATE) or "start duration" (TEXT). Use sibling context
   to resolve the ambiguity.

4. PREFER TEXT ONLY AS A LAST RESORT.
   Choose TEXT only when the attribute has no credible integer, numeric, or
   date interpretation after considering all context. Do not use sibling types
   or naming uncertainty alone as a reason to fall back to TEXT.

5. USE CONTEXT, NOT ASSUMPTIONS.
   Use the table name, sibling attribute names, and any metadata to understand
   what entity is being described. Let that understanding inform the semantic
   meaning of the attribute — then classify.

═══ OUTPUT ═══

Return only valid JSON matching the required schema. No explanation.
""".strip()

    #Check for correct upload
    # CHANGED: send richer context from your IR/build phase, not just attr + table
    user_payload = {
        "attribute_name": attr,
        "normalized_attribute_name": normalized_attr,
        "attribute_name_cyrillic": attr_cyrillic,
        "attribute_tokens": tokens,
        "table_name": table_name,
        "sibling_attributes": sibling_attributes,
        "extra_context": extra_context
    }

    user_prompt = f"""
Infer the SQL-like type of this database attribute.

Return ONLY valid JSON matching this schema:
{json.dumps(TYPE_INFERENCE_SCHEMA, ensure_ascii=False)}

Context:
{json.dumps(user_payload, ensure_ascii=False, indent=2)}

Reminders:
- Expand abbreviations and joined words in the attribute name before deciding.
- Sibling types are context clues about the entity, not a type to match.
- TIME fields → text. CALENDAR DATE fields → date.
- Fall back to "text" only if no other type fits after careful reasoning.
""".strip()

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": TYPE_INFERENCE_SCHEMA,
            "options": {
                "temperature": 0
            }
        },
        timeout=150,
    )
    response.raise_for_status()

    raw_content = response.json()["message"]["content"]
    parsed = safe_json_loads(raw_content)

    inferred_type = parsed.get("type", "text")
    confidence = float(parsed.get("confidence", 0.0))
    reason = parsed.get("reason", "")

    allowed = {"integer", "text", "numeric", "date"}
    if inferred_type not in allowed:
        inferred_type = "text"

    if confidence < 0:
        confidence = 0.0
    if confidence > 1:
        confidence = 1.0

    return {
        "type": inferred_type,
        "confidence": confidence,
        "reason": reason,
    }

# ============================================================
# TYPE INFERENCE & TABLE NAME HELPERS
# ============================================================

DIGRAPHS = [
    ("dzh", "џ"),
    ("dz", "ѕ"),
    ("gj", "ѓ"),
    ("kj", "ќ"),
    ("lj", "љ"),
    ("nj", "њ"),
    ("zh", "ж"),
    ("ch", "ч"),
    ("sh", "ш"),
    ("dj", "џ"),
]

SINGLE = {
    "a": "а", "b": "б", "v": "в", "g": "г", "d": "д", "e": "е",
    "z": "з", "i": "и", "j": "ј", "k": "к", "l": "л", "m": "м",
    "n": "н", "o": "о", "p": "п", "r": "р", "s": "с", "t": "т",
    "u": "у", "f": "ф", "h": "х", "c": "ц", "q": "к", "w": "в",
    "x": "кс", "y": "и",
}

def latin_mk_to_cyrillic(text: str) -> str:
    if not text:
        return text

    s = text.lower()

    for latin, cyr in DIGRAPHS:
        s = s.replace(latin, cyr)

    out = []
    for ch in s:
        out.append(SINGLE.get(ch, ch))
    return "".join(out)

# CHANGED: slightly safer threshold for self-reported LLM confidence
CONFIDENCE_THRESHOLD = 0.70

# ============================================================
# TYPE INFERENCE KEYWORDS — EXPANDED
# ============================================================

# INTEGER KEYWORDS
INTEGER_KEYWORDS = [
    r'(identifier|идентификатор|identification|ident|идентификација|ид|id)',
    r'(code|sifra|шифра|код)',
    r'(idx|index|indeks|индекс)',
    r'(num|number|nbr)',
    r'(count|qty|quantity|број|counter)',
    r'(rank|position|order|ordinal|sequence|seq|step|level_no|size)',
    r'(ранк|ранг|реден|позиција|редослед|секвенца|чекор|големина)',
    r'(duration_min|duration_in|minutes|траење_во|минути|возраст)',
    r'(age|year|година|години|возраст|yrs)',
    r'(size|број_на)',
    r'id\d+',
    r'id_\d+',
    r'[a-z]+_id',
    r'[a-z]+id',
    r'(^|_)бр(_|$)',
    r'(^|_)год(_|$)',
    r'(^|_)no(_|$)',
    r'(^|_)min(_|$)',
]

TEXT_KEYWORDS = [
    r'(name)',
    r'(title|label|caption|tag|alias|login)',
    r'(desc|details|summary|bio|about|content|text|message|body)',
    r'(note|comment|remark|status|state|condition)',
    r'(type|kind|category|group|class|brand|manufacturer|producer|company|vendor|model|version)',
    r'(address|street|city|country|place|location|region|municipality|postal_code|zip)',
    r'(mail|website|url|link)',
    r'(phone|mobile)',
    r'(ssn|social_security_number|passport_number|license|registration_number|account_number|bank_account|iban)',
    r'(serial)',
    r'(time|period|interval)',
    r'(duration)',
    r'(level|grade|tier)',
    r'(име)',
    r'(наслов|ознака|етикета|алијас|прекар|корисничко)',
    r'(опис|детали|содржина|текст|порака|биографија|забелешка|коментар|статус|состојба)',
    r'(тип|категорија|група|класа|бренд|производител|компанија|фирма|модел|верзија)',
    r'(^|_)вид(_|$)',
    r'(^|_)град(_|$)',
    r'(адреса|улица|држава|локација|регион|општина|поштенски_код)',
    r'(е_пошта|мејл|веб_страница|линк|врска)',
    r'(телефон|мобилен|број_за_контакт)',
    r'(ембг|мат_број|матичен|лиценца|лична_карта|пасош|возачка_дозвола|број_на_сметка|ибан|мбр)',
    r'(сериски)',
    r'(^|_)тел(_|$)',
    r'(време|време_на_денот|временски_период|период|термин|интервал)',
    r'(времетраење)',
    r'(timestamp|datetime|date_time)',
    r'(временска_ознака|датум_и_време)',
    r'(ниво|степен)',
]

DATE_KEYWORDS = [
    r'(date|created|updated|deleted|birthday)',
    r'(dob)',
    r'(valid_from|valid_until|valid_to|expires_at)',
    r'(датум|дата|креирано|ажурирано|избришано|дат|роденден)',
    r'(валиден_од|валиден_до|важи_од|важи_до|истекува_на)',
    r'(рок_на_доспевање)',
    r'(почеток|крај|почетен|краен)'
    r'(start|end|poceten|pocetok)'
]

NUMERIC_KEYWORDS = [
    r'(price|amount|cost|total|tax|fee|salary|wage|income|expense|profit|revenue|balance|deposit|budget|budzet|buzet)',
    r'(avg|average|mean|score|rating|rate|percentage|percent|ratio|share)',
    r'(weight|height|width|length|depth|volume|capacity|distance|area|speed|temperature)',
    r'(value|sum|mark)',
    r'(^|_)points(_|$)',
    r'(hours)',
    r'(износ|трошок|вкупно|меѓузбир|данок|надомест|плата|приход|расход|добивка|салдо|баланс|депозит|буџет|фонд)',
    r'(^|_)цена(_|$)',
    r'(просек|оценка|рејтинг|стапка|процент|сооднос|удел)',
    r'(тежина|висина|ширина|должина|длабочина|волумен|капацитет|растојание|површина|брзина|температура)',
    r'(вредност|збир|награда)',
    r'(^|_)поени(_|$)',
    r'(часови)',
]

INTEGER_RE = re.compile('|'.join(INTEGER_KEYWORDS), flags=re.IGNORECASE)
TEXT_RE = re.compile('|'.join(TEXT_KEYWORDS), flags=re.IGNORECASE)
DATE_RE = re.compile('|'.join(DATE_KEYWORDS), flags=re.IGNORECASE)
NUMERIC_RE = re.compile('|'.join(NUMERIC_KEYWORDS), flags=re.IGNORECASE)

def infer_sqlite_type_nli(
    attribute_name: str,
    table_name: Optional[str] = None,
    sibling_attributes: Optional[List[str]] = None,
    extra_context: Optional[Dict[str, Any]] = None,
    return_scores=False
):
    """Infer SQLite type using rule-based priority with Ollama fallback."""
    if not attribute_name or not isinstance(attribute_name, str):
        raise ValueError("Invalid input: Please provide a non-empty attribute name.")

    attr = attribute_name.strip()
    attr_cyrillic = latin_mk_to_cyrillic(attr)

    if DATE_RE.search(attr):
        return ("date", None) if return_scores else "date"
    if DATE_RE.search(attr_cyrillic):
        return ("date", None) if return_scores else "date"

    if NUMERIC_RE.search(attr):
        return ("numeric", None) if return_scores else "numeric"
    if NUMERIC_RE.search(attr_cyrillic):
        return ("numeric", None) if return_scores else "numeric"

    if TEXT_RE.search(attr):
        return ("text", None) if return_scores else "text"
    if TEXT_RE.search(attr_cyrillic):
        return ("text", None) if return_scores else "text"

    if INTEGER_RE.search(attr):
        return ("integer", None) if return_scores else "integer"
    if INTEGER_RE.search(attr_cyrillic):
        return ("integer", None) if return_scores else "integer"

    # CHANGED: replace zero-shot NLI fallback with local Ollama fallback
    try:
        llm_result = infer_type_with_ollama(
            attr=attr,
            table_name=table_name,
            sibling_attributes=sibling_attributes,
            attr_cyrillic=attr_cyrillic,
            extra_context=extra_context,
        )
    except Exception as e:
        print(f"Error calling Ollama for '{attr}': {e}. Defaulting to TEXT.")
        llm_result = {
            "type": "text",
            "confidence": 1.0,
            "reason": "fallback after Ollama error"
        }

    best_label = llm_result["type"]
    best_score = llm_result["confidence"]

    final_label = "text" if best_score < CONFIDENCE_THRESHOLD else best_label
    allowed = {"integer", "text", "numeric", "date"}
    if final_label not in allowed:
        final_label = "text"

    if return_scores:
        # CHANGED: keep debug-style scores compatible with your old pattern
        out_scores = {
            "integer": 1.0 if best_label == "integer" else 0.0,
            "text": 1.0 if best_label == "text" else 0.0,
            "numeric": 1.0 if best_label == "numeric" else 0.0,
            "date": 1.0 if best_label == "date" else 0.0,
            "best_label": best_label,
            "llm_confidence": best_score,
            "llm_reason": llm_result["reason"],
        }
        return final_label, out_scores

    return final_label

def clean_text(text):
    """Removes HTML tags, newlines, and special characters. Returns snake_case string."""
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\w]+', '_', text)
    return text.strip('_').lower()

def get_human_readable_name(xml_id, cells_map):
    """Resolves an XML ID to its human-readable label from the diagram."""
    if xml_id in cells_map:
        val = cells_map[xml_id].get('@value', '')
        if val is None:
            return ""
        return clean_text(str(val))
    return ""

def resolve_table_name(entity_obj, cells_map):
    """
    Determines the table name for an entity object.
    """
    if hasattr(entity_obj, '_patched_table_name'):
        return entity_obj._patched_table_name

    raw_id = entity_obj.id

    if ' $ ' in raw_id:
        parts = raw_id.split(' $ ')
        names = [get_human_readable_name(part, cells_map) for part in parts]
        names = [n for n in names if n]
        return "_".join(names) if names else "unnamed_relationship_table"

    name = get_human_readable_name(raw_id, cells_map)
    return name if name else "unnamed_table"

def get_sa_type(type_str):
    """Maps the inferred string to a SQLAlchemy Type object."""
    from sqlalchemy import Integer, Numeric, Date, Text
    type_str = type_str.lower()
    if type_str == 'integer': return Integer()
    if type_str == 'numeric': return Numeric()
    if type_str == 'date':    return Date()
    return Text()

from sqlalchemy import (
    MetaData, Table, Column, Integer, String, Text, Numeric, Date,
    ForeignKeyConstraint, PrimaryKeyConstraint
)

def build_sqlalchemy_metadata(ir):
    """
    Build and return a SQLAlchemy MetaData object from the IR.

    Returns
    -------
    metadata : sqlalchemy.MetaData
    entitycolmapping : dict[entityid][attrid] -> [colname1, colname2, ...]
    entitycoltypemapping : dict[entityid][attrid] -> [satype1, satype2, ...]
    """

    sorted_entities = ir['sorted_entities']
    entity_obj_map = ir['entity_obj_map']
    cells_map = ir['cells_map']

    metadata = MetaData()
    metadata.clear()

    entity_col_mapping = {}
    tablename_tracker = {}
    entity_col_type_mapping = {}
    entity_col_out_scores_mapping = {}

    for entity_id in sorted_entities:
        if entity_id not in entity_obj_map:
            continue

        entity = entity_obj_map[entity_id]

        base_table_name = resolve_table_name(entity, cells_map)
        if not base_table_name:
            continue

        if base_table_name in tablename_tracker:
            count = tablename_tracker[base_table_name] + 1
            tablename_tracker[base_table_name] = count
            table_name = f"{base_table_name}_{count}"
        else:
            tablename_tracker[base_table_name] = 1
            table_name = base_table_name

        sa_columns = []
        colname_tracker = set()

        entity_col_mapping[entity_id] = {}
        entity_col_type_mapping[entity_id] = {}
        entity_col_out_scores_mapping[entity_id] = {}

        def get_unique_col_name(base):
            cand = base
            c = 1
            while cand in colname_tracker:
                cand = f"{base}_{c}"
                c += 1
            colname_tracker.add(cand)
            return cand

        sorted_attr_ids = sorted(
            entity.attributes,
            key=lambda aid: get_human_readable_name(aid, cells_map)
        )

        # CHANGED: build resolved sibling list once per entity so we can send it to Ollama
        sibling_attribute_names = []
        for sibling_attr_id in sorted_attr_ids:
            sibling_raw = get_human_readable_name(sibling_attr_id, cells_map)
            if sibling_raw:
                sibling_attribute_names.append(sibling_raw)

        for attrid in sorted_attr_ids:
            raw_name = get_human_readable_name(attrid, cells_map)

            prefix = ""
            is_fk = False
            ref_fk = None
            referenced_entity_name = None

            for fk in entity.foreign_keys:
                if attrid in fk.attributes:
                    is_fk = True
                    ref_fk = fk
                    ref_entity = entity_obj_map.get(fk.entity)
                    if ref_entity:
                        prefix = resolve_table_name(ref_entity, cells_map)
                        referenced_entity_name = prefix
                    break

            if is_fk and prefix:
                clean = clean_text(raw_name)
                if clean in {"id", "identifier", "code"}:
                    prop_name = f"{prefix}_id"
                else:
                    prop_name = f"{prefix}_{raw_name}"
            else:
                prop_name = raw_name

            final_name = get_unique_col_name(prop_name)

            sa_type = None
            out_scores = None

            if is_fk and ref_fk:
                parent_entity = ref_fk.entity
                if parent_entity in entity_col_type_mapping and attrid in entity_col_type_mapping[parent_entity]:
                    sa_type = entity_col_type_mapping[parent_entity][attrid][0]
                    out_scores = "Foreign key"

            if sa_type is None:
                # CHANGED: construct extra IR-derived context for Ollama fallback
                extra_context = {
                    "original_attribute_name": raw_name,
                    "resolved_final_column_name": final_name,
                    "is_foreign_key": is_fk,
                    "is_primary_key_candidate": attrid in entity.primary_key,
                    "referenced_entity_name": referenced_entity_name,
                    "source_kind": "foreign_key_attribute" if is_fk else "entity_attribute",
                    "language_hint": "mk_latin_or_mk_cyrillic_or_english"
                }

                inferred_type, out_scores = infer_sqlite_type_nli(
                    final_name,
                    table_name=table_name,
                    sibling_attributes=sibling_attribute_names,
                    extra_context=extra_context,
                    return_scores=True
                )
                sa_type = get_sa_type(inferred_type)

            col_obj = Column(final_name, sa_type, nullable=True)
            sa_columns.append(col_obj)

            if attrid not in entity_col_type_mapping[entity_id]:
                entity_col_type_mapping[entity_id][attrid] = []
            entity_col_type_mapping[entity_id][attrid].append(sa_type)

            if attrid not in entity_col_mapping[entity_id]:
                entity_col_mapping[entity_id][attrid] = []
            entity_col_mapping[entity_id][attrid].append(final_name)

            if attrid not in entity_col_out_scores_mapping[entity_id]:
                entity_col_out_scores_mapping[entity_id][attrid] = []
            if out_scores is not None:
                entity_col_out_scores_mapping[entity_id][attrid].append(out_scores)

        table_args = []

        pk_col_names = []
        pk_usage_counter = {}

        for pkid in entity.primary_key:
            if pkid in entity_col_mapping[entity_id]:
                available_cols = entity_col_mapping[entity_id][pkid]
                idx = pk_usage_counter.get(pkid, 0)
                if idx < len(available_cols):
                    pk_col_names.append(available_cols[idx])
                    pk_usage_counter[pkid] = idx + 1

        if pk_col_names:
            table_args.append(PrimaryKeyConstraint(*pk_col_names))
            for col in sa_columns:
                if col.name in pk_col_names:
                    col.nullable = False

        fk_usage_counter = {}

        for fk in entity.foreign_keys:
            ref_entity = entity_obj_map.get(fk.entity)
            if not ref_entity:
                continue

            ref_table_name = resolve_table_name(ref_entity, cells_map)

            local_cols = []
            remote_cols = []

            for attrid in fk.attributes:
                if attrid in entity_col_mapping[entity_id]:
                    available = entity_col_mapping[entity_id][attrid]
                    idx = fk_usage_counter.get(attrid, 0)
                    if idx < len(available):
                        local_cols.append(available[idx])
                        fk_usage_counter[attrid] = idx + 1

                if fk.entity in entity_col_mapping and attrid in entity_col_mapping[fk.entity]:
                    parent_col_name = entity_col_mapping[fk.entity][attrid][0]
                    remote_cols.append(f"{ref_table_name}.{parent_col_name}")
                else:
                    remote_name = get_human_readable_name(attrid, cells_map)
                    remote_cols.append(f"{ref_table_name}.{remote_name}")

            if local_cols and remote_cols and len(local_cols) == len(remote_cols):
                table_args.append(ForeignKeyConstraint(local_cols, remote_cols))
                if fk.not_null:
                    for col in sa_columns:
                        if col.name in local_cols:
                            col.nullable = False

        Table(table_name, metadata, *sa_columns, *table_args, extend_existing=True)

    return metadata, entity_col_mapping, entity_col_type_mapping, entity_col_out_scores_mapping