#!/usr/bin/env python3
"""Generate the multilingual evaluation corpus fixture.

Six languages, one topic per document, with queries in the document's own
language plus a set of cross-language queries. Written as a script so the
fixture is reproducible rather than hand-maintained.
"""
import json
from pathlib import Path

# (id, language, text, topic)
DOCUMENTS = [
    # English
    ("en-terns", "en", "Arctic terns migrate from the Arctic to the Antarctic and back each year, "
     "the longest migration of any animal. A single bird may cover ninety thousand kilometres.", "terns"),
    ("en-vents", "en", "Hydrothermal vents on the ocean floor host ecosystems that depend on "
     "chemosynthesis rather than sunlight. Tube worms and vent crabs cluster around the plumes.", "vents"),
    ("en-press", "en", "The printing press, developed by Johannes Gutenberg around 1440, used movable "
     "metal type and an oil based ink to reproduce text at a scale manuscripts could not match.", "press"),
    ("en-kelp", "en", "Kelp forests grow in cold coastal waters and shelter sea otters, rockfish and "
     "urchins. A single frond can grow half a metre in a day given enough light.", "kelp"),
    # Russian
    ("ru-terns", "ru", "Полярные крачки ежегодно мигрируют из Арктики в Антарктику и обратно. "
     "Это самая длинная миграция среди животных: одна птица преодолевает девяносто тысяч километров.", "terns"),
    ("ru-vents", "ru", "Гидротермальные источники на дне океана поддерживают экосистемы, основанные "
     "на хемосинтезе, а не на солнечном свете. Вокруг них собираются трубчатые черви и крабы.", "vents"),
    ("ru-press", "ru", "Печатный станок Иоганна Гутенберга, созданный около 1440 года, использовал "
     "подвижные металлические литеры и масляную краску для печати текста.", "press"),
    # Turkish
    ("tr-terns", "tr", "Kuzey sumruları her yıl Kuzey Kutbu ile Güney Kutbu arasında göç eder. "
     "Bu, hayvanlar arasındaki en uzun göçtür ve bir kuş doksan bin kilometre yol alabilir.", "terns"),
    ("tr-vents", "tr", "Okyanus tabanındaki hidrotermal bacalar, güneş ışığı yerine kemosenteze "
     "dayanan ekosistemlere ev sahipliği yapar. Işık burada hiç ulaşmaz.", "vents"),
    ("tr-press", "tr", "Johannes Gutenberg'in 1440 civarında geliştirdiği matbaa, hareketli metal "
     "harfler ve yağlı mürekkep kullanarak metni büyük ölçekte çoğalttı.", "press"),
    # Arabic
    ("ar-terns", "ar", "تهاجر الخرشنة القطبية سنويا من القطب الشمالي إلى القطب الجنوبي والعودة. "
     "وهي أطول هجرة بين الحيوانات، إذ يقطع الطائر الواحد تسعين ألف كيلومتر.", "terns"),
    ("ar-vents", "ar", "تحتضن الفوهات الحرارية المائية في قاع المحيط أنظمة بيئية تعتمد على التخليق "
     "الكيميائي بدلا من ضوء الشمس. تتجمع الديدان الأنبوبية حول الأعمدة الحارة.", "vents"),
    ("ar-press", "ar", "استخدمت مطبعة يوهانس غوتنبرغ حوالي عام 1440 حروفا معدنية متحركة وحبرا زيتيا "
     "لإعادة إنتاج النصوص على نطاق واسع.", "press"),
    # Chinese
    ("zh-terns", "zh", "北极燕鸥每年在北极和南极之间往返迁徙，是所有动物中迁徙距离最长的物种，"
     "一只鸟可以飞行九万公里。", "terns"),
    ("zh-vents", "zh", "海底热液喷口孕育了依靠化能合成而非阳光的生态系统，管虫和螃蟹聚集在喷口周围。", "vents"),
    ("zh-press", "zh", "约翰内斯·古腾堡在一四四零年左右发明的印刷机使用活字金属和油性墨水，"
     "大规模复制文本。", "press"),
    # Japanese
    ("ja-terns", "ja", "キョクアジサシは毎年北極と南極の間を往復して渡りを行い、動物の中で最も長い"
     "距離を移動します。一羽が九万キロメートルを飛ぶこともあります。", "terns"),
    ("ja-vents", "ja", "深海の熱水噴出孔には、太陽光ではなく化学合成に依存する生態系が広がっており、"
     "チューブワームやカニが集まります。", "vents"),
    ("ja-press", "ja", "ヨハネス・グーテンベルクが一四四〇年頃に開発した活版印刷機は、"
     "可動式の金属活字と油性インクを用いて文章を大量に複製しました。", "press"),
]

# (id, text, slice, language-of-query, relevant-topic, restrict-to-language)
QUERIES = [
    ("q-en-terns", "arctic tern migration distance", "en", "terns", "en"),
    ("q-en-vents", "hydrothermal vents chemosynthesis", "en", "vents", "en"),
    ("q-en-press", "Gutenberg movable type printing", "en", "press", "en"),
    ("q-en-kelp", "kelp forest sea otters", "en", "kelp", "en"),
    ("q-ru-terns", "миграции полярных крачек", "ru", "terns", "ru"),
    ("q-ru-vents", "гидротермальные источники хемосинтез", "ru", "vents", "ru"),
    ("q-ru-press", "печатный станок Гутенберга", "ru", "press", "ru"),
    ("q-tr-terns", "kuzey sumrularının göçü", "tr", "terns", "tr"),
    ("q-tr-vents", "hidrotermal bacalar kemosentez", "tr", "vents", "tr"),
    ("q-tr-press", "GUTENBERG MATBAA HAREKETLİ HARF", "tr", "press", "tr"),
    ("q-ar-terns", "هِجرة الخرشنة القطبية", "ar", "terns", "ar"),
    ("q-ar-vents", "الفوهات الحرارية المائية", "ar", "vents", "ar"),
    ("q-ar-press", "مطبعة غوتنبرغ الحروف المعدنية", "ar", "press", "ar"),
    ("q-zh-terns", "北极燕鸥迁徙", "zh", "terns", "zh"),
    ("q-zh-vents", "海底热液喷口", "zh", "vents", "zh"),
    ("q-zh-single", "鸥", "zh-single-char", "terns", "zh"),
    ("q-ja-terns", "キョクアジサシ 渡り", "ja", "terns", "ja"),
    ("q-ja-vents", "熱水噴出孔 化学合成", "ja", "vents", "ja"),
    ("q-ja-press", "グーテンベルク 活版印刷", "ja", "press", "ja"),
]

# Cross-language: an English query that should reach the same topic in every
# language. Only a genuinely multilingual embedding space can answer these; the
# lexical channel cannot, which is exactly what makes them a useful slice.
CROSS_LANGUAGE = [
    ("q-x-terns", "arctic tern migration", "terns"),
    ("q-x-vents", "deep sea hydrothermal vents", "vents"),
    ("q-x-press", "invention of the printing press", "press"),
]

documents = [
    {"id": doc_id, "text": text, "language": language, "kind": "originalText",
     "metadata": {"language": language, "topic": topic}}
    for doc_id, language, text, topic in DOCUMENTS
]

queries = []
for query_id, text, slice_name, topic, language in QUERIES:
    relevance = {
        doc_id: 3
        for doc_id, doc_language, _, doc_topic in DOCUMENTS
        if doc_topic == topic and doc_language == language
    }
    queries.append({
        "id": query_id, "text": text, "slice": slice_name, "relevance": relevance,
        "filterKey": "language", "filterValue": language,
    })

for query_id, text, topic in CROSS_LANGUAGE:
    relevance = {
        doc_id: 3 for doc_id, _, _, doc_topic in DOCUMENTS if doc_topic == topic
    }
    queries.append({
        "id": query_id, "text": text, "slice": "cross-language", "relevance": relevance,
    })

corpus = {"name": "multilingual-v1", "documents": documents, "queries": queries}

out = Path(__file__).with_name("multilingual-corpus.json")
out.write_text(json.dumps(corpus, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"wrote {out} — {len(documents)} documents, {len(queries)} queries")
