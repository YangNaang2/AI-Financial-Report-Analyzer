from data_processor import document_from_text
from inference import analyze_document
from presentation import unprocessed_ranges, expression_changes


def test_missing_ranges_exactly_cover_the_unanalyzed_source():
    doc=document_from_text('원문 첫 문단\n\n실적 부진 둘째 문단\n\n지연 위험 마지막 문단')
    analysis=analyze_document(doc,settings={'max_segments':1})
    missing=unprocessed_ranges(doc,analysis)
    assert missing and missing[0]['text'].startswith('실적 부진')
    assert sum(len(r['text']) for r in missing)+analysis['metrics']['analyzed_chars']==len(doc['text'])
    for r in missing: assert r['text']==doc['pages'][0]['text'][r['start']:r['end']]
    assert unprocessed_ranges(doc,analyze_document(doc))==[]


def test_expression_changes_are_presence_not_causal_claims():
    a=analyze_document(document_from_text('실적 부진 우려'))
    b=analyze_document(document_from_text('실적 둔화 우려'))
    assert expression_changes(a,b)=={'added':['둔화'],'removed':['부진'],'shared':['우려']}
