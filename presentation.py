"""Pure presentation helpers; every missing range points into saved source text."""
def unprocessed_ranges(document, analysis):
    result=[]
    for page in document.get('pages',[]):
        text=page['text']; intervals=sorted((s['start'],s['end']) for s in analysis.get('segments',[]) if s['page']==page['number'])
        cursor=0
        for start,end in intervals:
            if start>cursor:
                result.append({'page':page['number'],'start':cursor,'end':start,'text':text[cursor:start]})
            cursor=max(cursor,end)
        if cursor<len(text): result.append({'page':page['number'],'start':cursor,'end':len(text),'text':text[cursor:]})
    return result


def expression_changes(a,b):
    def terms(analysis):
        return {hit['keyword'] for segment in analysis.get('segments',[]) for hit in segment.get('rule_hits',[])}
    old,new=terms(a),terms(b)
    return {'added':sorted(new-old),'removed':sorted(old-new),'shared':sorted(old&new)}
