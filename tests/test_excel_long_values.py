import json
import openpyxl
import pytest
import run_goal_intake as runtime
from aso_agents.excel_values import EXCEL_TEXT_UNITS, excel_text_parts, excel_text_units


@pytest.mark.parametrize('text',['x'*32768,'🧬'*17000,'a'*29999+'🧬'+'b'*40000])
def test_text_parts_roundtrip_without_splitting_unicode(text):
    parts=excel_text_parts(text)
    assert ''.join(parts)==text
    assert all(excel_text_units(part)<=EXCEL_TEXT_UNITS for part in parts)


def test_long_structured_values_are_linked_and_complete_across_grouped_sheets(tmp_path):
    raw=[{'label':f'site_{i}','status':'not_evaluated','scope':'unknown '+('x'*500)} for i in range(100)]
    expected=json.dumps(raw,ensure_ascii=True)
    path=tmp_path/'data.xlsx'
    result,error=runtime._write_aso_excel([{'ID':'A','evidence':raw,'ordinary':'unchanged'},
                                          {'ID':'B','evidence':'🧬'*17000,'ordinary':0}],str(path),
        sheet_groups=[('Evidence',['evidence']),('Long values',['ordinary'])])
    assert result==str(path) and error is None
    book=openpyxl.load_workbook(path,data_only=False)
    try:
        details=book['Long values 2']
        decoded={}
        for value_id,all_row,field,part,total,text in list(details.values)[1:]:
            assert excel_text_units(text)<=EXCEL_TEXT_UNITS
            decoded.setdefault((all_row,field),[]).append((part,total,text))
        actual={key:''.join(v[2] for v in sorted(parts)) for key,parts in decoded.items()}
        assert json.loads(actual[(2,'evidence')])==raw
        assert actual[(2,'evidence')]==expected
        assert actual[(3,'evidence')]=='🧬'*17000
        assert book['All']['B2'].value==book['Evidence']['A2'].value
        assert book['All']['B2'].hyperlink.target.startswith("#'Long values 2'!")
        assert book['All']['C2'].value=='unchanged' and book['All']['C3'].value==0
    finally:book.close()
