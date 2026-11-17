"""Binding geometry preserves coordinates and bounds dense annotation labels."""
import json
import shutil
import subprocess

import pytest

from aso_agents.report_binding_map import BINDING_MAP_SCRIPT


def javascript(expression):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is needed to check binding-map geometry')
    result = subprocess.run([node, '-e', BINDING_MAP_SCRIPT +
        '\nconsole.log(JSON.stringify(' + expression + '));'],
        text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


def test_regions_follow_every_binding_interval_and_leave_inputs_unchanged():
    rows = [{'intervals': [{'start': 1, 'end': 20}, {'start': 1001, 'end': 1020}]},
            {'intervals': [{'start': 6, 'end': 25}]},
            {'intervals': [{'start': 1090, 'end': 1110}]}]
    result = javascript('(()=>{const rows=' + json.dumps(rows) +
                        ';return {regions:bindingRegions(rows),rows};})()')
    assert result == {'regions': [[1, 37], [989, 1122]], 'rows': rows}


def test_requested_regulatory_sites_extend_map_and_tracks_stay_bounded():
    rows = [{'intervals': [{'start': 100, 'end': 119}]}]
    annotations = [{'kind': 'rbp', 'label': 'RBP '+str(i), 'start': 500+i, 'end': 519+i,
                    'evidence': 'eCLIP peak'} for i in range(1000)]
    annotations.append({'kind': 'mirna', 'label': 'miRNA', 'start': 90, 'end': 97, 'evidence': 'Predicted site'})
    expression = '(()=>{const rows='+json.dumps(rows)+',sites='+json.dumps(annotations)+';'
    expression += 'const regions=bindingRegions(rows,sites),axes=bindingAxes(regions,800,16,16);'
    expression += 'return {regions,tracks:bindingAnnotationTracks(sites,axes)};})()'
    result = javascript(expression)
    assert result['regions'] == [[78, 131], [488, 1530]]
    assert len(result['tracks']) == 2
    assert len(result['tracks'][0]['entries']) == 1000
    for track in result['tracks']:
        for entry in track['entries']:
            assert entry['width'] > 0 and entry['x'] + entry['width'] <= 784.001


@pytest.mark.parametrize('width', [320, 600, 1200])
def test_crowded_site_labels_fit_without_overlap_and_prioritize_the_target(width):
    result = javascript('bindingLabels(' + json.dumps([
        {'x': 20 + i * (width - 40) / 30, 'label': 'Nearby splice site ' + str(i), 'target': i == 15}
        for i in range(31)]) + f', text=>text.length*6, 16, {width - 16})')
    assert len(result) < 31  # Remaining exact labels are accessible in the site table.
    assert result[0]['target'] and result[0]['text'].startswith('Target · ')
    for i, label in enumerate(result):
        assert 16 <= label['left'] <= label['left'] + label['width'] <= width - 16
        for other in result[i + 1:]:
            if label['lane'] == other['lane']:
                assert (label['left'] + label['width'] < other['left'] or
                        other['left'] + other['width'] < label['left'])


def test_long_target_labels_keep_the_exact_source_and_use_a_bounded_display():
    label = 'An unusually long saved splice-site description ' * 10
    result = javascript('bindingLabels(' + json.dumps([
        {'x': 150, 'label': label, 'target': True}]) + ', text=>text.length*6, 16, 284)')[0]
    assert result['label'] == label
    assert result['text'].startswith('Target · ') and result['text'].endswith('…')
    assert result['width'] <= 268


def test_region_axes_share_a_scale_and_coordinate_ticks_do_not_collide():
    result = javascript('(()=>{const axes=bindingAxes([[156137533,156137955],'
                        '[156138276,156139292]],758,16,16);'
                        'return {axes,ticks:bindingTicks(axes,text=>text.length*6)};})()')
    axes, ticks = result['axes'], result['ticks']
    assert axes[0]['width'] / 423 == pytest.approx(axes[1]['width'] / 1017)
    assert axes[0]['x'] == 16 and axes[-1]['x'] + axes[-1]['width'] == pytest.approx(742)
    assert ticks and ticks[0]['value'] == 156137533
    for i, tick in enumerate(ticks):
        assert any(axis['a'] <= tick['value'] <= axis['b'] for axis in axes)
        for other in ticks[i + 1:]:
            assert tick['right'] < other['left'] or other['right'] < tick['left']


def test_tiny_regions_omit_ticks_that_cannot_fit():
    ticks = javascript('bindingTicks(bindingAxes([[100000001,100000002],'
                       '[200000001,200010001]],600,16,16),text=>text.length*6)')
    assert all(tick['value'] >= 200000001 for tick in ticks)


def test_large_overview_keeps_positive_regions_within_a_bounded_canvas():
    axes = javascript('bindingAxes(Array.from({length:1000},(_,i)=>[i*10000+1,i*10000+21]),4096,16,16)')
    assert len(axes) == 1000
    assert all(axis['width'] > 0 for axis in axes)
    assert axes[-1]['x'] + axes[-1]['width'] == pytest.approx(4080)
