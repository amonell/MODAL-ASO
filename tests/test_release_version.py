"""Reject release metadata drift before a tag or package is published."""
from pathlib import Path
import shutil

import pytest

from ci.check_version import check_version, release_notes


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def release_tree(tmp_path):
    for name in ['pyproject.toml', 'uv.lock', 'CITATION.cff', 'CHANGELOG.md', 'aso_agents/__init__.py']:
        dest = tmp_path/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, dest)
    return tmp_path


def test_repository_release_metadata_agrees():
    version = check_version(ROOT)
    assert check_version(ROOT, f'v{version}') == version
    assert release_notes(ROOT, version).strip()


@pytest.mark.parametrize('name', ['aso_agents/__init__.py', 'uv.lock', 'CITATION.cff'])
def test_stale_version_blocks_release(release_tree, name):
    version = check_version(release_tree)
    file = release_tree/name
    file.write_text(file.read_text().replace(version, '999.0.0'))
    with pytest.raises(ValueError, match='expected'):
        check_version(release_tree)


@pytest.mark.parametrize('tag', ['main', 'v999.0.0', '1.0.0'])
def test_tag_must_identify_the_package_version(tag):
    with pytest.raises(ValueError, match='does not match'):
        check_version(ROOT, tag)


def test_missing_release_notes_block_release(release_tree):
    (release_tree/'CHANGELOG.md').write_text('# Changelog\n\n## Unreleased\nNext changes.\n')
    with pytest.raises(ValueError, match='release notes'):
        check_version(release_tree)


def test_release_notes_do_not_include_unreleased_or_other_versions(tmp_path):
    (tmp_path/'CHANGELOG.md').write_text('# Changelog\n\n## Unreleased\nFuture.\n\n## 1.2.0 — 2026-11-12\nNew release.\n\n## 1.1.0\nPrevious.\n')
    assert release_notes(tmp_path, '1.2.0') == 'New release.\n'
