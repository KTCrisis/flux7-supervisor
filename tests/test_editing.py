"""Tests for editing the configuration and question sets through the admin API."""

import json

import httpx
import pytest

from sup7.admin import create_admin_app
from sup7.config import load_config
from sup7.editing import MASK, ConfigFiles, EditError, mask, sha, split_restart, unmask
from sup7.questions import socle_text
from sup7.runner import SupervisorRunner

CONFIG = """# production config
mesh:
  url: http://localhost:9090
memory:
  enabled: false
  token: mem-secret
admin:
  enabled: true
  token: admin-secret
evaluator:
  provider: none
  chain:
    - provider: jev
      jev:
        questions: [{qdir}/*.yaml]
rules:
  - name: reads
    condition: "tool contains read"
    action: approve
"""

FINANCE = """pack: finance
applies_to: ["tool:bank.*"]
questions:
  payment:
    type: noul
    group: danger
    instructions: The call initiates a payment.
"""


@pytest.fixture
def setup(tmp_path):
    qdir = tmp_path / "questions"
    qdir.mkdir()
    (qdir / "socle.yaml").write_text(socle_text())
    path = tmp_path / "sup7.yaml"
    path.write_text(CONFIG.format(qdir=qdir))
    return path, qdir


def test_tokens_are_masked_and_put_back():
    text = "admin:\n  token: s3cret\nmemory:\n  token: \"\"\n"
    masked = mask(text)
    assert "s3cret" not in masked and MASK in masked and 'token: ""' in masked
    assert unmask(masked.replace("admin:", "admin:  # edited"), text).count("s3cret") == 1
    with pytest.raises(EditError, match="masked tokens"):
        unmask(masked + "other:\n  token: " + MASK + "\n", text)


def test_files_lists_config_and_question_sets(setup):
    path, qdir = setup
    files = ConfigFiles(str(path), load_config(str(path)))
    assert [f.id for f in files.files()] == ["config", "questions/socle.yaml"]
    text, fingerprint = files.read("config")
    assert "admin-secret" not in text and "mem-secret" not in text and fingerprint == sha(path.read_text())


def test_write_config_validates_guards_backs_up(setup):
    path, qdir = setup
    files = ConfigFiles(str(path), load_config(str(path)))
    text, fingerprint = files.read("config")
    edited = text.replace("action: approve", "action: escalate")
    with pytest.raises(EditError) as e:  # stale fingerprint
        files.write("config", edited, "000000000000")
    assert e.value.status == 409
    with pytest.raises(EditError) as e:  # invalid rule, nothing written
        files.write("config", text.replace("tool contains read", "no operator here"), fingerprint)
    assert e.value.status == 400 and "rule 'reads'" in e.value.message
    config, change = files.write("config", edited, fingerprint)
    on_disk = path.read_text()
    assert "action: escalate" in on_disk and "admin-secret" in on_disk  # token put back
    assert "# production config" in on_disk  # comments survive: text, not re-serialized
    assert change["sha_before"] == fingerprint and change["sha_after"] == sha(on_disk)
    assert (tmp := path.parent / change["backup"].split("/")[-1]).exists() and "action: approve" in tmp.read_text()
    assert config.rules[0].action == "escalate"


def test_bad_question_set_is_refused_and_file_untouched(setup):
    path, qdir = setup
    files = ConfigFiles(str(path), load_config(str(path)))
    before = (qdir / "socle.yaml").read_text()
    with pytest.raises(EditError, match="questions: .*group must be"):
        files.write("questions/socle.yaml", before.replace("group: danger", "group: risk", 1), sha(before))
    assert (qdir / "socle.yaml").read_text() == before


def test_new_pack_is_created_in_a_configured_glob(setup):
    path, qdir = setup
    files = ConfigFiles(str(path), load_config(str(path)))
    with pytest.raises(EditError) as e:
        files.write("questions/finance.yaml", FINANCE, "123")
    assert e.value.status == 409
    files.write("questions/finance.yaml", FINANCE, "new")
    assert (qdir / "finance.yaml").exists()
    assert "questions/finance.yaml" in [f.id for f in files.files()]
    with pytest.raises(EditError) as e:  # outside the glob / odd name
        files.write("questions/../evil.yaml", FINANCE, "new")
    assert e.value.status == 400


def test_duplicate_question_in_new_pack_is_refused(setup):
    path, qdir = setup
    files = ConfigFiles(str(path), load_config(str(path)))
    dup = "pack: dup\nquestions:\n  deletes: {type: noul, group: danger, instructions: x}\n"
    with pytest.raises(EditError, match="defined twice"):
        files.write("questions/dup.yaml", dup, "new")
    assert not (qdir / "dup.yaml").exists()


def test_start_only_sections_wait_for_a_restart(setup):
    path, _ = setup
    old = load_config(str(path))
    new = old.model_copy(update={"mesh": old.mesh.model_copy(update={"url": "http://other:9090"}),
                                 "project_dirs": ["/p"]})
    applied, pending = split_restart(old, new)
    assert pending == ["mesh"] and applied.mesh.url == "http://localhost:9090" and applied.project_dirs == ["/p"]


# ── admin routes ──────────────────────────────────────────────
def _client(runner, token):
    app = create_admin_app(runner, token=token)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://sup7")


async def test_put_needs_a_configured_token(setup):
    path, _ = setup
    runner = SupervisorRunner(load_config(str(path)), config_path=str(path))
    async with _client(runner, "") as c:
        r = await c.put("/files/config", content="x", headers={"If-Match": "x"})
    assert r.status_code == 403 and "admin.token" in r.json()["error"]


async def test_edit_through_the_api_reloads_without_restart(setup, tmp_path):
    path, qdir = setup
    path.write_text(path.read_text() + f"decision_log: {tmp_path / 'decisions.jsonl'}\n")
    runner = SupervisorRunner(load_config(str(path)), config_path=str(path))
    runner._logger.open()
    auth = {"Authorization": "Bearer admin-secret"}
    async with _client(runner, "admin-secret") as c:
        assert (await c.put("/files/config", content="x", headers={"If-Match": "x"})).status_code == 401
        listing = (await c.get("/files", headers=auth)).json()["files"]
        assert [f["id"] for f in listing] == ["config", "questions/socle.yaml"]
        r = await c.get("/files/config", headers=auth)
        assert "admin-secret" not in r.text
        edited = r.text.replace("action: approve", "action: escalate")
        r = await c.put("/files/config", content=edited, headers={**auth, "If-Match": r.headers["etag"]})
        assert r.status_code == 200, r.text
        assert r.json()["restart_required"] == []
        r = await c.put("/files/questions/finance.yaml", content=FINANCE, headers={**auth, "If-Match": "new"})
        assert r.status_code == 200, r.text
        summary = (await c.get("/config", headers=auth)).json()
    runner._logger.close()
    assert summary["rules"][0]["action"] == "escalate"  # applied without restart
    assert [p["name"] for p in summary["questions"][0]["packs"]] == ["finance", "socle"]
    events = [json.loads(line) for line in open(tmp_path / "decisions.jsonl")]
    assert [e["file"] for e in events] == ["config", "questions/finance.yaml"]
    assert all(e["type"] == "config_change" and e["by"] == "admin-api" for e in events)


def test_write_keeps_the_file_mode(setup):
    # sup7.yaml holds admin.token: an edit must not turn 600 into 644
    import os
    import stat
    path, qdir = setup
    os.chmod(path, 0o600)
    files = ConfigFiles(str(path), load_config(str(path)))
    text, fingerprint = files.read("config")
    files.write("config", text + "# edited\n", fingerprint)
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    files.write("questions/finance.yaml", FINANCE, "new")
    assert stat.S_IMODE(os.stat(qdir / "finance.yaml").st_mode) == 0o600


def test_two_edits_in_one_second_keep_both_backups(setup):
    path, qdir = setup
    files = ConfigFiles(str(path), load_config(str(path)))
    original = path.read_text()
    for n in (1, 2):
        text, fingerprint = files.read("config")
        files.write("config", text + f"# edit {n}\n", fingerprint)
    backups = sorted(path.parent.glob("sup7.yaml.bak-*"))
    assert len(backups) == 2
    assert backups[0].read_text() == original  # the first backup is the original version
