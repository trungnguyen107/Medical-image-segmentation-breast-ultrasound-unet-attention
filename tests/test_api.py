"""Kiểm thử API của SonoSeg (chạy: python -m pytest -q tests). Dùng CSDL tạm để không ảnh hưởng dữ liệu thật."""
import base64, io, os, sys, tempfile, time, zipfile, shutil

import cv2
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import db  # noqa: E402

os.environ["SONOSEG_NO_STU"] = "1"   # không tải ảnh thật khi kiểm thử
_tmp = tempfile.mkdtemp(prefix="sonoseg_test_")
db.DATA = _tmp; db.DB_PATH = os.path.join(_tmp, "app.db")
import dataset_tools as T  # noqa: E402
T.SAMPLE_PLAN = [("train", 24, 1000, "A"), ("val", 6, 50000, "A"), ("test", 10, 90000, "A"), ("ext", 6, 130000, "B")]
import app as A  # noqa: E402
import trainer  # noqa: E402
trainer.USER_W = os.path.join(_tmp, "weights_user")


@pytest.fixture(scope="module")
def client():
    yield A.app.test_client()
    shutil.rmtree(_tmp, ignore_errors=True)


def dec(u):
    return base64.b64decode(u.split(",")[1])


def test_health_models_stats(client):
    assert client.get("/api/health").get_json()["status"] == "ok"
    ms = client.get("/api/models").get_json()
    assert [m["id"] for m in ms][:4] == ["M1", "M2", "M3", "M4"]
    st = client.get("/api/stats").get_json()
    assert sum(r["n"] for r in st["images"]) == 46


def test_images_and_detail(client):
    j = client.get("/api/images?split=test&per=5").get_json()
    assert j["total"] == 10 and len(j["rows"]) == 5 and j["rows"][0]["mask_url"]
    assert client.get(j["rows"][0]["url"]).status_code == 200


def test_segment_db_image_saves_case(client):
    iid = client.get("/api/images?split=test&kind=benign&per=1").get_json()["rows"][0]["id"]
    j = client.post("/api/segment", data=dict(image_id=str(iid), model="M4", px_mm="0,1"), content_type="multipart/form-data").get_json()
    assert j["metrics"]["dice"] > 0.5 and len(j["gates"]) == 4 and j["case_id"]
    c = j["confidence"]
    assert c["level"] in ("high", "medium", "low") and 0 <= c["stability"] <= 1 and c["loc"]["depth_mm"] is not None and j["uncertainty"]
    cs = client.get("/api/cases").get_json()
    assert cs["total"] >= 1 and cs["rows"][0]["model_id"] == "M4"
    assert client.get("/api/cases.csv").status_code == 200


def test_segment_upload_rgb_and_errors(client):
    img = (np.random.default_rng(0).random((300, 420, 3)) * 255).astype(np.uint8)
    j = client.post("/api/segment", data=dict(model="M1", save="0", image=(io.BytesIO(cv2.imencode(".jpg", img)[1].tobytes()), "x.jpg")),
                    content_type="multipart/form-data").get_json()
    assert j["size"] == [420, 300] and j["attention"] is None and j["case_id"] is None
    assert client.post("/api/segment", data=dict(image=(io.BytesIO(b"xx"), "x.png")), content_type="multipart/form-data").status_code == 400
    assert client.post("/api/segment", data={}, content_type="multipart/form-data").status_code == 400


def test_compare_robustness_batch(client):
    s = client.get("/api/sample?type=malignant&seed=5").get_json()
    img, gt = dec(s["image"]), dec(s["gt"])
    f = lambda **k: dict(k, image=(io.BytesIO(img), "a.png"), gt=(io.BytesIO(gt), "g.png"))
    j = client.post("/api/compare", data=f(), content_type="multipart/form-data").get_json()
    assert len(j["rows"]) >= 4 and 0 <= j["agreement"] <= 100
    j = client.post("/api/robustness", data=f(kind="gauss", mode="sweep", models="M1,M4"), content_type="multipart/form-data").get_json()
    assert set(j["series"]) == {"M1", "M4"} and len(j["series"]["M1"]) == 5
    data = {"model": "M2", "images": [(io.BytesIO(img), "c1.png"), (io.BytesIO(gt), "c1_mask.png"), (io.BytesIO(b"x"), "bad.png")]}
    j = client.post("/api/batch", data=data, content_type="multipart/form-data").get_json()
    assert j["summary"]["ok"] == 1 and j["summary"]["with_gt"] == 1
    assert "ket_qua.csv" in zipfile.ZipFile(io.BytesIO(client.get(j["zip"]).data)).namelist()


def test_upload_to_db_and_delete(client):
    s = client.get("/api/sample?type=benign&seed=9").get_json()
    data = {"kind": "benign", "images": [(io.BytesIO(dec(s["image"])), "ca1.png"), (io.BytesIO(dec(s["gt"])), "ca1_mask.png")]}
    j = client.post("/api/images/upload", data=data, content_type="multipart/form-data").get_json()
    assert j["added"] == 1
    iid = j["ids"][0]
    assert client.post(f"/api/images/{iid}/split", json={"split": "train"}).status_code == 200
    assert client.delete(f"/api/images/{iid}").status_code == 200
    sample_id = client.get("/api/images?split=test&per=1").get_json()["rows"][0]["id"]
    assert client.delete(f"/api/images/{sample_id}").status_code == 400


def test_evaluate(client):
    j = client.post("/api/evaluate", json={"models": ["M1"], "split": "test"}).get_json()
    assert j[0]["n"] == 10 and 0 < j[0]["dice"] <= 1
    ev = client.get("/api/evals").get_json()
    assert any(e["model_id"] == "M1" and e["split"] == "test" for e in ev)
    r = client.post("/api/evaluate/robustness", json={"models": ["M1"], "kind": "contrast", "n": 5}).get_json()
    assert len(r["series"]["M1"]) == 5


def test_train_short_run(client):
    r = client.post("/api/train", json={"model_type": "M1", "epochs": 1, "n_train": 24, "bs": 8, "size": 64, "base": 4})
    assert r.status_code == 200, r.get_json()
    rid = r.get_json()["run_id"]
    assert client.post("/api/train", json={"model_type": "M1", "epochs": 1}).status_code == 409   # đang bận
    t0 = time.time()
    while trainer.busy() and time.time() - t0 < 120:
        time.sleep(0.5)
    run = next(x for x in client.get("/api/runs").get_json()["runs"] if x["id"] == rid)
    assert run["status"] == "done" and run["test_dice"] is not None and len(run["history"]) == 1
    assert any(m["id"] == f"R{rid}" for m in client.get("/api/models").get_json())
    assert client.delete(f"/api/runs/{rid}").status_code == 200


def test_anatomy_m4_and_baseline(client):
    iid = client.get("/api/images?split=test&kind=malignant&per=1").get_json()["rows"][0]["id"]
    j = client.post("/api/anatomy", data=dict(image_id=str(iid), model="M4"), content_type="multipart/form-data").get_json()
    assert j["flags"] == dict(ag=True, cbam=True, ds=True)
    assert len(j["enc"]) == 4 and len(j["gates"]) == 4 and len(j["heads"]) == 4 and j["heads"][-1]["main"]
    assert j["cbam"]["n"] == len(j["cbam"]["channel"]) and all(0 <= v <= 1 for v in j["cbam"]["channel"])
    assert all(g["focus"] > 0 for g in j["gates"]) and 0 <= j["metrics"]["dice"] <= 1
    b = client.post("/api/anatomy", data=dict(image_id=str(iid), model="M1"), content_type="multipart/form-data").get_json()
    assert b["cbam"] is None and b["gates"] == [] and len(b["heads"]) == 1


def test_ablation_matrix_and_image(client):
    rows = client.get("/api/ablation").get_json()
    ids = {r["id"] for r in rows}
    assert {"M1", "M2", "M4"} <= ids and all(r["test"] and r["ext"] for r in rows)
    m4 = next(r for r in rows if r["id"] == "M4"); assert (m4["ag"], m4["cbam"], m4["ds"]) == (1, 1, 1)
    # biến thể ablation không lẫn vào danh sách mô hình hay bảng đánh giá
    assert not ({m["id"] for m in client.get("/api/models").get_json()} & set(A.ABL_ONLY))
    assert not ({e["model_id"] for e in client.get("/api/evals").get_json()} & set(A.ABL_ONLY))
    iid = client.get("/api/images?split=test&kind=benign&per=1").get_json()["rows"][0]["id"]
    j = client.post("/api/ablation/image", data=dict(image_id=str(iid), ref="M1"), content_type="multipart/form-data").get_json()
    assert len(j["rows"]) == len(rows) and j["diffs"]["M1"]["fixed"] == 0 and j["diffs"]["M1"]["broke"] == 0


def test_import_stu_real_split(client):
    """Nhập ảnh thật từ thư mục data/external/STU-Hospital nếu có sẵn (không tải mạng trong kiểm thử)."""
    if not os.path.isdir(T.STU_DIR):
        pytest.skip("chưa có data/external/STU-Hospital")
    n = T.import_stu(log=lambda *a: None)
    assert n == 42
    j = client.get("/api/images?split=real&per=5").get_json()
    assert j["total"] == 42 and j["rows"][0]["mask_url"]
    r = A.trainer.evaluate("M4", "real")
    assert r["n"] == 42 and 0 < r["dice"] < 1
