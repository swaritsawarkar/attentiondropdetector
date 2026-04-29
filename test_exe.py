"""Test the EXE server by uploading a video and checking results."""
import requests, time, json, sys

print("Uploading FACESCREEN.mp4 to EXE server...")
with open("FACESCREEN.mp4", "rb") as f:
    r = requests.post("http://localhost:7432/analyze",
        files={"video": ("FACESCREEN.mp4", f, "video/mp4")},
        data={"window": "5", "mode": "facecam"})

resp = r.json()
print(f"Response: {resp}")
if "error" in resp:
    print(f"Error: {resp['error']}")
    sys.exit(1)

job_id = resp["job_id"]
print(f"Job ID: {job_id}")

for i in range(120):
    time.sleep(2)
    s = requests.get(f"http://localhost:7432/status/{job_id}").json()
    step = s.get("progress", {}).get("step", 0)
    msg = s.get("progress", {}).get("msg", "")
    print(f"  [{i*2}s] step={step} status={s['status']} msg={msg}")

    if s["status"] == "done":
        result = s["result"]
        summary = result["summary"]
        print()
        print("=== RESULTS ===")
        print(f"Face pct: {summary['face_pct']}")
        print(f"Avg face conf: {summary['avg_face_conf']}")
        print(f"Avg audio: {summary['avg_audio']}")
        print(f"Avg score: {summary['avg_score']}")
        print(f"Drops: {summary['drop_count']}")
        for w in result["windows"]:
            print(f"  window {w['start']}-{w['end']}: face={w['face']} face_conf={w['face_conf']} audio={w['audio']} score={w['score']} label={w['label']}")
        break
    elif s["status"] == "error":
        print(f"ERROR: {s.get('error', 'unknown')}")
        break
else:
    print("TIMEOUT")
