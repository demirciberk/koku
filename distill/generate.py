"""Teacher data generation: for each perfume, ask a local Qwen3-8B (llama.cpp server)
for a short Turkish description plus Turkish search queries a real shopper might type.

The output (data/teacher.jsonl) is training data for both the distilled student LM
(perfume -> Turkish description) and the retriever (Turkish query -> perfume).

Server (separate terminal):
  tools/llama.cpp/llama-server.exe -m models/Qwen3-8B-Q4_K_M.gguf -ngl 99 -c 8192 -np 4 --port 8080
Run:
  uv run python distill/generate.py --limit 200      # pilot
  uv run python distill/generate.py                  # all perfumes, resumes where it stopped
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "eval")
from label import corpus  # noqa: E402

URL = "http://127.0.0.1:8080/v1/chat/completions"
OUT = "data/teacher.jsonl"
GENDER_TR = {"men": "erkek", "women": "kadın", "unisex": "unisex"}
# The teacher mistranslates note names (peony -> "pony", invents "muz"), so the prompt gets Turkish names directly.
with open("distill/tr_names.json", encoding="utf-8") as _f:
    TR = json.load(_f)

SYSTEM = (
    "Sen deneyimli bir parfüm danışmanısın. Sadece verilen nota ve akorları kullan, listede olmayan nota ekleme. "
    "Nota adlarını verildiği gibi yaz. İngilizce kelime kullanma. "
    "Sorguları gerçek bir müşterinin arama kutusuna yazacağı gibi kısa, günlük dille ve birbirinden farklı yaz. "
    "Marka ve parfüm adını sorgulara yazma. Yanıtın yalnızca geçerli JSON olsun."
)

PROMPT = """Örnek:
Parfüm: X Y (erkek) | akorlar: ferah baharatlı, narenciye, odunsu | üst: greyfurt, nane | orta: zencefil | alt: vetiver, sedir
{{"description": "Greyfurt ve naneyle açılan, zencefille canlanan ferah ve baharatlı bir koku. Vetiver ve sedirle temiz, odunsu bir bitiş yapar. Sıcak günlerde ve gündüz kullanımında rahat hissettirir.",
 "queries": {{"note": "greyfurtlu, zencefilli ferah bir erkek parfümü", "style": "temiz, enerjik ve sportif bir koku arıyorum", "occasion": "bisikletle şehir turuna çıkarken sürebileceğim bir şey"}}}}

Şimdi bu parfüm için aynı biçimde yaz:
Parfüm: {brand} {name} ({gender}) | akorlar: {accords} | üst: {top} | orta: {middle} | alt: {base}

Kurallar:
- description: 2-3 cümle; karakteri, hissi, uygun mevsim ve ortam. Akorların sırası kokunun ağırlığını gösterir.
- note: en belirgin 1-3 nota veya akoru anan doğal bir cümle.
- style: tarzı ve karakteri anlatan, hiç nota adı geçmeyen bir cümle.
- occasion: yalnızca ortam, durum, kişi veya his; hiç nota, akor ya da koku türü adı yok. Bu parfümün karakterine uymalı. Bu sorguyu şu açıdan yaz: {angle}."""

# One angle per perfume (rotated by id) so occasion queries do not collapse onto one stock phrase.
# Each angle has a pool of examples and every perfume sees only two of them: with a single fixed
# example list the teacher copied the examples verbatim. Examples also avoid the situations in
# eval/queries.jsonl to keep training data off the test set.
ANGLES = [
    ("belirli bir etkinlik veya an", ["mezuniyet töreni", "uzun bir uçak yolculuğu", "müze gezisi", "nişan yemeği",
                                      "bayram ziyareti", "tiyatro gecesi", "kitap kulübü buluşması", "yılbaşı sabahı"]),
    ("kime hediye alındığı veya kimin süreceği", ["ağabeyime", "öğretmenime", "yeni emekli olan birine", "kuzenime",
                                                   "iş arkadaşıma", "teyzeme", "yeni mezun bir gence", "eşime yıldönümünde"]),
    ("vermesi istenen his veya izlenim", ["nostaljik", "cesur", "sakin", "özenli", "sıcakkanlı", "asil",
                                          "rahat ve doğal", "otoriter"]),
    ("bir yer veya atmosfer benzetmesi", ["bağ evi", "karlı bir dağ köyü", "antika dükkânı", "eski bir istasyon",
                                          "limon bahçeli bir avlu", "kitap kokan bir sahaf", "yağmur sonrası orman", "bir kahve kavurma dükkânı"]),
    ("kullanıcının kaçındığı bir şey ve ihtiyacı", ["keskin olmasın", "yakın durunca hissedilsin", "çok genç durmasın",
                                                    "fazla erkeksi olmasın", "sıcakta bunaltmasın", "sabahtan akşama kalsın"]),
    ("günün saati ve gündelik rutin", ["öğle arası", "gece vardiyası", "hafta sonu pazarı", "sabah yürüyüşü",
                                       "akşam kitap okurken", "uzun bir toplantı günü"]),
]


def angle(pid: int) -> str:
    name, pool = ANGLES[pid % len(ANGLES)]
    k = (pid // len(ANGLES)) % len(pool)
    return f"{name} (ör. {pool[k]}, {pool[(k + 3) % len(pool)]}; bunları kopyalama, parfümün karakterine uyan kendi örneğini bul)"

KEYS = ("note", "style", "occasion")


def prompt(p: dict) -> str:
    j = lambda xs, table: ", ".join(table.get(x, x) for x in xs) if xs else "-"
    n = TR["notes"]
    return PROMPT.format(brand=p["brand"], name=p["name"], gender=GENDER_TR.get(p["gender"], p["gender"]),
                         accords=j(p["accords"], TR["accords"]), top=j(p["top"], n), middle=j(p["middle"], n), base=j(p["base"], n),
                         angle=angle(p["id"]))


def parse(text: str) -> dict | None:
    """Pull the JSON object out of the reply and check its shape; None means discard."""
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        d = json.loads(text[a : b + 1])
    except json.JSONDecodeError:
        return None
    q = d.get("queries")
    if not isinstance(d.get("description"), str) or not isinstance(q, dict):
        return None
    if not all(isinstance(q.get(k), str) and 8 <= len(q[k]) <= 200 for k in KEYS):
        return None
    if not 40 <= len(d["description"]) <= 800:
        return None
    return {"description": d["description"].strip(), "queries": {k: q[k].strip() for k in KEYS}}


def ask(p: dict) -> dict | None:
    body = {
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(p)}],
        "temperature": 0.7,
        "max_tokens": 400,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},  # Qwen3: skip <think>, we only want the JSON
    }
    for attempt in range(3):
        try:
            req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                text = json.load(r)["choices"][0]["message"]["content"]
            out = parse(text)
            if out:
                return {"id": p["id"], **out}
        except (OSError, KeyError, ValueError):
            time.sleep(2 * (attempt + 1))
    return None


def selftest() -> None:
    ok = '{"description": "' + "a" * 50 + '", "queries": {"note": "vanilyalı bir koku", "style": "sıcak ve tatlı", "occasion": "kış akşamı için"}}'
    assert parse("noise " + ok + " noise")["queries"]["style"] == "sıcak ve tatlı"
    assert parse('{"description": "kısa"}') is None
    assert parse("not json") is None


if __name__ == "__main__":
    selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)  # match the server's -np
    args = ap.parse_args()

    docs = sorted(corpus(), key=lambda p: -p["votes"])  # most-known perfumes first, so a pilot covers them
    done = set()
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            done = {json.loads(l)["id"] for l in f if l.strip()}
    todo = [p for p in docs if p["id"] not in done][: args.limit or None]
    print(f"{len(done)} done, {len(todo)} to go")

    t0, ok, fail = time.time(), 0, 0
    with open(OUT, "a", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as pool:
        for r in pool.map(ask, todo):
            if r:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                ok += 1
            else:
                fail += 1
            if (ok + fail) % 50 == 0:
                rate = (ok + fail) / (time.time() - t0)
                print(f"{ok + fail}/{len(todo)} ok={ok} fail={fail} {rate:.2f}/s eta={(len(todo) - ok - fail) / rate / 60:.0f} min", flush=True)
    print(f"done: ok={ok} fail={fail} in {(time.time() - t0) / 60:.1f} min")
