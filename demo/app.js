/* =========================
 Config & Flags
========================= */
const USE_MOCK = false;
const API_BASE = "http://localhost:8000/api";

/* =========================
 DOM Utilities
========================= */
const $ = (id) => document.getElementById(id);
const fmt = (s) => `${s.toFixed(2)} s`;

function setBtns(startEnabled, contEnabled, stopEnabled) {
    $("start").disabled = !startEnabled;
    $("cont").disabled = !contEnabled;
    $("stop").disabled = !stopEnabled;
}

function setStatus(t) {
    const el = $("status");
    el.textContent = USE_MOCK ? `${t} (Mock)` : t;
    el.setAttribute("data-state", t);
    const st = $("stateText");
    if (st) st.textContent = t;
}

function setErr(msg) {
    const err = $("err");
    err.textContent = msg || "";
    if (msg) err.focus();
}

function escapeHtml(s) {
    return String(s).replace(/[&<>'"]/g, (ch) =>
        ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[ch])
    );
}

function resetPanel() {
    $("terms").textContent = "";
    $("cond").textContent = "";
    $("t1").textContent = "-";
    $("tall").textContent = "0.00 s";
    $("log").textContent = "";
    $("summaryMain").textContent = "";
    $("summaryHigh").textContent = "";
    setErr("");
    state.log = [];
    state.allPotential = [];
}

/* =========================
 Countdown
========================= */
function countdown(sec) {
    clearInterval(state.timer);
    state.remain = sec;
    $("cd").textContent = `⏱ ${state.remain}s`;
    state.timer = setInterval(() => {
        state.remain--;
        $("cd").textContent = `⏱ ${Math.max(state.remain, 0)}s`;
        if (state.remain <= 0) {
            setStatus("Time limit reached (you may continue)");
        }
    }, 1000);
}

/* =========================
 Condition builder
========================= */
function buildCond(table, col, list) {
    const quoted = list.map((v) => `'${String(v).replace(/'/g, "''")}'`).join(", ");
    return `${table}.${col} IN (${quoted})`;
}

/* =========================
 App State
========================= */
const state = {
    running: false,
    iter: 0,
    startTs: 0,
    timer: null,
    remain: 0,
    sid: null,
    auto: true,
    alignedAll: [],      // high confidence
    seenTerms: new Set(),
    log: [],
    allPotential: [],    // potential values (unused; kept for compatibility)
    checkedTotal: 0,     // len(query.obj_scores) from server — cumulative LLM-checked table values
    t1Fixed: 0,      // 第一轮 reformulation 时间
};

function resetState() {
    state.running = false;
    state.iter = 0;
    state.startTs = 0;
    clearInterval(state.timer);
    state.timer = null;
    state.remain = 0;
    state.sid = null;
    state.auto = true;
    state.alignedAll = [];
    state.checkedTotal = 0;
    state.seenTerms.clear();
    state.t1Fixed = 0;
}

/* =========================
 Column fallback mapping
========================= */
let COLUMN_OPTIONS = {
    animal: ["animal"],
    chemical_compound: ["chemical_compound"],
    product: ["Product_ID", "Product_Title", "Merchant_ID"],
};

function syncColumnOptionsFromMeta(meta) {
    COLUMN_OPTIONS = {};
    for (const d of meta?.datasets ?? []) {
        COLUMN_OPTIONS[d.name] = d.columns ?? [];
    }
}

/* =========================
 Meta loading
========================= */
let META = null;

async function loadMeta() {
    const r = await fetch(`${API_BASE}/meta`);
    if (!r.ok) throw new Error(await r.text());
    return r.json();
}

async function loadMetaForPath(dbPath) {
    const r = await fetch(`${API_BASE}/meta_for_path`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ db_path: dbPath })
    });
    if (!r.ok) throw new Error("Failed to load meta");
    return await r.json(); // { datasets: [...] }
}

function fillTables() {
    $("table").innerHTML = (META?.datasets ?? [])
        .map((d) => `<option value="${escapeHtml(d.name)}">${escapeHtml(d.name)}</option>`)
        .join("");
}

function fillColumns() {
    const ds = $("table").value;
    const found = META?.datasets?.find((d) => d.name === ds);
    let cols = found?.columns;
    if (!cols || cols.length === 0) cols = COLUMN_OPTIONS[ds] ?? [];
    $("column").innerHTML = cols
        .map((c) => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`)
        .join("");
}

async function initMeta() {
    if (USE_MOCK) {
        META = {
            datasets: [
                { name: "animal", columns: ["animal"] },
                { name: "chemical_compound", columns: ["chemical_compound"] },
                { name: "product", columns: ["Product_Title"] },
            ],
        };
        fillTables();
        syncColumnOptionsFromMeta(META);
        fillColumns();
        setErr("");
        $("status").textContent += " (Mock)";
        return;
    }

    try {
        META = await loadMeta();
        fillTables();
        syncColumnOptionsFromMeta(META);
        fillColumns();
    } catch (e) {
        META = {
            datasets: [
                { name: "animal", columns: ["animal"] },
                { name: "chemical_compound", columns: ["chemical_compound"] },
                { name: "product", columns: ["Product_Title"] },
            ],
        };
        fillTables();
        syncColumnOptionsFromMeta(META);
        fillColumns();
        setErr(`Failed to load metadata. Using fallback: ${e?.message ?? e}`);
    }
}

/* =========================
 API (mock + real)
========================= */
let startController = null;
let iterController = null;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function startAPI(payload) {
    if (!USE_MOCK) {
        startController?.abort();
        startController = new AbortController();
        const r = await fetch(`${API_BASE}/iqe/start`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
            signal: startController.signal,
        });
        if (!r.ok) throw new Error(await r.text());
        return r.json();
    }

    await sleep(200);
    return {
        sessionId: `sess_${Date.now()}`,
        reformulatedTerms: [],
        t1Sec: 0.36,
    };
}

async function iterAPI(payload) {
    if (!USE_MOCK) {
        iterController?.abort();
        iterController = new AbortController();
        const r = await fetch(`${API_BASE}/iqe/iterate`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
            signal: iterController.signal,
        });
        if (!r.ok) throw new Error(await r.text());
        return r.json();
    }

    const dict = {
        "Chinese food": {
            terms: [
                ["Chinese cuisine", "Chinese dishes"],
                ["Sichuan food", "Cantonese food"],
                ["Peking food", "Hunan food"],
            ],
            vals: [
                "Sichuan food",
                "Cantonese food",
                "Peking food",
                "Hunan food",
                "Shanghai cuisine",
            ],
        },
    };

    const v = $("value").value.trim();
    const d =
        dict[v] ??
        {
            terms: [["Alt-1", "Alt-2"], ["Sub-1", "Sub-2"]],
            vals: ["Aligned-A", "Aligned-B", "Aligned-C"],
        };

    const i = Math.min(payload.iteration - 1, d.terms.length - 1);
    const t1 = 0.6 + Math.random() * 0.4;
    const start = (payload.iteration - 1) * 2;
    const now = d.vals.slice(start, start + 2);

    return {
        t1Sec: t1,
        reformulatedTerms: d.terms[i],
        alignedHigh: now.slice(0, 1),
        alignedPotential: now.slice(1),
        summary: `Round ${payload.iteration}: added ${now.slice(0, 1).length} high, ${now.slice(1).length} potential.`,
        done: start + 2 >= d.vals.length,
        checkedTotal: start + now.length,
    };
}

async function checkDbPath() {
    const path = $("dbpath").value.trim();
    $("pathErr").textContent = "";

    if (!path) {
        $("pathErr").textContent = "Please input a database path.";
        return;
    }

    try {
        const r = await fetch(`${API_BASE}/check_db_path`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ path })
        });

        const data = await r.json();

        if (!data.ok) {
            $("pathErr").textContent = data.message;
            $("start").disabled = true;
        } else {
            $("pathErr").textContent = "✅ Path exists";
            $("start").disabled = false;

            // 根据路径刷新 Table / Column
            const meta = await loadMetaForPath(path);
            META = meta;
            syncColumnOptionsFromMeta(META);
            fillTables();
            fillColumns();

            // 只有未运行时才允许 Start
            if (!state.running) {
                $("start").disabled = false;
            }
        }
    } catch (e) {
        $("pathErr").textContent = "Failed to check path.";
        $("start").disabled = true;
    }
}


/* =========================
 Sorting helper
========================= */
function sortEnCaseInsensitiveNumeric(arr) {
    return arr.sort((a, b) =>
        String(a).localeCompare(String(b), "en", { sensitivity: "base", numeric: true })
    );
}

/* =========================
 Flow Control
========================= */
async function startRun() {
    if (state.running) return;

    const table = $("table").value;
    const column = $("column").value;
    const value = $("value").value.trim();
    let limit = Number($("limit").value);

    if (!Number.isInteger(limit) || limit < 1) limit = 20;
    $("limit").value = String(limit);

    if (!table || !column || !value) {
        setErr("Please complete the inputs.");
        return;
    }

    resetPanel();
    resetState();

    state.auto = true;
    state.running = true;
    state.startTs = performance.now();

    setBtns(false, false, true);
    setStatus("Running");
    countdown(limit);

    try {
        const r = await startAPI({ db_path: $("dbpath").value.trim(), table, column, value, timeLimitSec: limit });
        state.t1Fixed = r.t1Sec;
        $("t1").textContent = fmt(state.t1Fixed);
        state.sid = r.sessionId;

        for (const t of r.reformulatedTerms ?? []) state.seenTerms.add(t);

        if (state.seenTerms.size > 0) {
            $("terms").textContent = Array.from(state.seenTerms).join(", ");
        }

        await next();
    } catch (e) {
        // 如果是用户主动 abort，静默处理
        if (e?.name === "AbortError") {
            return;
        }
        // 其他情况，才是真正的错误
        setErr(e?.message ?? String(e));
        stop();
    }
}

async function next() {
    if (!state.running) return;

    state.iter++;
    setBtns(false, false, true);
    setStatus("Processing");

    try {
        const r = await iterAPI({ sessionId: state.sid, iteration: state.iter });
        state.checkedTotal = r.checkedTotal ?? 0;

        r.suggestedStop = r.suggestedStop && state.alignedAll.length > 0;

        // soft stop 提示（不影响 Continue）
        if (r.suggestedStop) {
            setStatus("Algorithm suggests stopping (you may continue)");
        }
        if (r.timeUp) {
            setStatus("Time limit reached (you may continue)");
        }

        for (const t of r.reformulatedTerms ?? []) state.seenTerms.add(t);
        $("terms").textContent = Array.from(state.seenTerms).join(", ");

        // High confidence
        const highNew = r.alignedHigh ?? [];
        for (const h of highNew) state.alignedAll.push(h);

        // Log entry
        const hnswQueries = r.hnswQueries ?? [];
        const bm25Queries = r.bm25Queries ?? [];
        const logEntry = [
            `[Round] ${state.iter}:`,
            `  [Query terms for HNSW]: ${hnswQueries.length ? hnswQueries.join(", ") : "(none)"}`,
            `  [Query terms for BM25]: ${bm25Queries.length ? bm25Queries.join(", ") : "(none)"}`,
            `  [Highly Likely Aligned Values]: ${highNew.length ? highNew.join(", ") : "(none)"}`,
            `  [Potentially Aligned Values]: ${(r.alignedPotential ?? []).join(", ") || "(none)"}`,
            "",
        ].join("\n");

        state.log.push(logEntry);
        $("log").textContent = state.log.join("\n");
        $("cond").textContent = buildCond($("table").value, $("column").value, state.alignedAll);
        $("tall").textContent = fmt((performance.now() - state.startTs) / 1000);

        // ❶ 硬停止：budget 用尽
        if (r.done) {
            stop();
            return;
        }

        // ❷ 默认自动继续（没有被 early stop / time limit 拦住）
        if (!r.suggestedStop && !r.timeUp) {
            setStatus("Running");
            queueMicrotask(next);
            return;
        }

        // ❸ 自动暂停（soft stop），等待用户决定
        setBtns(false, true, true);

        if (r.suggestedStop && !r.timeUp) {
            setStatus(
                "Auto paused: algorithm suggests stopping (click Continue to continue, or Stop to terminate)"
            );
        } else if (r.timeUp && !r.suggestedStop) {
            setStatus(
                "Auto paused: time limit reached (click Continue to continue, or Stop to terminate)"
            );
        } else {
            setStatus(
                "Auto paused: algorithm suggests stopping & time limit reached"
            );
        }
    } catch (e) {
        // 如果是用户主动 abort，静默处理
        if (e?.name === "AbortError") {
            return;
        }
        // 其他情况，才是真正的错误
        setErr(e?.message ?? String(e));
        stop();
    }
}

/* =========================
 Stop & Summary
========================= */
function stop(timeout = false) {
    state.running = false;
    clearInterval(state.timer);
    startController?.abort?.();
    iterController?.abort?.();

    $("cd").textContent = timeout ? "⏱ Time limit reached" : "";
    setBtns(true, false, false);
    setStatus(timeout ? "Timeout" : "Stopped");

    const rounds = state.iter;
    const checked = state.checkedTotal;

    const summaryMainText = [
        `Total number of Rounds: ${rounds}.`,
        `Retrieved and Checked ${checked} Table Values.`,
    ].join("\n");
    $("summaryMain").textContent = summaryMainText;

    // High textbox
    $("summaryHigh").textContent =
        state.alignedAll.length ? state.alignedAll.join(", ") : "(none)";

    // 更新数量
    $("highCount").textContent = state.alignedAll.length;

}

/* =========================
 Bootstrap
========================= */
async function bootstrap() {
    await initMeta();
    $("table").addEventListener("change", fillColumns);
    $("start").addEventListener("click", startRun);
    $("checkPath").addEventListener("click", checkDbPath);
    $("cont").addEventListener("click", next);
    $("stop").addEventListener("click", () => stop(false));
    $("copy").addEventListener("click", async () => {
        const t = $("cond").textContent;
        if (!t) return;
        try {
            await navigator.clipboard.writeText(t);
            const toast = $("toast");
            toast.classList.add("show");
            setTimeout(() => toast.classList.remove("show"), 1200);
        } catch { }
    });
}

bootstrap();