/**
 * Browser Single-Page Application logic for the Multi-Site Paint Sample Collector (Section 5).
 */
(function () {
  let activeJob = null;
  let activeImageId = null;
  let wsConn = null;
  let taxonomy = { brands: [], solvents: [], roles: [], tree: {} };
  let draftEditsBySample = {};

  const logDrawer = document.getElementById("jobLogDrawer");
  const activeJobBadge = document.getElementById("activeJobBadge");
  const canvasEl = document.getElementById("segmentCanvas");

  function escapeHtml(val) {
    return String(val ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function safeHexColor(val) {
    const s = String(val ?? "").trim();
    return /^#[0-9a-fA-F]{3,8}$/.test(s) ? s : "#888888";
  }

  function updateLiveCropPreview(slot) {
    const preview = document.getElementById("liveCropPreviewImg");
    if (!preview) return;
    if (slot && slot.sampleTmpId) {
      preview.src = `/api/samples/${encodeURIComponent(slot.sampleTmpId)}/crop?t=${Date.now()}`;
      preview.style.display = "block";
    } else {
      preview.removeAttribute("src");
      preview.style.display = "none";
    }
  }

  function appendLog(msg, level = "") {
    const div = document.createElement("div");
    div.className = `log-line ${level}`;
    const ts = new Date().toLocaleTimeString();
    div.textContent = `[${ts}] ${msg}`;
    logDrawer.appendChild(div);
    logDrawer.scrollTop = logDrawer.scrollHeight;
  }

  function switchTab(tabName) {
    document.querySelectorAll(".tab-btn").forEach((b) => {
      b.classList.toggle("active", b.dataset.tab === tabName);
    });
    document.querySelectorAll(".panel-view").forEach((p) => {
      p.classList.toggle("active", p.id === `panel-${tabName}`);
    });
  }

  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
  });

  document.getElementById("dismissNoticeBtn").addEventListener("click", () => {
    document.getElementById("copyrightNoticeBar").style.display = "none";
  });

  const editor = new window.CollectorCanvasEditor(canvasEl, {
    onChange: (samples, activeIdx) => {
      renderSampleSlotsList(samples, activeIdx);
      updateLiveCropPreview(samples[activeIdx] || null);
    },
    onDeleteSample: async (sampleTmpId) => {
      await fetch(`/api/samples/${encodeURIComponent(sampleTmpId)}`, { method: "DELETE" });
      delete draftEditsBySample[sampleTmpId];
      await refreshActiveJob();
    },
    onAcceptAll: async () => {
      await acceptAllCurrentImageSamples();
    },
  });

  function renderSampleSlotsList(samples, activeIdx) {
    const box = document.getElementById("sampleSlotsList");
    box.innerHTML = samples
      .map(
        (s, i) =>
          `<div style="padding:4px 6px;margin-bottom:4px;border-radius:4px;cursor:pointer;background:${
            i === activeIdx ? "rgba(59,130,246,0.22)" : "transparent"
          };" data-slot-idx="${i}">
            <b>Badge #${i + 1}</b> — pos: ${s.pos.length}, neg: ${s.neg.length}, box: ${s.box ? "yes" : "no"}
          </div>`
      )
      .join("");
    box.querySelectorAll("[data-slot-idx]").forEach((el) => {
      el.addEventListener("click", () => {
        editor.activeSampleIndex = Number(el.dataset.slotIdx);
        editor.render();
        renderSampleSlotsList(editor.samples, editor.activeSampleIndex);
        const slot = editor.samples[editor.activeSampleIndex];
        updateLiveCropPreview(slot || null);
      });
    });
  }

  function connectJobWebSocket(jobId) {
    if (wsConn) {
      try { wsConn.close(); } catch (_) {}
    }
    return new Promise((resolve) => {
      let settled = false;
      const done = () => {
        if (!settled) {
          settled = true;
          resolve();
        }
      };
      const timer = setTimeout(done, 500);
      const proto = location.protocol === "https:" ? "wss:" : "ws:";
      wsConn = new WebSocket(`${proto}//${location.host}/ws/jobs/${encodeURIComponent(jobId)}`);
      wsConn.onopen = () => {
        clearTimeout(timer);
        done();
      };
      wsConn.onerror = () => {
        clearTimeout(timer);
        done();
      };
      wsConn.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data);
          if (data.type === "fetch_started") {
            appendLog(`Fetching ${data.source_count ?? ""} source(s)...`);
          } else if (data.type === "fetch_done") {
            appendLog(`Fetch complete: ${data.images_count ?? 0} candidate image(s).`, "ok");
          } else if (data.type === "image_found") {
            appendLog(`Image found: ${data.image_id} (${data.w}x${data.h}) from ${data.image_url}`, "ok");
          } else if (data.type === "vision_started") {
            appendLog(`Vision segmentation started on ${data.image_id} (${data.mode || "manual"})...`);
          } else if (data.type === "vision_done") {
            appendLog(`Vision segmentation done on ${data.image_id}: ${data.samples_count ?? 0} sample(s).`, "ok");
          } else if (data.type === "vision_failed") {
            appendLog(`Vision failed on ${data.image_id}: ${data.error}`, "error");
            document.getElementById("visionFailedBanner").style.display = "block";
          } else if (data.type === "sample_ready") {
            appendLog(`Sample segmented: ${data.sample_tmp_id}`, "ok");
          } else if (data.type === "ingested") {
            appendLog(`Ingested ${data.count} samples: ${(data.sample_ids || []).join(", ")}`, "ok");
          } else if (data.type === "error") {
            appendLog(`Error (${data.url || ""}): ${data.message}`, "error");
          }
        } catch (_) {}
      };
    });
  }

  function collectDefaultsFromForm() {
    return {
      brand_id: document.getElementById("defBrandId").value.trim(),
      solvent_type: document.getElementById("defSolventType").value.trim(),
      paint_role: document.getElementById("defPaintRole").value.trim(),
      product_line: document.getElementById("defProductLine").value.trim(),
      paint_code: document.getElementById("defPaintCode").value.trim(),
      paint_name_en: document.getElementById("defPaintNameEn").value.trim(),
      paint_name_ko: document.getElementById("defPaintNameKo").value.trim(),
      source_type: document.getElementById("defSourceType").value,
    };
  }

  async function refreshActiveJob() {
    if (!activeJob) return;
    const prevJobId = activeJob.job_id;
    const res = await fetch(`/api/jobs/${encodeURIComponent(activeJob.job_id)}`);
    if (!res.ok) return;
    activeJob = await res.json();
    if (activeJob.job_id !== prevJobId) {
      draftEditsBySample = {};
    }
    activeJobBadge.textContent = `Job ${activeJob.job_id} (${activeJob.images.length} imgs / ${activeJob.samples.length} samples)`;
    renderImagesGrid();
    populateSegmentImageDropdown();
    renderReviewTable();
  }

  document.getElementById("createAndFetchBtn").addEventListener("click", async () => {
    const urls = document
      .getElementById("sourceUrlsInput")
      .value.split("\n")
      .map((s) => s.trim())
      .filter(Boolean);
    if (!urls.length) {
      appendLog("Enter at least one URL.", "error");
      return;
    }
    const defaults = collectDefaultsFromForm();
    appendLog(`Creating job for ${urls.length} source URL(s)...`);
    const createRes = await fetch("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ urls, defaults }),
    });
    if (!createRes.ok) {
      appendLog(`Failed to create job: ${await createRes.text()}`, "error");
      return;
    }
    activeJob = await createRes.json();
    draftEditsBySample = {};
    await connectJobWebSocket(activeJob.job_id);
    const fetchRes = await fetch(`/api/jobs/${encodeURIComponent(activeJob.job_id)}/fetch`, { method: "POST" });
    if (fetchRes.ok) {
      activeJob = await fetchRes.json();
      await refreshActiveJob();
      switchTab("images");
    } else {
      appendLog(`Fetch failed: ${await fetchRes.text()}`, "error");
    }
  });

  document.getElementById("saveSourceRegistryBtn").addEventListener("click", async () => {
    const urls = document
      .getElementById("sourceUrlsInput")
      .value.split("\n")
      .map((s) => s.trim())
      .filter(Boolean);
    const defaults = collectDefaultsFromForm();
    for (const url of urls) {
      await fetch("/api/sources", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url, defaults }),
      });
    }
    appendLog(`Saved ${urls.length} URL(s) to config/collector_sources.json.`, "ok");
  });

  function renderImagesGrid() {
    const grid = document.getElementById("imagesThumbGrid");
    if (!activeJob || !activeJob.images.length) {
      grid.innerHTML = `<div style="color:var(--muted);">No candidate images extracted yet.</div>`;
      return;
    }
    const minPx = Number(document.getElementById("filterMinPxInput").value || 0);
    const filtered = activeJob.images.filter((im) => Math.min(im.w, im.h) >= minPx);
    grid.innerHTML = filtered
      .map(
        (im) => `
      <div class="thumb-card">
        <img src="/api/images/${encodeURIComponent(im.image_id)}" alt="${escapeHtml(im.image_id)}" />
        <label style="flex-direction:row;align-items:center;gap:6px;color:var(--text);">
          <input type="checkbox" class="img-select-cb" value="${escapeHtml(im.image_id)}" checked />
          <b>${escapeHtml(im.image_id)}</b> (${Number(im.w) || 0}&times;${Number(im.h) || 0})
        </label>
        <div style="font-size:11px;color:var(--muted);word-break:break-all;">${escapeHtml(im.image_url)}</div>
        <div style="font-size:11px;">Status: <b>${escapeHtml(im.status)}</b></div>
      </div>`
      )
      .join("");
  }

  document.getElementById("filterMinPxInput").addEventListener("input", renderImagesGrid);
  document.getElementById("selectAllImagesBtn").addEventListener("click", () => {
    document.querySelectorAll(".img-select-cb").forEach((cb) => (cb.checked = true));
  });

  function populateSegmentImageDropdown() {
    const sel = document.getElementById("segmentImageSelect");
    if (!activeJob) return;
    sel.innerHTML = activeJob.images
      .map(
        (im) =>
          `<option value="${escapeHtml(im.image_id)}">${escapeHtml(im.image_id)} (${Number(im.w) || 0}x${Number(im.h) || 0}) [${escapeHtml(im.status)}]</option>`
      )
      .join("");
    if (!activeImageId && activeJob.images.length) {
      activeImageId = activeJob.images[0].image_id;
    }
    if (activeImageId) {
      sel.value = activeImageId;
    }
  }

  async function openImageInEditor(imageId) {
    if (!imageId || !activeJob) return;
    activeImageId = imageId;
    document.getElementById("segmentImageSelect").value = imageId;
    document.getElementById("visionFailedBanner").style.display = "none";
    updateLiveCropPreview(null);
    await editor.loadImage(`/api/images/${encodeURIComponent(imageId)}`);
    const imgSamples = (activeJob.samples || []).filter((s) => s.image_id === imageId);
    editor.setSamplesFromJob(imgSamples);
    renderSampleSlotsList(editor.samples, editor.activeSampleIndex);
    updateLiveCropPreview(imgSamples.length ? editor.samples[editor.activeSampleIndex] : null);
  }

  document.getElementById("segmentImageSelect").addEventListener("change", (e) => {
    openImageInEditor(e.target.value);
  });

  document.getElementById("openSelectedInSegmentBtn").addEventListener("click", async () => {
    const checked = Array.from(document.querySelectorAll(".img-select-cb:checked")).map((c) => c.value);
    if (checked.length) {
      switchTab("segment");
      await openImageInEditor(checked[0]);
    }
  });

  function getEngineSelectEl() {
    return document.getElementById("segmentEngineSelect") || document.getElementById("segEngineSelect");
  }

  function syncToleranceSliderForEngine(engine) {
    const titleEl = document.getElementById("tolSliderTitle");
    const slider = document.getElementById("oklabTolSlider");
    const valLabel = document.getElementById("tolValueLabel");
    if (!slider) return;
    const curVal = Number(slider.value);
    if (engine === "sam2") {
      if (titleEl) titleEl.textContent = "SAM2 Tolerance";
      slider.min = "1.0";
      slider.max = "20.0";
      slider.step = "0.5";
      if (curVal < 1.0) {
        const mapped = Math.round((1.0 + ((curVal - 0.02) / 0.18) * 19.0) * 2.0) / 2.0;
        slider.value = String(Math.max(1.0, Math.min(20.0, mapped || 5.0)));
      }
    } else {
      if (titleEl) titleEl.textContent = "OKLab Tol (ΔE)";
      slider.min = "0.02";
      slider.max = "0.20";
      slider.step = "0.005";
      if (curVal >= 1.0) {
        const mapped = 0.02 + ((curVal - 1.0) / 19.0) * 0.18;
        const snapped = Math.round(mapped / 0.005) * 0.005;
        slider.value = String(Number(Math.max(0.02, Math.min(0.20, snapped || 0.06)).toFixed(3)));
      }
    }
    if (valLabel) valLabel.textContent = slider.value;
  }

  function computeToleranceParams(engine) {
    const slider = document.getElementById("oklabTolSlider");
    const raw = Number((slider && slider.value) || 0.06);
    if (engine === "sam2") {
      const tolerance = raw >= 1.0 ? raw : 1.0 + ((raw - 0.02) / 0.18) * 19.0;
      const oklabTol = Math.max(0.02, Math.min(0.20, 0.02 + ((tolerance - 1.0) / 19.0) * 0.18));
      return {
        oklabTol: Number(oklabTol.toFixed(4)),
        tolerance: Number(Math.max(1.0, Math.min(20.0, tolerance)).toFixed(2)),
      };
    }
    const oklabTol = raw < 1.0 ? raw : 0.02 + ((raw - 1.0) / 19.0) * 0.18;
    const tolerance = Math.max(1.0, Math.min(20.0, 1.0 + ((oklabTol - 0.02) / 0.18) * 19.0));
    return {
      oklabTol: Number(Math.max(0.02, Math.min(0.20, oklabTol)).toFixed(4)),
      tolerance: Number(tolerance.toFixed(2)),
    };
  }

  const segEngineEl = getEngineSelectEl();
  if (segEngineEl) {
    segEngineEl.addEventListener("change", (e) => {
      syncToleranceSliderForEngine(e.target.value);
    });
  }

  function setActiveVisionModeButtons(mode) {
    const manBtn = document.getElementById("modeManualBtn");
    const autoBtn = document.getElementById("modeAutoBtn");
    if (manBtn && autoBtn) {
      manBtn.classList.toggle("secondary", mode !== "manual");
      autoBtn.classList.toggle("secondary", mode !== "auto");
    }
  }

  document.getElementById("modeManualBtn").addEventListener("click", () => {
    setActiveVisionModeButtons("manual");
    document.getElementById("visionFailedBanner").style.display = "none";
    if (canvasEl && canvasEl.focus) canvasEl.focus();
    appendLog("Manual Click Mode active: Left-click (+), Right-click (-), Drag box, then Run Segmentation", "ok");
  });

  document.getElementById("runAutoBatchBtn").addEventListener("click", async () => {
    const checked = Array.from(document.querySelectorAll(".img-select-cb:checked")).map((c) => c.value);
    const customModel = document.getElementById("settingsModelCustom").value.trim();
    const dropdownModel = document.getElementById("settingsModelDropdown").value;
    const model = customModel || dropdownModel || undefined;
    const engine = (getEngineSelectEl() && getEngineSelectEl().value) || "grabcut";
    for (const imgId of checked) {
      appendLog(`Running Gemini Vision Auto Detection on ${imgId}...`);
      const res = await fetch(`/api/images/${encodeURIComponent(imgId)}/segment/auto`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model, engine }),
      });
      const data = await res.json();
      if (data.status === "vision_failed") {
        appendLog(`Auto vision failed on ${imgId}: ${data.error}`, "error");
      }
    }
    await refreshActiveJob();
    switchTab("review");
  });

  document.getElementById("oklabTolSlider").addEventListener("input", (e) => {
    document.getElementById("tolValueLabel").textContent = e.target.value;
  });

  document.getElementById("newSampleSlotBtn").addEventListener("click", () => editor.addNewSampleSlot());
  document.getElementById("deleteActiveSampleBtn").addEventListener("click", () => editor.deleteActiveSample());

  document.getElementById("runManualSegmentBtn").addEventListener("click", async () => {
    if (!activeImageId) return;
    const prompts = editor.samples
      .filter((s) => s.pos && s.pos.length > 0)
      .map((s) => ({ pos: s.pos, neg: s.neg || [], box: s.box || null }));
    if (!prompts.length) {
      appendLog("Click at least one positive seed point on the canvas first.", "error");
      return;
    }
    const engine = (getEngineSelectEl() && getEngineSelectEl().value) || "grabcut";
    const { oklabTol, tolerance } = computeToleranceParams(engine);
    const res = await fetch(`/api/images/${encodeURIComponent(activeImageId)}/segment/manual`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sample_prompts: prompts,
        engine,
        oklab_tol: oklabTol,
        tolerance,
        replace_existing: true,
      }),
    });
    if (!res.ok) {
      appendLog(`Manual segmentation failed: ${await res.text()}`, "error");
      return;
    }
    await refreshActiveJob();
    await openImageInEditor(activeImageId);
  });

  document.getElementById("modeAutoBtn").addEventListener("click", async () => {
    if (!activeImageId) return;
    setActiveVisionModeButtons("auto");
    const customModel = document.getElementById("settingsModelCustom").value.trim();
    const dropdownModel = document.getElementById("settingsModelDropdown").value;
    const model = customModel || dropdownModel || undefined;
    const engine = (getEngineSelectEl() && getEngineSelectEl().value) || "grabcut";
    const res = await fetch(`/api/images/${encodeURIComponent(activeImageId)}/segment/auto`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model, engine }),
    });
    const data = await res.json();
    await refreshActiveJob();
    if (data.status === "vision_failed") {
      document.getElementById("visionFailedBanner").style.display = "block";
    } else {
      await openImageInEditor(activeImageId);
    }
  });

  document.getElementById("switchToManualAfterFailBtn").addEventListener("click", () => {
    document.getElementById("visionFailedBanner").style.display = "none";
    setActiveVisionModeButtons("manual");
    editor.setSamplesFromJob([]);
    updateLiveCropPreview(null);
  });

  function readRowMetadata(row) {
    return {
      brand_id: row.querySelector(".meta-brand").value.trim(),
      solvent_type: row.querySelector(".meta-solvent").value.trim(),
      paint_role: row.querySelector(".meta-role").value.trim(),
      product_line: row.querySelector(".meta-line").value.trim(),
      paint_code: row.querySelector(".meta-code").value.trim(),
      paint_name_en: row.querySelector(".meta-name-en").value.trim(),
      paint_name_ko: row.querySelector(".meta-name-ko").value.trim(),
      surfacer_code: row.querySelector(".meta-sf").value.trim(),
      topcoat_code: row.querySelector(".meta-tc").value.trim(),
      final_finish: row.querySelector(".meta-finish").value.trim(),
    };
  }

  function snapshotReviewDrafts() {
    const tbody = document.getElementById("reviewTableBody");
    if (!tbody) return;
    tbody.querySelectorAll("tr[data-smp-id]").forEach((row) => {
      const smpId = row.dataset.smpId;
      if (smpId) {
        draftEditsBySample[smpId] = readRowMetadata(row);
      }
    });
  }

  function buildTaxonomySelectHtml(clsName, optionsList, currentValue, isAmbiguous, titleText, widthPx) {
    const cur = String(currentValue ?? "").trim();
    const opts = [];
    const seen = new Set();
    const addOpt = (v) => {
      if (v !== "" && !seen.has(v)) {
        seen.add(v);
        opts.push(v);
      }
    };
    addOpt("auto");
    (optionsList || []).forEach(addOpt);
    if (cur && !seen.has(cur)) {
      addOpt(cur);
    }
    const optHtml = opts
      .map((val) => {
        const selected = val === cur ? " selected" : "";
        return `<option value="${escapeHtml(val)}"${selected}>${escapeHtml(val)}</option>`;
      })
      .join("");
    const ambCls = isAmbiguous ? " ambiguous-field" : "";
    const titleAttr = titleText ? ` title="${escapeHtml(titleText)}"` : "";
    return `<select class="${clsName}${ambCls}"${titleAttr} style="width:${widthPx}px;margin-bottom:4px;">${optHtml}</select>`;
  }

  async function acceptAllCurrentImageSamples() {
    if (!activeJob || !activeImageId) return;
    snapshotReviewDrafts();
    const samples = (activeJob.samples || []).filter((s) => s.image_id === activeImageId);
    for (const s of samples) {
      const row = document.querySelector(`#reviewTableBody tr[data-smp-id="${CSS.escape(s.sample_tmp_id)}"]`);
      const draft = draftEditsBySample[s.sample_tmp_id];
      const metadata = row ? readRowMetadata(row) : (draft || s.metadata || {});
      await fetch(`/api/samples/${encodeURIComponent(s.sample_tmp_id)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ metadata, status: "accepted", needs_review: false }),
      });
      delete draftEditsBySample[s.sample_tmp_id];
    }
    await refreshActiveJob();
    appendLog(`Accepted ${samples.length} sample(s) for ${activeImageId}.`, "ok");
  }

  document.getElementById("acceptAllSamplesBtn").addEventListener("click", acceptAllCurrentImageSamples);

  function renderReviewTable() {
    const tbody = document.getElementById("reviewTableBody");
    snapshotReviewDrafts();
    if (!activeJob || !activeJob.samples.length) {
      tbody.innerHTML = `<tr><td colspan="7" style="color:var(--muted);">No segmented samples yet.</td></tr>`;
      return;
    }
    tbody.innerHTML = activeJob.samples
      .map((s) => {
        const baseMeta = s.metadata || {};
        const draft = draftEditsBySample[s.sample_tmp_id] || {};
        const m = { ...baseMeta, ...draft };
        const opt = s.calibrated_optical_params || {};
        const amb = baseMeta.ambiguous_fields || [];
        const isAmbBrand = amb.includes("brand_id");
        const isAmbSolvent = amb.includes("solvent_type");
        const isAmbRole = amb.includes("paint_role");
        const candSolvents = (baseMeta.candidates && baseMeta.candidates.solvent_type) || [];
        const candHint = candSolvents.map((c) => `${c.value} (${c.score})`).join(", ");
        const safeHex = safeHexColor(s.measured_srgb_hex);
        const reasonsText = (s.review_reasons || []).map((r) => escapeHtml(r)).join("; ") || "OK";

        const brandSelect = buildTaxonomySelectHtml("meta-brand", taxonomy.brands, m.brand_id || "auto", isAmbBrand, "", 140);
        const solventSelect = buildTaxonomySelectHtml(
          "meta-solvent",
          taxonomy.solvents,
          m.solvent_type || "auto",
          isAmbSolvent,
          candHint ? `Candidates: ${candHint}` : "",
          140
        );
        const roleSelect = buildTaxonomySelectHtml("meta-role", taxonomy.roles, m.paint_role || "auto", isAmbRole, "", 140);

        return `
        <tr class="${s.needs_review ? "needs-review-row" : ""}" data-smp-id="${escapeHtml(s.sample_tmp_id)}">
          <td>
            <div class="checkerboard" style="width:72px;height:72px;border-radius:4px;display:flex;align-items:center;justify-content:center;">
              <img src="/api/samples/${encodeURIComponent(s.sample_tmp_id)}/crop?t=${Date.now()}" style="max-width:68px;max-height:68px;object-fit:contain;" />
            </div>
          </td>
          <td>
            <b>${escapeHtml(s.sample_tmp_id)}</b><br/>
            <span class="badge">${escapeHtml(s.status)}</span>
            ${s.sample_id ? `<br/><b style="color:var(--success);">${escapeHtml(s.sample_id)}</b>` : ""}
          </td>
          <td>
            <span class="swatch-pill" style="background:${safeHex};"></span>
            <b>${escapeHtml(s.measured_srgb_hex || "")}</b><br/>
            <span style="font-size:11px;color:var(--muted);">
              gloss_index: ${escapeHtml(opt.gloss_index ?? opt.estimated_alpha_ggx ?? "")}<br/>
              margin: ${escapeHtml(opt.gloss_margin_to_next_class ?? "")}
            </span>
          </td>
          <td>
            ${brandSelect}<br/>
            ${solventSelect}<br/>
            ${roleSelect}<br/>
            <input type="text" class="meta-line" value="${escapeHtml(m.product_line || "")}" style="width:140px;" placeholder="product_line" />
          </td>
          <td>
            <input type="text" class="meta-code" value="${escapeHtml(m.paint_code || "")}" style="width:90px;margin-bottom:4px;" placeholder="Code" /><br/>
            <input type="text" class="meta-name-en" value="${escapeHtml(m.paint_name_en || "")}" style="width:140px;margin-bottom:4px;" placeholder="Name EN" /><br/>
            <input type="text" class="meta-name-ko" value="${escapeHtml(m.paint_name_ko || "")}" style="width:140px;" placeholder="Name KO" />
          </td>
          <td>
            <input type="text" class="meta-sf" value="${escapeHtml(m.surfacer_code || "none")}" style="width:120px;margin-bottom:4px;" placeholder="surfacer_code" /><br/>
            <input type="text" class="meta-tc" value="${escapeHtml(m.topcoat_code || "none")}" style="width:120px;margin-bottom:4px;" placeholder="topcoat_code" /><br/>
            <input type="text" class="meta-finish" value="${escapeHtml(m.final_finish || "gloss")}" style="width:120px;" placeholder="final_finish" />
          </td>
          <td>
            <div style="font-size:11px;color:#fcd34d;margin-bottom:6px;">
              ${reasonsText}
              ${candHint ? `<br/>Candidates: ${escapeHtml(candHint)}` : ""}
            </div>
            <div class="btn-row" style="margin-top:0;">
              <button class="btn success btn-accept-smp">Save &amp; Accept</button>
              <button class="btn danger btn-reject-smp">Reject</button>
            </div>
          </td>
        </tr>`;
      })
      .join("");

    tbody.querySelectorAll("tr[data-smp-id]").forEach((row) => {
      const smpId = row.dataset.smpId;
      row.querySelectorAll("input, select").forEach((ctrl) => {
        const trackDraft = () => {
          draftEditsBySample[smpId] = readRowMetadata(row);
        };
        ctrl.addEventListener("input", trackDraft);
        ctrl.addEventListener("change", trackDraft);
      });

      row.querySelector(".btn-accept-smp").addEventListener("click", async () => {
        const metadata = readRowMetadata(row);
        await fetch(`/api/samples/${encodeURIComponent(smpId)}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ metadata, status: "accepted", needs_review: false }),
        });
        delete draftEditsBySample[smpId];
        await refreshActiveJob();
      });

      row.querySelector(".btn-reject-smp").addEventListener("click", async () => {
        await fetch(`/api/samples/${encodeURIComponent(smpId)}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ status: "rejected" }),
        });
        delete draftEditsBySample[smpId];
        await refreshActiveJob();
      });
    });
  }

  document.getElementById("writeMasterIndexBtn").addEventListener("click", async () => {
    if (!activeJob) return;
    const res = await fetch(`/api/jobs/${encodeURIComponent(activeJob.job_id)}/ingest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    const box = document.getElementById("ingestResultsBox");
    if (!res.ok) {
      const errText = await res.text();
      box.innerHTML = `<span style="color:var(--danger);">Ingest error: ${escapeHtml(errText)}</span>`;
      appendLog(`Ingest blocked: ${errText}`, "error");
      return;
    }
    const data = await res.json();
    box.innerHTML = (data.rows || [])
      .map(
        (r) =>
          `<div><b>${escapeHtml(r.sample_id)}</b> &rarr; <code>sample/paint_samples/${escapeHtml(r.file_rel_path)}</code> (${escapeHtml(r.measured_srgb_hex)})</div>`
      )
      .join("");
    await refreshActiveJob();
  });

  // Settings & Auth
  async function loadSettingsAndModels() {
    const [setRes, modRes, taxRes] = await Promise.all([
      fetch("/api/settings"),
      fetch("/api/models"),
      fetch("/api/taxonomy"),
    ]);
    if (taxRes.ok) taxonomy = await taxRes.json();
    if (setRes.ok) {
      const s = await setRes.json();
      const eng = s.segmentation_engine || "grabcut";
      document.getElementById("settingsSegEngine").value = eng;
      const segSel = getEngineSelectEl();
      if (segSel) {
        segSel.value = eng;
        syncToleranceSliderForEngine(eng);
      }
      document.getElementById("settingsRateLimit").value = s.rate_limit_requests_per_sec ?? 1.0;
      document.getElementById("settingsMinImagePx").value = s.min_image_px ?? 300;
      if (s.auth_status) {
        document.getElementById("settingsAuthMode").value = s.auth_status.auth_mode || "api_key";
        document.getElementById("settingsProjectId").value = s.auth_status.project_id || "";
        document.getElementById("settingsLocation").value = s.auth_status.location || "us-central1";
        document.getElementById("authStatusLine").textContent =
          `Saved Auth: mode=${s.auth_status.auth_mode}, has_api_key=${s.auth_status.has_api_key}, ` +
          `service_account=${s.auth_status.client_email || "none"}, project_id=${s.auth_status.project_id || "none"}`;
      }
    }
    if (modRes.ok) {
      const m = await modRes.json();
      const dd = document.getElementById("settingsModelDropdown");
      dd.innerHTML = (m.models || []).map((id) => `<option value="${escapeHtml(id)}">${escapeHtml(id)}</option>`).join("");
      if (m.default_model) dd.value = m.default_model;
    }
  }

  document.getElementById("settingsSaFileInput").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    try {
      const text = await file.text();
      const parsed = JSON.parse(text);
      if (parsed.project_id) {
        document.getElementById("settingsProjectId").value = parsed.project_id;
      }
      document.getElementById("settingsSaFileInput").dataset.rawJson = text;
    } catch (err) {
      appendLog(`Invalid service-account JSON file: ${err}`, "error");
    }
  });

  document.getElementById("saveSecretsBtn").addEventListener("click", async () => {
    const authMode = document.getElementById("settingsAuthMode").value;
    const apiKey = document.getElementById("settingsApiKeyInput").value.trim();
    const projectId = document.getElementById("settingsProjectId").value.trim();
    const location = document.getElementById("settingsLocation").value.trim();
    const saRaw = document.getElementById("settingsSaFileInput").dataset.rawJson || null;

    const res = await fetch("/api/secrets", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        auth_mode: authMode,
        api_key: apiKey || undefined,
        project_id: projectId,
        location,
        service_account_json: saRaw || undefined,
      }),
    });
    if (res.ok) {
      document.getElementById("settingsApiKeyInput").value = "";
      await loadSettingsAndModels();
      appendLog("Credentials saved securely outside repository.", "ok");
    } else {
      appendLog(`Failed to save credentials: ${await res.text()}`, "error");
    }
  });

  document.getElementById("saveSettingsBtn").addEventListener("click", async () => {
    const customModel = document.getElementById("settingsModelCustom").value.trim();
    const dropdownModel = document.getElementById("settingsModelDropdown").value;
    const payload = {
      default_vision_model: customModel || dropdownModel || "gemini-3.8-flash",
      segmentation_engine: document.getElementById("settingsSegEngine").value,
      rate_limit_requests_per_sec: Number(document.getElementById("settingsRateLimit").value || 1.0),
      min_image_px: Number(document.getElementById("settingsMinImagePx").value || 300),
    };
    await fetch("/api/settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    await loadSettingsAndModels();
    appendLog("Updated collector settings.", "ok");
  });

  document.getElementById("testConnectionBtn").addEventListener("click", async () => {
    const customModel = document.getElementById("settingsModelCustom").value.trim();
    const dropdownModel = document.getElementById("settingsModelDropdown").value;
    const model = customModel || dropdownModel || "gemini-3.8-flash";
    const span = document.getElementById("testConnResult");
    span.textContent = "Testing...";
    const res = await fetch("/api/models/test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model }),
    });
    const data = await res.json();
    if (data.ok) {
      span.style.color = "var(--success)";
      span.textContent = `OK — ${data.model} (${data.latency_ms} ms)`;
    } else {
      span.style.color = "var(--danger)";
      span.textContent = `Failed (${data.model}): ${data.error}`;
    }
  });

  // Check URL query param `?job_id=...` (e.g. from CLI manual mode handoff)
  window.addEventListener("DOMContentLoaded", async () => {
    await loadSettingsAndModels();
    const params = new URLSearchParams(location.search);
    const jid = params.get("job_id");
    if (jid) {
      const res = await fetch(`/api/jobs/${encodeURIComponent(jid)}`);
      if (res.ok) {
        activeJob = await res.json();
        connectJobWebSocket(activeJob.job_id);
        await refreshActiveJob();
        switchTab("segment");
        if (activeJob.images.length) {
          await openImageInEditor(activeJob.images[0].image_id);
        }
      }
    }
  });
})();
