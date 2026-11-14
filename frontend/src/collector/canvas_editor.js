/**
 * Interactive <canvas> editor for Manual & Auto Vision sample segmentation (Section 3.1):
 * - Zoom & pan (mouse wheel, Space+drag)
 * - Single click: positive point on active sample
 * - Shift+click: start new independent sample with index badge (1, 2, 3...)
 * - Alt+click: negative background point on active sample
 * - Drag: optional bounding box prompt [x0, y0, x1, y1]
 * - Keyboard: Enter accepts all, Delete removes active sample, Ctrl+Z undoes last point/box
 */
(function (global) {
  class CollectorCanvasEditor {
    constructor(canvas, callbacks) {
      this.canvas = canvas;
      this.ctx = canvas.getContext("2d");
      this.callbacks = callbacks || {};
      this.image = null;
      this.scale = 1.0;
      this.offsetX = 0;
      this.offsetY = 0;
      this.spacePressed = false;
      this.isPanning = false;
      this.isDraggingBox = false;
      this.dragStartImg = null;
      this.panStart = { x: 0, y: 0 };

      // List of sample prompt objects: [{ pos: [[x,y]], neg: [[x,y]], box: [x0,y0,x1,y1]|null, maskImg: Image|null }]
      this.samples = [{ pos: [], neg: [], box: null, maskImg: null }];
      this.activeSampleIndex = 0;
      this.history = [];
      this._rafPending = false;

      this._bindEvents();
    }

    scheduleRender() {
      if (this._rafPending) return;
      this._rafPending = true;
      window.requestAnimationFrame(() => {
        this._rafPending = false;
        this.render();
      });
    }

    loadImage(url) {
      return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => {
          this.image = img;
          const scaleX = this.canvas.width / img.width;
          const scaleY = this.canvas.height / img.height;
          this.scale = Math.min(scaleX, scaleY, 1.0) * 0.95;
          this.offsetX = (this.canvas.width - img.width * this.scale) * 0.5;
          this.offsetY = (this.canvas.height - img.height * this.scale) * 0.5;
          this.render();
          resolve(img);
        };
        img.onerror = reject;
        img.src = url;
      });
    }

    setSamplesFromJob(jobSamples) {
      if (!jobSamples || !jobSamples.length) {
        this.samples = [{ pos: [], neg: [], box: null, maskImg: null, sampleTmpId: null }];
        this.activeSampleIndex = 0;
        this.render();
        return;
      }
      this.samples = jobSamples.map((s) => ({
        sampleTmpId: s.sample_tmp_id,
        pos: (s.prompts && s.prompts.pos) ? s.prompts.pos.map((p) => [p[0], p[1]]) : [],
        neg: (s.prompts && s.prompts.neg) ? s.prompts.neg.map((p) => [p[0], p[1]]) : [],
        box: (s.prompts && s.prompts.box) ? s.prompts.box.slice() : null,
        maskImg: null,
      }));
      this.activeSampleIndex = 0;
      this.samples.forEach((slot) => {
        if (slot.sampleTmpId) {
          const mImg = new Image();
          mImg.onload = () => this.render();
          mImg.src = `/api/samples/${encodeURIComponent(slot.sampleTmpId)}/mask?t=${Date.now()}`;
          slot.maskImg = mImg;
        }
      });
      this.render();
    }

    addNewSampleSlot() {
      this._pushHistory();
      this.samples.push({ pos: [], neg: [], box: null, maskImg: null, sampleTmpId: null });
      this.activeSampleIndex = this.samples.length - 1;
      this.render();
      if (this.callbacks.onChange) this.callbacks.onChange(this.samples, this.activeSampleIndex);
    }

    deleteActiveSample() {
      this._pushHistory();
      const removed = this.samples.splice(this.activeSampleIndex, 1)[0];
      if (!this.samples.length) {
        this.samples.push({ pos: [], neg: [], box: null, maskImg: null, sampleTmpId: null });
      }
      this.activeSampleIndex = Math.max(0, Math.min(this.activeSampleIndex, this.samples.length - 1));
      this.render();
      if (this.callbacks.onDeleteSample && removed && removed.sampleTmpId) {
        this.callbacks.onDeleteSample(removed.sampleTmpId);
      }
      if (this.callbacks.onChange) this.callbacks.onChange(this.samples, this.activeSampleIndex);
    }

    undo() {
      if (!this.history.length) return;
      const prev = this.history.pop();
      this.samples = JSON.parse(prev);
      this.activeSampleIndex = Math.min(this.activeSampleIndex, this.samples.length - 1);
      this.render();
      if (this.callbacks.onChange) this.callbacks.onChange(this.samples, this.activeSampleIndex);
    }

    _pushHistory() {
      const serializable = this.samples.map((s) => ({
        sampleTmpId: s.sampleTmpId || null,
        pos: s.pos,
        neg: s.neg,
        box: s.box,
      }));
      this.history.push(JSON.stringify(serializable));
      if (this.history.length > 40) this.history.shift();
    }

    _canvasToImageCoords(clientX, clientY) {
      const rect = this.canvas.getBoundingClientRect();
      const cx = clientX - rect.left;
      const cy = clientY - rect.top;
      const ix = Math.round((cx - this.offsetX) / this.scale);
      const iy = Math.round((cy - this.offsetY) / this.scale);
      return [ix, iy];
    }

    _bindEvents() {
      window.addEventListener("keydown", (e) => {
        if (e.code === "Space" && document.activeElement.tagName !== "INPUT" && document.activeElement.tagName !== "TEXTAREA") {
          this.spacePressed = true;
        }
        if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
          e.preventDefault();
          this.undo();
        } else if (e.key === "Delete" && document.activeElement.tagName !== "INPUT" && document.activeElement.tagName !== "TEXTAREA") {
          this.deleteActiveSample();
        } else if (e.key === "Enter" && document.activeElement.tagName !== "INPUT" && document.activeElement.tagName !== "TEXTAREA") {
          if (this.callbacks.onAcceptAll) this.callbacks.onAcceptAll();
        }
      });

      window.addEventListener("keyup", (e) => {
        if (e.code === "Space") this.spacePressed = false;
      });

      this.canvas.addEventListener("wheel", (e) => {
        e.preventDefault();
        const factor = e.deltaY < 0 ? 1.12 : 0.89;
        const rect = this.canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        this.offsetX = mx - (mx - this.offsetX) * factor;
        this.offsetY = my - (my - this.offsetY) * factor;
        this.scale = Math.max(0.1, Math.min(12.0, this.scale * factor));
        this.scheduleRender();
      });

      this.canvas.addEventListener("mousedown", (e) => {
        if (!this.image) return;
        if (this.spacePressed || e.button === 1) {
          this.isPanning = true;
          this.panStart = { x: e.clientX - this.offsetX, y: e.clientY - this.offsetY };
          return;
        }
        this.dragStartImg = this._canvasToImageCoords(e.clientX, e.clientY);
        this.isDraggingBox = false;
      });

      this.canvas.addEventListener("mousemove", (e) => {
        if (this.isPanning) {
          this.offsetX = e.clientX - this.panStart.x;
          this.offsetY = e.clientY - this.panStart.y;
          this.scheduleRender();
          return;
        }
        if (this.dragStartImg && e.buttons === 1) {
          const cur = this._canvasToImageCoords(e.clientX, e.clientY);
          if (Math.hypot(cur[0] - this.dragStartImg[0], cur[1] - this.dragStartImg[1]) > 8) {
            this.isDraggingBox = true;
            const slot = this.samples[this.activeSampleIndex];
            slot.box = [
              Math.min(this.dragStartImg[0], cur[0]),
              Math.min(this.dragStartImg[1], cur[1]),
              Math.max(this.dragStartImg[0], cur[0]),
              Math.max(this.dragStartImg[1], cur[1]),
            ];
            this.scheduleRender();
          }
        }
      });

      this.canvas.addEventListener("mouseup", (e) => {
        if (this.isPanning) {
          this.isPanning = false;
          return;
        }
        if (!this.image || !this.dragStartImg) return;
        const endPt = this._canvasToImageCoords(e.clientX, e.clientY);
        if (this.isDraggingBox) {
          this.dragStartImg = null;
          this.isDraggingBox = false;
          if (this.callbacks.onChange) this.callbacks.onChange(this.samples, this.activeSampleIndex);
          return;
        }
        this.dragStartImg = null;
        this._pushHistory();

        if (e.shiftKey && this.samples[this.activeSampleIndex].pos.length > 0) {
          this.samples.push({ pos: [endPt], neg: [], box: null, maskImg: null, sampleTmpId: null });
          this.activeSampleIndex = this.samples.length - 1;
        } else if (e.altKey) {
          this.samples[this.activeSampleIndex].neg.push(endPt);
        } else {
          this.samples[this.activeSampleIndex].pos.push(endPt);
        }
        this.render();
        if (this.callbacks.onChange) this.callbacks.onChange(this.samples, this.activeSampleIndex);
      });
    }

    render() {
      const ctx = this.ctx;
      ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
      if (!this.image) return;

      ctx.save();
      ctx.translate(this.offsetX, this.offsetY);
      ctx.scale(this.scale, this.scale);
      ctx.drawImage(this.image, 0, 0);

      const palette = ["#3b82f6", "#10b981", "#f59e0b", "#ec4899", "#8b5cf6", "#06b6d4"];
      this.samples.forEach((slot, idx) => {
        const color = palette[idx % palette.length];
        if (slot.maskImg && slot.maskImg.complete && slot.maskImg.naturalWidth > 0) {
          ctx.save();
          ctx.globalAlpha = idx === this.activeSampleIndex ? 0.35 : 0.2;
          ctx.drawImage(slot.maskImg, 0, 0);
          ctx.restore();
        }
        if (slot.box) {
          const [x0, y0, x1, y1] = slot.box;
          ctx.strokeStyle = color;
          ctx.lineWidth = 2 / this.scale;
          ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
        }
        slot.pos.forEach((pt, pIdx) => {
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(pt[0], pt[1], 6 / this.scale, 0, Math.PI * 2);
          ctx.fill();
          if (pIdx === 0) {
            ctx.fillStyle = "#fff";
            ctx.font = `bold ${Math.round(13 / this.scale)}px sans-serif`;
            ctx.fillText(String(idx + 1), pt[0] + 8 / this.scale, pt[1] - 6 / this.scale);
          }
        });
        slot.neg.forEach((pt) => {
          ctx.fillStyle = "#ef4444";
          ctx.beginPath();
          ctx.arc(pt[0], pt[1], 5 / this.scale, 0, Math.PI * 2);
          ctx.fill();
        });
      });

      ctx.restore();
    }
  }

  global.CollectorCanvasEditor = CollectorCanvasEditor;
})(window);
