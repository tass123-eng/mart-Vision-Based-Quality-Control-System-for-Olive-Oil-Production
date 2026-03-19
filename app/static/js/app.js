/* Olive Oil Quality Control – frontend logic */

(function () {
  'use strict';

  const form        = document.getElementById('upload-form');
  const fileInput   = document.getElementById('image-input');
  const dropZone    = document.getElementById('drop-zone');
  const dropLabel   = document.getElementById('drop-label');
  const preview     = document.getElementById('preview');
  const inspectBtn  = document.getElementById('inspect-btn');
  const spinner     = document.getElementById('spinner');
  const resultsCard = document.getElementById('results-card');
  const errorCard   = document.getElementById('error-card');
  const errorMsg    = document.getElementById('error-msg');
  const resetBtn    = document.getElementById('reset-btn');

  // ── File selection ──────────────────────────────────────────────────────────

  dropZone.addEventListener('click', () => fileInput.click());

  dropZone.addEventListener('dragover', e => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length) {
      fileInput.files = e.dataTransfer.files;
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length) handleFileSelected(fileInput.files[0]);
  });

  function handleFileSelected(file) {
    const reader = new FileReader();
    reader.onload = e => {
      preview.src = e.target.result;
      preview.hidden = false;
      dropLabel.hidden = true;
      inspectBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  // ── Form submit ─────────────────────────────────────────────────────────────

  form.addEventListener('submit', async e => {
    e.preventDefault();
    if (!fileInput.files.length) return;

    setLoading(true);
    hideResults();
    hideError();

    const formData = new FormData();
    formData.append('image', fileInput.files[0]);

    try {
      const resp = await fetch('/api/inspect', { method: 'POST', body: formData });
      const data = await resp.json();

      if (!resp.ok) {
        showError(data.error || `Server error (${resp.status})`);
        return;
      }
      renderResults(data);
    } catch (err) {
      showError('Network error: ' + err.message);
    } finally {
      setLoading(false);
    }
  });

  // ── Reset ───────────────────────────────────────────────────────────────────

  resetBtn.addEventListener('click', () => {
    form.reset();
    preview.hidden = true;
    preview.src = '';
    dropLabel.hidden = false;
    inspectBtn.disabled = true;
    hideResults();
    hideError();
  });

  // ── Render helpers ──────────────────────────────────────────────────────────

  function setLoading(on) {
    spinner.hidden = !on;
    inspectBtn.disabled = on;
  }

  function hideResults() { resultsCard.hidden = true; }
  function hideError()   { errorCard.hidden   = true; }

  function showError(msg) {
    errorMsg.textContent = '⚠️ ' + msg;
    errorCard.hidden = false;
  }

  function renderResults(data) {
    const overall = data.overall_valid;

    // Verdict banner
    const banner = document.getElementById('verdict-banner');
    const verdictText = document.getElementById('verdict-text');
    banner.className = 'verdict-banner ' + (overall ? 'verdict-pass' : 'verdict-fail');
    verdictText.textContent = overall ? '✅ PASS – Bottle meets quality standards'
                                      : '❌ FAIL – Quality defect(s) detected';

    // Annotated image
    const img = document.getElementById('annotated-img');
    img.src = 'data:image/jpeg;base64,' + data.annotated_image;

    // Alignment
    setCheck('check-alignment', data.alignment.is_aligned,
             data.alignment.message, null);

    // Fill
    const fillRatio = data.fill.fill_ratio !== undefined
      ? (data.fill.fill_ratio * 100).toFixed(1) + '%' : '';
    setCheck('check-fill', data.fill.is_valid, data.fill.message,
             fillRatio ? 'Measured fill: ' + fillRatio : null);

    // Cap & label
    setCheck('check-cap', data.cap_label.is_valid, data.cap_label.message, null);

    // Clarity
    const clarityDetail = data.clarity.clarity_score !== undefined
      ? `Clarity: ${data.clarity.clarity_score.toFixed(1)}/100  |  ` +
        `Turbidity: ${data.clarity.turbidity_score.toFixed(1)}/100`
      : null;
    setCheck('check-clarity', data.clarity.is_valid, data.clarity.message, clarityDetail);

    resultsCard.hidden = false;
  }

  function setCheck(id, isValid, message, detail) {
    const el   = document.getElementById(id);
    const icon = el.querySelector('.check-icon');
    const msg  = document.getElementById('msg-' + id.replace('check-', ''));
    const det  = document.getElementById('detail-' + id.replace('check-', ''));

    icon.textContent = isValid ? '✅' : '❌';
    if (msg)  msg.textContent  = message || '';
    if (det && detail !== null && detail !== undefined) {
      det.textContent = detail;
    }
  }
}());
