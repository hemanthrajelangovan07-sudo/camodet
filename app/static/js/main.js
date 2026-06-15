/**
 * CamouflageNet — Dashboard Controller
 * Handles: source switching, file upload, stream start/stop,
 *          SSE stats/events, detection list, clock, screenshot, fullscreen
 */

'use strict';

let activeSource = 'image';
let streaming    = false;
let esrganOn     = false;
let autoLog      = true;
let eventSource  = null;

const dom = {
  sysDot:    document.getElementById('sys-dot'),
  sysLabel:  document.getElementById('sys-label'),
  badgeFps:  document.getElementById('badge-fps'),
  badgeDet:  document.getElementById('badge-det'),
  clock:     document.getElementById('clock'),
  feedTitle: document.getElementById('feed-title'),

  placeholder:  document.getElementById('placeholder'),
  liveFeed:     document.getElementById('live-feed'),
  resultImg:    document.getElementById('result-img'),
  resultVid:    document.getElementById('result-vid'),
  feedLoading:  document.getElementById('feed-loading'),
  loadingText:  document.getElementById('loading-text'),
  feedWrap:     document.getElementById('feed-wrap'),

  mCurrent: document.getElementById('m-current'),
  mTotal:   document.getElementById('m-total'),
  mConf:    document.getElementById('m-conf'),
  mFps:     document.getElementById('m-fps'),
  detList:  document.getElementById('det-list'),
  eventLog: document.getElementById('event-log'),

  sConf:  document.getElementById('s-conf'),
  svConf: document.getElementById('sv-conf'),
  sIou:   document.getElementById('s-iou'),
  svIou:  document.getElementById('sv-iou'),
};

function updateClock() {
  const now = new Date();
  dom.clock.textContent = now.toISOString().slice(11, 19);
}
setInterval(updateClock, 1000);
updateClock();

document.querySelectorAll('.src-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const src = btn.dataset.src;
    if (src === activeSource) return;

    document.querySelectorAll('.src-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    document.querySelectorAll('.src-panel').forEach(p => p.classList.add('hidden'));
    document.getElementById(`panel-${src}`).classList.remove('hidden');

    activeSource = src;
  });
});

document.getElementById('conf-img').addEventListener('input', e => {
  document.getElementById('val-conf-img').textContent = (e.target.value / 100).toFixed(2);
});

dom.sConf.addEventListener('input', e => {
  dom.svConf.textContent = (e.target.value / 100).toFixed(2);
});
dom.sIou.addEventListener('input', e => {
  dom.svIou.textContent = (e.target.value / 100).toFixed(2);
});

document.getElementById('toggle-esrgan').addEventListener('click', function() {
  esrganOn = !esrganOn;
  this.classList.toggle('active', esrganOn);
  this.dataset.on = esrganOn;
});
document.getElementById('toggle-log').addEventListener('click', function() {
  autoLog = !autoLog;
  this.classList.toggle('active', autoLog);
  this.dataset.on = autoLog;
});

function showLoading(msg = 'ANALYSING…') {
  dom.placeholder.classList.add('hidden');
  dom.liveFeed.classList.add('hidden');
  dom.resultImg.classList.add('hidden');
  dom.resultVid.classList.add('hidden');
  dom.loadingText.textContent = msg;
  dom.feedLoading.classList.remove('hidden');
}

function showResult(type, src) {
  dom.feedLoading.classList.add('hidden');
  dom.placeholder.classList.add('hidden');

  if (type === 'image') {
    dom.resultImg.src = src + '?t=' + Date.now();  // cache-bust
    dom.resultImg.classList.remove('hidden');
    dom.liveFeed.classList.add('hidden');
    dom.resultVid.classList.add('hidden');
  } else if (type === 'video') {
    dom.resultVid.src = src;
    dom.resultVid.classList.remove('hidden');
    dom.liveFeed.classList.add('hidden');
    dom.resultImg.classList.add('hidden');
  } else if (type === 'stream') {
    dom.liveFeed.src = src;
    dom.liveFeed.classList.remove('hidden');
    dom.resultImg.classList.add('hidden');
    dom.resultVid.classList.add('hidden');
  }
}

function showIdle() {
  dom.feedLoading.classList.add('hidden');
  dom.liveFeed.src = '';
  dom.liveFeed.classList.add('hidden');
  dom.resultImg.classList.add('hidden');
  dom.resultVid.classList.add('hidden');
  dom.placeholder.classList.remove('hidden');
}

function setStatus(state) {
  // state: 'active' | 'idle' | 'alert'
  dom.sysDot.className = 'status-dot' + (state === 'idle' ? ' idle' : state === 'alert' ? ' alert' : '');
  dom.sysLabel.textContent = state === 'active' ? 'STREAM ACTIVE'
                           : state === 'alert'  ? 'TARGET DETECTED'
                           : 'SYSTEM READY';
}

const dzImage   = document.getElementById('dz-image');
const fileImage = document.getElementById('file-image');

dzImage.addEventListener('click', () => fileImage.click());
fileImage.addEventListener('change', e => handleImageFile(e.target.files[0]));

dzImage.addEventListener('dragover',  e => { e.preventDefault(); dzImage.classList.add('drag-over'); });
dzImage.addEventListener('dragleave', ()  => dzImage.classList.remove('drag-over'));
dzImage.addEventListener('drop',      e => {
  e.preventDefault();
  dzImage.classList.remove('drag-over');
  if (e.dataTransfer.files[0]) handleImageFile(e.dataTransfer.files[0]);
});

function handleImageFile(file) {
  if (!file) return;
  dzImage.querySelector('span:first-of-type').textContent = file.name.slice(0, 24);
}

document.getElementById('btn-detect-img').addEventListener('click', async () => {
  const file = fileImage.files[0];
  if (!file) { flash('DROP AN IMAGE FIRST'); return; }

  showLoading('RUNNING YOLO INFERENCE…');
  dom.feedTitle.textContent = `◈ ANALYSING — ${file.name.slice(0, 30)}`;

  const fd   = new FormData();
  const conf = document.getElementById('conf-img').value / 100;
  fd.append('file', file);
  fd.append('conf', conf);
  fd.append('iou',  dom.sIou.value / 100);

  try {
    const res  = await fetch('/api/detect/image', { method: 'POST', body: fd });
    const data = await res.json();

    if (!data.success) throw new Error(data.error || 'Detection failed');

    showResult('image', data.result_url);
    dom.feedTitle.textContent = `◈ RESULT — ${data.detection_count} TARGET(S) FOUND`;
    renderDetList(data.detections);
    setStatus(data.detection_count > 0 ? 'alert' : 'active');

  } catch (err) {
    dom.feedLoading.classList.add('hidden');
    showIdle();
    flash(`ERROR: ${err.message}`);
  }
});

const dzVideo   = document.getElementById('dz-video');
const fileVideo = document.getElementById('file-video');

dzVideo.addEventListener('click', () => fileVideo.click());
fileVideo.addEventListener('change', e => {
  if (e.target.files[0])
    dzVideo.querySelector('span:first-of-type').textContent = e.target.files[0].name.slice(0, 24);
});
dzVideo.addEventListener('dragover',  e => { e.preventDefault(); dzVideo.classList.add('drag-over'); });
dzVideo.addEventListener('dragleave', ()  => dzVideo.classList.remove('drag-over'));
dzVideo.addEventListener('drop',      e => {
  e.preventDefault(); dzVideo.classList.remove('drag-over');
  if (e.dataTransfer.files[0]) { fileVideo.files = e.dataTransfer.files; }
  dzVideo.querySelector('span:first-of-type').textContent = e.dataTransfer.files[0]?.name?.slice(0,24) || 'file dropped';
});

document.getElementById('btn-detect-vid').addEventListener('click', async () => {
  const file = fileVideo.files[0];
  if (!file) { flash('DROP A VIDEO FIRST'); return; }

  showLoading('PROCESSING VIDEO — THIS MAY TAKE SEVERAL MINUTES…');
  dom.feedTitle.textContent = `◈ PROCESSING — ${file.name.slice(0, 30)}`;

  const fd = new FormData();
  fd.append('file', file);
  fd.append('conf', dom.sConf.value / 100);
  fd.append('iou',  dom.sIou.value  / 100);

  try {
    const res  = await fetch('/api/detect/video', { method: 'POST', body: fd });
    const data = await res.json();

    if (!data.success) throw new Error(data.error || 'Processing failed');

    showResult('video', data.result_url);
    dom.feedTitle.textContent =
      `◈ VIDEO — ${data.frames_processed}f · ${data.total_detections} targets`;

  } catch (err) {
    dom.feedLoading.classList.add('hidden');
    showIdle();
    flash(`ERROR: ${err.message}`);
  }
});

function startStream(endpoint, urlParam, label, src) {
  if (streaming) stopStream();

  const streamUrl = urlParam ? `${endpoint}?url=${encodeURIComponent(urlParam)}` : endpoint;

  showResult('stream', streamUrl);
  dom.feedTitle.textContent = `◈ LIVE — ${label.toUpperCase()}`;
  streaming = true;
  setStatus('active');

  document.getElementById(`btn-start-${src}`).classList.add('hidden');
  document.getElementById(`btn-stop-${src}`).classList.remove('hidden');
}

async function stopStream() {
  if (!streaming) return;
  streaming = false;

  await fetch('/api/stream/stop', { method: 'POST' }).catch(() => {});

  showIdle();
  dom.feedTitle.textContent = '◈ STREAM STOPPED';
  setStatus('idle');

  document.querySelectorAll('.btn-stop').forEach(b => b.classList.add('hidden'));
  document.querySelectorAll('.btn-run').forEach(b => b.classList.remove('hidden'));
}

document.getElementById('btn-start-webcam').addEventListener('click', () =>
  startStream('/api/stream/webcam', null, 'webcam device-0', 'webcam'));
document.getElementById('btn-stop-webcam').addEventListener('click', stopStream);

document.getElementById('btn-start-rtsp').addEventListener('click', () => {
  const url = document.getElementById('url-rtsp').value.trim();
  if (!url) { flash('ENTER RTSP URL'); return; }
  startStream('/api/stream/rtsp', url, `RTSP ${url.slice(0,30)}`, 'rtsp');
});
document.getElementById('btn-stop-rtsp').addEventListener('click', stopStream);

document.getElementById('btn-start-ip').addEventListener('click', () => {
  const url = document.getElementById('url-ip').value.trim();
  if (!url) { flash('ENTER CAMERA URL'); return; }
  startStream('/api/stream/ip', url, `IP CAM ${url.slice(0,30)}`, 'ip');
});
document.getElementById('btn-stop-ip').addEventListener('click', stopStream);

document.getElementById('btn-start-rtmp').addEventListener('click', () => {
  const url = document.getElementById('url-rtmp').value.trim() || 'rtmp://localhost/live/stream';
  startStream('/api/stream/rtmp', url, `RTMP DRONE`, 'rtmp');
});
document.getElementById('btn-stop-rtmp').addEventListener('click', stopStream);

document.getElementById('btn-start-network').addEventListener('click', () => {
  const url = document.getElementById('url-network').value.trim();
  if (!url) { flash('ENTER FEED URL'); return; }
  startStream('/api/stream/network', url, `NET FEED ${url.slice(0,24)}`, 'network');
});
document.getElementById('btn-stop-network').addEventListener('click', stopStream);

function connectSSE() {
  if (eventSource) { eventSource.close(); }

  eventSource = new EventSource('/api/events');

  eventSource.addEventListener('stats', e => {
    const d = JSON.parse(e.data);
    updateMetrics(d);
  });

  eventSource.addEventListener('detection', e => {
    const d = JSON.parse(e.data);
    if (autoLog) addLogEntry(d);
  });

  eventSource.onerror = () => {
    setTimeout(connectSSE, 3000);
  };
}
connectSSE();

function updateMetrics(data) {
  dom.mCurrent.textContent = data.current  ?? 0;
  dom.mTotal.textContent   = data.total    ?? 0;
  dom.mConf.textContent    = data.avg_conf ? (data.avg_conf * 100).toFixed(0) + '%' : '—';
  dom.mFps.textContent     = data.fps      ?? 0;

  dom.badgeFps.textContent = `${data.fps ?? 0} FPS`;
  dom.badgeDet.textContent = `${data.current ?? 0} TARGETS`;

  if ((data.current ?? 0) > 0) {
    dom.mCurrent.closest('.metric-card').classList.remove('flash');
    void dom.mCurrent.closest('.metric-card').offsetWidth; // reflow
    dom.mCurrent.closest('.metric-card').classList.add('flash');
    setStatus('alert');
  }
}

function renderDetList(detections) {
  if (!detections || detections.length === 0) {
    dom.detList.innerHTML = '<p class="no-data">No detections</p>';
    return;
  }
  dom.detList.innerHTML = detections.map(d => `
    <div class="det-item">
      <span class="det-id">#${d.id}</span>
      <span class="det-class">${d.class.toUpperCase()}</span>
      <div class="det-bar">
        <div class="det-bar-fill" style="width:${(d.confidence*100).toFixed(0)}%"></div>
      </div>
      <span class="det-conf">${(d.confidence*100).toFixed(0)}%</span>
    </div>
  `).join('');
}

function addLogEntry(event) {
  const ph = dom.eventLog.querySelector('.placeholder-log');
  if (ph) ph.remove();

  const ts    = new Date(event.timestamp);
  const time  = ts.toISOString().slice(11, 19);
  const count = event.detection_count;
  const level = count >= 3 ? 'alert' : 'new';

  const entry = document.createElement('div');
  entry.className = `log-entry ${level}`;
  entry.innerHTML =
    `<span class="log-time">${time}</span>` +
    `[${event.source.toUpperCase()}] ` +
    `<span class="log-count">${count} target${count !== 1 ? 's' : ''}</span>`;

  dom.eventLog.insertBefore(entry, dom.eventLog.firstChild);

  const entries = dom.eventLog.querySelectorAll('.log-entry:not(.placeholder-log)');
  if (entries.length > 60) entries[entries.length - 1].remove();
}

document.getElementById('btn-clear-log').addEventListener('click', () => {
  dom.eventLog.innerHTML = '<div class="log-entry placeholder-log">Log cleared</div>';
});

document.getElementById('btn-screenshot').addEventListener('click', () => {
  const img = dom.liveFeed.classList.contains('hidden') ? dom.resultImg : dom.liveFeed;
  if (!img.src || img.src === window.location.href) return;

  const canvas = document.createElement('canvas');
  canvas.width  = img.naturalWidth  || img.width;
  canvas.height = img.naturalHeight || img.height;
  canvas.getContext('2d').drawImage(img, 0, 0);

  const a = document.createElement('a');
  a.href     = canvas.toDataURL('image/png');
  a.download = `camnet_${Date.now()}.png`;
  a.click();
});

document.getElementById('btn-fullscreen').addEventListener('click', () => {
  const el = dom.feedWrap;
  if (!document.fullscreenElement) {
    el.requestFullscreen().catch(() => {});
  } else {
    document.exitFullscreen();
  }
});

function flash(msg) {
  const prev = dom.sysLabel.textContent;
  dom.sysLabel.textContent = msg;
  dom.sysDot.classList.add('alert');
  setTimeout(() => {
    dom.sysLabel.textContent = prev;
    dom.sysDot.classList.remove('alert');
  }, 2400);
}

document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && streaming) stopStream();
});

(async () => {
  try {
    const res  = await fetch('/api/logs?limit=10');
    const data = await res.json();
    if (data.logs && data.logs.length > 0) {
      data.logs.forEach(addLogEntry);
    }
  } catch (_) { /* server may not be ready yet */ }
})();