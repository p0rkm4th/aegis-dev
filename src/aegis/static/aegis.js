const nodes = document.getElementById('nodes');
const edges = document.getElementById('edges');
const refresh = document.getElementById('refresh');
const nodeFilter = document.getElementById('node-filter');
const nodeFilterStatus = document.getElementById('node-filter-status');
const messageTimeoutMs = 120000;
const refreshRequestTimeoutMs = 10000;
const pendingStorageKey = 'aegis.pending-request';
const sessionStorageKey = 'aegis.session-id';
const contextStorageKey = 'aegis.context-correlation';
const themeStorageKey = 'aegis.theme';
const recoveryPollMs = 5000;
const recoveryRequestTimeoutMs = 10000;
const maxRecoveryPolls = 60;
let pendingCorrelationId = null;
let pendingOutcomeUnknown = false;
let conversationSessionId = null;
let conversationContextCorrelationId = null;
let selectedNode = null;
let renderedNodeCards = new Map();
let renderedNodeText = new Map();
let renderedNodeViews = new Map();
let renderedEdgeRows = [];
let activeView = 'home';
let pendingCapabilityFocus = null;
let latestResearch = null;
let researchHistory = [];
let authorizedProjectionLoaded = false;
let recoveryPollScheduled = false;
let recoveryPollAttempts = 0;
const themeToggle = document.getElementById('theme-toggle');
function applyTheme(theme) {
  const selected = theme === 'light' ? 'light' : 'dark';
  document.documentElement.dataset.theme = selected;
  themeToggle.textContent = selected === 'dark' ? 'Light mode' : 'Dark mode';
  themeToggle.setAttribute('aria-label', `Switch to ${selected === 'dark' ? 'light' : 'dark'} mode`);
  try { localStorage.setItem(themeStorageKey, selected); } catch (_) { /* optional */ }
}
let initialTheme = 'dark';
try {
  const savedTheme = localStorage.getItem(themeStorageKey);
  initialTheme = savedTheme || (matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
} catch (_) { /* optional */ }
applyTheme(initialTheme);
themeToggle.addEventListener('click', () =>
  applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'));
try {
  const savedSession = sessionStorage.getItem(sessionStorageKey);
  if (savedSession &&
      /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(savedSession))
    conversationSessionId = savedSession;
  else {
    conversationSessionId = crypto.randomUUID();
    sessionStorage.setItem(sessionStorageKey, conversationSessionId);
  }
} catch (_) { conversationSessionId = crypto.randomUUID(); }
try {
  const savedContext = localStorage.getItem(contextStorageKey);
  if (savedContext &&
      /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(savedContext))
    conversationContextCorrelationId = savedContext;
} catch (_) { /* local storage is optional */ }
function persistConversationContext(correlationId) {
  conversationContextCorrelationId = correlationId;
  try { localStorage.setItem(contextStorageKey, correlationId); } catch (_) { /* optional */ }
}
function clearConversationContext() {
  conversationContextCorrelationId = null;
  try { localStorage.removeItem(contextStorageKey); } catch (_) { /* optional */ }
}
const retryableCodes = new Set([
  'identity_unavailable', 'state_unavailable', 'request_unavailable', 'request_timeout'
]);
const errorLabels = Object.freeze({
  identity_unavailable: 'Identity unavailable', state_access_denied: 'Authorization denied',
  state_unavailable: 'State unavailable', health_unavailable: 'Runtime status unavailable',
  request_denied: 'Request denied', request_unavailable: 'Request unavailable',
  response_unavailable: 'Response unavailable', request_timeout: 'Request timed out',
  invalid_request: 'Invalid request', request_too_large: 'Request too large'
});
function errorLabel(code) { return errorLabels[code] || 'Request failed'; }
const lifecycleLabels = Object.freeze({
  proposed: 'Proposed', validated: 'Validated', authorized: 'Authorized',
  approval_required: 'Approval required', executing: 'Executing', observed: 'Observed',
  verified: 'Verified', completed: 'Completed', failed: 'Failed', blocked: 'Blocked',
  unknown: 'Outcome unknown'
});
function lifecycleLabel(state) { return lifecycleLabels[state] || state; }
function setOutcomeStatus(state) {
  const badge = document.getElementById('status-badge');
  const normalized = state || 'idle';
  badge.dataset.state = normalized;
  badge.textContent = normalized === 'idle' ? 'Ready' : lifecycleLabel(normalized);
}
function persistPendingRequest(utterance, correlationId, outcomeUnknown = false) {
  try {
    sessionStorage.setItem(pendingStorageKey, JSON.stringify(
    {utterance, correlation_id: correlationId, session_id: conversationSessionId,
      outcome_unknown: outcomeUnknown}));
  } catch (_) { /* session storage is optional; Core correlation remains authoritative. */ }
}
function clearPendingRequest() {
  try { sessionStorage.removeItem(pendingStorageKey); } catch (_) { /* optional storage */ }
}
function resizeComposer() {
  const input = document.getElementById('utterance');
  input.style.height = 'auto';
  input.style.height = `${Math.min(input.scrollHeight, 176)}px`;
  input.style.overflowY = input.scrollHeight > 176 ? 'auto' : 'hidden';
}
function appendConversationMessage(kind, text) {
  const conversation = document.getElementById('conversation');
  conversation.querySelector('.conversation-empty')?.remove();
  const wasNearBottom = conversation.scrollHeight - conversation.scrollTop - conversation.clientHeight < 80;
  const line = document.createElement('li');
  line.className = kind;
  line.textContent = text;
  conversation.append(line);
  if (wasNearBottom || kind === 'aegis-message')
    line.scrollIntoView({block: 'nearest', behavior: 'smooth'});
}
const composer = document.getElementById('utterance');
composer.addEventListener('input', resizeComposer);
composer.addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    if (!document.querySelector('#chat button').disabled)
      document.getElementById('chat').requestSubmit();
  }
});
resizeComposer();
function restorePendingRequest() {
  try {
    const saved = JSON.parse(sessionStorage.getItem(pendingStorageKey) || 'null');
    if (!saved || typeof saved.utterance !== 'string' || !saved.utterance.trim() ||
        typeof saved.correlation_id !== 'string' ||
        !/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i
          .test(saved.correlation_id)) {
      clearPendingRequest(); return;
    }
    pendingCorrelationId = saved.correlation_id;
    pendingOutcomeUnknown = saved.outcome_unknown === true;
    document.getElementById('utterance').value = saved.utterance;
    document.getElementById('activity').textContent = pendingOutcomeUnknown
      ? 'Outcome unknown. Recheck status will read canonical state without repeating the mutation.'
      : 'A previous request may still be in progress. Retry uses the same correlation.';
    document.querySelector('#chat button').textContent = pendingOutcomeUnknown
      ? 'Recheck status' : 'Retry';
  } catch (_) { clearPendingRequest(); }
  resizeComposer();
}
function scheduleRecoveryPoll() {
  if (recoveryPollScheduled || !pendingCorrelationId || recoveryPollAttempts >= maxRecoveryPolls) {
    if (pendingCorrelationId && recoveryPollAttempts >= maxRecoveryPolls) {
      document.getElementById('activity').textContent =
        'Status checks paused after five minutes. Retry remains explicit.';
    }
    return;
  }
  recoveryPollAttempts += 1;
  recoveryPollScheduled = true;
  setTimeout(() => {
    recoveryPollScheduled = false;
    recoverPendingRequest();
  }, recoveryPollMs);
}
async function recoverPendingRequest() {
  if (!pendingCorrelationId) return;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), recoveryRequestTimeoutMs);
  try {
    const response = await apiFetch(
      `/api/request-status?correlation_id=${pendingCorrelationId}`, {signal: controller.signal});
    if (!response.ok) {
      if (response.status === 401 || response.status === 403) {
        clearAuthorizedDisplays();
        document.getElementById('state-status').textContent =
          'Authorization lost; authorized state cleared.';
      } else {
        document.getElementById('activity').textContent =
          'Status check unavailable; retry remains explicit.';
        scheduleRecoveryPoll();
      }
      return;
    }
    const status = await response.json();
    if (status.state === 'unknown') {
      pendingOutcomeUnknown = true;
      document.querySelector('#chat button').textContent = 'Recheck status';
      persistPendingRequest(document.getElementById('utterance').value, pendingCorrelationId, true);
      document.getElementById('activity').textContent =
        'Outcome unknown; checking canonical status. No mutation will be repeated automatically.';
      scheduleRecoveryPoll(); return;
    }
    const inProgressStates = new Set([
      'proposed', 'validated', 'authorized', 'executing', 'observed'
    ]);
    if (inProgressStates.has(status.state)) {
      document.querySelector('#chat button').textContent = 'Recheck status';
      document.getElementById('activity').textContent =
        `Request status recovered: ${lifecycleLabel(status.state)}. Recheck status remains read-only.`;
      scheduleRecoveryPoll();
      return;
    }
    const recoveredCorrelationId = pendingCorrelationId;
    recoveryPollAttempts = 0;
    document.getElementById('answer').textContent = status.message || 'Request status recovered.';
    document.getElementById('activity').textContent = `Status: ${lifecycleLabel(status.state)}`;
    if (status.retryable === true) {
      pendingOutcomeUnknown = false;
      document.querySelector('#chat button').textContent = 'Retry';
      persistPendingRequest(document.getElementById('utterance').value, recoveredCorrelationId, false);
    } else {
      pendingCorrelationId = null; pendingOutcomeUnknown = false; clearPendingRequest();
      document.querySelector('#chat button').textContent = 'Send';
    }
    if (status.state === 'completed') refreshState();
  } catch (_) {
    document.getElementById('activity').textContent =
      'Status check unavailable; retry remains explicit.';
    scheduleRecoveryPoll();
  } finally {
    clearTimeout(timeout);
  }
}
async function loadHealth() {
  const response = await fetchWithTimeout('/api/health'); const report = await response.json();
  if (!response.ok) {
    const error = new Error(report.error || 'Runtime status unavailable.');
    error.code = report.code || 'health_unavailable'; throw error;
  }
  const required = (report.components || []).filter(component => component.required);
  const ready = report.ready ? 'READY' : 'NOT READY';
  document.getElementById('health').textContent =
    `Runtime: ${ready} · ${required.filter(component => component.healthy).length}` +
    `/${required.length} required checks OK`;
  const healthDetails = document.getElementById('health-details');
  healthDetails.replaceChildren(...(report.components || []).map(component => {
    const item = document.createElement('li');
    const requirement = component.required ? 'required' : 'optional';
    const status = component.healthy ? 'OK' : 'FAIL';
    item.textContent = `${component.name}: ${status} (${requirement}) — ${component.detail || ''}`;
    return item;
  }));
}
function renderDetailValue(value) {
  if (value === null) {
    const empty = document.createElement('span'); empty.textContent = '—'; return empty;
  }
  if (Array.isArray(value)) {
    const list = document.createElement('ul');
    if (!value.length) {
      const item = document.createElement('li');
      item.textContent = 'No canonical records available';
      list.append(item);
    }
    value.forEach(item => {
      const row = document.createElement('li');
      row.append(renderDetailValue(item)); list.append(row);
    });
    if (value.length > 12) {
      const disclosure = document.createElement('details');
      const summary = document.createElement('summary');
      summary.textContent = `Show ${value.length} canonical records`;
      disclosure.append(summary, list);
      return disclosure;
    }
    return list;
  }
  if (typeof value === 'object') {
    const definition = document.createElement('dl');
    Object.entries(value).forEach(([key, item]) => {
      const term = document.createElement('dt');
      const label = key.replaceAll('_', ' ');
      term.textContent = label.replace(/\\b\\w/g, character => character.toUpperCase());
      const description = document.createElement('dd'); description.append(renderDetailValue(item));
      definition.append(term, description);
    });
    return definition;
  }
  const text = document.createElement('span'); text.textContent = String(value); return text;
}
function clearAuthorizedDisplays() {
  nodes.replaceChildren(); edges.replaceChildren();
  renderedNodeCards = new Map();
  renderedNodeText = new Map();
  renderedNodeViews = new Map();
  renderedEdgeRows = [];
  authorizedProjectionLoaded = false;
  nodeFilter.value = '';
  document.getElementById('detail').replaceChildren();
  document.getElementById('activity').textContent = '';
  document.getElementById('step-status').textContent = '';
  nodeFilterStatus.textContent = 'Authorized nodes unavailable.';
  document.getElementById('answer').textContent = '';
  document.getElementById('conversation').replaceChildren();
  const empty = document.createElement('li');
  empty.className = 'conversation-empty';
  empty.textContent = 'Your conversation will appear here.';
  document.getElementById('conversation').append(empty);
  clearConversationContext();
  pendingCorrelationId = null;
  recoveryPollAttempts = 0;
  clearPendingRequest();
  document.querySelector('#chat button').textContent = 'Send';
}
function clearHealthDetails() {
  document.getElementById('health-details').replaceChildren();
}
function applyNodeFilter() {
  if (!authorizedProjectionLoaded) {
    nodeFilterStatus.textContent = 'Authorized nodes unavailable.';
    return;
  }
  const query = nodeFilter.value.trim().toLowerCase();
  let visibleCount = 0;
  renderedNodeCards.forEach((card, nodeId) => {
    const matchesView = activeView === 'home' || renderedNodeViews.get(nodeId)?.includes(activeView);
    card.hidden = !matchesView || Boolean(query && !renderedNodeText.get(nodeId).includes(query));
    if (!card.hidden) visibleCount += 1;
  });
  renderedEdgeRows.forEach(({item, edge}) => {
    item.hidden = renderedNodeCards.get(edge.source)?.hidden !== false
      && renderedNodeCards.get(edge.target)?.hidden !== false;
  });
  nodeFilterStatus.textContent = activeView === 'research'
    ? 'Research is available through conversation; ask for current, sourced information.'
    : query
    ? (visibleCount
      ? `Showing ${visibleCount} of ${renderedNodeCards.size} authorized nodes.`
      : `No authorized nodes match “${nodeFilter.value.trim()}”.`)
    : `${renderedNodeCards.size} authorized nodes.`;
}
function renderResearchSummary() {
  if (activeView !== 'research') return;
  const panel = document.getElementById('detail'); panel.replaceChildren();
  const section = document.createElement('section'); section.className = 'detail-card';
  const title = document.createElement('h3'); title.textContent = 'Save sourced research to Workspace';
  const form = document.createElement('form'); form.setAttribute('aria-label', 'Save research to Workspace');
  const query = document.createElement('input'); query.type = 'search'; query.required = true;
  query.placeholder = 'Research question'; query.setAttribute('aria-label', 'Research question');
  const path = document.createElement('input'); path.required = true;
  path.placeholder = 'Artifact path, e.g. notes.md'; path.setAttribute('aria-label', 'Research artifact path');
  const submit = document.createElement('button'); submit.type = 'submit'; submit.textContent = 'Research and save notes';
  form.append(query, path, submit);
  form.addEventListener('submit', event => {
    event.preventDefault();
    const question = query.value.trim(); const targetPath = path.value.trim();
    if (!question || !targetPath) return;
    document.getElementById('utterance').value = `Research ${question} and save notes as ${targetPath}`;
    document.getElementById('chat').requestSubmit();
  });
  const boundary = document.createElement('p'); boundary.className = 'muted';
  boundary.textContent = 'Public evidence remains non-canonical; saving requires the normal Core authorization and scoped Workspace verification.';
  section.append(title, form, boundary); panel.append(section);
  const draftSection = document.createElement('section'); draftSection.className = 'detail-card';
  const draftTitle = document.createElement('h3'); draftTitle.textContent = 'Draft a researched message';
  const draftForm = document.createElement('form'); draftForm.setAttribute('aria-label', 'Draft researched message');
  const recipient = document.createElement('input'); recipient.required = true;
  recipient.placeholder = 'Recipient'; recipient.setAttribute('aria-label', 'Research draft recipient');
  const subject = document.createElement('input'); subject.required = true;
  subject.placeholder = 'Subject'; subject.setAttribute('aria-label', 'Research draft subject');
  const draftQuery = document.createElement('input'); draftQuery.type = 'search'; draftQuery.required = true;
  draftQuery.placeholder = 'Research question'; draftQuery.setAttribute('aria-label', 'Research draft question');
  const draftPath = document.createElement('input'); draftPath.required = true;
  draftPath.placeholder = 'Artifact path, e.g. drafts/research.md';
  draftPath.setAttribute('aria-label', 'Research draft artifact path');
  const draftSubmit = document.createElement('button'); draftSubmit.type = 'submit';
  draftSubmit.textContent = 'Research and create unsent draft';
  draftForm.append(recipient, subject, draftQuery, draftPath, draftSubmit);
  draftForm.addEventListener('submit', event => {
    event.preventDefault();
    const values = [recipient.value.trim(), subject.value.trim(), draftQuery.value.trim(), draftPath.value.trim()];
    if (values.some(value => !value)) return;
    document.getElementById('utterance').value =
      `Draft researched message to ${values[0]} with subject ${values[1]} about ${values[2]}, save it as ${values[3]}`;
    document.getElementById('chat').requestSubmit();
  });
  const draftBoundary = document.createElement('p'); draftBoundary.className = 'muted';
  draftBoundary.textContent = 'Research is fixed as non-canonical evidence before the unsent draft is written; no message is sent.';
  draftSection.append(draftTitle, draftForm, draftBoundary); panel.append(draftSection);
  const sendSection = document.createElement('section'); sendSection.className = 'detail-card';
  const sendTitle = document.createElement('h3'); sendTitle.textContent = 'Send sourced research';
  const sendForm = document.createElement('form'); sendForm.setAttribute('aria-label', 'Send sourced research');
  const sendQuery = document.createElement('input'); sendQuery.type = 'search'; sendQuery.required = true;
  sendQuery.placeholder = 'Research question'; sendQuery.setAttribute('aria-label', 'Research send question');
  const sendSubmit = document.createElement('button'); sendSubmit.type = 'submit';
  sendSubmit.textContent = 'Send research'; sendForm.append(sendQuery, sendSubmit);
  sendForm.addEventListener('submit', event => {
    event.preventDefault(); const question = sendQuery.value.trim(); if (!question) return;
    document.getElementById('utterance').value = `Text me the research about ${question}`;
    document.getElementById('chat').requestSubmit();
  });
  const sendBoundary = document.createElement('p'); sendBoundary.className = 'muted';
  sendBoundary.textContent = 'Public evidence is fixed before sending and remains non-canonical; provider acceptance is not delivery proof.';
  sendSection.append(sendTitle, sendForm, sendBoundary); panel.append(sendSection);
  if (!latestResearch) {
    const empty = document.createElement('p'); empty.className = 'muted';
    empty.textContent = 'No saved research result is available yet. Ask a sourced question to add one.';
    panel.append(empty);
    return;
  }
  const heading = document.createElement('p');
  heading.textContent = `Latest external evidence · ${latestResearch.sources.length} source(s)`;
  panel.append(heading);
  panel.append(renderDetailValue({
    query: latestResearch.query,
    provider: latestResearch.provider_id,
    retrieved_at: latestResearch.retrieved_at,
    evidence_status: 'external evidence; not canonical personal truth',
    summary: latestResearch.summary,
  }));
  const sources = document.createElement('section'); sources.className = 'detail-card';
  const sourceHeading = document.createElement('h3'); sourceHeading.textContent = 'Sources';
  const sourceList = document.createElement('ul');
  latestResearch.sources.slice(0, 5).forEach(source => {
    const item = document.createElement('li');
    const link = document.createElement('a'); link.href = source.url; link.target = '_blank';
    link.rel = 'noopener noreferrer'; link.textContent = source.title || source.url;
    item.append(link); sourceList.append(item);
  });
  sources.append(sourceHeading, sourceList); panel.append(sources);
  if (researchHistory.length > 1) {
    const recent = document.createElement('section'); recent.className = 'detail-card';
    const recentHeading = document.createElement('h3'); recentHeading.textContent = 'Recent research';
    const recentList = document.createElement('ul');
    researchHistory.slice(0, 10).forEach(item => {
      const row = document.createElement('li');
      const button = document.createElement('button'); button.type = 'button';
      button.textContent = item.query || 'Untitled research';
      button.addEventListener('click', () => {
        latestResearch = item;
        renderResearchSummary();
      });
      row.append(button);
      if (item.retrieved_at) row.append(document.createTextNode(` · retrieved ${item.retrieved_at}`));
      recentList.append(row);
    });
    recent.append(recentHeading, recentList); panel.append(recent);
  }
}
async function loadResearch() {
  try {
    const response = await fetchWithTimeout('/api/research');
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'Research state unavailable.');
    researchHistory = Array.isArray(payload.results) ? payload.results : [];
    const result = researchHistory[0] || null;
    if (result) latestResearch = result;
    renderResearchSummary();
  } catch (_) {
    if (activeView !== 'research') return;
    const panel = document.getElementById('detail');
    const boundary = document.createElement('p'); boundary.className = 'muted';
    boundary.textContent = 'Saved research is unavailable; public evidence remains non-canonical.';
    panel.append(boundary);
  }
}
nodeFilter.addEventListener('input', applyNodeFilter);
document.querySelectorAll('[data-view]').forEach(button => button.addEventListener('click', () => {
  activeView = button.dataset.view || 'home';
  document.querySelectorAll('[data-view]').forEach(item =>
    item.setAttribute('aria-current', item === button ? 'page' : 'false'));
  const input = document.getElementById('utterance');
  input.placeholder = activeView === 'research' ? 'Ask for sourced research…' : 'Talk to AEGIS…';
  const viewCopy = {
    home: ['Today', 'Your conversation and authorized world at a glance.'],
    tasks: ['Tasks', 'Open and completed work from authorized canonical state.'],
    calendar: ['Calendar', 'Events and appointments currently visible to you.'],
    household: ['Household', 'Shared chores, groceries, and obligations.'],
    finance: ['Finance', 'Private balances and imported transactions with freshness.'],
    systems: ['Systems', 'Authorized hosts, services, and network state.'],
    weather: ['Weather', 'Current public conditions for explicit coordinates.'],
    'air-quality': ['Air quality', 'Current public air quality for explicit coordinates.'],
    devices: ['Devices', 'Authorized device state and bounded controls.'],
    communications: ['Communications', 'Authorized drafts and provider outcomes; delivery is never inferred.'],
    documents: ['Documents', 'Authorized documents and bounded transformations.'],
    'daily-driver': ['Daily driver', 'What is usable now, and what still needs a provider or approval.'],
    research: ['Research', 'Ask for current public information with sources.'],
    packs: ['Packs & capabilities', 'Installed capability areas and their current status.'],
    objectives: ['Active objectives', 'Objectives remain grounded in their canonical lifecycle.'],
    constellation: ['Constellation', 'The authorized semantic map: context and navigation, never authority.'],
    workspace: ['Workspace', 'Scoped artifacts and bounded digital work will appear here.']
    ,compositions: ['Compositions', 'Cross-capability workflows available through the trusted Core.']
  }[activeView] || ['Today', 'Your conversation and authorized world at a glance.'];
  document.getElementById('view-title').textContent = viewCopy[0];
  document.getElementById('view-description').textContent = viewCopy[1];
  if (activeView === 'research') { input.focus(); loadResearch(); }
  if (activeView === 'workspace') loadWorkspace();
  if (activeView === 'compositions') loadCompositions();
  if (activeView === 'packs') loadPacks();
  if (activeView === 'calendar') loadCalendar();
  if (activeView === 'devices') loadDevices();
  if (activeView === 'communications') loadCommunications();
  if (activeView === 'documents') loadDocuments();
  if (activeView === 'daily-driver') loadDailyDriver();
  if (activeView === 'systems') loadSystems();
  if (activeView === 'weather') loadWeather();
  if (activeView === 'air-quality') loadAirQuality();
  if (activeView === 'home') loadToday();
  if (activeView === 'tasks') loadTasks();
  if (activeView === 'household') loadHousehold();
  if (activeView === 'finance') loadFinance();
  if (activeView === 'objectives') loadObjectives();
  if (activeView === 'constellation') {
    document.querySelector('.secondary').open = true;
    loadState();
  }
  renderResearchSummary();
  applyNodeFilter();
}));
async function apiFetch(resource, options = {}) {
  const token = document.querySelector('meta[name="aegis-session-token"]')?.content;
  const headers = new Headers(options.headers || {});
  if (token) headers.set('x-aegis-session', token);
  const response = await fetch(resource, {...options, headers});
  if (response.status !== 401 || resource === '/') return response;
  // A normal owner-service token rotation should not discard the persistent
  // conversation. Refresh the transport token once, then let the endpoint's
  // normal authorization response decide whether the request is allowed.
  const root = await fetch('/');
  if (!root.ok) return response;
  const html = await root.text();
  const refreshed = html.match(/<meta name="aegis-session-token" content="([^"]*)"/);
  if (!refreshed || !refreshed[1]) return response;
  const meta = document.querySelector('meta[name="aegis-session-token"]');
  if (meta) meta.content = refreshed[1];
  const retryHeaders = new Headers(options.headers || {});
  retryHeaders.set('x-aegis-session', refreshed[1]);
  return fetch(resource, {...options, headers: retryHeaders});
}
async function fetchWithTimeout(resource, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), refreshRequestTimeoutMs);
  try {
    return await apiFetch(resource, {...options, signal: controller.signal});
  } finally {
    clearTimeout(timeout);
  }
}
async function loadState() {
  const response = await fetchWithTimeout('/api/constellation');
  const state = await response.json();
  if (!response.ok) {
    const error = new Error(state.error || 'State is unavailable.');
    error.code = state.code || 'state_unavailable'; throw error;
  }
  document.getElementById('state-status').textContent = '';
  nodeFilterStatus.textContent = '';
  const details = state.details || {};
  selectedNode = null;
  document.getElementById('detail').replaceChildren();
  const nodeCards = new Map();
  const selectNode = (node, card) => {
    if (selectedNode) selectedNode.setAttribute('aria-pressed', 'false');
    selectedNode = card; card.setAttribute('aria-pressed', 'true');
    const panel = document.getElementById('detail'); panel.replaceChildren();
    const heading = document.createElement('p');
    heading.textContent = `${node.label}: ${node.detail || 'No detail'}`;
    panel.append(heading);
    const nodeViews = (renderedNodeViews.get(node.id) || []).filter(view => view !== 'home');
    if (nodeViews.length) {
      const navigation = document.createElement('p');
      const targetView = nodeViews[0];
      const viewButton = document.createElement('button'); viewButton.type = 'button';
      const targetNav = document.querySelector(`[data-view="${targetView}"]`);
      viewButton.textContent = `Open ${targetNav?.textContent || targetView} view`;
      viewButton.addEventListener('click', () => targetNav?.click());
      navigation.append(viewButton); panel.append(navigation);
    }
    const focus = document.createElement('button'); focus.type = 'button';
    focus.textContent = `Ask about ${node.label}`;
    focus.addEventListener('click', () => {
      const input = document.getElementById('utterance');
      input.value = `Tell me about ${node.label}`; input.focus();
    });
    panel.append(focus);
    if (Object.prototype.hasOwnProperty.call(details, node.id)) {
      panel.append(renderDetailValue(details[node.id]));
    }
  };
  renderedNodeCards = new Map();
  renderedNodeText = new Map();
  renderedNodeViews = new Map();
  renderedEdgeRows = [];
  const constellationGroups = new Map();
  (state.nodes || []).forEach(node => {
    const card = document.createElement('button'); card.className = 'node'; card.type = 'button';
    const category = node.category || 'domain';
    card.dataset.category = category;
    card.setAttribute('aria-pressed', 'false');
    card.setAttribute('aria-label', `${node.label}: ${node.detail || 'No detail'}`);
    const title = document.createElement('h2'); title.textContent = node.label;
    const detail = document.createElement('p'); detail.textContent = node.detail || '';
    card.addEventListener('click', () => selectNode(node, card));
    nodeCards.set(node.id, card);
    renderedNodeText.set(node.id, `${node.label} ${node.detail || ''}`.toLowerCase());
    const searchable = `${node.id} ${node.label} ${node.detail || ''} ${node.category || ''}`.toLowerCase();
    const views = ['home'];
    const navigableViews = new Set(
      [...document.querySelectorAll('[data-view]')].map(item => item.dataset.view)
    );
    if (node.detail_view && navigableViews.has(node.detail_view)) views.push(node.detail_view);
    if (/task/.test(searchable)) views.push('tasks');
    if (/event|calendar|appointment/.test(searchable)) views.push('calendar');
    if (/household|chore|obligation|grocery|kitchen/.test(searchable)) views.push('household');
    if (/homelab|network|infrastructure|system/.test(searchable)) views.push('systems');
    if (node.id.startsWith('pack-')) views.push('packs');
    if (/objective|capability|need/.test(searchable)) views.push('objectives');
    if (/workspace|artifact|file/.test(searchable)) views.push('workspace');
    if (/document/.test(searchable)) views.push('documents');
    if (/daily.driver|readiness|release/.test(searchable)) views.push('daily-driver');
    renderedNodeViews.set(node.id, views);
    card.append(title, detail);
    if (!constellationGroups.has(category)) constellationGroups.set(category, []);
    constellationGroups.get(category).push(card);
  });
  const categoryLabels = {
    core: 'AEGIS', domain: 'Domains & Packs', capability: 'Semantic areas',
    objective: 'Active objectives', composition: 'Bounded relationships',
  };
  const categoryOrder = ['core', 'domain', 'capability', 'objective', 'composition'];
  nodes.replaceChildren(...categoryOrder.filter(category => constellationGroups.has(category)).map(category => {
    const layer = document.createElement('section');
    layer.className = `constellation-layer constellation-layer-${category}`;
    const title = document.createElement('h3'); title.textContent = categoryLabels[category] || category;
    const items = document.createElement('div'); items.className = 'constellation-layer-nodes';
    items.append(...constellationGroups.get(category));
    layer.append(title, items); return layer;
  }));
  renderedNodeCards = nodeCards;
  authorizedProjectionLoaded = true;
  const labels = Object.fromEntries((state.nodes || []).map(node => [node.id, node.label]));
  edges.replaceChildren(...(state.edges || []).map(edge => {
    const item = document.createElement('li');
    const link = document.createElement('button'); link.type = 'button';
    link.textContent =
      `${labels[edge.source] || 'Authorized node'} → ${labels[edge.target] || 'Authorized node'}`;
    link.setAttribute(
      'aria-label', `Open relationship to ${labels[edge.target] || 'authorized node'}`);
    link.addEventListener('click', () => {
      const target = nodeCards.get(edge.target);
      if (target) { target.focus(); target.click(); }
    });
    item.append(link);
    renderedEdgeRows.push({item, edge});
    return item;
  }));
  applyNodeFilter();
}
async function loadWorkspace() {
  const panel = document.getElementById('detail');
  panel.replaceChildren();
  try {
    const response = await fetchWithTimeout('/api/workspace');
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'Workspace unavailable.');
    const heading = document.createElement('p');
    heading.textContent = payload.workspaces?.length
      ? `${payload.workspaces.length} scoped workspace(s)`
      : 'No scoped workspaces yet. Ask AEGIS to create a bounded artifact.';
    panel.append(heading);
    const filterLabel = document.createElement('label');
    filterLabel.textContent = 'Find workspace artifacts';
    const filter = document.createElement('input');
    filter.type = 'search'; filter.placeholder = 'Filter by workspace or file name';
    filter.setAttribute('aria-label', 'Find workspace artifacts');
    filterLabel.append(filter); panel.append(filterLabel);
    const filterStatus = document.createElement('p'); filterStatus.className = 'muted';
    filterStatus.setAttribute('aria-live', 'polite'); panel.append(filterStatus);
    const searchForm = document.createElement('form');
    searchForm.setAttribute('aria-label', 'Search workspace contents');
    const searchInput = document.createElement('input');
    searchInput.type = 'search'; searchInput.required = true;
    searchInput.placeholder = 'Search authorized file contents';
    searchInput.setAttribute('aria-label', 'Search authorized file contents');
    const searchSubmit = document.createElement('button');
    searchSubmit.type = 'submit'; searchSubmit.textContent = 'Search contents';
    searchForm.append(searchInput, searchSubmit);
    searchForm.addEventListener('submit', event => {
      event.preventDefault();
      if (!searchInput.value.trim()) return;
      document.getElementById('utterance').value =
        `Search my workspace for ${searchInput.value.trim()}`;
      document.getElementById('chat').requestSubmit();
    });
    panel.append(searchForm);
    const cards = [];
    const applyWorkspaceFilter = () => {
      const query = filter.value.trim().toLowerCase();
      let visible = 0;
      cards.forEach(card => {
        const matches = !query || card.dataset.searchText.includes(query);
        card.hidden = !matches;
        if (matches) visible += 1;
      });
      filterStatus.textContent = query
        ? `${visible} matching workspace(s)`
        : `${cards.length} workspace(s) shown`;
    };
    filter.addEventListener('input', applyWorkspaceFilter);
    (payload.workspaces || []).forEach(workspace => {
      const card = document.createElement('section');
      card.className = 'detail-card';
      card.dataset.searchText = [workspace.workspace_id, ...(workspace.files || [])]
        .join(' ').toLowerCase();
      const title = document.createElement('h3');
      title.textContent = `Workspace ${workspace.workspace_id || 'unknown'}`;
      card.append(title);
      (workspace.files || []).forEach(path => {
        const button = document.createElement('button');
        button.type = 'button';
        button.textContent = `View ${path}`;
        button.addEventListener('click', async () => {
          button.disabled = true;
          try {
            const response = await fetchWithTimeout(`/api/workspace/file?workspace_id=${encodeURIComponent(workspace.workspace_id)}&path=${encodeURIComponent(path)}`);
            const file = await response.json();
            if (!response.ok) throw new Error(file.error || 'File unavailable.');
            const pre = document.createElement('pre');
            pre.className = 'detail-card';
            pre.textContent = file.content || '';
            card.append(pre);
            const digest = document.createElement('p'); digest.className = 'muted';
            digest.textContent = `Observed SHA-256: ${file.sha256 || 'unavailable'}`;
            card.append(digest);
          } catch (_) {
            button.textContent = 'File unavailable';
          } finally { button.disabled = false; }
        });
        card.append(button);
        const download = document.createElement('button');
        download.type = 'button'; download.textContent = `Download ${path}`;
        download.addEventListener('click', async () => {
          download.disabled = true;
          try {
            const response = await fetchWithTimeout(`/api/workspace/file?workspace_id=${encodeURIComponent(workspace.workspace_id)}&path=${encodeURIComponent(path)}`);
            const file = await response.json();
            if (!response.ok) throw new Error(file.error || 'Download unavailable.');
            const blobUrl = URL.createObjectURL(new Blob([file.content || ''], {type: 'text/plain;charset=utf-8'}));
            const anchor = document.createElement('a'); anchor.href = blobUrl;
            anchor.download = String(path).split('/').pop() || 'artifact'; anchor.click();
            setTimeout(() => URL.revokeObjectURL(blobUrl), 0);
          } catch (_) { download.textContent = 'Download unavailable'; }
          finally { download.disabled = false; }
        });
        card.append(download);
        const send = document.createElement('button');
        send.type = 'button'; send.textContent = `Send ${path}`;
        send.addEventListener('click', () => {
          document.getElementById('utterance').value =
            `Text me the workspace artifact ${workspace.workspace_id} at ${path}`;
          document.getElementById('chat').requestSubmit();
        });
        card.append(send);
        if (path.toLowerCase().endsWith('.html') || path.toLowerCase().endsWith('.htm')) {
          const preview = document.createElement('button');
          preview.type = 'button'; preview.textContent = `Preview ${path}`;
          preview.addEventListener('click', async () => {
            preview.disabled = true;
            try {
              const response = await fetchWithTimeout(`/api/workspace/file?workspace_id=${encodeURIComponent(workspace.workspace_id)}&path=${encodeURIComponent(path)}`);
              const file = await response.json();
              if (!response.ok) throw new Error(file.error || 'Preview unavailable.');
              const frame = document.createElement('iframe');
              frame.title = `Static preview of ${path}`;
              frame.setAttribute('sandbox', '');
              frame.style.cssText = 'width:100%;min-height:18rem;border:1px solid var(--border);border-radius:.5rem;background:white;margin-top:.6rem';
              frame.srcdoc = file.content || '';
              card.append(frame);
            } catch (_) { preview.textContent = 'Preview unavailable'; }
            finally { preview.disabled = false; }
          });
          card.append(preview);
        }
      });
      panel.append(card);
      cards.push(card);
    });
    applyWorkspaceFilter();
  } catch (_) {
    panel.textContent = 'Workspace inventory is unavailable; no artifact state was changed.';
  }
}
async function loadCalendar() {
  const panel = document.getElementById('detail');
  panel.replaceChildren();
  try {
    const response = await fetchWithTimeout('/api/calendar');
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'Calendar unavailable.');
    const events = payload.events || [];
    const heading = document.createElement('p');
    heading.textContent = events.length
      ? `${events.length} authorized calendar event(s)`
      : 'No authorized calendar events are currently visible.';
    panel.append(heading);
    const form = document.createElement('form');
    form.setAttribute('aria-label', 'Create calendar event');
    const title = document.createElement('input');
    title.type = 'text'; title.required = true; title.placeholder = 'Event title';
    title.setAttribute('aria-label', 'Event title');
    const starts = document.createElement('input');
    starts.type = 'datetime-local'; starts.required = true;
    starts.setAttribute('aria-label', 'Event start');
    const ends = document.createElement('input');
    ends.type = 'datetime-local'; ends.required = true;
    ends.setAttribute('aria-label', 'Event end');
    const submit = document.createElement('button');
    submit.type = 'submit'; submit.textContent = 'Create event';
    form.append(title, starts, ends, submit);
    form.addEventListener('submit', event => {
      event.preventDefault();
      const start = new Date(starts.value);
      const end = new Date(ends.value);
      if (!title.value.trim() || Number.isNaN(start.valueOf()) || Number.isNaN(end.valueOf()) || end <= start) {
        return;
      }
      const localIso = value => {
        const offset = -value.getTimezoneOffset();
        const sign = offset >= 0 ? '+' : '-';
        const hours = String(Math.floor(Math.abs(offset) / 60)).padStart(2, '0');
        const minutes = String(Math.abs(offset) % 60).padStart(2, '0');
        const iso = `${value.getFullYear()}-${String(value.getMonth() + 1).padStart(2, '0')}-` +
          `${String(value.getDate()).padStart(2, '0')}T${String(value.getHours()).padStart(2, '0')}:` +
          `${String(value.getMinutes()).padStart(2, '0')}:00${sign}${hours}:${minutes}`;
        return iso;
      };
      const clock = start.toLocaleTimeString([], {hour: 'numeric', minute: '2-digit'});
      document.getElementById('utterance').value =
        `Create a calendar event titled ${title.value.trim()} from ${localIso(start)} to ` +
        `${localIso(end)} at ${clock}`;
      document.getElementById('chat').requestSubmit();
    });
    panel.append(form);
    const snapshot = document.createElement('button');
    snapshot.type = 'button'; snapshot.textContent = 'Save calendar snapshot to Workspace';
    snapshot.addEventListener('click', () => {
      document.getElementById('utterance').value =
        'Save my calendar snapshot to Workspace as agenda.md';
      document.getElementById('chat').requestSubmit();
    });
    panel.append(snapshot);
    const sendSnapshot = document.createElement('button');
    sendSnapshot.type = 'button'; sendSnapshot.textContent = 'Send calendar snapshot';
    sendSnapshot.addEventListener('click', () => {
      document.getElementById('utterance').value = 'Text me my calendar';
      document.getElementById('chat').requestSubmit();
    });
    const sendSnapshotBoundary = document.createElement('p');
    sendSnapshotBoundary.className = 'muted';
    sendSnapshotBoundary.textContent = 'Uses the configured owner-approved destination; provider acceptance is not delivery proof.';
    panel.append(sendSnapshot, sendSnapshotBoundary);
    const draftSection = document.createElement('section'); draftSection.className = 'detail-card';
    const draftTitle = document.createElement('h3'); draftTitle.textContent = 'Draft an agenda message';
    const draftForm = document.createElement('form'); draftForm.setAttribute('aria-label', 'Draft calendar message');
    const recipient = document.createElement('input'); recipient.required = true; recipient.placeholder = 'Recipient';
    recipient.setAttribute('aria-label', 'Recipient');
    const targetPath = document.createElement('input'); targetPath.required = true; targetPath.placeholder = 'Workspace path, e.g. agenda.md';
    targetPath.setAttribute('aria-label', 'Workspace path');
    const draftSubmit = document.createElement('button'); draftSubmit.type = 'submit'; draftSubmit.textContent = 'Create unsent draft';
    draftForm.append(recipient, targetPath, draftSubmit);
    draftForm.addEventListener('submit', event => {
      event.preventDefault();
      if (!recipient.value.trim() || !targetPath.value.trim()) return;
      document.getElementById('utterance').value =
        `Draft my calendar for ${recipient.value.trim()} as ${targetPath.value.trim()}`;
      document.getElementById('chat').requestSubmit();
    });
    draftSection.append(draftTitle, draftForm); panel.append(draftSection);
    panel.append(renderDetailValue(events));
    const conflicts = payload.conflicts || [];
    const conflictSection = document.createElement('section'); conflictSection.className = 'detail-card';
    const conflictTitle = document.createElement('h3'); conflictTitle.textContent = 'Scheduling conflicts';
    conflictSection.append(conflictTitle);
    conflictSection.append(renderDetailValue(conflicts.length ? conflicts : 'No overlapping timed events detected.'));
    const conflictBoundary = document.createElement('p'); conflictBoundary.className = 'muted';
    conflictBoundary.textContent = payload.conflict_boundary || 'Conflict inspection is read-only.';
    conflictSection.append(conflictBoundary); panel.append(conflictSection);
    const sendAttention = document.createElement('button'); sendAttention.type = 'button';
    sendAttention.textContent = 'Send scheduling attention';
    sendAttention.addEventListener('click', () => {
      document.getElementById('utterance').value =
        'Text me the tasks due before my calendar events';
      document.getElementById('chat').requestSubmit();
    });
    const calendarSendBoundary = document.createElement('p'); calendarSendBoundary.className = 'muted';
    calendarSendBoundary.textContent =
      'Calendar and task state remain read-only context; provider acceptance is not delivery proof.';
    conflictSection.append(sendAttention, calendarSendBoundary);
    const holidays = payload.public_holidays?.holidays || [];
    const holidaySection = document.createElement('section'); holidaySection.className = 'detail-card';
    const holidayTitle = document.createElement('h3'); holidayTitle.textContent = 'Public holidays';
    holidaySection.append(holidayTitle, renderDetailValue(holidays.length ? holidays : 'No configured public-holiday feed.'));
    const holidayBoundary = document.createElement('p'); holidayBoundary.className = 'muted';
    holidayBoundary.textContent = 'Public holiday dates are external evidence, not canonical personal events.';
    holidaySection.append(holidayBoundary); panel.append(holidaySection);
    const sendHolidays = document.createElement('button');
    sendHolidays.type = 'button'; sendHolidays.textContent = 'Send public holidays';
    sendHolidays.addEventListener('click', () => {
      document.getElementById('utterance').value = 'Text me the public holidays';
      document.getElementById('chat').requestSubmit();
    });
    const sendHolidayBoundary = document.createElement('p'); sendHolidayBoundary.className = 'muted';
    sendHolidayBoundary.textContent = 'Public holiday data remains external/non-canonical; configured scope is fixed before communication and provider acceptance is not delivery proof.';
    holidaySection.append(sendHolidays, sendHolidayBoundary);
    holidays.forEach(holiday => {
      const date = String(holiday.date || '').trim();
      const name = String(holiday.name || '').trim();
      if (!date || !name) return;
      const button = document.createElement('button'); button.type = 'button';
      button.textContent = `Prepare for ${name}`;
      button.addEventListener('click', () => {
        document.getElementById('utterance').value =
          `Add a task to prepare for ${name} on ${date}`;
        document.getElementById('chat').requestSubmit();
      });
      holidaySection.append(button);
    });
    if (conflicts.length) {
      const actionNote = document.createElement('p'); actionNote.className = 'muted';
      actionNote.textContent = 'Create a follow-up task through the normal Core request path:';
      conflictSection.append(actionNote);
      conflicts.forEach(conflict => {
        const first = String(conflict.event_title || conflict.event_id || '').trim();
        const second = String(conflict.conflicting_title || conflict.conflicts_with || '').trim();
        if (!first || !second) return;
        const button = document.createElement('button'); button.type = 'button';
        button.textContent = `Make task: ${first} / ${second}`;
        button.addEventListener('click', () => {
          document.getElementById('utterance').value =
            `Add a task to resolve the calendar conflict between ${first} and ${second}`;
          document.getElementById('chat').requestSubmit();
        });
        conflictSection.append(button);
      });
    }
    if (events.length) {
      const cancelSection = document.createElement('section'); cancelSection.className = 'detail-card';
      const cancelTitle = document.createElement('h3'); cancelTitle.textContent = 'Cancel an authorized event';
      const cancelNote = document.createElement('p'); cancelNote.className = 'muted';
      cancelNote.textContent = 'Cancellation uses the provider event ID from the authorized list and requires normal Core authorization plus absence readback.';
      cancelSection.append(cancelTitle, cancelNote);
      events.forEach(eventRecord => {
        if (!eventRecord.event_id) return;
        const button = document.createElement('button'); button.type = 'button';
        button.textContent = `Cancel ${eventRecord.title || eventRecord.event_id}`;
        button.addEventListener('click', () => {
          document.getElementById('utterance').value = `Cancel calendar event ${eventRecord.event_id}`;
          document.getElementById('chat').requestSubmit();
        });
        cancelSection.append(button);
      });
      panel.append(cancelSection);
    }
    if (events.length) {
      const updateSection = document.createElement('section'); updateSection.className = 'detail-card';
      const updateTitle = document.createElement('h3'); updateTitle.textContent = 'Update an authorized event';
      const updateNote = document.createElement('p'); updateNote.className = 'muted';
      updateNote.textContent = 'The provider event ID stays bound to the selected authorized event; changed fields are independently read back.';
      updateSection.append(updateTitle, updateNote);
      events.forEach(eventRecord => {
        if (!eventRecord.event_id) return;
        const form = document.createElement('form'); form.setAttribute('aria-label', `Update ${eventRecord.title || eventRecord.event_id}`);
        const titleInput = document.createElement('input'); titleInput.required = true; titleInput.value = eventRecord.title || '';
        titleInput.setAttribute('aria-label', 'Updated event title');
        const startInput = document.createElement('input'); startInput.type = 'datetime-local'; startInput.required = true;
        const endInput = document.createElement('input'); endInput.type = 'datetime-local'; endInput.required = true;
        const toLocalInput = value => {
          const date = new Date(value); if (Number.isNaN(date.valueOf())) return '';
          const pad = number => String(number).padStart(2, '0');
          return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}`;
        };
        startInput.value = toLocalInput(eventRecord.starts_at);
        endInput.value = toLocalInput(eventRecord.ends_at || eventRecord.starts_at);
        const submit = document.createElement('button'); submit.type = 'submit'; submit.textContent = `Update ${eventRecord.title || eventRecord.event_id}`;
        form.append(titleInput, startInput, endInput, submit);
        form.addEventListener('submit', event => {
          event.preventDefault();
          const start = new Date(startInput.value); const end = new Date(endInput.value);
          if (!titleInput.value.trim() || Number.isNaN(start.valueOf()) || Number.isNaN(end.valueOf()) || end <= start) return;
          document.getElementById('utterance').value =
            `Update calendar event ${eventRecord.event_id} to ${titleInput.value.trim()} from ${start.toISOString()} to ${end.toISOString()}`;
          document.getElementById('chat').requestSubmit();
        });
        updateSection.append(form);
      });
      panel.append(updateSection);
    }
    const attention = payload.task_attention || [];
    const attentionSection = document.createElement('section'); attentionSection.className = 'detail-card';
    const attentionTitle = document.createElement('h3'); attentionTitle.textContent = 'Tasks before shared events';
    attentionSection.append(attentionTitle, renderDetailValue(attention));
    const attentionBoundary = document.createElement('p'); attentionBoundary.className = 'muted';
    attentionBoundary.textContent = payload.attention_boundary || 'Read-only Calendar + Tasks attention.';
    attentionSection.append(attentionBoundary); panel.append(attentionSection);
  } catch (_) {
    panel.textContent = 'Calendar state is unavailable; no event state was changed.';
  }
}
async function loadDevices() {
  const panel = document.getElementById('detail');
  panel.replaceChildren();
  try {
    const response = await fetchWithTimeout('/api/devices');
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'Devices unavailable.');
    const heading = document.createElement('p');
    heading.textContent = 'Authorized device state and bounded controls';
    panel.append(heading, renderDetailValue(payload));
    const snapshot = document.createElement('button');
    snapshot.type = 'button'; snapshot.textContent = 'Save device snapshot to Workspace';
    snapshot.addEventListener('click', () => {
      document.getElementById('utterance').value =
        'Save authorized device states to Workspace as devices.md';
      document.getElementById('chat').requestSubmit();
    });
    panel.append(snapshot);
    const sendStatus = document.createElement('button');
    sendStatus.type = 'button'; sendStatus.textContent = 'Send device status';
    sendStatus.addEventListener('click', () => {
      document.getElementById('utterance').value = 'Text me the device status';
      document.getElementById('chat').requestSubmit();
    });
    const sendBoundary = document.createElement('p'); sendBoundary.className = 'muted';
    sendBoundary.textContent = 'Device state is read-only context; Core fixes the authorized snapshot before communication, and provider acceptance is not delivery proof.';
    panel.append(sendStatus, sendBoundary);
    const control = payload.control_surface || {};
    const entities = Array.isArray(control.authorized_entities)
      ? control.authorized_entities : [];
    if (entities.length) {
      const section = document.createElement('section');
      section.className = 'detail-card';
      const title = document.createElement('h3'); title.textContent = 'Request a bounded control';
      const note = document.createElement('p');
      note.textContent = `${control.provider_state || 'unknown'} provider · explicit owner approval and independent readback required`;
      section.append(title, note);
      entities.forEach(entity => {
        const row = document.createElement('p');
        const label = document.createElement('strong'); label.textContent = entity;
        row.append(label, ' ');
        const research = document.createElement('button');
        research.type = 'button'; research.textContent = 'Research state';
        research.addEventListener('click', () => {
          document.getElementById('utterance').value = `Research the current state of ${entity} for why it is off`;
          document.getElementById('chat').requestSubmit();
        });
        row.append(research, ' ');
        ['on', 'off'].forEach(state => {
          const button = document.createElement('button');
          button.type = 'button'; button.textContent = `Turn ${state}`;
          button.addEventListener('click', () => {
            document.getElementById('utterance').value = `Turn ${state} ${entity} and verify`;
            document.getElementById('chat').requestSubmit();
          });
          row.append(button, ' ');
        });
        section.append(row);
      });
      panel.append(section);
    }
  } catch (_) {
    panel.textContent = 'Device state is unavailable; no device action was attempted.';
  }
}
async function loadSystems() {
  const panel = document.getElementById('detail');
  panel.replaceChildren();
  try {
    const response = await fetchWithTimeout('/api/systems');
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'Systems unavailable.');
    let networkReportsEnabled = false;
    try {
      const packsResponse = await apiFetch('/api/packs');
      const packsPayload = await packsResponse.json();
      networkReportsEnabled = packsResponse.ok && (packsPayload.packs || []).some(
        pack => pack.pack_id === 'network-reports' && pack.status === 'enabled'
      );
    } catch (_) {}
    const heading = document.createElement('p');
    heading.textContent = 'Authorized systems inventory (read-only view)';
    panel.append(heading, renderDetailValue(payload));
    const services = Array.isArray(payload.services) ? payload.services : [];
    if (services.length) {
      const actions = document.createElement('section'); actions.className = 'detail-card';
      const actionsTitle = document.createElement('h3'); actionsTitle.textContent = 'Bounded service actions';
      const actionsNote = document.createElement('p');
      actionsNote.textContent = 'Requests are sent through AEGIS Core authorization and independent health verification.';
      actions.append(actionsTitle, actionsNote);
      services.slice(0, 20).forEach(service => {
        if (!service || typeof service.service_id !== 'string') return;
        const row = document.createElement('div'); row.className = 'action-row';
        const label = document.createElement('span');
        label.textContent = `${service.name || service.service_id} · ${service.health || 'unknown'}`;
        const restart = document.createElement('button'); restart.type = 'button';
        restart.textContent = 'Request restart';
        restart.setAttribute('aria-label', `Request restart for ${service.service_id}`);
        restart.addEventListener('click', () => {
          document.getElementById('utterance').value = `Restart service ${service.service_id}`;
          document.getElementById('chat').requestSubmit();
        });
        const sendHealth = document.createElement('button'); sendHealth.type = 'button';
        sendHealth.textContent = 'Send health status';
        sendHealth.setAttribute('aria-label', `Send health status for ${service.service_id}`);
        sendHealth.addEventListener('click', () => {
          document.getElementById('utterance').value = `Text me the health of service ${service.service_id}`;
          document.getElementById('chat').requestSubmit();
        });
        row.append(label, restart, sendHealth); actions.append(row);
        if (service.health && service.health !== 'healthy') {
          const research = document.createElement('button'); research.type = 'button';
          research.textContent = 'Research likely cause';
          research.setAttribute('aria-label', `Research likely cause for ${service.service_id}`);
          research.addEventListener('click', () => {
            document.getElementById('utterance').value =
              `Research why service ${service.service_id} is unavailable`;
            document.getElementById('chat').requestSubmit();
          });
          row.append(research);
          const investigate = document.createElement('button'); investigate.type = 'button';
          investigate.textContent = 'Create investigation task';
          investigate.setAttribute('aria-label', `Create investigation task for ${service.service_id}`);
          investigate.addEventListener('click', () => {
            document.getElementById('utterance').value =
              `Create a task to investigate service ${service.service_id}`;
            document.getElementById('chat').requestSubmit();
          });
          row.append(investigate);
        }
      });
      panel.append(actions);
    }
    const communicationBoundary = document.createElement('p'); communicationBoundary.className = 'muted';
    communicationBoundary.textContent = 'Health status is read-only context; Core fixes the observation before communication, and provider acceptance is not delivery proof.';
    panel.append(communicationBoundary);
    const report = document.createElement('button');
    report.type = 'button'; report.textContent = 'Create verified health report';
    report.addEventListener('click', () => {
      document.getElementById('utterance').value = 'Create a homelab health report as health-report.html';
      document.getElementById('chat').requestSubmit();
    });
    panel.append(report);
    const networkReport = document.createElement('button');
    networkReport.type = 'button'; networkReport.textContent = 'Save network inventory to Workspace';
    networkReport.disabled = !networkReportsEnabled;
    networkReport.addEventListener('click', () => {
      document.getElementById('utterance').value =
        'Save the authorized network inventory to workspace as network-report.md';
      document.getElementById('chat').requestSubmit();
    });
    const networkReportBoundary = document.createElement('p'); networkReportBoundary.className = 'muted';
    networkReportBoundary.textContent = networkReportsEnabled
      ? 'Network inventory is read-only canonical state; the scoped Workspace artifact is independently verified.'
      : 'Approve network-reports in Packs & capabilities before exporting network inventory to Workspace.';
    panel.append(networkReport, networkReportBoundary);
    const probe = document.createElement('section'); probe.className = 'detail-card';
    const probeTitle = document.createElement('h3'); probeTitle.textContent = 'Probe an authorized network target';
    const probeForm = document.createElement('form'); probeForm.setAttribute('aria-label', 'Probe network target');
    const probeFields = [];
    [['address', 'Address'], ['scope_id', 'Authorized scope'], ['port', 'Port']].forEach(([name, placeholder]) => {
      const input = document.createElement('input'); input.name = name; input.required = true; input.placeholder = placeholder;
      input.setAttribute('aria-label', placeholder); probeFields.push(input);
    });
    const probeSubmit = document.createElement('button'); probeSubmit.type = 'submit'; probeSubmit.textContent = 'Submit probe';
    const reportPath = document.createElement('input'); reportPath.name = 'target_path';
    reportPath.required = false; reportPath.placeholder = 'Report path (optional)';
    reportPath.setAttribute('aria-label', 'Report path (optional)');
    const reportSubmit = document.createElement('button'); reportSubmit.type = 'submit';
    reportSubmit.textContent = 'Probe and save report';
    probeForm.append(...probeFields, reportPath, probeSubmit, reportSubmit);
    probeForm.addEventListener('submit', event => {
      event.preventDefault();
      const values = Object.fromEntries(probeFields.map(field => [field.name, field.value.trim()]));
      if (Object.values(values).some(value => !value)) return;
      if (event.submitter === reportSubmit) {
        const targetPath = reportPath.value.trim();
        if (!targetPath) return;
        document.getElementById('utterance').value =
          `Probe ${values.address} in scope ${values.scope_id} on port ${values.port} and save report as ${targetPath}`;
      } else {
        document.getElementById('utterance').value =
          `Probe ${values.address} in scope ${values.scope_id} on port ${values.port}`;
      }
      document.getElementById('chat').requestSubmit();
    });
    probe.append(probeTitle, probeForm); panel.append(probe);
  } catch (_) {
    panel.textContent = 'Systems inventory is unavailable; no system action was attempted.';
  }
}
async function loadWeather() {
  const panel = document.getElementById('detail'); panel.replaceChildren();
  try {
    const response = await fetchWithTimeout('/api/weather');
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'Weather unavailable.');
    const heading = document.createElement('p');
    heading.textContent = payload.reading ? 'Current public weather' : 'Weather is not configured.';
    panel.append(heading, renderDetailValue(payload));
    const boundary = document.createElement('p'); boundary.className = 'muted';
    boundary.textContent = payload.boundary || 'Weather is public evidence, not canonical personal truth. Data: Open-Meteo.';
    panel.append(boundary);
    if (Array.isArray(payload.forecast) && payload.forecast.length) {
      const forecast = document.createElement('section'); forecast.className = 'detail-card';
      const forecastTitle = document.createElement('h3'); forecastTitle.textContent = 'Three-day forecast';
      forecast.append(forecastTitle, renderDetailValue(payload.forecast)); panel.append(forecast);
      const reading = payload.reading || {};
      if (reading.latitude !== undefined && reading.longitude !== undefined) {
        const saveForecast = document.createElement('button'); saveForecast.type = 'button';
        saveForecast.textContent = 'Save forecast to Workspace';
        saveForecast.addEventListener('click', () => {
          document.getElementById('utterance').value =
            `Save my 3-day weather forecast at ${reading.latitude}, ${reading.longitude} as forecast.md`;
          document.getElementById('chat').requestSubmit();
        });
        forecast.append(saveForecast);
        const forecastBoundary = document.createElement('p'); forecastBoundary.className = 'muted';
        forecastBoundary.textContent =
          'The report preserves public forecast provenance; it is not canonical personal truth.';
        forecast.append(forecastBoundary);
      }
    }
    if (payload.reading) {
      const followup = document.createElement('button'); followup.type = 'button';
      followup.textContent = 'Create weather follow-up task';
      followup.addEventListener('click', () => {
        document.getElementById('utterance').value =
          'Add a task to check the weather before leaving';
        document.getElementById('chat').requestSubmit();
      });
      panel.append(followup);
      const sendForecast = document.createElement('button'); sendForecast.type = 'button';
      sendForecast.textContent = "Send tomorrow's weather";
      sendForecast.addEventListener('click', () => {
        document.getElementById('utterance').value = "Text me tomorrow's weather";
        document.getElementById('chat').requestSubmit();
      });
      panel.append(sendForecast);
      const sendForecastBoundary = document.createElement('p'); sendForecastBoundary.className = 'muted';
      sendForecastBoundary.textContent =
        'Sending uses the exact approved communication destination; provider acceptance is not delivery proof.';
      panel.append(sendForecastBoundary);
      const report = document.createElement('button'); report.type = 'button';
      report.textContent = 'Save verified air-quality report';
      report.addEventListener('click', () => {
        const reading = payload.reading || {};
        const latitude = reading.latitude; const longitude = reading.longitude;
        if (latitude === undefined || longitude === undefined) return;
        document.getElementById('utterance').value =
          `Save current air quality at ${latitude}, ${longitude} as air-quality.md`;
        document.getElementById('chat').requestSubmit();
      });
      panel.append(report);
    }
  } catch (_) { panel.textContent = 'Weather is unavailable; no personal state was changed.'; }
}
async function loadAirQuality() {
  const panel = document.getElementById('detail'); panel.replaceChildren();
  try {
    const response = await fetchWithTimeout('/api/air-quality');
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'Air quality unavailable.');
    const heading = document.createElement('p');
    heading.textContent = payload.reading ? 'Current public air quality' : 'Air quality is not configured.';
    panel.append(heading, renderDetailValue(payload));
    const boundary = document.createElement('p'); boundary.className = 'muted';
    boundary.textContent = payload.boundary || 'Air quality is public evidence, not canonical personal truth. Data: Open-Meteo / CAMS.';
    panel.append(boundary);
    if (payload.reading) {
      const followup = document.createElement('button'); followup.type = 'button';
      followup.textContent = 'Create air-quality follow-up task';
      followup.addEventListener('click', () => {
        document.getElementById('utterance').value =
          'Add a task to check air quality before going outside';
        document.getElementById('chat').requestSubmit();
      });
      panel.append(followup);
    }
  } catch (_) { panel.textContent = 'Air quality is unavailable; no personal state was changed.'; }
}
async function loadToday() {
  const panel = document.getElementById('detail');
  panel.replaceChildren();
  try {
    const response = await fetchWithTimeout('/api/today');
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'Today unavailable.');
    const heading = document.createElement('p');
    heading.textContent = 'Authorized state requiring attention today';
    panel.append(heading);
    appendTodayOverview(panel, payload);
    panel.append(renderDetailValue(payload));
    const report = document.createElement('button'); report.type = 'button';
    report.textContent = 'Save verified Today brief to Workspace';
    report.addEventListener('click', () => {
      document.getElementById('utterance').value = "Save today's brief to Workspace as today.md";
      document.getElementById('chat').requestSubmit();
    });
    panel.append(report);
    const sendBrief = document.createElement('button'); sendBrief.type = 'button';
    sendBrief.textContent = 'Send me today\'s brief';
    sendBrief.addEventListener('click', () => {
      document.getElementById('utterance').value = "Send me today's brief";
      document.getElementById('chat').requestSubmit();
    });
    const sendBoundary = document.createElement('p'); sendBoundary.className = 'muted';
    sendBoundary.textContent = 'Uses only one exact owner-approved communication target; provider acceptance is not delivery proof.';
    panel.append(sendBrief, sendBoundary);
    const activeObjectives = payload.active_objectives || [];
    appendTodaySection(panel, 'Active objectives', activeObjectives.length
      ? activeObjectives : 'No active objectives.');
    if (activeObjectives.length) {
      const objectivesButton = document.createElement('button'); objectivesButton.type = 'button';
      objectivesButton.textContent = 'Open active objectives';
      objectivesButton.addEventListener('click', () => {
        const objectives = document.querySelector('[data-view="objectives"]');
        if (objectives) objectives.click();
      });
      panel.append(objectivesButton);
    }
    const capabilityNeeds = payload.capability_needs || [];
    appendTodaySection(panel, 'Capability needs requiring attention', capabilityNeeds.length
      ? capabilityNeeds : 'No unresolved capability needs.');
    if (capabilityNeeds.length) {
      const needsButton = document.createElement('button'); needsButton.type = 'button';
      needsButton.textContent = 'Review capability needs';
      needsButton.addEventListener('click', () => {
        const objectives = document.querySelector('[data-view="objectives"]');
        if (objectives) objectives.click();
      });
      panel.append(needsButton);
    }
    const conflicts = payload.external_calendar?.conflicts || [];
    appendTodaySection(panel, 'Scheduling conflicts', conflicts.length
      ? conflicts : 'No overlapping timed events detected.');
    const conflictBoundary = document.createElement('p'); conflictBoundary.className = 'muted';
    conflictBoundary.textContent = payload.external_calendar?.conflict_boundary
      || 'Conflict inspection is read-only.';
    panel.append(conflictBoundary);
    const todayHolidays = payload.external_calendar?.public_holidays?.holidays || [];
    appendTodaySection(panel, 'Public holidays', todayHolidays.length
      ? todayHolidays : 'No configured public-holiday feed.');
    const holidayBoundary = document.createElement('p'); holidayBoundary.className = 'muted';
    holidayBoundary.textContent = 'Public holiday dates are external evidence, not canonical personal events.';
    panel.append(holidayBoundary);
    const airQuality = payload.external_calendar?.air_quality || {};
    const airQualitySection = document.createElement('section'); airQualitySection.className = 'detail-card';
    const airQualityTitle = document.createElement('h3'); airQualityTitle.textContent = 'Public air quality';
    airQualitySection.append(airQualityTitle, renderDetailValue(airQuality.reading || 'Unavailable'));
    const airQualityBoundary = document.createElement('p'); airQualityBoundary.className = 'muted';
    airQualityBoundary.textContent = airQuality.boundary || 'Public evidence only; no canonical personal state was changed.';
    airQualitySection.append(airQualityBoundary); panel.append(airQualitySection);
    const weather = payload.external_calendar?.weather || {};
    const weatherSection = document.createElement('section'); weatherSection.className = 'detail-card';
    const weatherTitle = document.createElement('h3'); weatherTitle.textContent = 'Weather outlook';
    weatherSection.append(weatherTitle, renderDetailValue(weather.forecast || weather.reading || 'Unavailable'));
    const weatherBoundary = document.createElement('p'); weatherBoundary.className = 'muted';
    weatherBoundary.textContent = weather.boundary || 'Weather is public evidence, not canonical personal truth. Data: Open-Meteo.';
    weatherSection.append(weatherBoundary); panel.append(weatherSection);
    todayHolidays.forEach(holiday => {
      const date = String(holiday.date || '').trim();
      const name = String(holiday.name || '').trim();
      if (!date || !name) return;
      const button = document.createElement('button'); button.type = 'button';
      button.textContent = `Prepare for ${name}`;
      button.addEventListener('click', () => {
        document.getElementById('utterance').value =
          `Add a task to prepare for ${name} on ${date}`;
        document.getElementById('chat').requestSubmit();
      });
      panel.append(button);
    });
    if (conflicts.length) {
      const actionNote = document.createElement('p'); actionNote.className = 'muted';
      actionNote.textContent = 'Scheduling follow-up stays a normal authorized task request.';
      panel.append(actionNote);
      conflicts.forEach(conflict => {
        const first = String(conflict.event_title || conflict.event_id || '').trim();
        const second = String(conflict.conflicting_title || conflict.conflicts_with || '').trim();
        if (!first || !second) return;
        const button = document.createElement('button'); button.type = 'button';
        button.textContent = `Make task: ${first} / ${second}`;
        button.addEventListener('click', () => {
          document.getElementById('utterance').value =
            `Add a task to resolve the calendar conflict between ${first} and ${second}`;
          document.getElementById('chat').requestSubmit();
        });
        panel.append(button);
      });
    }
  } catch (_) {
    panel.textContent = 'Today state is unavailable; no canonical state was changed.';
  }
}
function appendTodayOverview(panel, payload) {
  const canonical = payload.canonical || {};
  const external = payload.external_calendar || {};
  const items = [
    {
      label: 'Open tasks', count: (canonical.open_tasks || []).length, view: 'tasks',
      attention: (canonical.open_tasks || []).length > 0,
    },
    {
      label: 'Open chores', count: (canonical.open_chores || []).length, view: 'household',
      attention: (canonical.open_chores || []).length > 0,
    },
    {
      label: 'Up next', count: (canonical.upcoming_shared_events || []).length, view: 'calendar',
      attention: false,
    },
    {
      label: 'Capability needs', count: (payload.capability_needs || []).length, view: 'objectives',
      attention: (payload.capability_needs || []).length > 0,
    },
    {
      label: 'Conflicts', count: (external.conflicts || []).length, view: 'calendar',
      attention: (external.conflicts || []).length > 0,
    },
    {
      label: 'Active objectives', count: (payload.active_objectives || []).length, view: 'objectives',
      attention: false,
    },
  ];
  const unknowns = (payload.external_effects || {}).unknown || [];
  if (unknowns.length) {
    items.push({label: 'Outcome unknown', count: unknowns.length, view: 'compositions', attention: 'unknown'});
  }
  const overview = document.createElement('div');
  overview.className = 'today-overview';
  overview.setAttribute('aria-label', 'Today at a glance');
  items.forEach(item => {
    const button = document.createElement('button');
    button.type = 'button'; button.className = 'today-overview-card';
    if (item.attention) button.dataset.attention = String(item.attention);
    const count = document.createElement('strong'); count.textContent = String(item.count);
    const label = document.createElement('span'); label.textContent = item.label;
    button.append(count, label);
    button.addEventListener('click', () => {
      const destination = document.querySelector(`[data-view="${item.view}"]`);
      if (destination) destination.click();
    });
    overview.append(button);
  });
  panel.append(overview);
}
function appendTodaySection(panel, title, value) {
  const section = document.createElement('section'); section.className = 'detail-card';
  const heading = document.createElement('h3'); heading.textContent = title;
  section.append(heading, renderDetailValue(value)); panel.append(section);
}
function appendCompletableSection(panel, title, items, utterancePrefix) {
  const section = document.createElement('section'); section.className = 'detail-card';
  const heading = document.createElement('h3'); heading.textContent = title; section.append(heading);
  if (!items.length) {
    const empty = document.createElement('p'); empty.className = 'muted'; empty.textContent = 'None recorded.';
    section.append(empty);
  }
  items.forEach(item => {
    const row = document.createElement('p');
    row.append(document.createTextNode(item.title || 'Untitled'));
    if (item.due_at) row.append(document.createTextNode(` · due ${item.due_at}`));
    const complete = document.createElement('button'); complete.type = 'button';
    complete.textContent = 'Complete'; complete.style.marginLeft = '.5rem';
    complete.addEventListener('click', () => {
      const titleText = String(item.title || '').trim(); if (!titleText) return;
      const completionSuffix = utterancePrefix.includes('chore') ? ' as complete' : '';
      document.getElementById('utterance').value = `${utterancePrefix} ${titleText}${completionSuffix}`;
      document.getElementById('chat').requestSubmit();
    });
    row.append(complete); section.append(row);
  });
  panel.append(section);
}
async function loadTasks() {
  const panel = document.getElementById('detail'); panel.replaceChildren();
  try {
    const response = await fetchWithTimeout('/api/today');
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'Tasks unavailable.');
    const canonical = payload.canonical || {};
    appendCompletableSection(panel, 'Open and due', canonical.open_tasks || [], 'Complete the task');
    const sendTasks = document.createElement('button');
    sendTasks.type = 'button'; sendTasks.textContent = 'Send open tasks';
    sendTasks.addEventListener('click', () => {
      document.getElementById('utterance').value = 'Text me my open tasks';
      document.getElementById('chat').requestSubmit();
    });
    const sendBoundary = document.createElement('p'); sendBoundary.className = 'muted';
    sendBoundary.textContent = 'Canonical task state is fixed before communication; provider acceptance is not delivery proof.';
    panel.append(sendTasks, sendBoundary);
    const sendCompleted = document.createElement('button'); sendCompleted.type = 'button';
    sendCompleted.textContent = 'Send completed tasks';
    sendCompleted.addEventListener('click', () => {
      document.getElementById('utterance').value = 'Text me my completed tasks';
      document.getElementById('chat').requestSubmit();
    });
    const completedSendBoundary = document.createElement('p'); completedSendBoundary.className = 'muted';
    completedSendBoundary.textContent =
      'Completed task history is fixed before communication; provider acceptance is not delivery proof.';
    panel.append(sendCompleted, completedSendBoundary);
    appendTodaySection(panel, 'Completed', canonical.completed_tasks || []);
    const saveCompleted = document.createElement('button'); saveCompleted.type = 'button';
    saveCompleted.textContent = 'Save completed tasks to Workspace';
    saveCompleted.addEventListener('click', () => {
      document.getElementById('utterance').value =
        'Save my completed tasks to workspace as completed.md';
      document.getElementById('chat').requestSubmit();
    });
    const completedBoundary = document.createElement('p'); completedBoundary.className = 'muted';
    completedBoundary.textContent =
      'Completed task history is read-only canonical context; the scoped Workspace report is independently verified.';
    panel.append(saveCompleted, completedBoundary);
    const boundary = document.createElement('p'); boundary.className = 'muted';
    boundary.textContent = payload.truth_boundary || 'Task state is canonical authorized state.';
    panel.append(boundary);
  } catch (_) { panel.textContent = 'Task state is unavailable; no canonical state was changed.'; }
}
async function loadHousehold() {
  const panel = document.getElementById('detail'); panel.replaceChildren();
  try {
    const response = await fetchWithTimeout('/api/today');
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'Household unavailable.');
    let obligationsPackEnabled = false;
    try {
      const packsResponse = await apiFetch('/api/packs');
      const packsPayload = await packsResponse.json();
      obligationsPackEnabled = packsResponse.ok && (packsPayload.packs || []).some(
        pack => pack.pack_id === 'household-reports' && pack.status === 'enabled'
      );
    } catch (_) {}
    const canonical = payload.canonical || {};
    appendCompletableSection(panel, 'Open chores', canonical.open_chores || [], 'Mark the chore');
    appendTodaySection(panel, 'Groceries', canonical.groceries || []);
    const pantry = canonical.pantry_items || [];
    if (pantry.length) {
      const pantryRows = pantry.map(item => {
        const quantity = item.quantity == null ? 'quantity unknown' : `${item.quantity} ${item.unit || ''}`.trim();
        return `${item.display_name} · ${quantity}${item.storage_location ? ` · ${item.storage_location}` : ''}`;
      });
      appendTodaySection(panel, 'Pantry', pantryRows);
    } else {
      appendTodaySection(panel, 'Pantry', ['No canonical Pantry items recorded yet.']);
    }
    const lowPantry = canonical.pantry_low_items || [];
    appendTodaySection(panel, 'Pantry items to review', lowPantry.length
      ? lowPantry.map(item => `${item.display_name} · ${item.quantity} ${item.unit || ''} · minimum ${item.minimum_quantity}`.trim())
      : ['No low-stock projection; unknown quantities are not treated as low.']);
    const sendGroceries = document.createElement('button');
    sendGroceries.type = 'button'; sendGroceries.textContent = 'Send grocery list';
    sendGroceries.addEventListener('click', () => {
      document.getElementById('utterance').value = 'Text me the grocery list';
      document.getElementById('chat').requestSubmit();
    });
    const sendBoundary = document.createElement('p'); sendBoundary.className = 'muted';
    sendBoundary.textContent = 'Canonical groceries are fixed before communication; one approved target is required and provider acceptance is not delivery proof.';
    panel.append(sendGroceries, sendBoundary);
    const sendChores = document.createElement('button');
    sendChores.type = 'button'; sendChores.textContent = 'Send open chores';
    sendChores.addEventListener('click', () => {
      document.getElementById('utterance').value = 'Text me my open chores';
      document.getElementById('chat').requestSubmit();
    });
    const choreBoundary = document.createElement('p'); choreBoundary.className = 'muted';
      choreBoundary.textContent = 'Canonical chore state is fixed before communication; provider acceptance is not delivery proof.';
      panel.append(sendChores, choreBoundary);
      const sendObligations = document.createElement('button');
      sendObligations.type = 'button'; sendObligations.textContent = 'Send open obligations';
      sendObligations.addEventListener('click', () => {
        document.getElementById('utterance').value = 'Text me my open obligations';
        document.getElementById('chat').requestSubmit();
      });
      const obligationBoundary = document.createElement('p'); obligationBoundary.className = 'muted';
      obligationBoundary.textContent = 'Canonical obligation state is fixed before communication; provider acceptance is not delivery proof.';
      panel.append(sendObligations, obligationBoundary);
      const saveObligations = document.createElement('button');
      saveObligations.type = 'button'; saveObligations.textContent = 'Save obligations to Workspace';
      saveObligations.disabled = !obligationsPackEnabled;
      saveObligations.addEventListener('click', () => {
        document.getElementById('utterance').value = 'Save my open obligations to workspace as obligations.md';
        document.getElementById('chat').requestSubmit();
      });
      const obligationsWorkspaceBoundary = document.createElement('p');
      obligationsWorkspaceBoundary.className = 'muted';
      obligationsWorkspaceBoundary.textContent = obligationsPackEnabled
        ? 'Open obligations are read-only canonical context; the scoped Workspace artifact is independently verified.'
        : 'Approve household-reports in Packs & capabilities before exporting obligations to Workspace.';
      panel.append(saveObligations, obligationsWorkspaceBoundary);
    appendTodaySection(panel, 'Upcoming shared events', canonical.upcoming_shared_events || []);
    const boundary = document.createElement('p'); boundary.className = 'muted';
    boundary.textContent = payload.truth_boundary || 'Household state is canonical authorized state.';
    panel.append(boundary);
  } catch (_) { panel.textContent = 'Household state is unavailable; no canonical state was changed.'; }
}
async function loadFinance() {
  const panel = document.getElementById('detail'); panel.replaceChildren();
  try {
    const response = await apiFetch('/api/finance');
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'Finance unavailable.');
    const title = document.createElement('h3');
    title.textContent = payload.provider_state === 'available' ? 'Private finance' : 'Finance not configured';
    panel.append(title);
    if (payload.provider_state === 'available') {
      const source = document.createElement('p'); source.className = 'muted';
      source.textContent = `Source: ${payload.provider_id || 'unknown'} · As of: ${payload.captured_at || 'unknown'}`;
      panel.append(source);
      const importSection = document.createElement('section'); importSection.className = 'detail-card';
      const importTitle = document.createElement('h3'); importTitle.textContent = 'Import a private CSV';
      const importHint = document.createElement('p'); importHint.className = 'muted';
      importHint.textContent = 'Owner-controlled import; rows stay private and duplicate sources are ignored.';
      const file = document.createElement('input'); file.type = 'file'; file.accept = '.csv,text/csv';
      file.setAttribute('aria-label', 'Finance CSV file');
      const importButton = document.createElement('button'); importButton.type = 'button'; importButton.textContent = 'Import CSV';
      const importStatus = document.createElement('p'); importStatus.className = 'muted'; importStatus.setAttribute('aria-live', 'polite');
      importButton.addEventListener('click', async () => {
        const selected = file.files?.[0];
        if (!selected) { importStatus.textContent = 'Choose a CSV file first.'; return; }
        if (selected.size > 900000) { importStatus.textContent = 'CSV is too large (900 KB maximum).'; return; }
        importButton.disabled = true; importStatus.textContent = 'Importing…';
        try {
          const content = await selected.text();
          const result = await apiFetch('/api/finance/import', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({account_id: payload.accounts?.[0]?.account_id || '', source_id: selected.name, content, currency: payload.accounts?.[0]?.currency || 'USD'})
          });
          const imported = result.imported_transaction_ids?.length || 0;
          importStatus.textContent = `Imported ${imported} transaction${imported === 1 ? '' : 's'}; source hash recorded.`;
          await loadFinance();
        } catch (error) { importStatus.textContent = error.message || 'Finance import unavailable.'; }
        finally { importButton.disabled = false; }
      });
      importSection.append(importTitle, importHint, file, importButton, importStatus); panel.append(importSection);
      appendTodaySection(panel, 'Accounts', payload.accounts || []);
      appendTodaySection(panel, 'Recent transactions', payload.transactions || []);
    }
    const boundary = document.createElement('p'); boundary.className = 'muted';
    boundary.textContent = payload.boundary || 'Finance is private Principal-scoped canonical state.';
    panel.append(boundary);
  } catch (_) { panel.textContent = 'Finance is unavailable; no financial state was changed.'; }
}
async function loadObjectives() {
  const panel = document.getElementById('detail');
  panel.replaceChildren();
  try {
    const response = await fetchWithTimeout('/api/objectives');
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'Objectives unavailable.');
    const objectives = payload.objectives || [];
    const heading = document.createElement('p');
    heading.textContent = objectives.length ? `${objectives.length} authorized objective(s)` : 'No active objectives.';
    panel.append(heading);
    objectives.forEach(objective => {
      const card = document.createElement('section'); card.className = 'detail-card';
      const title = document.createElement('h3'); title.textContent = `${objective.state} · ${objective.objective_id}`;
      const text = document.createElement('p'); text.textContent = objective.utterance || 'Objective without conversational text';
      card.append(title, text);
      if (objective.capability_needs?.length) {
        const needs = document.createElement('h4'); needs.textContent = 'Capability needs'; card.append(needs);
        objective.capability_needs.forEach(need => {
          const needCard = document.createElement('section'); needCard.className = 'detail-card';
          const needTitle = document.createElement('h5');
          needTitle.textContent = `${need.status || 'open'} · ${need.requirement_id || need.effect_id || 'requirement'}`;
          const effect = document.createElement('p');
          effect.textContent = need.normalized_effect || need.requested_effect || 'Unresolved requested effect';
          needCard.append(needTitle, effect);
          const investigation = need.investigation || need.investigation_state;
          if (investigation) {
            const state = document.createElement('p'); state.className = 'muted';
            state.textContent = `Investigation: ${investigation}`; needCard.append(state);
          }
          const candidates = need.candidate_resolutions || need.candidates;
          if (Array.isArray(candidates) && candidates.length) {
            const candidateTitle = document.createElement('strong'); candidateTitle.textContent = 'Candidate resolutions';
            needCard.append(candidateTitle, renderDetailValue(candidates));
            candidates.forEach(candidate => {
              if (!candidate || candidate.requires_owner_input !== true) return;
              const forgeReview = document.createElement('section');
              forgeReview.className = 'forge-review detail-card';
              forgeReview.setAttribute('aria-label', 'Forge candidate review');
              const forgeTitle = document.createElement('h6');
              forgeTitle.textContent = 'Forge review · candidate only';
              const forgeStatus = document.createElement('span');
              forgeStatus.className = 'status-badge';
              forgeStatus.dataset.state = 'candidate';
              forgeStatus.textContent = String(candidate.status || 'candidate').toUpperCase();
              const forgeDetails = document.createElement('p'); forgeDetails.className = 'muted';
              const permissions = Array.isArray(candidate.permissions) && candidate.permissions.length
                ? ` Requested permissions: ${candidate.permissions.join(', ')}.` : '';
              forgeDetails.textContent = `Owner review is required before a Pack proposal can be prepared.${permissions}`;
              const forgeBoundary = document.createElement('p'); forgeBoundary.className = 'muted';
              forgeBoundary.textContent = 'Research and preview do not install, enable, approve, grant permissions, or execute a candidate.';
              forgeReview.append(forgeTitle, forgeStatus, forgeDetails, forgeBoundary);
              needCard.append(forgeReview);
              const review = document.createElement('button');
              review.type = 'button'; review.textContent = 'Review Packs & capabilities';
              review.addEventListener('click', () => {
                pendingCapabilityFocus = String(candidate.capability || '').trim() || null;
                const packs = document.querySelector('[data-view="packs"]');
                if (packs) packs.click();
              });
              needCard.append(review);
              const research = document.createElement('button');
              research.type = 'button'; research.textContent = 'Research candidate path';
              research.addEventListener('click', () => {
                const effectText = need.requested_effect || need.normalized_effect || 'this requirement';
                const needId = need.need_id || need.requirement_id || 'requirement';
                document.getElementById('utterance').value =
                  `Research a safe path for ${effectText} and save notes as capability-needs/${needId}.md`;
                document.getElementById('chat').requestSubmit();
              });
              needCard.append(research);
            });
          }
          const boundary = document.createElement('p'); boundary.className = 'muted';
          boundary.textContent = 'Investigation is read-only: discovery does not grant installation, enablement, approval, or execution authority.';
          needCard.append(boundary); card.append(needCard);
        });
      }
      panel.append(card);
    });
  } catch (_) { panel.textContent = 'Objective state is unavailable; no objective state was changed.'; }
}
async function loadCompositions() {
  const panel = document.getElementById('detail');
  panel.replaceChildren();
  try {
    const response = await fetchWithTimeout('/api/compositions');
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'Compositions unavailable.');
    const compositions = payload.compositions || [];
    const heading = document.createElement('p');
    heading.textContent = compositions.length
      ? `${compositions.length} trusted cross-capability workflow(s)`
      : 'No cross-capability workflows are currently available.';
    panel.append(heading);
    compositions.forEach(composition => {
      const card = document.createElement('section'); card.className = 'detail-card';
      const title = document.createElement('h3');
      title.textContent = composition.label || composition.id || 'Composition';
      const description = document.createElement('p');
      description.textContent = composition.description || 'Bounded cross-capability workflow';
      card.append(title, description);
      if (Array.isArray(composition.surfaces) && composition.surfaces.length) {
        const surfaces = document.createElement('p'); surfaces.className = 'muted';
        surfaces.textContent = `Surfaces: ${composition.surfaces.join(' · ')}`;
        card.append(surfaces);
      }
      const authority = document.createElement('p'); authority.className = 'muted';
      authority.textContent = `Authority: ${composition.authority || 'Core authorization required'}`;
      card.append(authority); panel.append(card);
    });
  } catch (_) {
    panel.textContent = 'Composition metadata is unavailable; no action state was changed.';
  }
}
async function loadCommunications() {
  const panel = document.getElementById('detail'); panel.replaceChildren();
  try {
    const response = await fetchWithTimeout('/api/communications');
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'Communications unavailable.');
    const messages = payload.messages || [];
    const heading = document.createElement('p');
    heading.textContent = messages.length
      ? `${messages.length} authorized communication outcome(s)`
      : 'No communication drafts or sends yet.';
    panel.append(heading, renderDetailValue({
      provider_boundary: payload.provider_boundary,
      target_boundary: payload.target_boundary,
      outcomes: messages,
    }));
    if (messages.length) {
      const outcomeSection = document.createElement('section'); outcomeSection.className = 'detail-card';
      const outcomeTitle = document.createElement('h3'); outcomeTitle.textContent = 'Provider outcome status';
      outcomeSection.append(outcomeTitle);
      messages.forEach(message => {
        const row = document.createElement('p');
        const status = String(message.provider_status || 'DRAFTED');
        const readback = message.provider_readback_proven === true
          ? 'provider readback proven' : 'provider readback unavailable';
        const delivery = message.delivery_proven === true ? 'delivery proven' : 'delivery not proven';
        row.textContent = `${status} · ${readback} · ${delivery} · ${message.target || 'target not recorded'}`;
        outcomeSection.append(row);
      });
      panel.append(outcomeSection);
    }
    const section = document.createElement('section'); section.className = 'detail-card';
    const title = document.createElement('h3'); title.textContent = 'Send an explicit message';
    const form = document.createElement('form'); form.setAttribute('aria-label', 'Send message');
    const fields = [];
    [['target', 'Approved target'], ['channel', 'Channel'], ['account', 'Account'], ['body', 'Message']].forEach(([name, placeholder]) => {
      const input = document.createElement(name === 'body' ? 'textarea' : 'input');
      input.name = name; input.required = true; input.placeholder = placeholder;
      input.setAttribute('aria-label', placeholder); fields.push(input);
    });
    const source = document.createElement('select');
    source.name = 'source'; source.setAttribute('aria-label', 'Message source');
    [['message', 'Write message'], ['groceries', 'Use authorized grocery list']].forEach(([value, label]) => {
      const option = document.createElement('option'); option.value = value; option.textContent = label;
      source.append(option);
    });
    const sourceNote = document.createElement('p'); sourceNote.className = 'muted';
    sourceNote.textContent = 'Canonical sources are resolved by Core; this control never chooses a recipient or grants send authority.';
    const send = document.createElement('button'); send.type = 'submit'; send.textContent = 'Submit send request';
    form.append(...fields, source, sourceNote, send);
    form.addEventListener('submit', event => {
      event.preventDefault();
      const values = Object.fromEntries(fields.map(field => [field.name, field.value.trim()]));
      if (Object.values(values).some(value => !value)) return;
      document.getElementById('utterance').value = source.value === 'groceries'
        ? `Text my grocery list to ${values.target} via ${values.channel} account ${values.account}`
        : `Send a message to ${values.target} via ${values.channel} account ${values.account} saying ${values.body}`;
      document.getElementById('chat').requestSubmit();
    });
    section.append(title, form); panel.append(section);
  } catch (_) {
    panel.textContent = 'Communications state is unavailable; no message was sent.';
  }
}
async function loadDocuments() {
  const panel = document.getElementById('detail'); panel.replaceChildren();
  try {
    const response = await fetchWithTimeout('/api/documents');
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'Documents unavailable.');
    const documents = payload.documents || [];
    const heading = document.createElement('p');
    heading.textContent = documents.length
      ? `${documents.length} authorized document(s)`
      : 'No authorized documents are currently visible.';
    panel.append(heading);
    const searchSection = document.createElement('section'); searchSection.className = 'detail-card';
    const searchTitle = document.createElement('h3'); searchTitle.textContent = 'Search authorized documents';
    const searchForm = document.createElement('form'); searchForm.setAttribute('aria-label', 'Search authorized documents');
    const searchInput = document.createElement('input'); searchInput.type = 'search'; searchInput.required = true;
    searchInput.placeholder = 'Search by text'; searchInput.setAttribute('aria-label', 'Document search');
    const searchSubmit = document.createElement('button'); searchSubmit.type = 'submit'; searchSubmit.textContent = 'Search';
    searchForm.append(searchInput, searchSubmit);
    searchForm.addEventListener('submit', event => {
      event.preventDefault();
      const query = searchInput.value.trim(); if (!query) return;
      document.getElementById('utterance').value = `Find my documents for ${query}`;
      document.getElementById('chat').requestSubmit();
    });
    const searchBoundary = document.createElement('p'); searchBoundary.className = 'muted';
    searchBoundary.textContent = 'Search is bounded to documents authorized for the current owner.';
    searchSection.append(searchTitle, searchForm, searchBoundary); panel.append(searchSection);
    const exportForm = document.createElement('form'); exportForm.setAttribute('aria-label', 'Search documents to Workspace');
    const exportQuery = document.createElement('input'); exportQuery.type = 'search'; exportQuery.required = true;
    exportQuery.placeholder = 'Search query'; exportQuery.setAttribute('aria-label', 'Workspace search query');
    const exportPath = document.createElement('input'); exportPath.required = true;
    exportPath.placeholder = 'Artifact path, e.g. results.md'; exportPath.setAttribute('aria-label', 'Artifact path');
    const exportSubmit = document.createElement('button'); exportSubmit.type = 'submit'; exportSubmit.textContent = 'Save results to Workspace';
    exportForm.append(exportQuery, exportPath, exportSubmit);
    exportForm.addEventListener('submit', event => {
      event.preventDefault();
      const query = exportQuery.value.trim(); const path = exportPath.value.trim();
      if (!query || !path) return;
      document.getElementById('utterance').value = `Find my documents for ${query} and save results as ${path}`;
      document.getElementById('chat').requestSubmit();
    });
    const exportBoundary = document.createElement('p'); exportBoundary.className = 'muted';
    exportBoundary.textContent = 'Results become a scoped Workspace artifact only after the normal Core authorization and independent verification path.';
    searchSection.append(exportForm, exportBoundary);
    const sendSearchForm = document.createElement('form');
    sendSearchForm.setAttribute('aria-label', 'Send document search results');
    const sendSearchQuery = document.createElement('input'); sendSearchQuery.type = 'search';
    sendSearchQuery.required = true; sendSearchQuery.placeholder = 'Search query';
    sendSearchQuery.setAttribute('aria-label', 'Communication search query');
    const sendSearchSubmit = document.createElement('button'); sendSearchSubmit.type = 'submit';
    sendSearchSubmit.textContent = 'Send search results';
    sendSearchForm.append(sendSearchQuery, sendSearchSubmit);
    sendSearchForm.addEventListener('submit', event => {
      event.preventDefault();
      const query = sendSearchQuery.value.trim(); if (!query) return;
      document.getElementById('utterance').value =
        `Text me the document search results for ${query}`;
      document.getElementById('chat').requestSubmit();
    });
    const sendSearchBoundary = document.createElement('p'); sendSearchBoundary.className = 'muted';
    sendSearchBoundary.textContent = 'Core fixes authorized matches before sending; provider acceptance is not delivery proof.';
    searchSection.append(sendSearchForm, sendSearchBoundary);
    documents.forEach(documentRecord => {
      const card = document.createElement('section'); card.className = 'detail-card';
      const title = document.createElement('h3');
      title.textContent = documentRecord.title || documentRecord.document_id || 'Document';
      const metadata = document.createElement('p'); metadata.className = 'muted';
      metadata.textContent = `${documentRecord.source || 'authorized source'} · ${documentRecord.document_id || 'unknown id'}`;
      const button = document.createElement('button'); button.type = 'button';
      button.textContent = 'Read document';
      button.addEventListener('click', async () => {
        button.disabled = true;
        try {
          const result = await fetchWithTimeout(`/api/documents/file?document_id=${encodeURIComponent(documentRecord.document_id)}`);
          const file = await result.json();
          if (!result.ok) throw new Error(file.error || 'Document unavailable.');
          const pre = document.createElement('pre'); pre.className = 'detail-card';
          pre.textContent = file.text || '';
          card.append(pre);
        } catch (_) { button.textContent = 'Document unavailable'; }
        finally { button.disabled = false; }
      });
      const summarize = document.createElement('button'); summarize.type = 'button';
      summarize.textContent = 'Create summary artifact';
      summarize.addEventListener('click', () => {
        const documentId = String(documentRecord.document_id || '').trim();
        const documentTitle = String(documentRecord.title || documentId).trim();
        if (!documentId || !documentTitle) return;
        document.getElementById('utterance').value =
          `Summarize ${documentTitle} to ${documentId}-summary.md`;
        document.getElementById('chat').requestSubmit();
      });
      const exportArtifact = document.createElement('button'); exportArtifact.type = 'button';
      exportArtifact.textContent = 'Export document to Workspace';
      exportArtifact.addEventListener('click', () => {
        const documentId = String(documentRecord.document_id || '').trim();
        const documentTitle = String(documentRecord.title || documentId).trim();
        if (!documentId || !documentTitle) return;
        document.getElementById('utterance').value =
          `Export ${documentTitle} to ${documentId}.md`;
        document.getElementById('chat').requestSubmit();
      });
      card.append(title, metadata, button, summarize, exportArtifact); panel.append(card);
    });
    if (payload.transformation_boundary) {
      const boundary = document.createElement('p'); boundary.className = 'muted';
      boundary.textContent = payload.transformation_boundary; panel.append(boundary);
    }
  } catch (_) { panel.textContent = 'Documents are unavailable; no document state was changed.'; }
}
async function loadDailyDriver() {
  const panel = document.getElementById('detail'); panel.replaceChildren();
  try {
    const response = await fetchWithTimeout('/api/daily-driver');
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'Daily-driver status unavailable.');
    const heading = document.createElement('p');
    heading.textContent = 'Capability status from the current release truth';
    panel.append(heading);
    appendDailyDriverStatuses(panel, payload.statuses || {});
    const gates = Array.isArray(payload.provider_gates) ? payload.provider_gates : [];
    const gateSection = document.createElement('section'); gateSection.className = 'detail-card';
    const gateTitle = document.createElement('h3'); gateTitle.textContent = 'Provider gates';
    gateSection.append(gateTitle);
    const gateList = document.createElement('ul');
    if (!gates.length) {
      const item = document.createElement('li'); item.textContent = 'No configured provider gate is currently recorded.';
      gateList.append(item);
    } else gates.forEach(gate => {
      const item = document.createElement('li'); item.textContent = String(gate); gateList.append(item);
    });
    gateSection.append(gateList);
    panel.append(gateSection, renderDetailValue({
      source_basis_sha: payload.source_basis_sha,
      metrics: payload.metrics,
      boundary: payload.boundary,
    }));
  } catch (_) {
    panel.textContent = 'Daily-driver status is unavailable; no capability state was changed.';
  }
}
function appendDailyDriverStatuses(panel, statuses) {
  const section = document.createElement('section'); section.className = 'today-overview';
  section.setAttribute('aria-label', 'Capability readiness');
  Object.entries(statuses).forEach(([key, value]) => {
    const card = document.createElement('div'); card.className = 'today-overview-card';
    const state = String(value || 'NONE').toLowerCase();
    card.dataset.attention = ['partial', 'fixture', 'none'].includes(state) ? 'true' : 'false';
    const status = document.createElement('strong'); status.textContent = String(value || 'NONE');
    const label = document.createElement('span'); label.textContent = key.replaceAll('_', ' ');
    card.append(status, label); section.append(card);
  });
  panel.append(section);
}
async function loadPacks() {
  const panel = document.getElementById('detail');
  panel.replaceChildren();
  try {
    const response = await fetchWithTimeout('/api/packs');
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'Packs unavailable.');
    const heading = document.createElement('p');
    const focusText = pendingCapabilityFocus
      ? ` Candidate capability: ${pendingCapabilityFocus}.`
      : '';
    heading.textContent = payload.packs?.length
      ? `${payload.packs.length} Pack(s) in the lifecycle registry`
      : 'No Pack metadata is currently available.';
    heading.textContent += focusText;
    panel.append(heading);
    (payload.packs || []).forEach(pack => {
      const card = document.createElement('section');
      card.className = 'detail-card';
      const title = document.createElement('h2');
      title.textContent = `${pack.label || pack.pack_id} · ${pack.status || 'unknown'}`;
      card.append(title, renderDetailValue(pack));
      if (pendingCapabilityFocus && JSON.stringify(pack).includes(pendingCapabilityFocus)) {
        const match = document.createElement('p'); match.className = 'status-badge';
        match.textContent = `Matches candidate capability: ${pendingCapabilityFocus}`;
        card.prepend(match);
      }
      if (pack.status !== 'enabled' && Array.isArray(pack.permissions)) {
        const enable = document.createElement('button');
        enable.type = 'button';
        enable.textContent = 'Approve & enable';
        enable.addEventListener('click', async () => {
          enable.disabled = true;
          try {
            const response = await apiFetch('/api/packs/enable', {
              method: 'POST', headers: {'content-type': 'application/json'},
              body: JSON.stringify({pack_id: pack.pack_id, permissions: pack.permissions, confirm: true})
            });
            const result = await response.json();
            if (!response.ok) throw new Error(result.error || 'Pack enablement was denied.');
            await loadPacks();
            await loadState();
          } catch (_) {
            enable.disabled = false;
            enable.textContent = 'Enablement denied';
          }
        });
        card.append(enable);
      }
      panel.append(card);
    });
  } catch (_) {
    panel.textContent = 'Pack state is unavailable; no lifecycle or permission state was changed.';
  }
}
async function refreshState() {
  refresh.disabled = true;
  nodes.setAttribute('aria-busy', 'true');
  try {
    try {
      await loadHealth();
    } catch (error) {
      const code = error.code || 'health_unavailable';
      clearHealthDetails();
      document.getElementById('health').textContent =
        `${errorLabel(code)} (${code}).`;
    }
    try {
      await loadState();
    } catch (error) {
      const code = error.code || 'state_unavailable';
      document.getElementById('state-status').textContent =
        `State refresh failed — ${errorLabel(code)} (${code}). Use Refresh state to try again.`;
      if (code === 'identity_unavailable' || code === 'state_access_denied') {
        clearAuthorizedDisplays();
      }
    }
  } finally {
    nodes.setAttribute('aria-busy', 'false');
    refresh.disabled = false;
  }
}
refresh.addEventListener('click', refreshState);
refreshState().catch(() => {
  clearHealthDetails();
  document.getElementById('health').textContent = 'Runtime status unavailable.';
});
document.getElementById('chat').addEventListener('submit', async event => {
  event.preventDefault(); const form = event.currentTarget;
  const input = document.getElementById('utterance');
  const send = form.querySelector('button');
  const utterance = input.value.trim(); if (!utterance || send.disabled) return;
  if (pendingOutcomeUnknown) {
    recoverPendingRequest();
    return;
  }
  const correlationId = pendingCorrelationId || crypto.randomUUID();
  if (!pendingCorrelationId) appendConversationMessage('owner-message', `You: ${utterance}`);
  send.disabled = true; input.disabled = true;
  form.setAttribute('aria-busy', 'true');
  persistPendingRequest(utterance, correlationId);
  document.getElementById('activity').textContent = 'Status: working';
  setOutcomeStatus('executing');
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), messageTimeoutMs);
  try {
    const requestBody = {utterance, correlation_id:correlationId, session_id:conversationSessionId};
    if (!pendingCorrelationId && conversationContextCorrelationId)
      requestBody.context_correlation_id = conversationContextCorrelationId;
    const response = await apiFetch('/api/message', {method:'POST',
      headers:{'content-type':'application/json'},
      body:JSON.stringify(requestBody), signal:controller.signal});
    const result = await response.json();
    const answer = result.message || result.error || 'No response';
    document.getElementById('answer').textContent = answer;
    appendConversationMessage('aegis-message', `AEGIS: ${answer}`);
    const sources = document.getElementById('research-sources');
    const researchPanel = document.getElementById('research-panel');
    sources.replaceChildren();
    (result.sources || []).forEach(source => {
      const item = document.createElement('li');
      const link = document.createElement('a');
      link.textContent = `${source.title} · retrieved ${source.retrieved_at}`;
      link.href = source.url; link.target = '_blank'; link.rel = 'noopener noreferrer';
      item.append(link); sources.append(item);
    });
    if (result.sources && result.sources.length) {
      latestResearch = {
        query: utterance,
        summary: answer,
        provider_id: 'bounded public research',
        retrieved_at: new Date().toISOString(),
        sources: result.sources,
      };
      renderResearchSummary();
    }
    researchPanel.hidden = !(result.sources && result.sources.length);
    if (result.steps && result.steps.length) document.getElementById('step-status').textContent =
      result.steps.map(step =>
        `${step.action_id}: ${lifecycleLabel(step.state)} · ${step.message}`).join(' | ');
    else document.getElementById('step-status').textContent = '';
    if (result.state) {
      const outcomeUnknown = result.evidence && result.evidence.assurance === 'OUTCOME_UNKNOWN';
      setOutcomeStatus(outcomeUnknown ? 'unknown' : result.state);
      document.getElementById('activity').textContent =
      `Status: ${outcomeUnknown ? 'Outcome unknown' : lifecycleLabel(result.state)}${result.detail ? ` · ${result.detail}` : ''}`;
    } else if (!response.ok) {
      setOutcomeStatus('failed');
      document.getElementById('activity').textContent =
      `Status: ${errorLabel(result.code)} (${result.code || 'request_failed'})`;
    }
    if (response.ok) {
      const feedback = document.getElementById('feedback');
      feedback.hidden = !result.correlation_id;
      feedback.dataset.correlationId = result.correlation_id || '';
      document.getElementById('feedback-status').textContent = '';
      if (result.retryable === true) {
        pendingOutcomeUnknown = false;
        pendingCorrelationId = correlationId; send.textContent = 'Retry';
        persistPendingRequest(utterance, correlationId, false);
      } else if (result.evidence && result.evidence.assurance === 'OUTCOME_UNKNOWN') {
        pendingOutcomeUnknown = true;
        pendingCorrelationId = correlationId; send.textContent = 'Recheck status';
        persistPendingRequest(utterance, correlationId, true);
      } else {
        pendingCorrelationId = null; pendingOutcomeUnknown = false; send.textContent = 'Send';
        clearPendingRequest();
        if (result.state === 'completed' && result.correlation_id)
          persistConversationContext(result.correlation_id);
      }
      if (result.state === 'completed') refreshState();
    } else {
      const authorizationLost = result.code === 'identity_unavailable' ||
        result.code === 'state_access_denied';
      if (authorizationLost) {
        clearAuthorizedDisplays();
        document.getElementById('state-status').textContent =
          'Authorization lost; authorized state cleared.';
        send.textContent = 'Send';
      } else {
        pendingCorrelationId = retryableCodes.has(result.code) ? correlationId : null;
        if (!pendingCorrelationId) clearPendingRequest();
        send.textContent = pendingCorrelationId ? 'Retry' : 'Send';
      }
    }
  } catch (error) {
    const timedOut = error && error.name === 'AbortError';
    document.getElementById('answer').textContent = timedOut
      ? 'AEGIS did not respond in time. The outcome is unknown.'
      : 'AEGIS is unavailable.';
    setOutcomeStatus('unknown');
    document.getElementById('activity').textContent = timedOut
      ? 'Status: request_timeout · Retry uses the same correlation.'
      : 'Status: unavailable';
    pendingOutcomeUnknown = timedOut;
    pendingCorrelationId = correlationId; send.textContent = timedOut ? 'Recheck status' : 'Retry';
    persistPendingRequest(utterance, correlationId, timedOut);
  } finally {
    clearTimeout(timeout);
    form.setAttribute('aria-busy', 'false');
    send.disabled = false; input.disabled = false;
  }
});
document.querySelectorAll('[data-feedback]').forEach(button =>
  button.addEventListener('click', async event => {
  const feedback = document.getElementById('feedback');
  const correlationId = feedback.dataset.correlationId;
  if (!correlationId) return;
  for (const item of document.querySelectorAll('[data-feedback]')) item.disabled = true;
  try {
    const response = await apiFetch('/api/feedback', {method:'POST',
      headers:{'content-type':'application/json'}, body:JSON.stringify({
        correlation_id:correlationId, outcome:event.currentTarget.dataset.feedback})});
    const result = await response.json();
    document.getElementById('feedback-status').textContent = response.ok ? 'Recorded.' :
      (result.error || 'Feedback unavailable.');
  } catch (_) {
    document.getElementById('feedback-status').textContent = 'Feedback unavailable.';
  }
}));
restorePendingRequest();
recoverPendingRequest();

// Small progressive enhancement for the owner shell. Core routing remains in
// the existing bounded browser adapter; this only controls presentation.
document.addEventListener('DOMContentLoaded', () => {
  const nav = document.querySelector('.product-nav');
  if (!nav) return;
  const toggle = document.createElement('button');
  toggle.type = 'button';
  toggle.className = 'nav-more';
  toggle.textContent = 'More views';
  toggle.setAttribute('aria-expanded', 'false');
  toggle.addEventListener('click', () => {
    const expanded = nav.classList.toggle('show-advanced');
    toggle.setAttribute('aria-expanded', String(expanded));
    toggle.textContent = expanded ? 'Fewer views' : 'More views';
  });
  nav.append(toggle);
});
