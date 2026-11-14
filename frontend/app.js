"use strict";

// Set window.TIBB_API_BASE before this script to call an API on another origin.
const API_BASE = String(window.TIBB_API_BASE || "").replace(/\/$/, "");
const STORAGE_KEY = "tibb-chats-v2";
const LEGACY_STORAGE_KEY = "tibb-chat-v1"; // single conversation, before the list existed
const SIDEBAR_KEY = "tibb-sidebar-collapsed";
const MAX_CHATS = 200;
const HISTORY_TURNS = 6;
const GENERIC_ERROR = "تعذّر إكمال الإجابة. يرجى المحاولة مرة أخرى.";
const NETWORK_ERROR = "تعذّر الاتصال بالخادم. تحقق من اتصالك ثم حاول مجدداً.";
const DEFAULT_TITLE = "الطب النبوي";
const DEFAULT_PRESETS = [
  "كيف تكون العلاقة بين التداوي والقدر؟",
  "ما هدي النبي ﷺ في التداوي والأمر به؟",
  "ما علاج المفؤود كما ورد في الكتاب؟",
];

const ICONS = {
  send: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 19V5M5 12l7-7 7 7"/></svg>',
  stop: '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>',
  edit: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/></svg>',
  book: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 5h6a3 3 0 0 1 3 3v12a2 2 0 0 0-2-2H4zM20 5h-4a3 3 0 0 0-3 3v12a2 2 0 0 1 2-2h5z"/></svg>',
  chevron: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M15 6l-6 6 6 6"/></svg>',
  stopped: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/><rect x="9" y="9" width="6" height="6" rx="1"/></svg>',
  trash: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M10 11v6M14 11v6M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12M9 7V4h6v3"/></svg>',
};

const els = {
  welcome: document.getElementById("welcome"),
  presets: document.getElementById("presets"),
  transcript: document.getElementById("transcript"),
  composer: document.getElementById("composer"),
  input: document.getElementById("input"),
  send: document.getElementById("send"),
  chatList: document.getElementById("chat-list"),
  chatTitle: document.getElementById("chat-title"),
  backdrop: document.getElementById("backdrop"),
};

const desktopQuery = window.matchMedia("(min-width: 900px)");

/*
 * chat: {id, title, createdAt, updatedAt, messages}
 * user message: {role: "user", content}
 * assistant message: {role: "assistant", answer, allSources, sources, followUps, status, error, sourcesOpen}
 *   status: "streaming" | "done" | "stopped" | "error"
 */
const state = {
  chats: [], // newest activity first
  activeId: null, // null = a new conversation that has no question yet
  presets: DEFAULT_PRESETS,
  controllers: new Map(), // chat id -> AbortController of its running answer
  editingIndex: null,
};

/* ---------- persistence ---------- */

function newId() {
  return window.crypto?.randomUUID ? crypto.randomUUID() : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

function titleFrom(question) {
  const text = String(question || "").replace(/\s+/g, " ").trim();
  return text.length > 80 ? `${text.slice(0, 80)}…` : text || DEFAULT_TITLE;
}

function restoreMessage(message) {
  if (!message || (message.role !== "user" && message.role !== "assistant")) return null;
  if (message.role === "user") return { role: "user", content: String(message.content || "") };
  return {
    role: "assistant",
    answer: String(message.answer || ""),
    allSources: [],
    sources: Array.isArray(message.sources) ? message.sources : [],
    followUps: Array.isArray(message.followUps) ? message.followUps : [],
    // An answer that was streaming when the page closed can't resume.
    status: message.status === "streaming" ? "stopped" : message.status || "done",
    error: message.error || "",
    sourcesOpen: Boolean(message.sourcesOpen),
  };
}

function restoreChat(chat) {
  if (!chat || !Array.isArray(chat.messages)) return null;
  const messages = chat.messages.map(restoreMessage).filter(Boolean);
  if (!messages.length) return null;
  const now = Date.now();
  return {
    id: String(chat.id || newId()),
    title: String(chat.title || titleFrom(messages[0].content)),
    createdAt: Number(chat.createdAt) || now,
    updatedAt: Number(chat.updatedAt) || now,
    messages,
  };
}

function loadChats() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
    if (saved && Array.isArray(saved.chats)) {
      const chats = saved.chats.map(restoreChat).filter(Boolean).sort((a, b) => b.updatedAt - a.updatedAt);
      const activeId = chats.some((chat) => chat.id === saved.activeId) ? saved.activeId : null;
      return { chats, activeId };
    }
    // Carry the single saved conversation from the earlier version into the list.
    const legacy = restoreChat({ messages: JSON.parse(localStorage.getItem(LEGACY_STORAGE_KEY) || "[]") });
    if (legacy) return { chats: [legacy], activeId: legacy.id, migrated: true };
  } catch {
    /* unreadable storage: start fresh */
  }
  return { chats: [], activeId: null };
}

/** Keep only what is needed to redisplay a chat; passage texts would fill storage quickly. */
function serializeMessage(message) {
  if (message.role === "user") return { role: "user", content: message.content };
  return {
    role: "assistant",
    answer: message.answer,
    status: message.status === "streaming" ? "stopped" : message.status,
    error: message.error || undefined,
    followUps: message.followUps || [],
    sources: (message.sources || []).map(({ id, title, source_url }) => ({ id, title, source_url })),
    sourcesOpen: message.sourcesOpen || undefined,
  };
}

function saveChats() {
  const chats = state.chats.slice(0, MAX_CHATS).map((chat) => ({ ...chat, messages: chat.messages.map(serializeMessage) }));
  // If storage is full, drop the oldest conversations until it fits.
  for (let keep = chats.length; keep >= 0; keep = keep > 0 ? Math.min(keep - 1, Math.floor(keep * 0.8)) : -1) {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ activeId: state.activeId, chats: chats.slice(0, keep) }));
      localStorage.removeItem(LEGACY_STORAGE_KEY);
      return;
    } catch {
      /* quota exceeded or storage unavailable: try with fewer chats */
    }
  }
}

/* ---------- chat helpers ---------- */

function activeChat() {
  return state.chats.find((chat) => chat.id === state.activeId) || null;
}

function activeMessages() {
  return activeChat()?.messages || [];
}

function isBusy(chatId = state.activeId) {
  return chatId !== null && state.controllers.has(chatId);
}

function touch(chat) {
  chat.updatedAt = Date.now();
  state.chats = [chat, ...state.chats.filter((item) => item !== chat)];
}

function stopGeneration(chatId = state.activeId) {
  state.controllers.get(chatId)?.abort();
}

/* ---------- text rendering ---------- */

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function safeSourceUrl(value) {
  try {
    const url = new URL(String(value || "").trim());
    return url.protocol === "https:" && url.hostname === "ar.wikisource.org" ? url.href : null;
  } catch {
    return null;
  }
}

function cleanAnswer(answer, streaming) {
  let text = String(answer || "").replace(/\\([\[\]*_])/g, "$1");
  // Hide a citation that is still arriving, e.g. "[S" or "[S1".
  if (streaming) text = text.replace(/\[S?\d*$/, "");
  return text;
}

/** Number cited sources by first appearance, like the Streamlit app. */
function citationOrder(answer, sourcesById) {
  const order = new Map();
  for (const match of answer.matchAll(/\[(S\d+)\]/g)) {
    if (sourcesById.has(match[1]) && !order.has(match[1])) order.set(match[1], order.size + 1);
  }
  return order;
}

function citationHtml(id, order, sourcesById) {
  const number = order.get(id);
  if (!number) return "";
  const source = sourcesById.get(id);
  const url = safeSourceUrl(source.source_url);
  const title = escapeHtml(source.title || "");
  return url
    ? `<a class="cite" href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer" title="${title}">${number}</a>`
    : `<span class="cite" title="${title}">${number}</span>`;
}

function inlineMarkdown(text, cite) {
  let html = escapeHtml(text);
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, "$1<em>$2</em>");
  return html.replace(/\[(S\d+)\]/g, (_, id) => cite(id));
}

/** Small Markdown subset: paragraphs, headings, lists, quotes, bold, italic. HTML is always escaped. */
function renderMarkdown(markdown, cite) {
  let html = "";
  let paragraph = [];
  let list = null;
  const flushParagraph = () => {
    if (paragraph.length) html += `<p>${inlineMarkdown(paragraph.join("\n"), cite).replace(/\n/g, "<br>")}</p>`;
    paragraph = [];
  };
  const closeList = () => {
    if (list) html += `</${list}>`;
    list = null;
  };
  for (const rawLine of markdown.replace(/\r\n?/g, "\n").split("\n")) {
    const line = rawLine.trim();
    let m;
    if (!line) {
      flushParagraph();
      closeList();
    } else if ((m = line.match(/^(#{1,6})\s+(.*)$/))) {
      flushParagraph();
      closeList();
      const level = Math.min(m[1].length + 2, 6);
      html += `<h${level}>${inlineMarkdown(m[2], cite)}</h${level}>`;
    } else if ((m = line.match(/^[-*•]\s+(.*)$/))) {
      flushParagraph();
      if (list !== "ul") { closeList(); html += "<ul>"; list = "ul"; }
      html += `<li>${inlineMarkdown(m[1], cite)}</li>`;
    } else if ((m = line.match(/^[\d٠-٩]+[.)]\s+(.*)$/))) {
      flushParagraph();
      if (list !== "ol") { closeList(); html += "<ol>"; list = "ol"; }
      html += `<li>${inlineMarkdown(m[1], cite)}</li>`;
    } else if ((m = line.match(/^>\s?(.*)$/))) {
      flushParagraph();
      closeList();
      html += `<blockquote>${inlineMarkdown(m[1], cite)}</blockquote>`;
    } else {
      closeList();
      paragraph.push(line);
    }
  }
  flushParagraph();
  closeList();
  return html;
}

/* ---------- DOM helpers ---------- */

function el(tag, className, html) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (html !== undefined) node.innerHTML = html;
  return node;
}

function suggestionButton(text) {
  const button = el("button", "suggestion");
  button.type = "button";
  button.textContent = text;
  button.addEventListener("click", () => ask(text));
  return button;
}

/* ---------- sidebar ---------- */

function groupLabel(timestamp) {
  const startOfDay = (date) => new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime();
  const days = Math.round((startOfDay(new Date()) - startOfDay(new Date(timestamp))) / 86400000);
  if (days <= 0) return "اليوم";
  if (days === 1) return "أمس";
  if (days < 7) return "آخر 7 أيام";
  if (days < 30) return "آخر 30 يوماً";
  return "أقدم";
}

function renderSidebar() {
  if (!state.chats.length) {
    els.chatList.replaceChildren(el("p", "chat-empty", "لا توجد محادثات بعد."));
    return;
  }
  const nodes = [];
  let currentGroup = null;
  for (const chat of state.chats) {
    const group = groupLabel(chat.updatedAt);
    if (group !== currentGroup) {
      currentGroup = group;
      nodes.push(el("div", "chat-group", group));
    }
    const active = chat.id === state.activeId;
    const item = el("div", `chat-item${active ? " active" : ""}`);
    const open = el("button", "chat-open");
    open.type = "button";
    open.title = chat.title;
    if (active) open.setAttribute("aria-current", "page");
    if (isBusy(chat.id)) open.append(el("span", "chat-busy"));
    const title = el("span", "title");
    title.textContent = chat.title;
    open.append(title);
    open.addEventListener("click", () => openChat(chat.id));
    const remove = el("button", "chat-delete", ICONS.trash);
    remove.type = "button";
    remove.title = "حذف المحادثة";
    remove.setAttribute("aria-label", `حذف المحادثة: ${chat.title}`);
    remove.addEventListener("click", () => deleteChat(chat.id));
    item.append(open, remove);
    nodes.push(item);
  }
  els.chatList.replaceChildren(...nodes);
}

function setSidebarOpen(open) {
  if (desktopQuery.matches) {
    document.body.classList.toggle("sidebar-collapsed", !open);
    try {
      localStorage.setItem(SIDEBAR_KEY, open ? "0" : "1");
    } catch {
      /* preference just won't persist */
    }
  } else {
    document.body.classList.toggle("sidebar-open", open);
  }
}

function toggleSidebar() {
  const open = desktopQuery.matches
    ? document.body.classList.contains("sidebar-collapsed")
    : !document.body.classList.contains("sidebar-open");
  setSidebarOpen(open);
}

function closeDrawer() {
  document.body.classList.remove("sidebar-open");
}

/* ---------- transcript ---------- */

function renderUser(message, index) {
  const wrap = el("div", "msg user");
  if (state.editingIndex === index) {
    const box = el("div", "edit-box");
    const area = document.createElement("textarea");
    area.value = message.content;
    area.setAttribute("aria-label", "تعديل السؤال");
    area.maxLength = 2000;
    const actions = el("div", "edit-actions");
    const cancel = el("button", "btn", "إلغاء");
    cancel.type = "button";
    const submit = el("button", "btn primary", "إرسال");
    submit.type = "button";
    const doSubmit = () => submitEdit(index, area.value);
    cancel.addEventListener("click", cancelEdit);
    submit.addEventListener("click", doSubmit);
    area.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        doSubmit();
      } else if (event.key === "Escape") {
        cancelEdit();
      }
    });
    actions.append(cancel, submit);
    box.append(area, actions);
    wrap.append(box);
    requestAnimationFrame(() => {
      area.focus();
      area.setSelectionRange(area.value.length, area.value.length);
    });
    return wrap;
  }

  const bubble = el("div", "bubble");
  bubble.textContent = message.content;
  wrap.append(bubble);
  if (!isBusy()) {
    const actions = el("div", "msg-actions");
    const edit = el("button", "icon-btn", `${ICONS.edit}<span>تعديل</span>`);
    edit.type = "button";
    edit.addEventListener("click", () => startEdit(index));
    actions.append(edit);
    wrap.append(actions);
  }
  return wrap;
}

function renderAssistant(message, index, messages) {
  const wrap = el("div", "msg assistant");
  wrap.append(el("div", "avatar", ICONS.book));
  const content = el("div", "content");
  wrap.append(content);

  const streaming = message.status === "streaming";
  const answer = cleanAnswer(message.answer, streaming);

  if (streaming && !answer) {
    content.append(el("div", "thinking", '<span class="dots"><span></span><span></span><span></span></span>أبحث في النص وأعدّ الإجابة…'));
    return wrap;
  }

  // While streaming, link against every retrieved passage; once done, only cited ones.
  const pool = streaming ? message.allSources || [] : message.sources || [];
  const sourcesById = new Map(pool.map((s) => [s.id, s]));
  const order = citationOrder(answer, sourcesById);
  const cite = (id) => citationHtml(id, order, sourcesById);

  if (answer) {
    const body = el("div", "markdown", renderMarkdown(answer, cite));
    if (streaming) {
      const last = body.lastElementChild?.matches("ul, ol") ? body.lastElementChild.lastElementChild : body.lastElementChild;
      (last || body).append(el("span", "caret"));
    }
    content.append(body);
  }

  if (message.status === "stopped") {
    content.append(el("div", "note", `${ICONS.stopped}<span>تم إيقاف الإجابة.</span>`));
  } else if (message.status === "error") {
    const note = el("div", "note error");
    note.textContent = message.error || GENERIC_ERROR;
    content.append(note);
  }

  if (message.status === "done" && order.size) {
    const details = el("details", "sources");
    details.open = Boolean(message.sourcesOpen);
    details.addEventListener("toggle", () => {
      message.sourcesOpen = details.open;
      saveChats();
    });
    details.append(el("summary", "", `${ICONS.chevron}<span>المصادر (${order.size})</span>`));
    const list = el("ol");
    for (const [id, number] of order) {
      const source = sourcesById.get(id);
      const item = el("li");
      item.append(el("span", "cite", String(number)));
      const url = safeSourceUrl(source.source_url);
      const label = url ? el("a") : el("span");
      label.textContent = source.title || "";
      if (url) {
        label.href = url;
        label.target = "_blank";
        label.rel = "noopener noreferrer";
      }
      item.append(label);
      list.append(item);
    }
    details.append(list);
    content.append(details);
  }

  // Suggestions only under the latest finished answer, and never while
  // another answer is being prepared or a question is being edited.
  const isLast = index === messages.length - 1;
  if (isLast && message.status === "done" && !isBusy() && state.editingIndex === null) {
    const followups = el("div", "followups");
    for (const question of (message.followUps || []).slice(0, 3)) followups.append(suggestionButton(question));
    if (followups.childElementCount) content.append(followups);
  }
  return wrap;
}

function nearBottom() {
  const scroller = document.scrollingElement;
  return scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 120;
}

function scrollToBottom() {
  const scroller = document.scrollingElement;
  scroller.scrollTo({ top: scroller.scrollHeight });
}

function render({ scroll = false } = {}) {
  const follow = nearBottom() && isBusy();
  const chat = activeChat();
  const messages = activeMessages();

  els.welcome.hidden = messages.length > 0;
  if (!messages.length) els.presets.replaceChildren(...state.presets.map(suggestionButton));
  els.transcript.replaceChildren(
    ...messages.map((message, index) =>
      message.role === "user" ? renderUser(message, index) : renderAssistant(message, index, messages),
    ),
  );
  els.chatTitle.textContent = chat ? chat.title : DEFAULT_TITLE;
  document.title = chat ? `${chat.title} | الطب النبوي` : "الطب النبوي | اسأل عن النص";
  renderSidebar();
  updateComposer();
  if (scroll || follow) scrollToBottom();
}

/**
 * While text streams in, redraw only the streaming message (once per frame).
 * Rebuilding the whole page on every chunk would swallow clicks elsewhere.
 */
const pendingStreamUpdates = new Map(); // reply -> chat
let streamFrameQueued = false;
function scheduleStreamUpdate(chat, reply) {
  pendingStreamUpdates.set(reply, chat);
  if (streamFrameQueued) return;
  streamFrameQueued = true;
  requestAnimationFrame(() => {
    streamFrameQueued = false;
    const updates = [...pendingStreamUpdates];
    pendingStreamUpdates.clear();
    for (const [updatedReply, updatedChat] of updates) {
      if (updatedChat.id !== state.activeId) continue;
      const index = updatedChat.messages.indexOf(updatedReply);
      const node = els.transcript.children[index];
      if (index === -1 || !node) continue;
      const follow = nearBottom();
      node.replaceWith(renderAssistant(updatedReply, index, updatedChat.messages));
      if (follow) scrollToBottom();
    }
  });
}

function updateComposer() {
  const busy = isBusy();
  els.send.classList.toggle("stop", busy);
  els.send.innerHTML = busy ? ICONS.stop : ICONS.send;
  els.send.setAttribute("aria-label", busy ? "إيقاف" : "إرسال");
  els.send.title = busy ? "إيقاف" : "إرسال";
  els.send.disabled = !busy && !els.input.value.trim();
}

function autoGrow() {
  els.input.style.height = "auto";
  els.input.style.height = `${Math.min(els.input.scrollHeight, 200)}px`;
}

/* ---------- conversation actions ---------- */

function buildHistory(messages) {
  const turns = [];
  for (const message of messages) {
    const content = message.role === "user" ? message.content : message.answer;
    if (content && content.trim()) turns.push({ role: message.role, content: content.trim().slice(0, 1500) });
  }
  return turns.slice(-HISTORY_TURNS);
}

function openChat(chatId) {
  state.activeId = chatId;
  state.editingIndex = null;
  closeDrawer();
  saveChats();
  render({ scroll: true });
}

function newChat() {
  // Any answer still running in another conversation keeps going.
  state.activeId = null;
  state.editingIndex = null;
  closeDrawer();
  saveChats();
  render({ scroll: true });
  els.input.focus();
}

function deleteChat(chatId) {
  const chat = state.chats.find((item) => item.id === chatId);
  if (!chat || !window.confirm(`حذف المحادثة «${chat.title}»؟`)) return;
  stopGeneration(chatId);
  state.chats = state.chats.filter((item) => item.id !== chatId);
  if (state.activeId === chatId) {
    state.activeId = null;
    state.editingIndex = null;
  }
  saveChats();
  render();
}

function startEdit(index) {
  state.editingIndex = index;
  render();
}

function cancelEdit() {
  state.editingIndex = null;
  render();
}

function submitEdit(index, text) {
  const question = text.trim();
  const chat = activeChat();
  if (!question || !chat) return;
  stopGeneration(chat.id);
  // Like regenerating from an edited message: drop this turn and everything after it.
  chat.messages.splice(index);
  if (index === 0) chat.title = titleFrom(question);
  state.editingIndex = null;
  ask(question);
}

async function readEvents(body, onEvent) {
  const reader = body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value.replace(/\r\n/g, "\n");
    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const data = block
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n");
      if (data) onEvent(JSON.parse(data));
    }
  }
}

function applyEvent(chat, reply, event) {
  if (event.type === "sources") {
    reply.allSources = event.sources || [];
  } else if (event.type === "delta") {
    reply.answer += event.text || "";
  } else if (event.type === "done") {
    reply.answer = event.answer || reply.answer;
    reply.sources = event.sources || [];
    reply.followUps = event.follow_up_questions || [];
    reply.status = "done";
  } else if (event.type === "error") {
    reply.status = "error";
    reply.error = event.message || GENERIC_ERROR;
  }
  scheduleStreamUpdate(chat, reply);
}

async function ask(text) {
  const question = String(text || "").trim();
  if (!question) return;

  let chat = activeChat();
  if (!chat) {
    const now = Date.now();
    chat = { id: newId(), title: titleFrom(question), createdAt: now, updatedAt: now, messages: [] };
    state.chats.unshift(chat);
    state.activeId = chat.id;
  }
  stopGeneration(chat.id); // asking while an answer is running replaces it, as in Claude

  const history = buildHistory(chat.messages);
  const reply = { role: "assistant", answer: "", allSources: [], sources: [], followUps: [], status: "streaming" };
  chat.messages.push({ role: "user", content: question }, reply);
  touch(chat);
  state.editingIndex = null;
  const controller = new AbortController();
  state.controllers.set(chat.id, controller);
  saveChats();
  render({ scroll: true });

  try {
    const response = await fetch(`${API_BASE}/api/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history }),
      signal: controller.signal,
    });
    if (!response.ok) {
      let detail = "";
      try {
        detail = (await response.json()).detail;
      } catch {
        /* not JSON */
      }
      throw new Error(typeof detail === "string" && detail ? detail : GENERIC_ERROR);
    }
    await readEvents(response.body, (event) => applyEvent(chat, reply, event));
    if (reply.status === "streaming") throw new Error(NETWORK_ERROR); // stream ended early
  } catch (error) {
    if (error.name === "AbortError") {
      reply.status = "stopped";
    } else if (reply.status === "streaming") {
      reply.status = "error";
      reply.error = error instanceof TypeError ? NETWORK_ERROR : error.message || GENERIC_ERROR;
    }
  } finally {
    if (state.controllers.get(chat.id) === controller) state.controllers.delete(chat.id);
    saveChats();
    render();
  }
}

/* ---------- wiring ---------- */

els.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  if (isBusy()) {
    stopGeneration();
    return;
  }
  const question = els.input.value;
  if (!question.trim()) return;
  els.input.value = "";
  autoGrow();
  ask(question);
});

els.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    if (!isBusy()) els.composer.requestSubmit();
  }
});

els.input.addEventListener("input", () => {
  autoGrow();
  updateComposer();
});

for (const button of document.querySelectorAll(".js-new-chat")) button.addEventListener("click", newChat);
for (const button of document.querySelectorAll(".js-toggle-sidebar")) button.addEventListener("click", toggleSidebar);
els.backdrop.addEventListener("click", closeDrawer);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeDrawer();
});
desktopQuery.addEventListener("change", closeDrawer);

// Other tabs: pick up their changes to the conversation list.
window.addEventListener("storage", (event) => {
  if (event.key !== STORAGE_KEY || state.controllers.size) return;
  const loaded = loadChats();
  state.chats = loaded.chats;
  if (!state.chats.some((chat) => chat.id === state.activeId)) state.activeId = null;
  render();
});

try {
  if (localStorage.getItem(SIDEBAR_KEY) === "1") document.body.classList.add("sidebar-collapsed");
} catch {
  /* default: sidebar shown on wide screens */
}

const loaded = loadChats();
state.chats = loaded.chats;
state.activeId = loaded.activeId;
if (loaded.migrated) saveChats(); // store in the new format and drop the old key

fetch(`${API_BASE}/api/config`)
  .then((response) => (response.ok ? response.json() : null))
  .then((config) => {
    if (config && Array.isArray(config.presets) && config.presets.length) {
      state.presets = config.presets;
      if (!activeMessages().length) render();
    }
  })
  .catch(() => {});

render({ scroll: true });
els.input.focus();
