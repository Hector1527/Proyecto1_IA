const form = document.getElementById("ticketForm");
const subjectInput = document.getElementById("subject");
const descriptionInput = document.getElementById("description");
const channelInput = document.getElementById("ticketChannel");
const clearBtn = document.getElementById("clearBtn");
const apiStatus = document.getElementById("apiStatus");
const vocabSize = document.getElementById("vocabSize");
const classCount = document.getElementById("classCount");
const nextTicketLabel = document.getElementById("nextTicketLabel");
const emptyState = document.getElementById("emptyState");
const emptyStateMessage = document.getElementById("emptyStateMessage");
const resultState = document.getElementById("resultState");
const resultTicketId = document.getElementById("resultTicketId");
const resultCategory = document.getElementById("resultCategory");
const resultTokens = document.getElementById("resultTokens");
const resultChannel = document.getElementById("resultChannel");
const resultCreatedAt = document.getElementById("resultCreatedAt");
const resultSubject = document.getElementById("resultSubject");
const normalizedText = document.getElementById("normalizedText");
const probabilityList = document.getElementById("probabilityList");
const ticketLookupInput = document.getElementById("ticketLookupInput");
const ticketLookupBtn = document.getElementById("ticketLookupBtn");
const ticketCount = document.getElementById("ticketCount");
const ticketList = document.getElementById("ticketList");
const ticketListEmpty = document.getElementById("ticketListEmpty");
const ticketListEmptyMessage = document.getElementById("ticketListEmptyMessage");

const examples = {
  billing: {
    subject: "Duplicate charge on my card",
    description: "I was charged twice for the same invoice and need the billing team to review the duplicate payment."
  },
  technical: {
    subject: "I cannot log into my account",
    description: "The system rejects my password and I still cannot access the account even after trying to reset it."
  },
  cancel: {
    subject: "I want to cancel the service",
    description: "I want to cancel my subscription and avoid future charges because I will no longer use the platform."
  }
};

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "\"": "&quot;",
    "'": "&#39;"
  }[char]));
}

function formatDate(value) {
  if (!value) {
    return "-";
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return date.toLocaleString("en-US", {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit"
  });
}

function showEmptyState(message) {
  emptyStateMessage.textContent = message;
  emptyState.classList.remove("hidden");
  resultState.classList.add("hidden");
}

function renderProbabilities(probabilities) {
  probabilityList.innerHTML = "";

  Object.entries(probabilities || {}).forEach(([label, value]) => {
    const row = document.createElement("div");
    row.className = "probability-row";
    row.innerHTML = `
      <div class="probability-meta">
        <strong>${escapeHtml(label)}</strong>
        <span>${(value * 100).toFixed(2)}%</span>
      </div>
      <div class="probability-bar">
        <div class="probability-fill" style="width: ${value * 100}%"></div>
      </div>
    `;
    probabilityList.appendChild(row);
  });
}

function showResult(ticket) {
  emptyState.classList.add("hidden");
  resultState.classList.remove("hidden");

  resultTicketId.textContent = ticket.ticket_id || "-";
  resultCategory.textContent = ticket.category || "-";
  resultTokens.textContent = ticket.tokens_used ?? "-";
  resultChannel.textContent = ticket.channel || "-";
  resultCreatedAt.textContent = formatDate(ticket.created_at);
  resultSubject.textContent = ticket.subject || "(No subject)";
  normalizedText.textContent = ticket.normalized_text || "-";
  renderProbabilities(ticket.probabilities);
}

function setLoadingState(isLoading) {
  const submitButton = form.querySelector('button[type="submit"]');
  submitButton.disabled = isLoading;
  submitButton.textContent = isLoading ? "Sending..." : "Send ticket";
}

function renderTicketList(tickets) {
  ticketCount.textContent = tickets.length;
  ticketList.innerHTML = "";

  if (!tickets.length) {
    ticketListEmptyMessage.textContent = "No tickets have been sent yet. New tickets will accumulate here.";
    ticketListEmpty.classList.remove("hidden");
    ticketList.classList.add("hidden");
    return;
  }

  ticketListEmpty.classList.add("hidden");
  ticketList.classList.remove("hidden");

  tickets.forEach((ticket) => {
    const card = document.createElement("article");
    card.className = "ticket-card";
    card.innerHTML = `
      <div class="ticket-card-top">
        <div>
          <p class="ticket-card-id">${escapeHtml(ticket.ticket_id)}</p>
          <h3>${escapeHtml(ticket.subject || "Untitled ticket")}</h3>
        </div>
        <button
          type="button"
          class="ghost-chip ticket-open-btn"
          data-ticket-id="${escapeHtml(ticket.ticket_id)}"
        >
          Open
        </button>
      </div>
      <div class="ticket-card-meta">
        <span>${escapeHtml(ticket.category || "-")}</span>
        <span>${escapeHtml(ticket.channel || "-")}</span>
        <span>${escapeHtml(formatDate(ticket.created_at))}</span>
      </div>
    `;
    ticketList.appendChild(card);
  });
}

async function loadHealth() {
  try {
    const response = await fetch("/health");
    const data = await response.json();
    apiStatus.textContent = data.status === "ok" ? "Backend available" : "No response";
    vocabSize.textContent = Number(data.vocabulary_size || 0).toLocaleString("en-US");
    classCount.textContent = Array.isArray(data.classes) ? data.classes.length : "-";
  } catch (error) {
    apiStatus.textContent = "Backend unavailable";
  }
}

async function loadTickets() {
  try {
    const response = await fetch("/tickets");
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "The tickets could not be loaded.");
    }

    renderTicketList(data.tickets || []);
  } catch (error) {
    ticketListEmptyMessage.textContent = error.message;
    ticketListEmpty.classList.remove("hidden");
    ticketList.classList.add("hidden");
  }
}

async function openTicketById(ticketId) {
  const normalizedId = ticketId.trim();
  if (!normalizedId) {
    showEmptyState("Enter a ticket ID to open a stored ticket.");
    return;
  }

  try {
    const response = await fetch(`/tickets/${encodeURIComponent(normalizedId)}`);
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "The ticket could not be found.");
    }

    ticketLookupInput.value = data.ticket_id;
    showResult(data);
    resultState.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    showEmptyState(error.message);
  }
}

document.querySelectorAll("[data-example]").forEach((button) => {
  button.addEventListener("click", () => {
    const example = examples[button.dataset.example];
    subjectInput.value = example.subject;
    descriptionInput.value = example.description;
  });
});

ticketList.addEventListener("click", (event) => {
  const trigger = event.target.closest("[data-ticket-id]");
  if (!trigger) {
    return;
  }
  openTicketById(trigger.dataset.ticketId);
});

ticketLookupBtn.addEventListener("click", () => {
  openTicketById(ticketLookupInput.value);
});

ticketLookupInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    openTicketById(ticketLookupInput.value);
  }
});

clearBtn.addEventListener("click", () => {
  form.reset();
  nextTicketLabel.textContent = "Generated automatically when sent";
  showEmptyState("Submit a ticket to see the predicted category and class probability distribution.");
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setLoadingState(true);

  const payload = {
    subject: subjectInput.value.trim(),
    description: descriptionInput.value.trim(),
    channel: channelInput.value
  };

  try {
    const response = await fetch("/tickets", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "The ticket could not be sent.");
    }

    nextTicketLabel.textContent = `Last created: ${data.ticket_id}`;
    ticketLookupInput.value = data.ticket_id;
    showResult(data);
    form.reset();
    await loadTickets();
  } catch (error) {
    showEmptyState(error.message);
  } finally {
    setLoadingState(false);
  }
});

showEmptyState("Submit a ticket to see the predicted category and class probability distribution.");
loadHealth();
loadTickets();
