const form = document.getElementById("ticketForm");
const ticketIdInput = document.getElementById("ticketId");
const subjectInput = document.getElementById("subject");
const descriptionInput = document.getElementById("description");
const clearBtn = document.getElementById("clearBtn");
const apiStatus = document.getElementById("apiStatus");
const vocabSize = document.getElementById("vocabSize");
const classCount = document.getElementById("classCount");
const emptyState = document.getElementById("emptyState");
const resultState = document.getElementById("resultState");
const resultTicketId = document.getElementById("resultTicketId");
const resultCategory = document.getElementById("resultCategory");
const resultTokens = document.getElementById("resultTokens");
const normalizedText = document.getElementById("normalizedText");
const probabilityList = document.getElementById("probabilityList");

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

function buildTicketId() {
  const now = new Date();
  const stamp = [
    now.getFullYear(),
    String(now.getMonth() + 1).padStart(2, "0"),
    String(now.getDate()).padStart(2, "0"),
    String(now.getHours()).padStart(2, "0"),
    String(now.getMinutes()).padStart(2, "0"),
    String(now.getSeconds()).padStart(2, "0")
  ].join("");
  const suffix = Math.floor(Math.random() * 900 + 100);
  return `TCK-${stamp}-${suffix}`;
}

function resetTicketId() {
  ticketIdInput.value = buildTicketId();
}

function showResult(data) {
  emptyState.classList.add("hidden");
  resultState.classList.remove("hidden");

  resultTicketId.textContent = data.ticket_id || ticketIdInput.value;
  resultCategory.textContent = data.category;
  resultTokens.textContent = data.tokens_used;
  normalizedText.textContent = data.normalized_text || "-";

  probabilityList.innerHTML = "";

  const probabilities = data.probabilities || {};

  Object.entries(probabilities).forEach(([label, value]) => {
    const row = document.createElement("div");
    row.className = "probability-row";

    row.innerHTML = `
      <div class="probability-meta">
        <strong>${label}</strong>
        <span>${(value * 100).toFixed(2)}%</span>
      </div>
      <div class="probability-bar">
        <div class="probability-fill" style="width: ${value * 100}%"></div>
      </div>
    `;

    probabilityList.appendChild(row);
  });
}

function setLoadingState(isLoading) {
  const submitButton = form.querySelector('button[type="submit"]');
  submitButton.disabled = isLoading;
  submitButton.textContent = isLoading ? "Classifying..." : "Classify ticket";
}

async function loadHealth() {
  try {
    const response = await fetch("/health");
    const data = await response.json();
    apiStatus.textContent = data.status === "ok" ? "Backend available" : "No response";
    vocabSize.textContent = Number(data.vocabulary_size || 0).toLocaleString("en-US");
    classCount.textContent = Array.isArray(data.classes)
      ? data.classes.length
      : "-";
  } catch (error) {
    apiStatus.textContent = "Backend unavailable";
  }
}

document.querySelectorAll("[data-example]").forEach((button) => {
  button.addEventListener("click", () => {
    const example = examples[button.dataset.example];
    subjectInput.value = example.subject;
    descriptionInput.value = example.description;
  });
});

clearBtn.addEventListener("click", () => {
  form.reset();
  resetTicketId();
  emptyState.classList.remove("hidden");
  resultState.classList.add("hidden");
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setLoadingState(true);

  const payload = {
    ticket_id: ticketIdInput.value,
    subject: subjectInput.value.trim(),
    description: descriptionInput.value.trim(),
    channel: document.getElementById("ticketChannel").value
  };

  try {
    const response = await fetch("/classify", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "The ticket could not be classified.");
    }

    showResult(data);
  } catch (error) {
    emptyState.classList.remove("hidden");
    resultState.classList.add("hidden");
    emptyState.textContent = error.message;
  } finally {
    setLoadingState(false);
  }
});

resetTicketId();
loadHealth();
