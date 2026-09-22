const questionInput = document.getElementById("question");

const askButton = document.getElementById("askButton");

const buttonText = document.getElementById("buttonText");

const answerSection = document.getElementById("answerSection");

const answerElement = document.getElementById("answer");

const sourcesSection = document.getElementById("sourcesSection");

const sourcesElement = document.getElementById("sources");

const abstentionNotice = document.getElementById("abstentionNotice");

const errorCard = document.getElementById("errorCard");

const errorMessage = document.getElementById("errorMessage");

const metricsSection = document.getElementById("metricsSection");

const totalLatency = document.getElementById("totalLatency");

const retrievalLatency = document.getElementById("retrievalLatency");

const generationLatency = document.getElementById("generationLatency");

const contextCount = document.getElementById("contextCount");

const totalTokens = document.getElementById("totalTokens");

const estimatedCost = document.getElementById("estimatedCost");


function setLoading(isLoading) {

    askButton.disabled = isLoading;

    if (isLoading) {

        askButton.classList.add("loading");

        buttonText.textContent = "Thinking...";

    } else {

        askButton.classList.remove("loading");

        buttonText.textContent = "Ask question";

    }

}


function hideError() {

    errorCard.classList.add("hidden");

    errorMessage.textContent = "";

}


function showError(message) {

    errorMessage.textContent = message;

    errorCard.classList.remove("hidden");

}


function clearAnswer() {

    answerSection.classList.add("hidden");

    answerElement.textContent = "";

    sourcesElement.innerHTML = "";

    sourcesSection.classList.add("hidden");

    abstentionNotice.classList.add("hidden");

    metricsSection.classList.add("hidden");

}


function renderSources(citations) {

    sourcesElement.innerHTML = "";

    if (!citations || citations.length === 0) {

        sourcesSection.classList.add("hidden");

        return;

    }

    const uniqueSources = [];

    const seen = new Set();

    for (const citation of citations) {

        const key =
            `${citation.source}|${citation.section}|${citation.chunk_index}`;

        if (!seen.has(key)) {

            seen.add(key);

            uniqueSources.push(citation);

        }

    }


    for (const citation of uniqueSources) {

        const card = document.createElement("div");

        card.className = "source-card";


        const section = document.createElement("div");

        section.className = "source-section";

        section.textContent =
            citation.section || "Unknown section";


        const file = document.createElement("div");

        file.className = "source-file";

        file.textContent =
            citation.source || "Unknown source";


        card.appendChild(section);

        card.appendChild(file);


        if (
            citation.chunk_index !== null &&
            citation.chunk_index !== undefined
        ) {

            const chunk = document.createElement("div");

            chunk.className = "source-chunk";

            chunk.textContent =
                `Chunk ${citation.chunk_index}`;

            card.appendChild(chunk);

        }


        sourcesElement.appendChild(card);

    }


    sourcesSection.classList.remove("hidden");

}


function formatMilliseconds(value) {

    if (
        value === null ||
        value === undefined ||
        Number.isNaN(Number(value))
    ) {

        return "—";

    }

    return `${Math.round(Number(value))} ms`;

}


function formatTokens(value) {

    if (
        value === null ||
        value === undefined
    ) {

        return "—";

    }

    return Number(value).toLocaleString();

}


function formatCost(value) {

    if (
        value === null ||
        value === undefined
    ) {

        return "—";

    }

    const amount = Number(value);

    if (amount === 0) {

        return "$0.00";

    }

    return `$${amount.toFixed(5)}`;

}


function renderMetrics(metrics) {

    if (!metrics) {

        metricsSection.classList.add("hidden");

        return;

    }


    totalLatency.textContent =
        formatMilliseconds(metrics.total_latency_ms);

    retrievalLatency.textContent =
        formatMilliseconds(metrics.retrieval_latency_ms);

    generationLatency.textContent =
        formatMilliseconds(metrics.generation_latency_ms);

    contextCount.textContent =
        metrics.context_count ?? "—";

    totalTokens.textContent =
        formatTokens(metrics.total_tokens);

    estimatedCost.textContent =
        formatCost(metrics.estimated_cost_usd);


    metricsSection.classList.remove("hidden");

}


function renderResponse(data) {

    answerSection.classList.remove("hidden");

    answerElement.textContent =
        data.answer || "No answer returned.";


    if (data.abstained) {

        abstentionNotice.classList.remove("hidden");

    } else {

        abstentionNotice.classList.add("hidden");

    }


    renderSources(data.citations);

    renderMetrics(data.metrics);

}


async function askQuestion() {

    const question =
        questionInput.value.trim();


    if (!question) {

        showError("Please enter a question.");

        questionInput.focus();

        return;

    }


    hideError();

    clearAnswer();

    setLoading(true);


    try {

        const response =
            await fetch("/api/ask", {

                method: "POST",

                headers: {

                    "Content-Type": "application/json"

                },

                body: JSON.stringify({

                    question: question,

                    top_k: 5

                })

            });


        let data;

        try {

            data = await response.json();

        } catch {

            throw new Error(
                `Server returned HTTP ${response.status}.`
            );

        }


        if (!response.ok) {

            throw new Error(
                data.detail ||
                `Request failed with HTTP ${response.status}.`
            );

        }


        renderResponse(data);


    } catch (error) {

        showError(
            error.message ||
            "Something went wrong while processing the request."
        );

    } finally {

        setLoading(false);

    }

}


askButton.addEventListener(
    "click",
    askQuestion
);


questionInput.addEventListener(
    "keydown",
    (event) => {

        if (
            event.key === "Enter" &&
            (event.metaKey || event.ctrlKey)
        ) {

            event.preventDefault();

            askQuestion();

        }

    }
);