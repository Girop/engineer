document.getElementById("article-req").addEventListener("submit", async function (event) {
    event.preventDefault();
    const arxivId = document.getElementById("arxiv-id").value;
    const method = document.getElementById("method").value;


    const server_address = "http://127.0.0.1:8000";

    try {
        const response = await fetch(`${server_address}/recommend/id/${arxivId}/${method}`, {
            method: "GET"
        });

        if (!response.ok) {
            throw new Error("Failed to fetch recommendations.");
        }

        const data = await response.json();

        const resultsTable = document.getElementById("results-table");
        const tbody = resultsTable.querySelector("tbody");

        // Clear any previous results
        tbody.innerHTML = "";

        if (data["error"] != "None") {
            throw new Error("Request failed: " + data["error"]);
        }
         
        data["result"].forEach((item) => {
            const row = document.createElement("tr");
            row.innerHTML = `
                        <td>${item.arxiv_id}</td>
                        <td>${item.similarity_score.toFixed(2)}</td>
                        <td>${item.title}</td>
                        <td>${item.date}</td>
                    `;
            tbody.appendChild(row);
        });

        resultsTable.style.display = "table";
    } catch (error) {
        console.error("Error:", error);
        alert("Failed to fetch recommendations. Please try again later.");
    }
});
