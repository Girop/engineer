function createArxivUrl(arxiv_id) {
    return `https://arxiv.org/pdf/${arxiv_id}`
}


async function fillTable(table, arxivId, mode) {
    const url = new URL( "http://127.0.0.1:8000/recommend/");
    url.searchParams.append('document_id', arxivId);
    url.searchParams.append('mode', mode)
    url.searchParams.append('limit', 10);
    console.log("Request sent to: ", url.toString());

    const response = await fetch(url.toString(), { method: "GET" });

    if (!response.ok) {
        throw new Error("Failed to fetch recommendations.");
    }

    const data = await response.json();

    const tbody = table.querySelector("tbody");
    // Clear any previous results
    tbody.innerHTML = "";

    if (data["error"] != "None") {
        throw new Error("Request failed: " + data["error"]);
    }
     
    data["result"].forEach((item) => {
        const row = document.createElement("tr");
        row.innerHTML = `
            <td><a href=${createArxivUrl(item.arxiv_id)}>${item.arxiv_id}</a></td>
            <td>${item.similarity_score.toFixed(2)}</td>
            <td>${item.title}</td>
            <td>${item.authors}</td>
            <td>${item.category}</td>
            <td>${item.date}</td>
        `;
        tbody.appendChild(row);
    });
}


function update_table(tableName, arxivId, method) {
    const resultsTable = document.getElementById(tableName);
    resultsTable.style.display = "none";
    fillTable(resultsTable, arxivId, method);
    resultsTable.style.display = "table";
}


document.getElementById("article-req").addEventListener("submit", async function (event) {
    event.preventDefault();
    const arxivId = document.getElementById("arxiv-id").value;
    
    try {
        update_table("bert-table", arxivId, 1);
        update_table("glove-table", arxivId, 2);
        update_table("tf-table", arxivId, 3);

    } catch (error) {
        console.error("Error:", error);
        alert("Failed to fetch recommendations. Please try again later.");
    }
});

