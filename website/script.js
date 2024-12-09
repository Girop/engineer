function createArxivUrl(arxiv_id) {
    return `https://arxiv.org/pdf/${arxiv_id}`
}


function getActiveTarget() {
    const button = document.querySelector(".input-button.active");
    const target = button.getAttribute('data-target')
    if (target === "arxiv-input") {
        return 3;
    }
    if (target === "text-input") {
        return 2;
    }
    return 1;
}


function getPayload(target) {
    if (target === 1) {
        return document.getElementById("pdf-file").files[0];
    }

    if (target === 2) {
        return document.getElementById("text-sample").value;
    }

    if (target === 3) {
        return document.getElementById("arxiv-id").value;
    }
}

function getAddress(method) {
    if (method === 1) {
        return "http://127.0.0.1:8000/recommend/pdf/";
    }
    return "http://127.0.0.1:8000/recommend/"
}

async function fillTable(table, mode) {
    const tbody = table.querySelector("tbody");
    tbody.innerHTML = "";
    const activeTarget = getActiveTarget();
    const payload = getPayload(activeTarget);

    form = new FormData();
    form.append("payload", payload);
    form.append('limit', 10);
    form.append('req_type', activeTarget);
    form.append("mode", mode);

    const address = getAddress(activeTarget);
    const response = await fetch(address, {
        method: "POST",
        body: form
    });

    if (!response.ok) {
        throw new Error("Failed to fetch recommendations.");
    }

    const data = await response.json();

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


function update_table(tableName, method) {
    const resultsTable = document.getElementById(tableName);
    resultsTable.style.display = "none";
    fillTable(resultsTable, method);
    resultsTable.style.display = "table";
}


const rating = document.getElementById("rating");

document.getElementById("article-req").addEventListener("submit", async function (event) {
    event.preventDefault();
    
    try {
        const updates = [
            await update_table("tf-table", 3),
            await update_table("glove-table", 2),
            await update_table("bert-table", 1),
        ];
        await Promise.all(updates);
        rating.style.display = "block";
    } catch (error) {
        console.error("Error:", error);
        alert("Failed to fetch recommendations. Please try again later.");
    }
});


const input_buttons = document.querySelectorAll(".input-button");
const input_fields = document.querySelectorAll(".input-field");

input_buttons.forEach((button) => {
    button.addEventListener("click", () => {
        input_buttons.forEach((btn) => btn.classList.remove("active"));

        button.classList.add("active");

        input_fields.forEach((field) => {
            field.style.display = "none";
        });

        const target = button.getAttribute("data-target");
        document.getElementById(target).style.display = "block";
    });
});


const selectOptions = document.querySelectorAll(".rating-select");

selectOptions.forEach((elem) => {
    elem.addEventListener("change", () => {
        handleMutualExclusion();
    });
});

function handleMutualExclusion() {
    const selectedValues = Array.from(selectOptions)
        .map(select => select.value)
        .filter(value => value !== "");

    selectOptions.forEach(select => {
        const currentValue = select.value;

        Array.from(select.options).forEach(option => {
            if (selectedValues.includes(option.value) && option.value !== currentValue) {
                option.style.display = "none";
            } else {
                option.style.display = "";
            }
        });
    });
}


rating.addEventListener("submit", async (event) => {
    event.preventDefault();
    const rankings = {"bert": null, "glove": null, "tf": null};

    const shouldSend = true;

    selectOptions.forEach((select, index) => {
        const name = ["glove", "bert", "tf"]


        if (select.value == "" || select.value == null) {
            shouldSend = false;
            return;
        }

        const key = name[index];
        rankings[key] = parseInt(select.value, 10);
        select.value = "";
    });

    if (!shouldSend) {
        alert("Feedback submitted");
        return;
    }


    const result = JSON.stringify(rankings)
    console.log("Submitted Rankings:", result);
    const address = "http://127.0.0.1:8000/rating/";

    await fetch(address, {
        'method': "Post",
        "headers": {
            'Content-Type': 'application/json',
        },
        body: result,
    });

    alert("Thanks for feedback!");
    rating.style.display = "none";
});
