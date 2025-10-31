# 🚀 Gemini MLOps Benchmarks: Cost, Carbon & Performance Simulation

This repository contains a comprehensive MLOps project for benchmarking the **Gemini 2.5 Flash** large language model. It goes beyond standard performance metrics (like latency and tokens/sec) to include detailed **financial cost** and **simulated environmental impact (carbon footprint)**.

The entire system runs from a single Colab notebook, launching a public-facing FastAPI server, performing observability with Langfuse, and simulating deployments across different GPU architectures and cloud locations.

### Demo: Final Efficiency Dashboard
![Efficiency Comparison Dashboard](https://i.imgur.com/example.png) ---

## 💡 Key Features

* **🚀 API-Driven Benchmarking:** Deploys a **FastAPI** server (`app.py`) on-the-fly, serving a `/chat` and `/batch` endpoint.
* **🌐 Public Tunneling:** Uses **pyngrok** to instantly create a public URL for the local Colab server, allowing it to be accessed like a production API.
* **🔄 Smart Prompt Routing:** The `smart_router` intelligently categorizes user prompts (e.g., `Precise`, `Coder`, `Creative`, `Analyst`) to apply **optimal temperature settings** (from $0.1$ to $0.9$) for the task, all while using the same `gemini-2.5-flash` model.
* **📊 Full-Stack Observability:** Integrates **LiteLLM** as a universal LLM interface and automatically traces every single API call (prompt, response, latency, tokens) to **Langfuse** for deep monitoring.
* **🌍 Sustainability Simulation:** This is the core feature. The benchmark *simulates* running on three different GPU/Datacenter profiles (`T4_COLAB`, `A100_SIMULATED`, `L40S_SIMULATED`) by calculating the **Total Energy ($\text{kWh}$)** and **Carbon Footprint ($\text{gCO2e}$)** based on hardware TDP, datacenter PUE, and regional Grid Carbon Intensity (CI).
* **💰 Total Cost of Ownership (TCO):** The final reports combine the direct **API Cost** (from `calculate_cost`) with the simulated **Energy Cost** to give a complete picture of deployment expenses.
* **🎯 Accuracy Checking:** Includes a `check_accuracy` function to validate responses for known factual prompts, providing a $100.0\%$ accuracy score in the final test run.
* **📈 Rich Data Analysis:** Automatically generates a detailed **CSV file** and **six interactive Plotly dashboards** (saved as HTML files) analyzing every aspect of the benchmark run.

---

## ⚙️ How It Works: Architecture & Data Flow

This project follows a complete MLOps pipeline from request to analysis:

1.  **Setup (Cells 1-2):** Installs all dependencies (`fastapi`, `litellm`, `langfuse`, `pandas`, `plotly`, etc.) and collects API keys for Gemini, ngrok, and Langfuse.
2.  **Server Launch (Cells 4-5):**
    * The `app.py` script (containing all API logic) is written to the local filesystem.
    * `uvicorn` starts the FastAPI server as a background process.
    * `pyngrok` connects to the local port $8000$ and generates a public `SERVER_URL`.
3.  **Benchmark Initialization (Cells 7-9):**
    * The 3 `GPU_PROFILES` are defined, containing the sustainability metrics (TDP, CI, PUE) for the T4, A100, and L40S.
    * The `BatchProcessor` class is initialized with the `SERVER_URL`.
    * A list of 12 diverse `TEST_PROMPTS` is loaded.
4.  **Batch Execution (Cell 10):**
    * The `BatchProcessor` loops through each of the 3 GPU Profiles.
    * For each profile, it sends all 12 test prompts (one by one) to the public `SERVER_URL`.
    * The FastAPI server receives the request at the `/chat` endpoint.
    * The `smart_router` function analyzes the prompt, assigns a name (e.g., `Flash-Coder`), and sets a temperature.
    * `litellm` makes the call to the `gemini-2.5-flash` API.
    * `langfuse` (via `litellm.callbacks`) logs the entire trace in the background.
    * The API returns a JSON response with latency, token counts, and API cost.
    * The `BatchProcessor` receives this response and calls `calculate_enhanced_metrics` to *add* the simulated energy ($\text{kWh}$) and carbon ($\text{gCO2e}$) costs based on the *current* GPU profile being tested.
    * A 5-second `time.sleep(5.0)` is added to manage API rate limits (though $9$ requests still failed due to 429 errors in the test run).
5.  **Data Analysis & Reporting (Cells 11-19):**
    * All $27$ successful results are saved to a comprehensive CSV file.
    * A series of Plotly dashboards are generated, analyzing results by GPU, model route, and location.
    * All dashboards are saved as interactive `.html` files.
    * A final, text-based comprehensive report is printed to the console.

---

## 📊 Final Benchmark Results & Key Insights

The final analysis is based on 27 successful requests run across the 3 simulated environments.

### 🏆 Efficiency Rankings (per 1,000 Tokens)

This table shows the most efficient GPU environment for running the Gemini Flash model, combining both API and simulated energy costs.

| Rank | Metric | NVIDIA Tesla T4 | NVIDIA L40S | NVIDIA A100 80GB |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **💰 Total Cost / 1K Tokens** | **\$0.000081** | \$0.000113 | \$0.000110 |
| **1** | **🌱 Carbon / 1K Tokens ($\text{gCO2e}$)** | **$0.032371$** | $0.090464$ | $0.126973$ |

**Conclusion:** The **NVIDIA Tesla T4** was overwhelmingly the most cost-effective and carbon-efficient environment. This is because its low 70W TDP resulted in significantly lower simulated energy consumption, which had a greater impact than the datacenter's high carbon intensity (CI) grid.

### ⚡ Key Insight: Idle Energy vs. Compute Energy

A critical finding was the ratio of energy consumed by the model (`Compute`) versus the energy consumed by the server waiting for requests (`Idle`).

* **Idle Energy:** $0.001383 \text{ kWh}$ ($\mathbf{83.7\%}$ of total)
* **Compute Energy:** $0.000048 \text{ kWh}$ ($\mathbf{2.9\%}$ of total)

**Conclusion:** For fast models like Gemini Flash, the vast majority of energy is wasted by the server *idling* between requests. This proves that **efficient request batching** and **scaling to zero** are the most important optimizations for sustainable MLOps.

### ⏱️ Performance by Smart Router Task

The `smart_router` successfully applied different temperature settings, and the average latency varied by the assigned task type:

| Model Route | Requests | Avg. Latency (sec) |
| :--- | :--- | :--- |
| `Flash-Analyst` | 5 | **$1.124\text{s}$ (Fastest)** |
| `Flash-Balanced`| 3 | $1.321\text{s}$ |
| `Flash-Precise` | 15 | $2.150\text{s}$ |
| `Flash-Coder` | 4 | **$4.184\text{s}$ (Slowest)** |

---

## 🚀 How to Run

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    ```
2.  **Open in Google Colab:**
    Upload and open the `final_aidevops (1).ipynb` notebook.
3.  **Install Dependencies:**
    Run the first code cell (Cell 1) to install all required `pip` packages.
4.  **Add API Keys:**
    Run the second code cell (Cell 2) and enter your secret keys when prompted:
    * `GEMINI_API_KEY`
    * `GROQ_API_KEY` (Note: This is not used in the final script)
    * `NGROK_AUTHTOKEN`
    * `LANGFUSE_PUBLIC_KEY` & `LANGFUSE_SECRET_KEY`
5.  **Run All Cells:**
    From the "Runtime" menu, select "Run all". The notebook will:
    * Start the API server.
    * Run all 36 benchmark tests (which will take several minutes due to the $5.0\text{s}$ sleep).
    * Save the `ai_metrics_*.csv` file.
    * Display all Plotly dashboards and save them as `.html` files.
    * Print the final comprehensive report.

## 📦 Generated Outputs

After a successful run, you will find the following files in your Colab environment's file explorer:

* **`ai_metrics_[timestamp].csv`**: The raw data from all 27 successful benchmark runs.
* `gpu_comparison_dashboard.html`: Plotly chart comparing GPUs on Carbon, Cost, Latency, and Energy.
* `model_performance_analysis.html`: Plotly chart comparing the `smart_router` models on cost, throughput, and accuracy.
* `carbon_energy_analysis.html`: Plotly chart analyzing carbon by location and the idle vs. compute energy breakdown.
* `time_series_analysis.html`: Plotly chart showing performance over the time of the batch run.
* `advanced_hardware_metrics.html`: Plotly chart analyzing TDP vs. Carbon and Memory vs. Performance.
* `efficiency_comparison.html`: The summary dashboard showing the final cost and carbon efficiency rankings.
