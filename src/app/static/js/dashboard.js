document.addEventListener("DOMContentLoaded", () => {
    // --- DOM Elements ---
    const timestampSelect = document.getElementById("historical-timestamp");
    const btnReset = document.getElementById("btn-reset-defaults");
    const btnSubmit = document.getElementById("btn-submit-simulation");
    const sandboxForm = document.getElementById("sandbox-form");
    
    // Form Inputs
    const inputAirportTemp = document.getElementById("airport_temp");
    const inputAirportDew = document.getElementById("airport_dew");
    const inputWindSpeed = document.getElementById("airport_wind_speed");
    const inputWindDir = document.getElementById("airport_wind_dir");
    const inputUrbanTemp = document.getElementById("urban_temp");
    const inputRuralTemp = document.getElementById("rural_temp");
    const inputAod500 = document.getElementById("AOD_500nm");
    const inputAod440 = document.getElementById("AOD_440nm");
    
    // Re-engineered display panels
    const displayRh = document.getElementById("val-rh");
    const displayDpd = document.getElementById("val-dpd");
    const displayWsi = document.getElementById("val-wsi");
    const displayAsep = document.getElementById("val-asep");
    
    // Z3 Audit Elements
    const z3AuditCard = document.getElementById("z3-audit-card");
    const auditBadge = document.getElementById("audit-badge");
    const auditStatusLarge = document.getElementById("audit-status-large");
    const auditDescription = document.getElementById("audit-description");
    const violationsContainer = document.getElementById("violations-list-container");
    const violationsList = document.getElementById("violations-list");
    
    // Global Header status indicators
    const systemStatusIndicator = document.getElementById("system-status-indicator");
    
    // Metrics table
    const metricsTableBody = document.querySelector("#metrics-data-table tbody");
    
    // S3 & Mongo status pill overrides
    const mongoStatusText = document.getElementById("mongo-status");
    const s3StatusText = document.getElementById("s3-status");

    // Default cached values for resetting
    let historicalDefaults = null;
    let forecastChart = null;

    // --- Chart.js Configuration ---
    const ctx = document.getElementById("forecastTimelineChart").getContext("2d");
    
    function initChart(rfData, xgbData, gruData, z3Data) {
        if (forecastChart) {
            forecastChart.destroy();
        }
        
        forecastChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: ['t+1h', 't+2h', 't+3h', 't+4h', 't+5h', 't+6h'],
                datasets: [
                    {
                        label: 'Random Forest',
                        data: rfData,
                        borderColor: '#7f8c8d',
                        backgroundColor: 'rgba(127, 140, 141, 0.1)',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        pointStyle: 'circle',
                        pointRadius: 6,
                        tension: 0.15
                    },
                    {
                        label: 'XGBoost',
                        data: xgbData,
                        borderColor: '#e67e22',
                        backgroundColor: 'rgba(230, 126, 34, 0.1)',
                        borderWidth: 2,
                        pointStyle: 'triangle',
                        pointRadius: 6,
                        tension: 0.15
                    },
                    {
                        label: 'Deep GRU',
                        data: gruData,
                        borderColor: '#9b59b6',
                        backgroundColor: 'rgba(155, 89, 182, 0.1)',
                        borderWidth: 2,
                        pointStyle: 'rect',
                        pointRadius: 6,
                        tension: 0.15
                    },
                    {
                        label: 'Z3-Verified',
                        data: z3Data,
                        borderColor: '#27ae60',
                        backgroundColor: 'rgba(39, 174, 96, 0.15)',
                        borderWidth: 3.5,
                        pointStyle: 'rectRot',
                        pointRadius: 7,
                        pointHoverRadius: 9,
                        tension: 0.15,
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false // We use our custom legend in HTML
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        backgroundColor: 'rgba(15, 16, 22, 0.95)',
                        titleFont: { family: 'Outfit', size: 13 },
                        bodyFont: { family: 'Inter', size: 12 },
                        borderColor: 'rgba(255, 255, 255, 0.1)',
                        borderWidth: 1,
                        callbacks: {
                            label: function(context) {
                                return `${context.dataset.label}: ${context.raw} meters`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.04)'
                        },
                        ticks: {
                            color: '#94a3b8',
                            font: { family: 'Outfit', size: 12 }
                        }
                    },
                    y: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.04)'
                        },
                        ticks: {
                            color: '#94a3b8',
                            font: { family: 'Outfit', size: 12 }
                        },
                        title: {
                            display: true,
                            text: 'Runway Visibility Range (meters)',
                            color: '#94a3b8',
                            font: { family: 'Outfit', size: 12, weight: 600 }
                        }
                    }
                }
            }
        });
    }

    // --- Dynamic Derived Parameter Calculation (Local Preview) ---
    function updateDerivedCalculations() {
        const temp = parseFloat(inputAirportTemp.value);
        const dew = parseFloat(inputAirportDew.value);
        const speed = parseFloat(inputWindSpeed.value);
        const dir = parseFloat(inputWindDir.value);
        const aod500 = parseFloat(inputAod500.value);
        const aod440 = parseFloat(inputAod440.value);

        if (isNaN(temp) || isNaN(dew)) return;

        // 1. Dew Point Depression (DPD)
        const dpd = Math.max(temp - dew, 0.0);
        displayDpd.textContent = `${dpd.toFixed(1)}°C`;

        // 2. Relative Humidity (RH) - Magnus-Tetens
        const rh = 100.0 * Math.exp((17.625 * dew) / (243.04 + dew) - (17.625 * temp) / (243.04 + temp));
        const rhClamped = Math.min(Math.max(rh, 0.0), 100.0);
        displayRh.textContent = `${rhClamped.toFixed(1)}%`;

        // 3. Wind Stagnation Index (WSI)
        if (!isNaN(speed)) {
            const wsi = speed < 1.5 ? 1 : 0;
            displayWsi.textContent = wsi;
        }

        // 4. Aerosol Scattering Extinction Proxy (ASEP)
        if (!isNaN(aod500) && !isNaN(aod440) && aod440 !== 0) {
            const asep = aod500 / aod440;
            displayAsep.textContent = asep.toFixed(3);
        }
    }

    // Bind event listeners for input fields to update local previews immediately
    [inputAirportTemp, inputAirportDew, inputWindSpeed, inputWindDir, inputAod500, inputAod440].forEach(input => {
        input.addEventListener("input", updateDerivedCalculations);
    });

    // --- Fetch list of Timestamps ---
    async function fetchTimestamps() {
        try {
            const res = await fetch("/api/timestamps?limit=150");
            const data = await res.json();
            
            if (data.timestamps && data.timestamps.length > 0) {
                timestampSelect.innerHTML = "";
                data.timestamps.forEach(ts => {
                    const opt = document.createElement("option");
                    opt.value = ts;
                    opt.textContent = ts;
                    timestampSelect.appendChild(opt);
                });
                
                // Select first, load defaults and predict
                const firstTs = data.timestamps[0];
                await loadDefaults(firstTs);
            } else {
                timestampSelect.innerHTML = "<option value=''>Error: No timestamps generated</option>";
            }
        } catch (err) {
            console.error("Error fetching timestamps:", err);
            timestampSelect.innerHTML = "<option value=''>Error connecting to server</option>";
        }
    }

    // --- Load Default Values for Selected Timestamp ---
    async function loadDefaults(timestamp) {
        try {
            const res = await fetch(`/api/sequence?timestamp=${encodeURIComponent(timestamp)}`);
            const data = await res.json();
            
            if (data.defaults) {
                historicalDefaults = data.defaults;
                applyDefaultsToInputs(data.defaults);
                updateDerivedCalculations();
                
                // Trigger prediction immediately for initial view
                triggerPrediction(timestamp, null);
            }
        } catch (err) {
            console.error("Error loading sequence defaults:", err);
        }
    }

    function applyDefaultsToInputs(defaults) {
        inputAirportTemp.value = defaults.airport_temp;
        inputAirportDew.value = defaults.airport_dew;
        inputWindSpeed.value = defaults.airport_wind_speed;
        inputWindDir.value = defaults.airport_wind_dir;
        inputUrbanTemp.value = defaults.urban_temp;
        inputRuralTemp.value = defaults.rural_temp;
        inputAod500.value = defaults.AOD_500nm;
        inputAod440.value = defaults.AOD_440nm;
    }

    // Dropdown change listener
    timestampSelect.addEventListener("change", (e) => {
        loadDefaults(e.target.value);
    });

    // Reset button listener
    btnReset.addEventListener("click", () => {
        if (historicalDefaults) {
            applyDefaultsToInputs(historicalDefaults);
            updateDerivedCalculations();
        }
    });

    // --- Trigger Prediction / Verification ---
    async function triggerPrediction(timestamp, overrides) {
        btnSubmit.disabled = true;
        btnSubmit.innerHTML = `<span>Computing...</span> <i class="fa-solid fa-circle-notch fa-spin"></i>`;
        
        try {
            const res = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    timestamp: timestamp,
                    overrides: overrides
                })
            });
            const data = await res.json();
            
            if (data.predictions) {
                // Renders Timeline Chart
                initChart(
                    data.predictions.random_forest,
                    data.predictions.xgboost,
                    data.predictions.deep_gru,
                    data.predictions.z3_verified
                );

                // Update Derived Metric Panel (with exact server values)
                const derived = data.meteorological_conditions;
                displayRh.textContent = `${derived.relative_humidity}%`;
                displayDpd.textContent = `${derived.dew_point_depression}°C`;
                displayWsi.textContent = derived.wind_stagnation_index;
                displayAsep.textContent = derived.aerosol_optical_depth;

                // Update S3 / MongoDB telemetry Status indicators
                // Since this call completes, we verify if they logged successfully
                // In local mode, logs go to JSONL so the server sends it back
                // We keep defaults but can update if required

                // Render Z3 Guardrail Audit Dashboard
                updateAuditPanel(data.z3_audit);
            } else if (data.error) {
                alert(`Error: ${data.error}`);
            }
        } catch (err) {
            console.error("Error executing prediction API:", err);
            alert("Failed to communicate with prediction server.");
        } finally {
            btnSubmit.disabled = false;
            btnSubmit.innerHTML = `<span>Verify &amp; Forecast Runway Visibility</span> <i class="fa-solid fa-arrow-right"></i>`;
        }
    }

    function updateAuditPanel(audit) {
        // Clear violations list
        violationsList.innerHTML = "";
        
        if (audit.status === "SAT") {
            // SAT state CSS classes
            z3AuditCard.className = "glass-card status-display-card sat-state";
            auditBadge.textContent = "SAT";
            auditStatusLarge.textContent = "SATISFIED";
            auditDescription.textContent = "Predictions comply with all deterministic meteorological physical axioms. No corrections applied.";
            violationsContainer.classList.add("hidden");
            
            // Header system pill
            systemStatusIndicator.className = "status-pill sat";
            systemStatusIndicator.innerHTML = `<span class="dot"></span> System Ready`;
        } else {
            // UNSAT state CSS classes
            z3AuditCard.className = "glass-card status-display-card unsat-state";
            auditBadge.textContent = "UNSAT";
            auditStatusLarge.textContent = "CORRECTED";
            auditDescription.textContent = "Meteorological violations detected in raw models. Real-time safety filters applied physics-safe clamping limits.";
            
            // Populate violations list
            violationsContainer.classList.remove("hidden");
            audit.violated_rules.forEach(rule => {
                const li = document.createElement("li");
                li.textContent = rule;
                violationsList.appendChild(li);
            });

            // Header system pill
            systemStatusIndicator.className = "status-pill unsat";
            systemStatusIndicator.innerHTML = `<span class="dot"></span> Active Clamping`;
        }
    }

    // Form Submission Override handler
    sandboxForm.addEventListener("submit", (e) => {
        e.preventDefault();
        const selectedTimestamp = timestampSelect.value;
        if (!selectedTimestamp) return;

        // Build overrides payload
        const overrides = {
            airport_temp: parseFloat(inputAirportTemp.value),
            airport_dew: parseFloat(inputAirportDew.value),
            airport_wind_speed: parseFloat(inputWindSpeed.value),
            airport_wind_dir: parseFloat(inputWindDir.value),
            urban_temp: parseFloat(inputUrbanTemp.value),
            rural_temp: parseFloat(inputRuralTemp.value),
            AOD_500nm: parseFloat(inputAod500.value),
            AOD_440nm: parseFloat(inputAod440.value)
        };

        triggerPrediction(selectedTimestamp, overrides);
    });

    // --- Load Research Calibration Metrics Table ---
    async function loadMetrics() {
        try {
            const res = await fetch("/api/metrics");
            const data = await res.json();
            
            if (data.metrics && data.metrics.length > 0) {
                // Clear initial message
                metricsTableBody.innerHTML = "";
                
                // Group by horizon/threshold to render cleanly or just loop
                data.metrics.forEach(row => {
                    const tr = document.createElement("tr");
                    
                    // Style the Verified row to stand out
                    if (row.Model === "Z3-Verified") {
                        tr.style.background = "rgba(39, 174, 96, 0.04)";
                        tr.style.fontWeight = "600";
                    }

                    tr.innerHTML = `
                        <td><strong>${row.Horizon}</strong></td>
                        <td>&lt; ${row.Threshold}m</td>
                        <td style="color: ${row.Model === "Z3-Verified" ? "#27ae60" : row.Model === "Deep GRU" ? "#9b59b6" : row.Model === "XGBoost" ? "#e67e22" : "#7f8c8d"}">${row.Model}</td>
                        <td>${row.Precision.toFixed(4)}</td>
                        <td>${row["Recall (POD)"].toFixed(4)}</td>
                        <td>${row["False Alarm Ratio (FAR)"].toFixed(4)}</td>
                        <td>${row["False Alarm Rate (FPR)"].toFixed(4)}</td>
                        <td>${row["F1-Score"].toFixed(4)}</td>
                        <td>${row["Brier Score"].toFixed(4)}</td>
                        <td style="font-family: monospace; font-size: 0.75rem; color: #94a3b8">
                            ${row.TP}/${row.FP}/${row.FN}/${row.TN}
                        </td>
                    `;
                    metricsTableBody.appendChild(tr);
                });
            } else {
                metricsTableBody.innerHTML = `<tr><td colspan="10" class="text-center">⚠️ No metrics logged. Complete the training pipeline first.</td></tr>`;
            }
        } catch (err) {
            console.error("Error fetching calibration metrics:", err);
            metricsTableBody.innerHTML = `<tr><td colspan="10" class="text-center">❌ Failed to retrieve evaluation statistics.</td></tr>`;
        }
    }

    // --- Bootstrapping Execution ---
    fetchTimestamps();
    loadMetrics();
});
