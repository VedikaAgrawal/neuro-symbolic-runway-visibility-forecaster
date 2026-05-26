document.addEventListener("DOMContentLoaded", () => {
    // --- DOM Elements ---
    const timestampSelect = document.getElementById("historical-timestamp");
    const btnReset = document.getElementById("btn-reset-defaults");
    const btnSubmit = document.getElementById("btn-submit-simulation");
    const sandboxForm = document.getElementById("sandbox-form");
    
    // Numeric Form Inputs
    const inputAirportTemp = document.getElementById("airport_temp");
    const inputAirportDew = document.getElementById("airport_dew");
    const inputWindSpeed = document.getElementById("airport_wind_speed");
    const inputWindDir = document.getElementById("airport_wind_dir");
    const inputUrbanTemp = document.getElementById("urban_temp");
    const inputRuralTemp = document.getElementById("rural_temp");
    const inputAod500 = document.getElementById("AOD_500nm");
    const inputAod440 = document.getElementById("AOD_440nm");
    
    // Bidirectional Slider Elements
    const syncInputs = [
        { number: "airport_temp", range: "slider_airport_temp" },
        { number: "airport_dew", range: "slider_airport_dew" },
        { number: "airport_wind_speed", range: "slider_airport_wind_speed" },
        { number: "airport_wind_dir", range: "slider_airport_wind_dir" },
        { number: "urban_temp", range: "slider_urban_temp" },
        { number: "rural_temp", range: "slider_rural_temp" },
        { number: "AOD_500nm", range: "slider_AOD_500nm" },
        { number: "AOD_440nm", range: "slider_AOD_440nm" }
    ];

    // Re-engineered display panels & progress bars
    const displayRh = document.getElementById("val-rh");
    const displayDpd = document.getElementById("val-dpd");
    const displayWsi = document.getElementById("val-wsi");
    const displayAsep = document.getElementById("val-asep");
    
    const gaugeRh = document.getElementById("gauge-rh");
    const gaugeDpd = document.getElementById("gauge-dpd");
    const gaugeAsep = document.getElementById("gauge-asep");
    const wsiLampBox = document.getElementById("wsi-lamp-box");
    
    // Z3 Audit Elements
    const z3AuditCard = document.getElementById("z3-audit-card");
    const auditBadge = document.getElementById("audit-badge");
    const auditStatusLarge = document.getElementById("audit-status-large");
    const auditDescription = document.getElementById("audit-description");
    const violationsContainer = document.getElementById("violations-list-container");
    const violationsList = document.getElementById("violations-list");
    
    const clampingContainer = document.getElementById("clamping-diffs-container");
    const clampingList = document.getElementById("clamping-list");
    
    // Global Header status indicators
    const systemStatusIndicator = document.getElementById("system-status-indicator");
    
    // Metrics table
    const metricsTableBody = document.querySelector("#metrics-data-table tbody");
    
    // Metrics interactive chart elements
    const thresholdFilter = document.getElementById("filter-threshold");
    const horizonFilter = document.getElementById("filter-horizon");

    // Global caches
    let historicalDefaults = null;
    let forecastChart = null;
    let metricsChart = null;
    let globalMetricsData = [];

    // --- Bidirectional Input-Slider Synchronization ---
    syncInputs.forEach(pair => {
        const numEl = document.getElementById(pair.number);
        const rngEl = document.getElementById(pair.range);
        if (numEl && rngEl) {
            numEl.addEventListener("input", () => {
                rngEl.value = numEl.value;
                updateDerivedCalculations();
            });
            rngEl.addEventListener("input", () => {
                numEl.value = rngEl.value;
                updateDerivedCalculations();
            });
        }
    });

    // --- Chart.js Configuration for Forecast Timeline ---
    const ctxForecast = document.getElementById("forecastTimelineChart").getContext("2d");
    
    function initForecastChart(rfData, xgbData, gruData, z3Data) {
        if (forecastChart) {
            forecastChart.destroy();
        }
        
        forecastChart = new Chart(ctxForecast, {
            type: 'line',
            data: {
                labels: ['t+1h', 't+2h', 't+3h', 't+4h', 't+5h', 't+6h'],
                datasets: [
                    {
                        label: 'Random Forest (Raw)',
                        data: rfData,
                        borderColor: '#94a3b8',
                        backgroundColor: 'rgba(148, 163, 184, 0.05)',
                        borderWidth: 2,
                        borderDash: [4, 4],
                        pointStyle: 'circle',
                        pointRadius: 5,
                        tension: 0.15
                    },
                    {
                        label: 'XGBoost',
                        data: xgbData,
                        borderColor: '#f59e0b',
                        backgroundColor: 'rgba(245, 158, 11, 0.05)',
                        borderWidth: 2,
                        pointStyle: 'triangle',
                        pointRadius: 5,
                        tension: 0.15
                    },
                    {
                        label: 'Deep GRU',
                        data: gruData,
                        borderColor: '#ec4899',
                        backgroundColor: 'rgba(236, 72, 153, 0.05)',
                        borderWidth: 2,
                        pointStyle: 'rect',
                        pointRadius: 5,
                        tension: 0.15
                    },
                    {
                        label: 'Z3-Verified',
                        data: z3Data,
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.12)',
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
                        display: false // Using custom legends in index.html
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        backgroundColor: 'rgba(11, 13, 22, 0.95)',
                        titleFont: { family: 'Outfit', size: 13, weight: 600 },
                        bodyFont: { family: 'Inter', size: 12 },
                        borderColor: 'rgba(255, 255, 255, 0.08)',
                        borderWidth: 1,
                        callbacks: {
                            label: function(context) {
                                return ` ${context.dataset.label}: ${context.raw} m`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.03)'
                        },
                        ticks: {
                            color: '#94a3b8',
                            font: { family: 'Outfit', size: 12 }
                        }
                    },
                    y: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.03)'
                        },
                        ticks: {
                            color: '#94a3b8',
                            font: { family: 'Outfit', size: 12 }
                        },
                        title: {
                            display: true,
                            text: 'Visibility Range (meters)',
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
        const aod500 = parseFloat(inputAod500.value);
        const aod440 = parseFloat(inputAod440.value);

        if (isNaN(temp) || isNaN(dew)) return;

        // 1. Dew Point Depression (DPD)
        const dpd = Math.max(temp - dew, 0.0);
        displayDpd.textContent = `${dpd.toFixed(1)}°C`;
        if (gaugeDpd) {
            // scale 0 to 15°C
            const pct = Math.min((dpd / 15) * 100, 100);
            gaugeDpd.style.width = `${pct}%`;
        }

        // 2. Relative Humidity (RH) - Magnus-Tetens
        const rh = 100.0 * Math.exp((17.625 * dew) / (243.04 + dew) - (17.625 * temp) / (243.04 + temp));
        const rhClamped = Math.min(Math.max(rh, 0.0), 100.0);
        displayRh.textContent = `${rhClamped.toFixed(1)}%`;
        if (gaugeRh) {
            gaugeRh.style.width = `${rhClamped}%`;
        }

        // 3. Wind Stagnation Index (WSI)
        if (!isNaN(speed)) {
            const wsiActive = speed < 1.5;
            if (wsiLampBox && displayWsi) {
                if (wsiActive) {
                    wsiLampBox.className = "wsi-status-box stagnant-active";
                    displayWsi.textContent = "STAGNANT";
                } else {
                    wsiLampBox.className = "wsi-status-box stagnant-inactive";
                    displayWsi.textContent = "VENTILATED";
                }
            }
        }

        // 4. Aerosol Scattering Extinction Proxy (ASEP)
        if (!isNaN(aod500) && !isNaN(aod440) && aod440 !== 0) {
            const asep = aod500 / aod440;
            displayAsep.textContent = asep.toFixed(3);
            if (gaugeAsep) {
                // scale 0.5 to 2.0
                const pct = Math.min(Math.max(((asep - 0.5) / 1.5) * 100, 0), 100);
                gaugeAsep.style.width = `${pct}%`;
            }
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

        // Apply to sliders too
        syncInputs.forEach(pair => {
            const numVal = document.getElementById(pair.number);
            const rngEl = document.getElementById(pair.range);
            if (numVal && rngEl) {
                rngEl.value = numVal.value;
            }
        });
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
        btnSubmit.innerHTML = `<span>Simulating...</span> <i class="fa-solid fa-circle-notch fa-spin"></i>`;
        
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
                initForecastChart(
                    data.predictions.random_forest,
                    data.predictions.xgboost,
                    data.predictions.deep_gru,
                    data.predictions.z3_verified
                );

                // Update Derived Metric Panel (with exact server values)
                const derived = data.meteorological_conditions;
                displayRh.textContent = `${derived.relative_humidity}%`;
                displayDpd.textContent = `${derived.dew_point_depression}°C`;
                displayAsep.textContent = derived.aerosol_optical_depth;

                // Sync derived progress bars
                if (gaugeRh) gaugeRh.style.width = `${derived.relative_humidity}%`;
                if (gaugeDpd) gaugeDpd.style.width = `${Math.min((derived.dew_point_depression / 15) * 100, 100)}%`;
                if (gaugeAsep) gaugeAsep.style.width = `${Math.min((derived.aerosol_optical_depth / 5) * 100, 100)}%`;

                if (wsiLampBox && displayWsi) {
                    if (derived.wind_stagnation_index === 1) {
                        wsiLampBox.className = "wsi-status-box stagnant-active";
                        displayWsi.textContent = "STAGNANT";
                    } else {
                        wsiLampBox.className = "wsi-status-box stagnant-inactive";
                        displayWsi.textContent = "VENTILATED";
                    }
                }

                // Render Z3 Guardrail Audit Dashboard and Clamping details
                updateAuditPanel(data.z3_audit, data.predictions);
            } else if (data.error) {
                alert(`Error: ${data.error}`);
            }
        } catch (err) {
            console.error("Error executing prediction API:", err);
            alert("Failed to communicate with prediction server.");
        } finally {
            btnSubmit.disabled = false;
            btnSubmit.innerHTML = `<span>Execute Flight Audit Simulation</span> <i class="fa-solid fa-arrow-right"></i>`;
        }
    }

    function updateAuditPanel(audit, predictions) {
        // Clear lists
        violationsList.innerHTML = "";
        if (clampingList) clampingList.innerHTML = "";
        
        if (audit.status === "SAT") {
            z3AuditCard.className = "glass-card status-display-card sat-state";
            auditBadge.textContent = "SAT";
            auditStatusLarge.textContent = "SATISFIED";
            auditDescription.textContent = "Predictions comply with all deterministic meteorological physical axioms. No corrections applied.";
            violationsContainer.classList.add("hidden");
            if (clampingContainer) clampingContainer.classList.add("hidden");
            
            // Header system pill
            systemStatusIndicator.className = "status-pill sat";
            systemStatusIndicator.innerHTML = `<span class="dot"></span> System Ready`;
        } else {
            z3AuditCard.className = "glass-card status-display-card unsat-state";
            auditBadge.textContent = "UNSAT";
            auditStatusLarge.textContent = "CORRECTED";
            auditDescription.textContent = "Meteorological violations detected in raw models. Real-time safety filters applied physics-safe clamping limits.";
            
            // Populate violations list
            violationsContainer.classList.remove("hidden");
            audit.violated_rules.forEach(rule => {
                const li = document.createElement("li");
                li.innerHTML = `<i class="fa-solid fa-circle-exclamation" style="margin-right:0.3rem"></i> ${rule}`;
                violationsList.appendChild(li);
            });

            // Populate Z3 Corrections comparison list
            let hasCorrections = false;
            if (predictions && predictions.random_forest && predictions.z3_verified) {
                const rf = predictions.random_forest;
                const z3 = predictions.z3_verified;
                for (let h = 0; h < rf.length; h++) {
                    if (Math.abs(rf[h] - z3[h]) > 0.5) {
                        hasCorrections = true;
                        const div = document.createElement("div");
                        div.className = "clamping-item";
                        div.innerHTML = `
                            <span class="clamping-horizon">t+${h+1}h Horizon</span>
                            <div class="clamping-values">
                                <span class="clamping-old">${rf[h]}m</span>
                                <span class="clamping-arrow"><i class="fa-solid fa-arrow-right-long"></i></span>
                                <span class="clamping-new">${z3[h]}m</span>
                            </div>
                        `;
                        clampingList.appendChild(div);
                    }
                }
            }

            if (hasCorrections && clampingContainer) {
                clampingContainer.classList.remove("hidden");
            } else if (clampingContainer) {
                clampingContainer.classList.add("hidden");
            }

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

    // --- Interactive Analytics Model Performance Comparisons ---
    function initMetricsChart(filteredData) {
        const modelOrder = ["Random Forest", "XGBoost", "Deep GRU", "Z3-Verified"];
        
        const precisionVals = [];
        const recallVals = [];
        const f1Vals = [];
        
        modelOrder.forEach(modelName => {
            const row = filteredData.find(r => r.Model === modelName);
            if (row) {
                precisionVals.push(parseFloat(row.Precision));
                recallVals.push(parseFloat(row["Recall (POD)"]));
                f1Vals.push(parseFloat(row["F1-Score"]));
            } else {
                precisionVals.push(0.0);
                recallVals.push(0.0);
                f1Vals.push(0.0);
            }
        });

        const ctxMetrics = document.getElementById("modelMetricsComparisonChart").getContext("2d");
        
        if (metricsChart) {
            metricsChart.destroy();
        }

        metricsChart = new Chart(ctxMetrics, {
            type: 'bar',
            data: {
                labels: modelOrder,
                datasets: [
                    {
                        label: 'Precision',
                        data: precisionVals,
                        backgroundColor: 'rgba(6, 182, 212, 0.75)',
                        borderColor: '#06b6d4',
                        borderWidth: 1.5,
                        borderRadius: 4
                    },
                    {
                        label: 'Recall (POD)',
                        data: recallVals,
                        backgroundColor: 'rgba(236, 72, 153, 0.75)',
                        borderColor: '#ec4899',
                        borderWidth: 1.5,
                        borderRadius: 4
                    },
                    {
                        label: 'F1-Score',
                        data: f1Vals,
                        backgroundColor: 'rgba(95, 63, 252, 0.75)',
                        borderColor: '#5f3ffc',
                        borderWidth: 1.5,
                        borderRadius: 4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            color: '#94a3b8',
                            font: { family: 'Outfit', size: 11, weight: 600 }
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(11, 13, 22, 0.95)',
                        titleFont: { family: 'Outfit', size: 12, weight: 600 },
                        bodyFont: { family: 'Inter', size: 11 },
                        borderColor: 'rgba(255, 255, 255, 0.08)',
                        borderWidth: 1
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.03)' },
                        ticks: { color: '#94a3b8', font: { family: 'Outfit', size: 11 } }
                    },
                    y: {
                        min: 0,
                        max: 1.0,
                        grid: { color: 'rgba(255, 255, 255, 0.03)' },
                        ticks: { color: '#94a3b8', font: { family: 'Outfit', size: 11 } },
                        title: {
                            display: true,
                            text: 'Operational Performance Score',
                            color: '#94a3b8',
                            font: { family: 'Outfit', size: 11, weight: 600 }
                        }
                    }
                }
            }
        });
    }

    function filterAndUpdateMetrics() {
        if (!globalMetricsData || globalMetricsData.length === 0) return;
        
        const threshVal = parseFloat(thresholdFilter.value);
        const horizVal = horizonFilter.value;

        // Filter the dataset rows matching threshold and horizon
        const filtered = globalMetricsData.filter(row => {
            const rowThresh = parseFloat(row.Threshold);
            return Math.abs(rowThresh - threshVal) < 0.1 && row.Horizon === horizVal;
        });

        initMetricsChart(filtered);
    }

    // Add filter change triggers
    if (thresholdFilter && horizonFilter) {
        thresholdFilter.addEventListener("change", filterAndUpdateMetrics);
        horizonFilter.addEventListener("change", filterAndUpdateMetrics);
    }

    // --- Load Research Calibration Metrics Table ---
    async function loadMetrics() {
        try {
            const res = await fetch("/api/metrics");
            const data = await res.json();
            
            if (data.metrics && data.metrics.length > 0) {
                // Cache metrics globally
                globalMetricsData = data.metrics;
                
                // Draw initial metrics comparison chart
                filterAndUpdateMetrics();

                // Clear initial table loading message
                metricsTableBody.innerHTML = "";
                
                // Populate metrics table rows
                data.metrics.forEach(row => {
                    const tr = document.createElement("tr");
                    
                    // Highlight the formal verified row to draw contrast
                    if (row.Model === "Z3-Verified") {
                        tr.className = "highlight-verified";
                    }

                    tr.innerHTML = `
                        <td><strong>${row.Horizon}</strong></td>
                        <td>&lt; ${row.Threshold}m</td>
                        <td style="font-weight: 700; color: ${row.Model === "Z3-Verified" ? "#10b981" : row.Model === "Deep GRU" ? "#ec4899" : row.Model === "XGBoost" ? "#f59e0b" : "#94a3b8"}">${row.Model}</td>
                        <td>${row.Precision.toFixed(4)}</td>
                        <td>${row["Recall (POD)"].toFixed(4)}</td>
                        <td>${row["False Alarm Ratio (FAR)"].toFixed(4)}</td>
                        <td>${row["False Alarm Rate (FPR)"].toFixed(4)}</td>
                        <td>${row["F1-Score"].toFixed(4)}</td>
                        <td>${row["Brier Score"].toFixed(4)}</td>
                        <td style="font-family: monospace; font-size: 0.78rem; color: #94a3b8">
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
            metricsTableBody.innerHTML = `<tr><td colspan="10" class="text-center">❌ Failed to retrieve evaluation statistics from server.</td></tr>`;
        }
    }

    // --- Tab Switching Logic ---
    const navTabs = document.querySelectorAll(".nav-tab");
    const tabContents = document.querySelectorAll(".tab-content");

    navTabs.forEach(tab => {
        tab.addEventListener("click", () => {
            const targetId = tab.getAttribute("data-target");

            // Update active state on tab buttons
            navTabs.forEach(t => t.classList.remove("active"));
            tab.classList.add("active");

            // Update active state on tab contents
            tabContents.forEach(content => {
                if (content.id === targetId) {
                    content.classList.add("active");
                } else {
                    content.classList.remove("active");
                }
            });

            // Prevent Chart.js layout collapse on hidden transitions
            if (targetId === "tab-simulator" && forecastChart) {
                setTimeout(() => forecastChart.resize(), 50);
            }
            if (targetId === "tab-performance" && metricsChart) {
                setTimeout(() => metricsChart.resize(), 50);
            }
        });
    });

    // --- Bootstrapping Execution ---
    fetchTimestamps();
    loadMetrics();
});
