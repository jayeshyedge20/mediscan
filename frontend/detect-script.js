// MediScan AI - Enhanced Multi-Mode Diagnostic Engine Script
document.addEventListener('DOMContentLoaded', function() {
    const API_URL = window.location.origin.includes('localhost') || window.location.origin.includes('127.0.0.1') 
        ? window.location.origin 
        : 'http://localhost:8000';

    let currentMode = 'fracture'; // 'fracture' | 'disease' | 'comprehensive'
    let currentFile = null;

    // DOM Elements
    const modeTabs = document.querySelectorAll('.mode-tab');
    const samplesGrid = document.getElementById('samples-grid');
    const dropArea = document.getElementById('drop-area');
    const browseBtn = document.getElementById('browse-btn');
    const fileInput = document.getElementById('file-input');
    const filePreview = document.getElementById('file-preview');
    const previewImage = document.getElementById('preview-image');
    const detectBtn = document.getElementById('detect-button');
    const detectBtnText = document.getElementById('detect-btn-text');
    const cancelPreviewBtn = document.getElementById('cancel-preview-btn');
    const loadingBox = document.getElementById('loading');
    const progressBar = document.getElementById('progress-bar');
    const progressInfo = document.getElementById('progress-info');
    const resultsArea = document.getElementById('results-area');
    const resetBtn = document.getElementById('reset-btn');

    // Visual result cards
    const resOrigImg = document.getElementById('res-orig-img');
    const resPlotImg = document.getElementById('res-plot-img');
    const resGradcamImg = document.getElementById('res-gradcam-img');
    const resDiseaseImg = document.getElementById('res-disease-img');
    const cardFractureResult = document.getElementById('card-fracture-result');
    const cardGradcam = document.getElementById('card-gradcam');
    const cardDiseaseHeatmap = document.getElementById('card-disease-heatmap');

    // Clinical cards
    const triageBanner = document.getElementById('triage-banner');
    const triageTitle = document.getElementById('triage-title');
    const triageSubtitle = document.getElementById('triage-subtitle');
    const triageBadgeContainer = document.getElementById('triage-badge-container');
    const clinicalFractureCard = document.getElementById('clinical-fracture-card');
    const clinicalDiseaseCard = document.getElementById('clinical-disease-card');
    const fractureSummaryContent = document.getElementById('fracture-summary-content');
    const detectionListItems = document.getElementById('detection-list-items');
    const diseaseSummaryContent = document.getElementById('disease-summary-content');
    const recommendationsList = document.getElementById('recommendations-list');

    // Initialize Mode Switcher
    modeTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            modeTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            currentMode = tab.getAttribute('data-mode');
            updateModeUI();
        });
    });

    function updateModeUI() {
        if (currentMode === 'fracture') {
            detectBtnText.textContent = 'Detect Bone Fractures';
        } else if (currentMode === 'disease') {
            detectBtnText.textContent = 'Analyze Bone Disease & Pathology';
        } else {
            detectBtnText.textContent = 'Run Comprehensive Full Scan';
        }
    }

    // Load Samples Gallery
    fetchSamples();

    function fetchSamples() {
        fetch(`${API_URL}/samples`)
            .then(res => res.json())
            .then(data => {
                renderSamples(data.samples || []);
            })
            .catch(err => {
                console.warn('Using default sample manifest', err);
                renderDefaultSamples();
            });
    }

    function renderDefaultSamples() {
        const defaultSamples = [
            { id: 'wrist_fracture', name: 'Wrist Fracture', category: 'Fracture', url: '/sample_xrays/wrist_fracture.jpg' },
            { id: 'fingers_fracture', name: 'Hand / Finger', category: 'Fracture', url: '/sample_xrays/fingers_fracture.jpg' },
            { id: 'forearm_fracture', name: 'Forearm Fracture', category: 'Fracture', url: '/sample_xrays/forearm_fracture.jpg' },
            { id: 'elbow_fracture', name: 'Elbow Region', category: 'Fracture', url: '/sample_xrays/elbow_fracture.jpg' },
            { id: 'osteoarthritis_knee', name: 'Knee Osteoarthritis', category: 'Bone Disease', url: '/sample_xrays/osteoarthritis_knee.jpg' },
            { id: 'osteoporosis_spine', name: 'Spine Osteoporosis', category: 'Bone Disease', url: '/sample_xrays/osteoporosis_spine.jpg' },
            { id: 'bone_lesion', name: 'Bone Lesion / Tumor', category: 'Bone Disease', url: '/sample_xrays/bone_lesion.jpg' },
            { id: 'normal_bone', name: 'Normal Bone', category: 'Normal', url: '/sample_xrays/normal_bone.jpg' }
        ];
        renderSamples(defaultSamples);
    }

    function renderSamples(samples) {
        samplesGrid.innerHTML = '';
        samples.forEach(sample => {
            const card = document.createElement('div');
            card.className = 'sample-card';
            
            let badgeClass = 'badge-fracture';
            if (sample.category === 'Bone Disease') badgeClass = 'badge-disease';
            if (sample.category === 'Normal Baseline' || sample.category === 'Normal') badgeClass = 'badge-normal';

            card.innerHTML = `
                <img src="${sample.url}" alt="${sample.name}" onerror="this.src='images/wrist-xray.jpg'" />
                <div class="sample-name">${sample.name}</div>
                <span class="sample-badge ${badgeClass}">${sample.category}</span>
            `;

            card.addEventListener('click', () => {
                loadSampleImage(sample);
            });
            samplesGrid.appendChild(card);
        });
    }

    function loadSampleImage(sample) {
        fetch(sample.url)
            .then(res => res.blob())
            .then(blob => {
                currentFile = new File([blob], sample.filename || `${sample.id}.jpg`, { type: 'image/jpeg' });
                previewImage.src = URL.createObjectURL(currentFile);
                dropArea.style.display = 'none';
                filePreview.style.display = 'flex';
                
                // Adjust active mode based on sample category
                if (sample.category === 'Bone Disease' && currentMode === 'fracture') {
                    setMode('disease');
                } else if (sample.category === 'Fracture' && currentMode === 'disease') {
                    setMode('fracture');
                }
                
                // Auto-trigger scan on 1-click test
                runAnalysis();
            })
            .catch(err => {
                console.error('Failed to load sample image:', err);
                alert('Could not load sample image. Please try uploading manually.');
            });
    }

    function setMode(mode) {
        currentMode = mode;
        modeTabs.forEach(t => {
            if (t.getAttribute('data-mode') === mode) {
                t.classList.add('active');
            } else {
                t.classList.remove('active');
            }
        });
        updateModeUI();
    }

    // Drag & Drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(name => {
        dropArea.addEventListener(name, e => { e.preventDefault(); e.stopPropagation(); }, false);
    });
    ['dragenter', 'dragover'].forEach(name => dropArea.addEventListener(name, () => dropArea.classList.add('highlight')));
    ['dragleave', 'drop'].forEach(name => dropArea.addEventListener(name, () => dropArea.classList.remove('highlight')));

    dropArea.addEventListener('drop', e => {
        const files = e.dataTransfer.files;
        if (files.length > 0) handleFileSelection(files[0]);
    });

    browseBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', e => {
        if (e.target.files.length > 0) handleFileSelection(e.target.files[0]);
    });

    function handleFileSelection(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload a valid radiographic image (JPG or PNG).');
            return;
        }
        currentFile = file;
        const reader = new FileReader();
        reader.onload = e => {
            previewImage.src = e.target.result;
            dropArea.style.display = 'none';
            filePreview.style.display = 'flex';
        };
        reader.readAsDataURL(file);
    }

    cancelPreviewBtn.addEventListener('click', resetStudio);
    resetBtn.addEventListener('click', resetStudio);

    function resetStudio() {
        currentFile = null;
        fileInput.value = '';
        resultsArea.style.display = 'none';
        filePreview.style.display = 'none';
        loadingBox.style.display = 'none';
        dropArea.style.display = 'block';
    }

    detectBtn.addEventListener('click', runAnalysis);

    function runAnalysis() {
        if (!currentFile) {
            alert('Please select an X-ray image first.');
            return;
        }

        // Show loading
        filePreview.style.display = 'none';
        resultsArea.style.display = 'none';
        loadingBox.style.display = 'block';
        progressBar.style.width = '10%';
        progressInfo.textContent = 'Uploading radiograph...';

        let endpoint = `${API_URL}/detect`;
        if (currentMode === 'disease') endpoint = `${API_URL}/detect-disease`;
        if (currentMode === 'comprehensive') endpoint = `${API_URL}/comprehensive-scan`;

        const formData = new FormData();
        formData.append('file', currentFile);

        let progress = 15;
        const progressTimer = setInterval(() => {
            progress += Math.random() * 15;
            if (progress > 90) progress = 90;
            progressBar.style.width = `${progress}%`;
            progressInfo.textContent = `Analyzing osseous structures... ${Math.round(progress)}%`;
        }, 180);

        fetch(endpoint, {
            method: 'POST',
            body: formData
        })
        .then(res => {
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            return res.json();
        })
        .then(data => {
            clearInterval(progressTimer);
            progressBar.style.width = '100%';
            progressInfo.textContent = 'Diagnostics finalized!';
            setTimeout(() => {
                loadingBox.style.display = 'none';
                displayResults(data);
            }, 400);
        })
        .catch(err => {
            clearInterval(progressTimer);
            loadingBox.style.display = 'none';
            filePreview.style.display = 'flex';
            console.error('Diagnostic error:', err);
            alert(`Analysis failed: ${err.message}. Ensure the backend server is running at ${API_URL}`);
        });
    }

    function displayResults(data) {
        resultsArea.style.display = 'block';

        // Set original image
        resOrigImg.src = URL.createObjectURL(currentFile);

        if (data.mode === 'fracture') {
            displayFractureResults(data);
        } else if (data.mode === 'disease') {
            displayDiseaseResults(data);
        } else if (data.mode === 'comprehensive') {
            displayComprehensiveResults(data);
        }
    }

    function displayFractureResults(data) {
        cardFractureResult.style.display = 'block';
        cardGradcam.style.display = 'block';
        cardDiseaseHeatmap.style.display = 'none';
        clinicalFractureCard.style.display = 'block';
        clinicalDiseaseCard.style.display = 'none';

        resPlotImg.src = `${API_URL}${data.result_image}`;
        resGradcamImg.src = `${API_URL}${data.gradcam_image}`;

        const summary = data.clinical_summary;
        applyTriageBanner(summary.triage, summary.status, summary.severity);

        fractureSummaryContent.innerHTML = `
            <div class="metric-row">
                <span class="metric-label">Diagnosis Status:</span>
                <span class="metric-val" style="color: ${summary.severity === 'Severe' ? '#dc2626' : '#16a34a'}">${summary.status}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Detected Sites:</span>
                <span class="metric-val">${data.detections_count}</span>
            </div>
            <div style="margin-top: 1rem;">
                <h4 style="font-size: 0.95rem; color: #1e3a8a; margin-bottom: 0.5rem;">Radiographic Findings:</h4>
                <ul class="bullet-list">
                    ${summary.findings.map(f => `<li>${f}</li>`).join('')}
                </ul>
            </div>
        `;

        detectionListItems.innerHTML = '';
        if (data.detections && data.detections.length > 0) {
            data.detections.forEach(det => {
                const pill = document.createElement('div');
                pill.className = 'detection-pill';
                pill.innerHTML = `
                    <div>
                        <strong style="color: #0048c2;">#${det.id} ${det.formatted_class}</strong>
                        <div style="font-size: 0.8rem; color: #64748b;">Coords: [x:${det.box.x1}, y:${det.box.y1} to x:${det.box.x2}, y:${det.box.y2}]</div>
                    </div>
                    <span style="font-weight: 700; color: #0062ff;">${det.confidence}% Conf</span>
                `;
                detectionListItems.appendChild(pill);
            });
        }

        renderRecommendations(summary.recommendations);
    }

    function displayDiseaseResults(data) {
        cardFractureResult.style.display = 'none';
        cardGradcam.style.display = 'none';
        cardDiseaseHeatmap.style.display = 'block';
        clinicalFractureCard.style.display = 'none';
        clinicalDiseaseCard.style.display = 'block';

        resDiseaseImg.src = `${API_URL}${data.disease_heatmap}`;
        const report = data.report;
        applyTriageBanner(report.triage_level, report.primary_disease, report.severity);

        let probBars = '';
        for (const [disease, prob] of Object.entries(report.probabilities)) {
            probBars += `
                <div style="margin-bottom: 0.5rem;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 2px;">
                        <span>${disease}</span>
                        <strong>${prob}%</strong>
                    </div>
                    <div style="height: 6px; background: #e2e8f0; border-radius: 3px; overflow: hidden;">
                        <div style="width: ${prob}%; height: 100%; background: #0062ff;"></div>
                    </div>
                </div>
            `;
        }

        diseaseSummaryContent.innerHTML = `
            <div class="metric-row">
                <span class="metric-label">Primary Pathology:</span>
                <span class="metric-val" style="color: #0048c2; font-size: 1.1rem;">${report.primary_disease}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">AI Confidence:</span>
                <span class="metric-val">${report.confidence}%</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">KL Severity Grade:</span>
                <span class="metric-val">${report.kl_grade}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Bone Density Status:</span>
                <span class="metric-val">${report.bmd_status}</span>
            </div>
            <div style="margin-top: 1rem;">
                <h4 style="font-size: 0.95rem; color: #1e3a8a; margin-bottom: 0.5rem;">Radiographic Biomarkers:</h4>
                <div class="metric-row"><span class="metric-label">Radiodensity Score:</span><span class="metric-val">${report.biomarkers["Bone Radiodensity Score"]}</span></div>
                <div class="metric-row"><span class="metric-label">Joint Narrowing:</span><span class="metric-val">${report.biomarkers["Joint Narrowing Index"]}</span></div>
                <div class="metric-row"><span class="metric-label">Trabecular Heterogeneity:</span><span class="metric-val">${report.biomarkers["Trabecular Heterogeneity"]}</span></div>
            </div>
            <div style="margin-top: 1.25rem;">
                <h4 style="font-size: 0.95rem; color: #1e3a8a; margin-bottom: 0.5rem;">Pathology Probability Distribution:</h4>
                ${probBars}
            </div>
            <div style="margin-top: 1rem;">
                <h4 style="font-size: 0.95rem; color: #1e3a8a; margin-bottom: 0.5rem;">Radiological Findings:</h4>
                <ul class="bullet-list">
                    ${report.findings.map(f => `<li>${f}</li>`).join('')}
                </ul>
            </div>
        `;

        renderRecommendations(report.recommendations);
    }

    function displayComprehensiveResults(data) {
        cardFractureResult.style.display = 'block';
        cardGradcam.style.display = 'block';
        cardDiseaseHeatmap.style.display = 'block';
        clinicalFractureCard.style.display = 'block';
        clinicalDiseaseCard.style.display = 'block';

        resPlotImg.src = `${API_URL}${data.fracture_result_image}`;
        resGradcamImg.src = `${API_URL}${data.fracture_gradcam_image}`;
        resDiseaseImg.src = `${API_URL}${data.disease_heatmap_image}`;

        const fSum = data.fracture_analysis.summary;
        const dRep = data.disease_analysis;

        applyTriageBanner(data.master_triage, `Comprehensive Scan: ${fSum.status} | ${dRep.primary_disease}`, fSum.severity === 'Severe' || dRep.severity === 'Severe' ? 'Severe' : 'Moderate');

        // Fracture card
        fractureSummaryContent.innerHTML = `
            <div class="metric-row">
                <span class="metric-label">Fracture Status:</span>
                <span class="metric-val">${fSum.status}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Sites Detected:</span>
                <span class="metric-val">${data.fracture_analysis.count}</span>
            </div>
            <ul class="bullet-list" style="margin-top: 0.5rem;">
                ${fSum.findings.map(f => `<li>${f}</li>`).join('')}
            </ul>
        `;

        // Disease card
        diseaseSummaryContent.innerHTML = `
            <div class="metric-row">
                <span class="metric-label">Disease Assessment:</span>
                <span class="metric-val" style="color: #0048c2;">${dRep.primary_disease} (${dRep.confidence}%)</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Bone Density / KL:</span>
                <span class="metric-val">${dRep.bmd_status}</span>
            </div>
            <ul class="bullet-list" style="margin-top: 0.5rem;">
                ${dRep.findings.map(f => `<li>${f}</li>`).join('')}
            </ul>
        `;

        const combinedRecs = [...new Set([...fSum.recommendations, ...dRep.recommendations])];
        renderRecommendations(combinedRecs);
    }

    function applyTriageBanner(triage, title, severity) {
        triageBanner.className = 'triage-banner';
        if (severity === 'Severe' || triage.toLowerCase().includes('urgent')) {
            triageBanner.classList.add('triage-urgent');
            triageTitle.innerHTML = `<i class="fas fa-triangle-exclamation"></i> URGENT CLINICAL ATTENTION`;
        } else if (severity === 'Moderate' || triage.toLowerCase().includes('moderate')) {
            triageBanner.classList.add('triage-moderate');
            triageTitle.innerHTML = `<i class="fas fa-circle-info"></i> CLINICAL REVIEW RECOMMENDED`;
        } else {
            triageBanner.classList.add('triage-routine');
            triageTitle.innerHTML = `<i class="fas fa-circle-check"></i> ROUTINE / INTACT OSSEOUS FINDINGS`;
        }
        triageSubtitle.textContent = title;
    }

    function renderRecommendations(recs) {
        recommendationsList.innerHTML = '';
        (recs || []).forEach(r => {
            const li = document.createElement('li');
            li.textContent = r;
            recommendationsList.appendChild(li);
        });
    }
});