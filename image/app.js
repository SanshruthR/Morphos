document.addEventListener('DOMContentLoaded', async () => {
    const classifier = new ImageClassifier();
    let currentClassId = null;
    let webcam = null;
    const colors = ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff'];

    // DOM elements
    const classesContainer = document.getElementById('classesContainer');
    const trainBtn = document.getElementById('trainBtn');
    const trainingSection = document.querySelector('.training-section');
    const previewContent = document.getElementById('previewContent');
    const exportTab = document.getElementById('exportTab');
    const webcamModal = document.createElement('div');
    webcamModal.className = 'webcam-modal';
    webcamModal.id = 'webcamModal';
    webcamModal.innerHTML = `
        <div class="webcam-modal-content">
            <video id="webcamVideo" autoplay playsinline></video>
            <button id="captureButton">Capture</button>
            <button id="closeModal">Close</button>
        </div>
    `;
    document.body.appendChild(webcamModal);
    const webcamVideo = document.getElementById('webcamVideo');
    const captureButton = document.getElementById('captureButton');
    const closeModal = document.getElementById('closeModal');
    const fileInput = document.getElementById('fileInput');

    // Add hyperparameters directly to training section
    const hyperParams = document.createElement('div');
    hyperParams.className = 'hyper-params';
    hyperParams.style.padding = '15px';
    hyperParams.innerHTML = `
        <div class="option-group">
            <label>Epochs:</label>
            <input type="number" id="epochs" value="10" min="1" max="100">
        </div>
        <div class="option-group">
            <label>Batch Size:</label>
            <input type="number" id="batchSize" value="16" min="1" max="64">
        </div>
        <div class="option-group">
            <label>Learning Rate:</label>
            <input type="number" id="learningRate" value="0.001" min="0.0001" max="0.1" step="0.0001">
        </div>
    `;
    trainingSection.appendChild(hyperParams);

    // Add training status element
    const trainingStatus = document.createElement('div');
    trainingStatus.className = 'training-status';
    trainingStatus.style.padding = '10px';
    trainingStatus.style.textAlign = 'center';
    trainingSection.appendChild(trainingStatus);

    // Load classifier
    await classifier.load();

    // Preview webcam setup
    const previewWebcamVideo = document.createElement('video');
    previewWebcamVideo.id = 'preview-webcam';
    previewWebcamVideo.autoplay = true;
    previewWebcamVideo.playsinline = true;
    previewWebcamVideo.style.width = '224px';
    previewWebcamVideo.style.height = '224px';
    previewContent.appendChild(previewWebcamVideo);
    const previewWebcam = new Webcam(previewWebcamVideo);
    await previewWebcam.setup();

    // Add initial classes
    for (let i = 0; i < 2; i++) {
        addNewClass();
    }

    // Add "Add a class" button
    const addClassBtn = document.createElement('button');
    addClassBtn.className = 'add-class-btn';
    addClassBtn.innerHTML = '<i class="bi bi-plus-lg add-icon"></i> Add a class';
    addClassBtn.addEventListener('click', addNewClass);
    classesContainer.appendChild(addClassBtn);

    // Train button
    trainBtn.addEventListener('click', async () => {
        const epochs = parseInt(document.getElementById('epochs').value) || 10;
        const batchSize = parseInt(document.getElementById('batchSize').value) || 16;
        const learningRate = parseFloat(document.getElementById('learningRate').value) || 0.001;

        trainBtn.disabled = true;
        trainBtn.textContent = 'Training...';
        trainingStatus.textContent = 'Preparing training data...';

        const success = await classifier.train(epochs, batchSize, learningRate, (epoch, totalEpochs) => {
            trainingStatus.textContent = `Training: Epoch ${epoch + 1} of ${totalEpochs}`;
        });

        if (success) {
            trainingStatus.textContent = 'Training completed!';
            const classLabels = classifier.getClassLabels();
            const predictionBars = document.createElement('div');
            predictionBars.className = 'prediction-bars';
            classLabels.forEach((classId, index) => {
                const barDiv = document.createElement('div');
                barDiv.className = 'prediction-bar';
                const color = colors[index % colors.length];
                barDiv.innerHTML = `
                    <span class="class-name">${classId}</span>
                    <div class="bar-container">
                        <div id="prediction-bar-${index}" class="bar" style="width: 0%; background-color: ${color}"></div>
                    </div>
                    <span id="prediction-value-${index}" class="prediction-value">0%</span>
                `;
                predictionBars.appendChild(barDiv);
            });
            previewContent.appendChild(predictionBars);
            setTimeout(() => trainingStatus.textContent = '', 3000); // Clear status after 3 seconds
        } else {
            trainingStatus.textContent = 'Training failed. Check console for details.';
            setTimeout(() => trainingStatus.textContent = '', 3000);
        }

        trainBtn.disabled = false;
        trainBtn.textContent = 'Train Model';
    });

    // Export model
    exportTab.addEventListener('click', () => classifier.export());

    // Start webcam for a class
    function startWebcamForClass(classId) {
        currentClassId = classId;
        webcamModal.style.display = 'flex';
        if (!webcam) {
            webcam = new Webcam(webcamVideo);
            webcam.setup().then(() => {
                console.log('Webcam ready for class:', classId);
            });
        }
    }

    // Capture image from webcam
    captureButton.addEventListener('click', () => {
        if (webcam && currentClassId) {
            const capture = webcam.capture();
            classifier.addExample(currentClassId, capture.tensor);
            addSampleToGrid(currentClassId, capture.imageData);
        }
    });

    // Close webcam modal
    closeModal.addEventListener('click', () => {
        webcamModal.style.display = 'none';
        if (webcam) {
            webcam.stop();
            webcam = null;
        }
    });

    // Trigger file upload for a class
    function triggerUploadForClass(classId) {
        currentClassId = classId;
        fileInput.click();
    }

    // Handle file upload
    fileInput.addEventListener('change', (event) => {
        const files = event.target.files;
        for (let file of files) {
            const reader = new FileReader();
            reader.onload = (e) => {
                const img = new Image();
                img.src = e.target.result;
                img.onload = () => {
                    const tensor = preprocessImage(img);
                    classifier.addExample(currentClassId, tensor);
                    addSampleToGrid(currentClassId, e.target.result);
                };
            };
            reader.readAsDataURL(file);
        }
    });

    // Preprocess image for upload
    function preprocessImage(img) {
        return tf.tidy(() => {
            const tensor = tf.browser.fromPixels(img);
            const resized = tf.image.resizeBilinear(tensor, [224, 224]);
            return resized.toFloat().div(tf.scalar(127.5)).sub(tf.scalar(1)).expandDims(0);
        });
    }

    // Add sample to grid
    function addSampleToGrid(classId, imageData) {
        const samplesGrid = document.getElementById(`samples-${classId}`);
        if (samplesGrid) {
            const img = document.createElement('img');
            img.src = imageData;
            img.className = 'sample-image';
            samplesGrid.appendChild(img);
        }
    }

    // Add new class
    function addNewClass() {
        const classId = `class-${Date.now()}`;
        const card = document.createElement('div');
        card.className = 'class-card';
        card.id = classId;
        card.innerHTML = `
            <div class="class-header">
                <div class="class-title">
                    <span class="class-name">${classId}</span>
                    <div class="class-icons">
                        <i class="bi bi-pencil edit-icon"></i>
                        <i class="bi bi-trash delete-icon"></i>
                    </div>
                </div>
                <div class="menu-dots">
                    <i class="bi bi-three-dots-vertical"></i>
                </div>
            </div>
            <div class="class-content">
                <div class="samples-label">Add Image Samples:</div>
                <div class="sample-buttons">
                    <button class="sample-button webcam-btn">
                        <i class="bi bi-camera-video sample-icon"></i>
                        <span class="sample-label">Webcam</span>
                    </button>
                    <button class="sample-button upload-btn">
                        <i class="bi bi-upload sample-icon"></i>
                        <span class="sample-label">Upload</span>
                    </button>
                </div>
                <div class="samples-grid" id="samples-${classId}"></div>
            </div>
        `;

        card.querySelector('.webcam-btn').addEventListener('click', () => startWebcamForClass(classId));
        card.querySelector('.upload-btn').addEventListener('click', () => triggerUploadForClass(classId));
        classesContainer.insertBefore(card, addClassBtn);
    }

    // Prediction loop for preview
    async function updatePrediction() {
        if (classifier.model && previewWebcam) {
            const capture = previewWebcam.capture();
            const predictions = await classifier.predict(capture.tensor);
            if (predictions) {
                for (let i = 0; i < predictions.length; i++) {
                    const percentage = (predictions[i] * 100).toFixed(0);
                    const bar = document.getElementById(`prediction-bar-${i}`);
                    const value = document.getElementById(`prediction-value-${i}`);
                    if (bar && value) {
                        bar.style.width = `${percentage}%`;
                        value.textContent = `${percentage}%`;
                    }
                }
            }
        }
        requestAnimationFrame(updatePrediction);
    }

    // Start prediction loop
    updatePrediction();
});