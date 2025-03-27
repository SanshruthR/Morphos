class ImageClassifier {
    constructor() {
        this.mobilenet = null;
        this.model = null;
        this.trainingData = {};
        this.labels = [];
    }

    async load() {
        this.mobilenet = await mobilenet.load({ version: 2, alpha: 0.5 });
    }

    addExample(classId, tensor) {
        if (!this.trainingData[classId]) {
            this.trainingData[classId] = [];
            this.labels.push(classId);
        }
        const features = tf.tidy(() => this.mobilenet.infer(tensor, true).squeeze());
        this.trainingData[classId].push(features);
    }

    async train(epochs = 10, batchSize = 16, learningRate = 0.001, onEpochUpdate = () => {}) {
        if (Object.keys(this.trainingData).length < 2) {
            alert('Add at least two classes with samples.');
            return false;
        }

        const xs = [];
        const ys = [];
        const numClasses = this.labels.length;

        this.labels.forEach((label, index) => {
            this.trainingData[label].forEach(feature => {
                xs.push(feature);
                ys.push(tf.oneHot(index, numClasses));
            });
        });

        const xsTensor = tf.stack(xs);
        const ysTensor = tf.stack(ys);

        this.model = tf.sequential({
            layers: [
                tf.layers.dense({ inputShape: [xs[0].shape[0]], units: 128, activation: 'relu' }),
                tf.layers.dropout({ rate: 0.2 }),
        
                tf.layers.dense({ units: 64, activation: 'relu' }),
                tf.layers.dropout({ rate: 0.2 }),
        
                tf.layers.dense({ units: 32, activation: 'relu' }),
                tf.layers.dropout({ rate: 0.2 }),
        
                tf.layers.dense({ units: 16, activation: 'relu' }),
                tf.layers.dropout({ rate: 0.2 }),
        
                tf.layers.dense({ units: numClasses, activation: 'softmax' }) // Output layer
            ]
        });
        
        this.model.compile({
            optimizer: tf.train.adam(learningRate),
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });

        await this.model.fit(xsTensor, ysTensor, {
            epochs,
            batchSize,
            shuffle: true,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    console.log(`Epoch ${epoch + 1}: Loss = ${logs.loss.toFixed(4)}, Accuracy = ${logs.acc.toFixed(4)}`);
                    onEpochUpdate(epoch, epochs); // Report current epoch and total epochs
                }
            }
        });

        xsTensor.dispose();
        ysTensor.dispose();
        return true;
    }

    async predict(tensor) {
        if (!this.model || !this.mobilenet) return null;
        const features = tf.tidy(() => this.mobilenet.infer(tensor, true).squeeze());
        const prediction = this.model.predict(features.expandDims(0));
        return prediction.dataSync();
    }

    getClassLabels() {
        return this.labels;
    }

    async export() {
        if (this.model) {
            await this.model.save('downloads://teachable-machine-model');
        } else {
            alert('No model trained yet.');
        }
    }
}