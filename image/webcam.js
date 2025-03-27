class Webcam {
    constructor(videoElement) {
        this.videoElement = videoElement;
        this.stream = null;
    }

    async setup() {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 224 }, height: { ideal: 224 } },
                audio: false
            });
            this.videoElement.srcObject = this.stream;
            return new Promise((resolve) => {
                this.videoElement.onloadedmetadata = () => resolve();
            });
        } else {
            throw new Error('Webcam not supported');
        }
    }

    capture() {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = this.videoElement.videoWidth;
        canvas.height = this.videoElement.videoHeight;
        ctx.drawImage(this.videoElement, 0, 0, canvas.width, canvas.height);
        return {
            imageData: canvas.toDataURL('image/png'),
            tensor: tf.tidy(() => {
                const image = tf.browser.fromPixels(canvas);
                const resized = tf.image.resizeBilinear(image, [224, 224]);
                return resized.toFloat().div(tf.scalar(127.5)).sub(tf.scalar(1)).expandDims(0);
            })
        };
    }

    stop() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
            this.videoElement.srcObject = null;
        }
    }
}