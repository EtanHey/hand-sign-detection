'use client';

import { useState, useRef, useCallback } from 'react';
import { useCompletion } from 'ai/react';

interface DetectionResult {
  class: 'hand' | 'arm' | 'not_hand';
  confidence: number;
  probabilities: {
    hand: number;
    arm: number;
    not_hand: number;
  };
  isHand: boolean;
  isArm: boolean;
}

export function HandDetector() {
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isWebcamActive, setIsWebcamActive] = useState(false);

  // Use Vercel AI SDK for streaming responses
  const { complete, completion, isLoading: isAiLoading } = useCompletion({
    api: '/api/ai-describe',
  });

  // Handle file upload
  const handleFileUpload = async (file: File) => {
    setIsLoading(true);

    // Preview image
    const reader = new FileReader();
    reader.onloadend = () => {
      setImagePreview(reader.result as string);
    };
    reader.readAsDataURL(file);

    // Send to detection API
    const formData = new FormData();
    formData.append('image', file);

    try {
      const response = await fetch('/api/detect-hand', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Detection failed');

      const data: DetectionResult = await response.json();
      setResult(data);

      // Generate AI description based on detection
      if (data.isHand) {
        await complete(
          `Describe what gesture or sign this hand might be making with ${(
            data.confidence * 100
          ).toFixed(1)}% confidence.`
        );
      }
    } catch (error) {
      console.error('Detection error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Start webcam
  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user' },
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsWebcamActive(true);
      }
    } catch (error) {
      console.error('Webcam error:', error);
    }
  };

  // Capture from webcam
  const captureFromWebcam = useCallback(() => {
    if (!videoRef.current) return;

    const canvas = document.createElement('canvas');
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    const ctx = canvas.getContext('2d');

    if (ctx) {
      ctx.drawImage(videoRef.current, 0, 0);
      canvas.toBlob((blob) => {
        if (blob) {
          const file = new File([blob], 'webcam-capture.jpg', {
            type: 'image/jpeg',
          });
          handleFileUpload(file);
        }
      }, 'image/jpeg');
    }
  }, []);

  // Stop webcam
  const stopWebcam = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      setIsWebcamActive(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h1 className="text-2xl font-bold mb-6">Hand Detection</h1>

        {/* Model Info */}
        <div className="mb-6 p-4 bg-blue-50 rounded-lg">
          <p className="text-sm text-blue-800">
            Model:{' '}
            <a
              href="https://huggingface.co/EtanHey/hand-detection-3class"
              target="_blank"
              rel="noopener noreferrer"
              className="underline"
            >
              EtanHey/hand-detection-3class
            </a>
          </p>
          <p className="text-xs text-blue-600 mt-1">
            YOLOv8 • 3 classes (hand, arm, not_hand) • 96% accuracy
          </p>
        </div>

        {/* Upload Section */}
        <div className="mb-6">
          <div className="flex gap-4">
            <button
              onClick={() => fileInputRef.current?.click()}
              className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
              disabled={isLoading}
            >
              Upload Image
            </button>

            {!isWebcamActive ? (
              <button
                onClick={startWebcam}
                className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
              >
                Use Webcam
              </button>
            ) : (
              <>
                <button
                  onClick={captureFromWebcam}
                  className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
                >
                  Capture
                </button>
                <button
                  onClick={stopWebcam}
                  className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600"
                >
                  Stop Webcam
                </button>
              </>
            )}
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) handleFileUpload(file);
            }}
            className="hidden"
          />
        </div>

        {/* Webcam Preview */}
        {isWebcamActive && (
          <div className="mb-6">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              className="w-full max-w-md rounded-lg"
            />
          </div>
        )}

        {/* Image Preview */}
        {imagePreview && !isWebcamActive && (
          <div className="mb-6">
            <img
              src={imagePreview}
              alt="Preview"
              className="max-w-md rounded-lg shadow"
            />
          </div>
        )}

        {/* Loading State */}
        {isLoading && (
          <div className="mb-6">
            <div className="animate-pulse">
              <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
              <div className="h-4 bg-gray-200 rounded w-1/2"></div>
            </div>
          </div>
        )}

        {/* Results */}
        {result && (
          <div className="space-y-4">
            {/* Main Result */}
            <div
              className={`p-4 rounded-lg ${
                result.isHand
                  ? 'bg-green-100 border-green-500'
                  : result.isArm
                  ? 'bg-yellow-100 border-yellow-500'
                  : 'bg-gray-100 border-gray-500'
              } border-2`}
            >
              <h2 className="text-xl font-semibold mb-2">
                {result.class === 'hand' && '✋ Hand Detected'}
                {result.class === 'arm' && '💪 Arm Detected'}
                {result.class === 'not_hand' && '❌ No Hand/Arm'}
              </h2>
              <p className="text-lg">
                Confidence: {(result.confidence * 100).toFixed(1)}%
              </p>
            </div>

            {/* Probability Breakdown */}
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="font-semibold mb-2">All Probabilities:</h3>
              <div className="space-y-2">
                {Object.entries(result.probabilities).map(([cls, prob]) => (
                  <div key={cls} className="flex justify-between">
                    <span className="capitalize">{cls.replace('_', ' ')}:</span>
                    <div className="flex items-center gap-2">
                      <div className="w-32 bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-blue-500 h-2 rounded-full"
                          style={{ width: `${prob * 100}%` }}
                        />
                      </div>
                      <span className="text-sm">
                        {(prob * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* AI Description (if hand detected) */}
            {result.isHand && completion && (
              <div className="bg-blue-50 p-4 rounded-lg">
                <h3 className="font-semibold mb-2">AI Analysis:</h3>
                <p className="text-gray-700">{completion}</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}