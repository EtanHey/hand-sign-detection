// app/api/detect-hand/route.ts
import { NextRequest, NextResponse } from 'next/server';

/**
 * Hand detection API route using Python backend
 * The Python service loads the model from HuggingFace
 */
export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const image = formData.get('image') as File;

    if (!image) {
      return NextResponse.json(
        { error: 'No image provided' },
        { status: 400 }
      );
    }

    // Forward to Python backend
    const backendFormData = new FormData();
    backendFormData.append('file', image);

    const response = await fetch(
      process.env.PYTHON_API_URL || 'http://localhost:8000/predict',
      {
        method: 'POST',
        body: backendFormData,
      }
    );

    if (!response.ok) {
      throw new Error('Detection failed');
    }

    const result = await response.json();

    // Format for frontend
    return NextResponse.json({
      class: result.class,
      confidence: result.confidence,
      probabilities: result.all_probs,
      isHand: result.class === 'hand',
      isArm: result.class === 'arm',
    });
  } catch (error) {
    console.error('Detection error:', error);
    return NextResponse.json(
      { error: 'Detection failed' },
      { status: 500 }
    );
  }
}

/**
 * Stream detection results using Vercel AI SDK
 */
export async function GET(request: NextRequest) {
  const encoder = new TextEncoder();

  // Create a streaming response
  const stream = new ReadableStream({
    async start(controller) {
      // Send initial message
      controller.enqueue(
        encoder.encode(
          `data: ${JSON.stringify({
            message: 'Hand detection model ready',
            modelUrl: 'https://huggingface.co/EtanHey/hand-detection-3class',
          })}\n\n`
        )
      );

      // Keep connection alive
      const interval = setInterval(() => {
        controller.enqueue(encoder.encode(': ping\n\n'));
      }, 30000);

      // Cleanup
      request.signal.addEventListener('abort', () => {
        clearInterval(interval);
        controller.close();
      });
    },
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
    },
  });
}