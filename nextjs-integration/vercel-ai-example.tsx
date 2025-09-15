'use client';

// Correct imports for Vercel AI SDK
import { useChat, useCompletion } from 'ai/react';
import { OpenAIStream, StreamingTextResponse } from 'ai';
import { useState } from 'react';

// For React components - useChat hook
export function ChatWithHandDetection() {
  const { messages, input, handleInputChange, handleSubmit, isLoading } = useChat({
    api: '/api/chat',
  });

  const [detectionResult, setDetectionResult] = useState<any>(null);

  // After detecting a hand, ask AI about it
  const analyzeGesture = async () => {
    if (detectionResult?.isHand) {
      // Automatically send a message about the detection
      await handleSubmit({
        preventDefault: () => {},
        currentTarget: {
          input: {
            value: `I detected a hand with ${detectionResult.confidence}% confidence. What gesture could this be?`
          }
        }
      } as any);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      {/* Chat messages */}
      <div className="flex flex-col gap-2">
        {messages.map(m => (
          <div key={m.id} className={`p-2 rounded ${
            m.role === 'user' ? 'bg-blue-100' : 'bg-gray-100'
          }`}>
            <strong>{m.role === 'user' ? 'You: ' : 'AI: '}</strong>
            {m.content}
          </div>
        ))}
      </div>

      {/* Input form */}
      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          value={input}
          onChange={handleInputChange}
          placeholder="Ask about hand gestures..."
          className="flex-1 p-2 border rounded"
          disabled={isLoading}
        />
        <button
          type="submit"
          disabled={isLoading}
          className="px-4 py-2 bg-blue-500 text-white rounded disabled:opacity-50"
        >
          {isLoading ? 'Thinking...' : 'Send'}
        </button>
      </form>

      {/* Analyze detection button */}
      {detectionResult?.isHand && (
        <button
          onClick={analyzeGesture}
          className="px-4 py-2 bg-green-500 text-white rounded"
        >
          Ask AI about this gesture
        </button>
      )}
    </div>
  );
}

// For API routes - streaming response
// app/api/chat/route.ts
import { OpenAI } from 'openai';
import { OpenAIStream, StreamingTextResponse } from 'ai';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

export const runtime = 'edge';

export async function POST(req: Request) {
  const { messages } = await req.json();

  // Create chat completion
  const response = await openai.chat.completions.create({
    model: 'gpt-4',
    stream: true,
    messages: [
      {
        role: 'system',
        content: 'You are an expert at interpreting hand gestures and sign language.'
      },
      ...messages
    ],
  });

  // Convert to stream
  const stream = OpenAIStream(response);

  // Return streaming response
  return new StreamingTextResponse(stream);
}

// For completion (non-chat) use case
export function CompletionExample() {
  const { complete, completion, isLoading } = useCompletion({
    api: '/api/completion',
  });

  const handleDetection = async (detectionResult: any) => {
    if (detectionResult.isHand) {
      // Generate a completion based on detection
      await complete(`The hand detection model found a ${detectionResult.class} with ${detectionResult.confidence}% confidence. This gesture likely represents`);
    }
  };

  return (
    <div>
      <button onClick={() => handleDetection({ isHand: true, class: 'hand', confidence: 95 })}>
        Analyze Gesture
      </button>

      {isLoading && <p>Analyzing...</p>}

      {completion && (
        <div className="mt-4 p-4 bg-gray-100 rounded">
          <p>{completion}</p>
        </div>
      )}
    </div>
  );
}

// Installation commands:
// npm install ai openai
// npm install @vercel/ai-sdk

// Environment variables needed:
// OPENAI_API_KEY=sk-...

// Package.json dependencies:
/*
{
  "dependencies": {
    "ai": "^3.0.0",
    "openai": "^4.0.0",
    "react": "^18.0.0",
    "next": "^14.0.0"
  }
}
*/