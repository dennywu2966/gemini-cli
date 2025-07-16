/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { OpenAIContentGenerator } from './openaiContentGenerator.js';
import { GenerateContentParameters, Type } from '@google/genai';

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe('OpenAIContentGenerator', () => {
  let generator: OpenAIContentGenerator;
  const mockApiKey = 'test-api-key';
  const mockApiUrl = 'https://test.api.url';

  beforeEach(() => {
    // Disable retries for most tests to avoid timeouts
    generator = new OpenAIContentGenerator(mockApiKey, mockApiUrl, {}, { retryConfig: { maxRetries: 0 }});
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.resetAllMocks();
  });

  describe('generateContent', () => {
    it('should successfully generate content with basic request', async () => {
      const mockOpenAIResponse = {
        choices: [
          {
            message: {
              content: 'Hello, this is a test response',
            },
            finish_reason: 'stop',
          },
        ],
        usage: {
          prompt_tokens: 10,
          completion_tokens: 8,
          total_tokens: 18,
        },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockOpenAIResponse),
        text: () => Promise.resolve(JSON.stringify(mockOpenAIResponse)),
      });

      const request: GenerateContentParameters = {
        model: 'openai-plus',
        contents: [
          {
            role: 'user',
            parts: [{ text: 'Hello, how are you?' }],
          },
        ],
      };

      const result = await generator.generateContent(request);

      expect(mockFetch).toHaveBeenCalledWith(
        `${mockApiUrl}/chat/completions`,
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${mockApiKey}`,
          }),
          body: expect.stringContaining('"model":"openai-plus"'),
        }),
      );

      expect(result.candidates).toHaveLength(1);
      expect(result.candidates?.[0]?.content?.parts?.[0]?.text).toBe(
        'Hello, this is a test response',
      );
      expect(result.usageMetadata?.totalTokenCount).toBe(18);
    });

    it('should handle function calling', async () => {
      const mockOpenAIResponse = {
        choices: [
          {
            message: {
              role: 'assistant',
              function_call: {
                name: 'test_function',
                arguments: '{"arg1":"value1"}',
              },
            },
            finish_reason: 'function_call',
          },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockOpenAIResponse),
        text: () => Promise.resolve(JSON.stringify(mockOpenAIResponse)),
      });

      const request: GenerateContentParameters = {
        model: 'openai-plus',
        contents: [{ role: 'user', parts: [{ text: 'Call a function' }] }],
        config: {
          tools: [
            {
              functionDeclarations: [
                {
                  name: 'test_function',
                  description: 'A test function',
                  parameters: {
                    type: Type.OBJECT,
                    properties: {
                      arg1: { type: Type.STRING },
                    },
                  },
                },
              ],
            },
          ],
        },
      };

      const response = await generator.generateContent(request);

      expect(response.candidates).toBeDefined();
      expect(response.candidates?.[0]?.content?.parts?.[0]?.functionCall).toEqual({
        name: 'test_function',
        args: { arg1: 'value1' },
      });
      expect(mockFetch.mock.calls[0][1]?.body).toContain(
        '"function_call":"auto"',
      );
    });

    it('should handle system instructions', async () => {
      const mockOpenAIResponse = {
        choices: [
          { message: { content: 'System response' }, finish_reason: 'stop' },
        ],
      };
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockOpenAIResponse),
        text: () => Promise.resolve(JSON.stringify(mockOpenAIResponse)),
      });

      const request: GenerateContentParameters = {
        model: 'openai-plus',
        contents: [{ role: 'user', parts: [{ text: 'User query' }] }],
        config: {
          systemInstruction: 'You are a helpful assistant.',
        },
      };

      await generator.generateContent(request);

      const sentBody = JSON.parse(mockFetch.mock.calls[0][1]?.body as string);
      expect(sentBody.messages[0]).toEqual({
        role: 'system',
        content: 'You are a helpful assistant.',
      });
      expect(sentBody.messages[1]).toEqual({
        role: 'user',
        content: 'User query',
      });
    });

    it('should handle 429 Rate Limit error', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 429,
        statusText: 'Too Many Requests',
        json: () => Promise.resolve({}),
        text: () => Promise.resolve(''),
      });
      const request: GenerateContentParameters = {
        model: 'openai-plus',
        contents: [{ role: 'user', parts: [{ text: 'test' }] }],
      };
      await expect(generator.generateContent(request)).rejects.toThrow(
        'OpenAI API error (429): Too Many Requests',
      );
    });

    it('should handle 503 Service Unavailable error', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 503,
        statusText: 'Service Unavailable',
        json: () => Promise.resolve({}),
        text: () => Promise.resolve(''),
      });
      const request: GenerateContentParameters = {
        model: 'openai-plus',
        contents: [{ role: 'user', parts: [{ text: 'test' }] }],
      };
      await expect(generator.generateContent(request)).rejects.toThrow(
        'OpenAI API error (503): Service Unavailable',
      );
    });

    it('should include system instruction in request', async () => {
      const mockOpenAIResponse = {
        choices: [
          {
            message: { content: 'Response with system instruction' },
            finish_reason: 'stop',
          },
        ],
        usage: { prompt_tokens: 20, completion_tokens: 5, total_tokens: 25 },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockOpenAIResponse),
        text: () => Promise.resolve(JSON.stringify(mockOpenAIResponse)),
      });

      const request: GenerateContentParameters = {
        model: 'openai-plus',
        contents: [
          {
            role: 'user',
            parts: [{ text: 'Test message' }],
          },
        ],
        config: {
          systemInstruction: { text: 'You are a helpful assistant' },
        },
      };

      await generator.generateContent(request);

      const fetchCall = mockFetch.mock.calls[0];
      const requestBody = JSON.parse(fetchCall[1].body);

      expect(requestBody.messages).toHaveLength(2);
      expect(requestBody.messages[0]).toEqual({
        role: 'system',
        content: 'You are a helpful assistant',
      });
    });

    it('should throw error on API failure', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 401,
        statusText: 'Unauthorized',
        json: () => Promise.resolve({}),
        text: () => Promise.resolve(''),
      });

      const request: GenerateContentParameters = {
        model: 'openai-plus',
        contents: [
          {
            role: 'user',
            parts: [{ text: 'Hello' }],
          },
        ],
      };

      await expect(generator.generateContent(request)).rejects.toThrow(
        'OpenAI API error (401): Unauthorized',
      );
    });

    it('should throw error when no choices in response', async () => {
      const mockOpenAIResponse = {
        choices: [],
        usage: { prompt_tokens: 5, completion_tokens: 0, total_tokens: 5 },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockOpenAIResponse),
        text: () => Promise.resolve(JSON.stringify(mockOpenAIResponse)),
      });

      const request: GenerateContentParameters = {
        model: 'openai-plus',
        contents: [
          {
            role: 'user',
            parts: [{ text: 'Hello' }],
          },
        ],
      };

      await expect(generator.generateContent(request)).rejects.toThrow(
        'No choices in OpenAI response',
      );
    });
  });

  describe('generateContentStream', () => {
    it('should handle streaming response correctly', async () => {
      const streamData = [
        'data: {"choices":[{"delta":{"content":"Hello"}}]}',
        'data: {"choices":[{"delta":{"content":" world"}}]}',
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}','data: [DONE]'
      ];

      const mockReader = {
        read: vi
          .fn()
          .mockResolvedValueOnce({
            done: false,
            value: new TextEncoder().encode(streamData.join('\n\n')),
          })
          .mockResolvedValueOnce({ done: true }),
        releaseLock: vi.fn(),
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        body: {
          getReader: () => mockReader,
        },
      });

      const request: GenerateContentParameters = {
        model: 'openai-plus',
        contents: [
          {
            role: 'user',
            parts: [{ text: 'Hello' }],
          },
        ],
      };

      const stream = await generator.generateContentStream(request);
      const chunks = [];

      for await (const chunk of stream) {
        chunks.push(chunk);
      }

      expect(chunks).toHaveLength(2);
      expect(chunks[0].candidates?.[0]?.content?.parts?.[0]?.text).toBe('Hello');
      expect(chunks[1].candidates?.[0]?.content?.parts?.[0]?.text).toBe(' world');
      expect(mockReader.releaseLock).toHaveBeenCalled();
    });

    it('should handle streaming errors gracefully', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error',
        text: () => Promise.resolve(''),
      });

      const request: GenerateContentParameters = {
        model: 'openai-plus',
        contents: [
          {
            role: 'user',
            parts: [{ text: 'Hello' }],
          },
        ],
      };

      await expect(
        async () => {
          const stream = await generator.generateContentStream(request);
          // eslint-disable-next-line @typescript-eslint/no-unused-vars
          for await (const chunk of stream) {
            // This should throw before we get here
          }
        },
      ).rejects.toThrow('OpenAI API error (500): Internal Server Error');
    });
  });

  describe('countTokens', () => {
    it('should estimate token count correctly', async () => {
      const request = {
        model: 'openai-plus',
        contents: [
          {
            role: 'user',
            parts: [{ text: 'This is a test message with some text content' }],
          },
        ],
      };

      const result = await generator.countTokens(request);

      expect(result.totalTokens).toBeGreaterThan(0);
      // With 4 chars per token estimation, this should be around 11-12 tokens
      expect(result.totalTokens).toBeLessThan(20);
    });

    it('should return 0 for empty contents', async () => {
      const request = {
        model: 'openai-plus',
        contents: [],
      };

      const result = await generator.countTokens(request);

      expect(result.totalTokens).toBe(0);
    });
  });

  describe('embedContent', () => {
    it('should throw error for unsupported embedding', async () => {
      const request = {
        model: 'openai-plus',
        contents: ['test text'],
      };

      await expect(generator.embedContent(request)).rejects.toThrow(
        'Embedding not supported for OpenAI models',
      );
    });
  });

  describe('getTier', () => {
    it('should return undefined for tier information', async () => {
      const result = await generator.getTier();
      expect(result).toBeUndefined();
    });
  });

  describe('request conversion', () => {
    it('should convert generation config correctly', async () => {
      const mockOpenAIResponse = {
        choices: [
          {
            message: { content: 'Test response' },
            finish_reason: 'stop',
          },
        ],
        usage: { prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockOpenAIResponse),
        text: () => Promise.resolve(JSON.stringify(mockOpenAIResponse)),
      });

      const request: GenerateContentParameters = {
        model: 'openai-plus',
        contents: [
          {
            role: 'user',
            parts: [{ text: 'Test' }],
          },
        ],
        config: {
          temperature: 0.7,
          topP: 0.9,
          maxOutputTokens: 2000,
        },
      };

      await generator.generateContent(request);

      const fetchCall = mockFetch.mock.calls[0];
      const requestBody = JSON.parse(fetchCall[1].body);

      expect(requestBody.temperature).toBe(0.7);
      expect(requestBody.top_p).toBe(0.9);
      expect(requestBody.max_tokens).toBe(2000);
    });

    it('should handle multiple content roles correctly', async () => {
      const mockOpenAIResponse = {
        choices: [
          {
            message: { content: 'Multi-turn response' },
            finish_reason: 'stop',
          },
        ],
        usage: { prompt_tokens: 15, completion_tokens: 3, total_tokens: 18 },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockOpenAIResponse),
        text: () => Promise.resolve(JSON.stringify(mockOpenAIResponse)),
      });

      const request: GenerateContentParameters = {
        model: 'openai-plus',
        contents: [
          {
            role: 'user',
            parts: [{ text: 'Hello' }],
          },
          {
            role: 'model',
            parts: [{ text: 'Hi there!' }],
          },
          {
            role: 'user',
            parts: [{ text: 'How are you?' }],
          },
        ],
      };

      await generator.generateContent(request);

      const fetchCall = mockFetch.mock.calls[0];
      const requestBody = JSON.parse(fetchCall[1].body);

      expect(requestBody.messages).toHaveLength(3);
      expect(requestBody.messages[0].role).toBe('user');
      expect(requestBody.messages[1].role).toBe('assistant');
      expect(requestBody.messages[2].role).toBe('user');
    });
  });

  describe('finish reason mapping', () => {
    it('should map finish reasons correctly', async () => {
      const testCases = [
        { openaiReason: 'stop', expectedGeminiReason: 'STOP' },
        { openaiReason: 'length', expectedGeminiReason: 'MAX_TOKENS' },
        { openaiReason: 'function_call', expectedGeminiReason: 'STOP' },
        { openaiReason: 'unknown', expectedGeminiReason: 'OTHER' },
      ];

      for (const { openaiReason, expectedGeminiReason } of testCases) {
        const mockOpenAIResponse = {
          choices: [
            {
              message: { content: 'Test response' },
              finish_reason: openaiReason,
            },
          ],
          usage: { prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 },
        };

        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(mockOpenAIResponse),
          text: () => Promise.resolve(JSON.stringify(mockOpenAIResponse)),
        });

        const request: GenerateContentParameters = {
          model: 'openai-plus',
          contents: [
            {
              role: 'user',
              parts: [{ text: 'Test' }],
            },
          ],
        };

        const result = await generator.generateContent(request);

        expect(result.candidates![0].finishReason).toBe(expectedGeminiReason);
      }
    });
  });
});