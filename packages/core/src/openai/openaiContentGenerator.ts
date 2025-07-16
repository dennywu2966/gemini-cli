/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  CountTokensResponse,
  GenerateContentResponse,
  GenerateContentParameters,
  CountTokensParameters,
  EmbedContentResponse,
  EmbedContentParameters,
  Content,
  PartUnion,
  Part,
  GenerateContentConfig,
  FinishReason,
} from '@google/genai';
import { ContentGenerator } from '../core/contentGenerator.js';
import { UserTierId } from '../code_assist/types.js';
import { 
  OpenAIError, 
  OpenAIErrorType, 
  RetryHandler, 
  RetryConfig, 
  DEFAULT_RETRY_CONFIG 
} from './openaiErrors.js';
import {
  ExtendedGenerateContentConfig,
  OpenAISpecificConfig,
  getMergedOpenAIConfig,
  openaiConfigToApiParams,
} from './openaiConfig.js';
import {
  OpenAIConfigValidator,
  validateEnvironmentOrThrow,
  validateConfigOrThrow,
} from './openaiValidator.js';

export interface OpenAIContentGeneratorConfig {
  timeout?: number;
  retryConfig?: Partial<RetryConfig>;
}

/**
 * Content generator that adapts OpenAI API to Gemini interface
 */
export class OpenAIContentGenerator implements ContentGenerator {
  private retryHandler: RetryHandler;
  private timeout: number;

  constructor(
    private apiKey: string,
    private apiUrl: string,
    private httpOptions: any = {},
    config: OpenAIContentGeneratorConfig = {},
  ) {
    // Validate environment configuration
    validateEnvironmentOrThrow({
      apiKey: this.apiKey,
      apiUrl: this.apiUrl,
      timeout: config.timeout,
    });

    this.timeout = config.timeout || 30000; // 30 seconds default
    this.retryHandler = new RetryHandler({
      ...DEFAULT_RETRY_CONFIG,
      ...config.retryConfig,
    });
  }

  async generateContent(
    request: GenerateContentParameters,
  ): Promise<GenerateContentResponse> {
    const abortController = new AbortController();
    const timeoutId = setTimeout(() => abortController.abort(), this.timeout);

    try {
      return await this.retryHandler.executeWithRetry(async () => {
        const openaiRequest = this.convertToOpenAIRequest(request);
        
        let response: Response;
        try {
          response = await fetch(`${this.apiUrl}/chat/completions`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${this.apiKey}`,
              ...this.httpOptions.headers,
            },
            body: JSON.stringify(openaiRequest),
            signal: abortController.signal,
          });
        } catch (error) {
          if (abortController.signal.aborted) {
            throw OpenAIError.fromTimeoutError();
          }
          throw OpenAIError.fromNetworkError(error);
        }

        if (!response.ok) {
          let responseBody: string;
          try {
            responseBody = await response.text();
          } catch {
            responseBody = '';
          }
          throw OpenAIError.fromHttpStatus(response.status, response.statusText, responseBody);
        }

        let openaiResponse: any;
        try {
          openaiResponse = await response.json();
        } catch (error) {
          throw new OpenAIError(
            OpenAIErrorType.API_ERROR,
            'Failed to parse JSON response from OpenAI API',
            response.status,
            false,
            error,
          );
        }

        return this.convertToGeminiResponse(openaiResponse);
      }, abortController.signal);
    } finally {
      clearTimeout(timeoutId);
    }
  }

  async generateContentStream(
    request: GenerateContentParameters,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    return this.generateContentStreamInternal(request);
  }

  private async *generateContentStreamInternal(
    request: GenerateContentParameters,
  ): AsyncGenerator<GenerateContentResponse> {
    const abortController = new AbortController();
    const timeoutId = setTimeout(() => abortController.abort(), this.timeout);

    try {
      const openaiRequest = {
        ...this.convertToOpenAIRequest(request),
        stream: true,
      };

      let response: Response;
      try {
        response = await fetch(`${this.apiUrl}/chat/completions`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${this.apiKey}`,
            ...this.httpOptions.headers,
          },
          body: JSON.stringify(openaiRequest),
          signal: abortController.signal,
        });
      } catch (error) {
        if (abortController.signal.aborted) {
          throw OpenAIError.fromTimeoutError();
        }
        throw OpenAIError.fromNetworkError(error);
      }

      if (!response.ok) {
        let responseBody: string;
        try {
          responseBody = await response.text();
        } catch {
          responseBody = '';
        }
        throw OpenAIError.fromHttpStatus(response.status, response.statusText, responseBody);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new OpenAIError(
          OpenAIErrorType.API_ERROR,
          'Failed to get response stream reader',
        );
      }

      const decoder = new TextDecoder();
      let buffer = '';

      try {
        while (true) {
          if (abortController.signal.aborted) {
            throw OpenAIError.fromTimeoutError();
          }

          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = line.slice(6);
              if (data === '[DONE]') {
                return;
              }
              try {
                const chunk = JSON.parse(data);
                const geminiChunk = this.convertStreamChunkToGemini(chunk);
                if (geminiChunk) {
                  yield geminiChunk;
                }
              } catch (e) {
                // Skip invalid JSON chunks but log warning
                console.warn('Failed to parse streaming chunk:', e);
              }
            }
          }
        }
      } finally {
        reader.releaseLock();
      }
    } finally {
      clearTimeout(timeoutId);
    }
  }

//  private ensureContentArray(input: Content[] | PartUnion[]): Content[] {
//    if (input.length === 0) return [];
//  
//    // 检查第一个元素的类型
//    if (typeof input[0] === 'string') {
//      return input.map(part => ({
//        // 根据你的实际 Content 结构调整
//        type: 'text',
//        text: part as string
//      }));
//    }
//  
//    return input as Content[];
//  }

  private ensureContentArray(input: Content[] | PartUnion[]): Content[] {
    // 1. 处理空数组
    if (input.length === 0) return [];
  
    // 2. 如果已经是 Content[] 类型，直接返回
    if (this.isContentArray(input)) {
      return input;
    }
  
    // 3. 处理 PartUnion[] 类型
    const parts: Part[] = [];
  
    for (const item of input) {
      if (typeof item === 'string') {
        // 字符串类型转换为 text Part
        parts.push({ text: item });
      } else {
        // Part 对象直接使用
        parts.push(item);
      }
    }
  
    // 4. 创建 Content 对象（使用默认角色）
    return [{
      role: 'user', // 默认角色，可根据需要调整
      parts
    }];
  }
  
  // 类型守卫函数：检查是否是 Content[]
  private isContentArray(arr: any): arr is Content[] {
    return Array.isArray(arr) &&
           arr.length > 0 &&
           typeof arr[0] === 'object' &&
           'role' in arr[0] &&
           'parts' in arr[0];
  }

  async countTokens(request: CountTokensParameters): Promise<CountTokensResponse> {
    const contents = Array.isArray(request.contents) ? request.contents : [];
    const finalContents = this.ensureContentArray(contents);
    // OpenAI doesn't have token counting API, so we estimate
    const text = this.extractTextFromContents(
      finalContents
    );
    const estimatedTokens = Math.ceil(text.length / 4); // Rough estimation: 4 chars per token
    
    return {
      totalTokens: estimatedTokens,
    };
  }

  async embedContent(request: EmbedContentParameters): Promise<EmbedContentResponse> {
    // OpenAI embedding API is different, would need specific implementation
    throw new Error('Embedding not supported for OpenAI models');
  }

  async getTier(): Promise<UserTierId | undefined> {
    return undefined;
  }

  private convertToOpenAIRequest(request: GenerateContentParameters): any {
    const contents = Array.isArray(request.contents) ? request.contents : [];
    const messages = this.convertContentsToMessages(this.ensureContentArray(contents));
    
    // Add system instruction if present
    if (request.config?.systemInstruction) {
      const systemParts = Array.isArray(request.config.systemInstruction) 
        ? request.config.systemInstruction 
        : [request.config.systemInstruction];
      const systemContent = this.extractTextFromParts(this.convertToParts(systemParts));
      if (systemContent) {
        messages.unshift({
          role: 'system',
          content: systemContent,
        });
      }
    }

    // Validate model
    const modelValidation = OpenAIConfigValidator.validateModel(request.model);
    if (modelValidation.warnings.length > 0) {
      console.warn('Model validation warnings:', modelValidation.warnings.join(', '));
    }

    // Get merged OpenAI configuration
    const extendedConfig = request.config as ExtendedGenerateContentConfig;
    const openaiConfig = getMergedOpenAIConfig(request.model, extendedConfig?.openai);
    
    // Validate the merged configuration
    validateConfigOrThrow(openaiConfig);

    const openaiRequest: any = {
      model: request.model,
      messages,
      temperature: request.config?.temperature || 0,
      top_p: request.config?.topP || 1,
      max_tokens: request.config?.maxOutputTokens || 4000,
    };

    // Add OpenAI-specific parameters
    const openaiParams = openaiConfigToApiParams(openaiConfig);
    Object.assign(openaiRequest, openaiParams);

    // Handle function calling if tools are present
    if (request.config?.tools && request.config.tools.length > 0) {
      const functions = this.convertToolsToFunctions(request.config.tools);
      if (functions.length > 0) {
        openaiRequest.functions = functions;
        openaiRequest.function_call = 'auto';
      }
    }

    // Handle OpenAI-specific tools if no Gemini tools
    if (!openaiRequest.functions && openaiConfig.tools && openaiConfig.tools.length > 0) {
      openaiRequest.tools = openaiConfig.tools;
    }

    return openaiRequest;
  }

  private convertContentsToMessages(contents: Content[]): any[] {
    return contents.map(content => ({
      role: content.role === 'model' ? 'assistant' : content.role,
      content: this.extractTextFromParts(content.parts || []),
    })).filter(msg => msg.content.trim());
  }

  private extractTextFromParts(parts: Part[]): string {
    return parts
      .map(part => {
        if (part.text) return part.text;
        if (part.functionCall) {
          return `[Function Call: ${part.functionCall.name}(${JSON.stringify(part.functionCall.args)})]`;
        }
        if (part.functionResponse) {
          return `[Function Response: ${JSON.stringify(part.functionResponse.response)}]`;
        }
        return '';
      })
      .join('\n')
      .trim();
  }

  private extractTextFromContents(contents: Content[]): string {
    return contents
      .map(content => this.extractTextFromParts(content.parts || []))
      .join('\n');
  }

  private convertToParts(input: (Content | PartUnion)[]): Part[] {
    // 1. 处理空输入
    if (input.length === 0) return [];
  
    // 2. 创建结果数组
    const result: Part[] = [];
  
    // 3. 遍历所有输入项
    for (const item of input) {
      if (this.isContent(item)) {
        // 处理 Content 类型
        if (item.parts && Array.isArray(item.parts)) {
          // 过滤掉 undefined 并确保所有元素都是有效的 Part
          const validParts = item.parts.filter(
            part => part !== null && part !== undefined
          ) as Part[];
  
          result.push(...validParts);
        }
      } else {
        // 处理 PartUnion 类型
        if (typeof item === 'string') {
          // 字符串转换为 text Part
          result.push({ text: item });
        } else if (item !== null && item !== undefined) {
          // 有效的 Part 对象直接添加
          result.push(item);
        }
      }
    }
  
    return result;
  }
  
  // 类型守卫：检查是否是 Content 对象
  private isContent(item: any): item is Content {
    return (
      typeof item === 'object' &&
      item !== null &&
      'role' in item &&
      'parts' in item
    );
  }

//  private convertToParts(input: Content[] | PartUnion[]): Part[] {
//    // 1. 处理空输入
//    if (input.length === 0) return [];
//  
//    // 2. 检查输入类型
//    const isContentArray = input.every(item =>
//      typeof item === 'object' &&
//      'role' in item &&
//      'parts' in item
//    );
//  
//    // 3. 处理 Content[] 类型
//    if (isContentArray) {
//      const contents = input as Content[];
//      // 提取所有 parts 并展平
//      return contents.flatMap(content => content.parts);
//    }
//  
//    // 4. 处理 PartUnion[] 类型
//    const parts = input as PartUnion[];
//    return parts.map(item => {
//      if (typeof item === 'string') {
//        // 将字符串转换为 text Part
//        return { text: item };
//      } else {
//        // 直接返回 Part 对象
//        return item;
//      }
//    });
//  }
//
  private convertToGeminiResponse(openaiResponse: any): GenerateContentResponse {
    const choice = openaiResponse.choices?.[0];
    if (!choice) {
      throw new Error('No choices in OpenAI response');
    }

    const parts: Part[] = [];
    
    if (choice.message?.content) {
      parts.push({ text: choice.message.content });
    }

    if (choice.message?.function_call) {
      parts.push({
        functionCall: {
          name: choice.message.function_call.name,
          args: JSON.parse(choice.message.function_call.arguments || '{}'),
        },
      });
    }

    return {
      text: "", // 根据实际响应设置
      functionCalls: undefined,
      executableCode: undefined,
      codeExecutionResult: undefined,
      data: undefined,
      candidates: [
        {
          content: {
            role: 'model',
            parts,
          },
          finishReason: this.mapFinishReason(choice.finish_reason),
        },
      ],
      usageMetadata: {
        promptTokenCount: openaiResponse.usage?.prompt_tokens,
        candidatesTokenCount: openaiResponse.usage?.completion_tokens,
        totalTokenCount: openaiResponse.usage?.total_tokens,
      },
    };
  }

  private convertStreamChunkToGemini(chunk: any): GenerateContentResponse | null {
    const delta = chunk.choices?.[0]?.delta;
    if (!delta) return null;

    const parts: Part[] = [];
    
    if (delta.content) {
      parts.push({ text: delta.content });
    }

    if (delta.function_call) {
      parts.push({
        functionCall: {
          name: delta.function_call.name,
          args: JSON.parse(delta.function_call.arguments || '{}'),
        },
      });
    }

    if (parts.length === 0) return null;

    return {
      text: "", // 根据实际响应设置
      functionCalls: undefined,
      executableCode: undefined,
      codeExecutionResult: undefined,
      data: undefined,
      candidates: [
        {
          content: {
            role: 'model',
            parts,
          },
          finishReason: chunk.choices?.[0]?.finish_reason 
            ? this.mapFinishReason(chunk.choices[0].finish_reason)
            : undefined,
        },
      ],
    };
  }

  private convertToolsToFunctions(tools: any[]): any[] {
    const functions: any[] = [];
    
    for (const tool of tools) {
      if (tool.functionDeclarations) {
        for (const func of tool.functionDeclarations) {
          functions.push({
            name: func.name,
            description: func.description,
            parameters: func.parameters,
          });
        }
      }
    }
    
    return functions;
  }

  private mapFinishReason(openaiReason: string): FinishReason {
    switch (openaiReason) {
      case 'stop':
        return FinishReason.STOP;
      case 'length':
        return FinishReason.MAX_TOKENS;
      case 'function_call':
        return FinishReason.STOP;
      default:
        return FinishReason.OTHER;
    }
  }
}
