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
        openaiRequest.model = "qwen-max-latest";
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

          console.log(`OpenAI API error: ${this.apiUrl} ${this.apiKey} ${response.status} ${response.statusText} - ${responseBody}`);
          throw OpenAIError.fromHttpStatus(response.status, response.statusText + " " + this.apiUrl + " " + this.apiKey, responseBody);
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
      openaiRequest.stream = true;
      openaiRequest.model = "qwen-max-latest";
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

      console.log(`Hello world wzd.`);

      if (!response.ok) {
        let responseBody: string;
        try {
          responseBody = await response.text();
        } catch {
          responseBody = '';
        }
        console.log(`OpenAI API error: ${this.apiUrl} ${this.apiKey} ${response.status} ${response.statusText} - ${responseBody}`);
        throw OpenAIError.fromHttpStatus(response.status, response.statusText + " " + openaiRequest.model + " wzd" + " " + this.apiUrl + " " + this.apiKey + " " + JSON.stringify(openaiRequest), responseBody);
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

    // 调试：打印 tools 转换前后
    console.log('[convertToOpenAIRequest] config.tools:', request.config?.tools);

    // Handle function calling if tools are present
    if (request.config?.tools && request.config.tools.length > 0) {
      const functions = this.convertToolsToFunctions(request.config.tools);
      console.log('[convertToOpenAIRequest] Converted functions:', functions);
      if (functions.length > 0) {
        openaiRequest.functions = functions;
        openaiRequest.function_call = 'auto';
      }
    }

    // Handle OpenAI-specific tools if no Gemini tools
    if (!openaiRequest.functions && openaiConfig.tools && openaiConfig.tools.length > 0) {
      openaiRequest.tools = openaiConfig.tools;
      console.log('[convertToOpenAIRequest] OpenAI-specific tools:', openaiConfig.tools);
    }

    // 调试：打印最终 openaiRequest
    console.log('[convertToOpenAIRequest] Final OpenAI request:', JSON.stringify(openaiRequest, null, 2));

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

  private convertToGeminiResponse(openaiResponse: any): GenerateContentResponse {
    const choice = openaiResponse.choices?.[0];
    if (!choice) {
      throw new Error('No choices in OpenAI response');
    }

    console.log('[OpenAIContentGenerator] Raw OpenAI response:', JSON.stringify(openaiResponse, null, 2));

    const parts: Part[] = [];
    
    // 调试：打印 message 内容
    console.log('[convertToGeminiResponse] choice.message:', choice.message);

    if (choice.message?.content) {
      parts.push({ text: choice.message.content });
    }

    if (choice.message?.function_call) {
      console.log('[OpenAIContentGenerator] Detected function_call:', choice.message.function_call);
      let parsedArgs;
      try {
        parsedArgs = JSON.parse(choice.message.function_call.arguments || '{}');
      } catch (e) {
        console.warn('[convertToGeminiResponse] Failed to parse function_call arguments:', choice.message.function_call.arguments, e);
        parsedArgs = {};
      }
      parts.push({
        functionCall: {
          name: choice.message.function_call.name,
          args: parsedArgs,
        },
      });
    }

    // 调试 functionCalls 字段
    let functionCalls = undefined;
    if (choice.message?.function_call) {
      let parsedArgs;
      try {
        parsedArgs = JSON.parse(choice.message.function_call.arguments || '{}');
      } catch (e) {
        console.warn('[convertToGeminiResponse] Failed to parse function_call arguments for functionCalls:', choice.message.function_call.arguments, e);
        parsedArgs = {};
      }
      functionCalls = [{
        id: choice.message.id || 'openai-func-' + Date.now(),
        name: choice.message.function_call.name,
        args: parsedArgs,
      }];
      console.log('[OpenAIContentGenerator] Gemini functionCalls:', functionCalls);
    }

    // 调试 thoughts 相关内容
    if (choice.message?.content && choice.message.content.startsWith('Thought:')) {
      parts.push({
        thought: true,
        text: choice.message.content,
      });
      console.log('[OpenAIContentGenerator] Detected thought:', choice.message.content);
    }

    // 调试 candidates 结构
    const candidates = [
      {
        content: {
          role: 'model',
          parts,
        },
        finishReason: this.mapFinishReason(choice.finish_reason),
      },
    ];
    console.log('[convertToGeminiResponse] candidates:', JSON.stringify(candidates, null, 2));

    return {
      text: "", // 根据实际响应设置
      functionCalls,
      executableCode: undefined,
      codeExecutionResult: undefined,
      data: undefined,
      candidates,
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

    console.log('[OpenAIContentGenerator] Streaming chunk:', JSON.stringify(chunk, null, 2));

    const parts: Part[] = [];
    
    if (delta.content) {
      parts.push({ text: delta.content });
    }

    if (delta.function_call) {
      console.log('[OpenAIContentGenerator] Streaming function_call:', delta.function_call);
      let parsedArgs;
      try {
        parsedArgs = JSON.parse(delta.function_call.arguments || '{}');
      } catch (e) {
        console.warn('[convertStreamChunkToGemini] Failed to parse function_call arguments:', delta.function_call.arguments, e);
        parsedArgs = {};
      }
      parts.push({
        functionCall: {
          name: delta.function_call.name,
          args: parsedArgs,
        },
      });
    }

    // 调试 functionCalls 字段
    let functionCalls = undefined;
    if (delta.function_call) {
      let parsedArgs;
      try {
        parsedArgs = JSON.parse(delta.function_call.arguments || '{}');
      } catch (e) {
        console.warn('[convertStreamChunkToGemini] Failed to parse function_call arguments for functionCalls:', delta.function_call.arguments, e);
        parsedArgs = {};
      }
      functionCalls = [{
        id: delta.id || 'openai-func-' + Date.now(),
        name: delta.function_call.name,
        args: parsedArgs,
      }];
      console.log('[convertStreamChunkToGemini] Streaming Gemini functionCalls:', functionCalls);
    }

    // 调试 thoughts 相关内容
    if (delta.content && delta.content.startsWith('Thought:')) {
      parts.push({
        thought: true,
        text: delta.content,
      });
      console.log('[convertStreamChunkToGemini] Streaming thought:', delta.content);
    }

    if (parts.length === 0) return null;

    // 调试 candidates 结构
    const candidates = [
      {
        content: {
          role: 'model',
          parts,
        },
        finishReason: chunk.choices?.[0]?.finish_reason 
          ? this.mapFinishReason(chunk.choices[0].finish_reason)
          : undefined,
      },
    ];
    console.log('[convertStreamChunkToGemini] candidates:', JSON.stringify(candidates, null, 2));

    return {
      text: "",
      functionCalls,
      executableCode: undefined,
      codeExecutionResult: undefined,
      data: undefined,
      candidates,
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
