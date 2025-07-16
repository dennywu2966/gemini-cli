/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { AuthType } from '@google/gemini-cli-core';
import { loadEnvironment } from './settings.js';

export const validateAuthMethod = (authMethod: string): string | null => {
  loadEnvironment();
  if (
    authMethod === AuthType.LOGIN_WITH_GOOGLE ||
    authMethod === AuthType.CLOUD_SHELL
  ) {
    return null;
  }

  if (authMethod === AuthType.USE_GEMINI) {
    if (!process.env.GEMINI_API_KEY) {
      return 'GEMINI_API_KEY environment variable not found. Add that to your environment and try again (no reload needed if using .env)!';
    }
    return null;
  }

  if (authMethod === AuthType.USE_VERTEX_AI) {
    const hasVertexProjectLocationConfig =
      !!process.env.GOOGLE_CLOUD_PROJECT && !!process.env.GOOGLE_CLOUD_LOCATION;
    const hasGoogleApiKey = !!process.env.GOOGLE_API_KEY;
    if (!hasVertexProjectLocationConfig && !hasGoogleApiKey) {
      return (
        'When using Vertex AI, you must specify either:\n' +
        '• GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION environment variables.\n' +
        '• GOOGLE_API_KEY environment variable (if using express mode).\n' +
        'Update your environment and try again (no reload needed if using .env)!'
      );
    }
    return null;
  }

  if (authMethod === AuthType.USE_OPENAI) {
    if (!process.env.OPENAI_API_KEY) {
      return 'OPENAI_API_KEY environment variable not found. Add that to your environment and try again (no reload needed if using .env)!';
    }
    return null;
  }

  return 'Invalid auth method selected.';
};

export function getAuthOptions() {
  loadEnvironment();
  const options = [
    {
      label: 'Login with Google',
      value: AuthType.LOGIN_WITH_GOOGLE,
      description: 'Default for most users. Provides a generous free tier.',
      hint: null,
    },
    {
      label: 'Use Gemini API Key',
      value: AuthType.USE_GEMINI,
      description: 'Use your own API key for higher rate limits.',
      hint: process.env.GEMINI_API_KEY
        ? 'Existing API key detected (GEMINI_API_KEY). Select "Gemini API Key" option to use it.'
        : 'Set the GEMINI_API_KEY environment variable to use this option.',
    },
    {
      label: 'Vertex AI',
      value: AuthType.USE_VERTEX_AI,
      description: 'For Google Cloud users. Supports service accounts.',
      hint:
        process.env.GOOGLE_API_KEY ||
        (process.env.GOOGLE_CLOUD_PROJECT && process.env.GOOGLE_CLOUD_LOCATION)
          ? 'Existing Vertex AI configuration detected. Select "Vertex AI" option to use it.'
          : 'Set GOOGLE_API_KEY or GOOGLE_CLOUD_PROJECT/LOCATION to use this option.',
    },
    {
      label: 'OpenAI Compatible',
      value: AuthType.USE_OPENAI,
      description: 'Use an OpenAI-compatible API, like Alibaba Cloud.',
      hint: process.env.OPENAI_API_KEY
        ? 'Existing API key detected (OPENAI_API_KEY). Select this option to use it.'
        : 'Set the OPENAI_API_KEY environment variable to use this option.',
    },
  ];

  if (process.env.GOOGLE_CLOUD_SHELL === 'true') {
    options.unshift({
      label: 'Cloud Shell',
      value: AuthType.CLOUD_SHELL,
      description: 'Use your Cloud Shell credentials.',
      hint: null,
    });
  }

  return options;
}
